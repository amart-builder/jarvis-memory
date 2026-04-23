const JARVIS_MEMORY_API = process.env.JARVIS_MEMORY_API ?? "http://localhost:3500";
const MAX_MESSAGES = Number.parseInt(process.env.JARVIS_HOOK_MAX_MESSAGES ?? "24", 10);
const DEDUPE_FILE = process.env.JARVIS_HOOK_DEDUPE_FILE ?? "/tmp/jarvis-openclaw-hook-cache.json";
const DRY_RUN = /^(1|true|yes)$/i.test(process.env.JARVIS_HOOK_DRY_RUN ?? "");

// Project routing — add patterns to route handoffs to named group_ids.
// Each entry: { id: "group-name", patterns: [/keyword/i, /keyword/i] }.
// Leave empty to send everything to `group_id=system`.
const PROJECT_PATTERNS: Array<{ id: string; patterns: RegExp[] }> = [
  // { id: "example-project", patterns: [/\bexample\b/i] },
];

function extractText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((part) => part && part.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("\n");
}

function normalizeWhitespace(value) {
  return value.replace(/\s+/g, " ").trim();
}

function getSessionMessages(event) {
  if (Array.isArray(event?.session?.messages)) return event.session.messages;
  return [];
}

function getSessionKey(event) {
  return event?.sessionKey ?? event?.session?.key ?? event?.session?.sessionKey ?? "unknown";
}

function summarizeText(value, max = 500) {
  const clean = normalizeWhitespace(value || "");
  return clean.length <= max ? clean : `${clean.slice(0, max - 1)}…`;
}

function isHeartbeatText(value) {
  const text = normalizeWhitespace(String(value || ""));
  if (!text) return true;
  if (text === "HEARTBEAT_OK") return true;
  if (/Read HEARTBEAT\.md if it exists/i.test(text)) return true;
  if (/When reading HEARTBEAT\.md, use workspace file/i.test(text)) return true;
  if (/Do not infer or repeat old tasks from prior chats/i.test(text)) return true;
  return false;
}

function getRelevantMessages(event) {
  return getSessionMessages(event)
    .filter((message) => message && (message.role === "user" || message.role === "assistant"))
    .map((message) => ({
      role: message.role,
      text: normalizeWhitespace(extractText(message.content || "")),
    }))
    .filter((message) => message.text)
    .slice(-MAX_MESSAGES);
}

function isHeartbeatOnly(messages) {
  return messages.length > 0 && messages.every((message) => isHeartbeatText(message.text));
}

function inferGroupId(messages) {
  const corpus = messages.map((message) => message.text).join("\n");
  const scored = PROJECT_PATTERNS.map((project) => ({
    id: project.id,
    score: project.patterns.reduce((total, pattern) => total + (pattern.test(corpus) ? 1 : 0), 0),
  })).sort((a, b) => b.score - a.score);

  if (!scored[0] || scored[0].score === 0) {
    return { groupId: "system", confidence: "low" };
  }

  if (scored[1] && scored[0].score === scored[1].score) {
    return { groupId: scored[0].id, confidence: "medium" };
  }

  return { groupId: scored[0].id, confidence: scored[0].score >= 2 ? "high" : "medium" };
}

function buildTranscript(messages) {
  return messages
    .map((message) => `${message.role === "user" ? "User" : "Assistant"}: ${summarizeText(message.text, 900)}`)
    .join("\n\n");
}

function getLastByRole(messages, role) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === role && !isHeartbeatText(messages[index].text)) {
      return messages[index].text;
    }
  }
  return "";
}

function buildContent({ trigger, sessionKey, groupId, confidence, messages }) {
  const recentUser = getLastByRole(messages, "user");
  const recentAssistant = getLastByRole(messages, "assistant");
  const transcript = buildTranscript(messages);

  return [
    `[COMPLETION] OpenClaw automatic handoff snapshot (${trigger})`,
    `WHY: Preserve cross-system continuity before compaction or session reset without depending on a manual handoff.`,
    `IMPACT: Claude and OpenClaw can recover recent state, task context, and next-step clues from Jarvis memory.`,
    "",
    `SESSION_KEY: ${sessionKey}`,
    `GROUP_ID: ${groupId}`,
    `GROUP_CONFIDENCE: ${confidence}`,
    recentUser ? `RECENT_USER_REQUEST: ${summarizeText(recentUser, 700)}` : null,
    recentAssistant ? `RECENT_ASSISTANT_STATE: ${summarizeText(recentAssistant, 700)}` : null,
    "",
    "TRANSCRIPT_EXCERPT:",
    transcript,
  ].filter(Boolean).join("\n");
}

async function sha256(value) {
  const crypto = await import("node:crypto");
  return crypto.createHash("sha256").update(value).digest("hex");
}

async function loadDedupeCache() {
  try {
    const fs = await import("node:fs/promises");
    const raw = await fs.readFile(DEDUPE_FILE, "utf8");
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

async function saveDedupeCache(cache) {
  const fs = await import("node:fs/promises");
  await fs.writeFile(DEDUPE_FILE, JSON.stringify(cache, null, 2));
}

async function shouldSkipDuplicate({ sessionKey, trigger, fingerprint }) {
  const cache = await loadDedupeCache();
  const key = `${sessionKey}:${trigger}`;
  const previous = cache[key];
  if (previous?.fingerprint === fingerprint) {
    return true;
  }
  cache[key] = { fingerprint, at: new Date().toISOString() };
  await saveDedupeCache(cache);
  return false;
}

async function postMemory(payload) {
  const response = await fetch(`${JARVIS_MEMORY_API}/api/v1/add`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(10000),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Jarvis write failed (${response.status}): ${body}`);
  }

  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

const handler = async (event) => {
  const trigger = event?.action ? `${event.type}:${event.action}` : String(event?.type ?? "unknown");
  const supported = trigger === "session:compact:before" || trigger === "command:new" || trigger === "command:reset";
  if (!supported) return;

  const sessionKey = getSessionKey(event);
  const messages = getRelevantMessages(event);

  if (messages.length === 0) {
    console.log(`[jarvis-handoff-enforcer] No user/assistant transcript for ${sessionKey}, skipping.`);
    return;
  }

  if (isHeartbeatOnly(messages)) {
    console.log(`[jarvis-handoff-enforcer] Heartbeat-only session ${sessionKey}, skipping.`);
    return;
  }

  const { groupId, confidence } = inferGroupId(messages);
  const content = buildContent({ trigger, sessionKey, groupId, confidence, messages });
  const fingerprint = await sha256(`${trigger}\n${sessionKey}\n${content}`);

  if (await shouldSkipDuplicate({ sessionKey, trigger, fingerprint })) {
    console.log(`[jarvis-handoff-enforcer] Duplicate snapshot for ${sessionKey} (${trigger}), skipping.`);
    return;
  }

  const payload = {
    content,
    agent_id: "openclaw-hook",
    memory_type: "meta",
    group_id: groupId,
    metadata: {
      source: "hook:jarvis-handoff-enforcer",
      trigger,
      session_key: sessionKey,
      group_confidence: confidence,
      fingerprint,
      workspace: process.cwd(),
      message_count: messages.length,
      captured_at: new Date().toISOString(),
    },
  };

  if (DRY_RUN) {
    console.log(`[jarvis-handoff-enforcer] Dry run payload for ${sessionKey}:`);
    console.log(JSON.stringify(payload, null, 2));
    return;
  }

  const result = await postMemory(payload);
  const memoryId = result?.id ?? result?.memory_id ?? "unknown";
  console.log(`[jarvis-handoff-enforcer] Wrote Jarvis snapshot ${memoryId} for ${sessionKey} (${groupId}).`);
};

export default handler;
