// jarvis-handoff-enforcer
//
// Fires on `session:compact:before`, `command:new`, and `command:reset`.
// Writes a snapshot + retrievable [HANDOFF] Episode to jarvis-memory via
// POST /api/v2/session/handoff (the v1.1 contract endpoint — atomic, idempotent,
// strict group_id).
//
// Configuration via env:
//   JARVIS_MEMORY_API          base URL (default http://localhost:3500)
//   JARVIS_API_BEARER_TOKEN    bearer token if the API requires auth
//   JARVIS_HOOK_MAX_MESSAGES   number of trailing messages to include (default 24)
//   JARVIS_HOOK_DRY_RUN        1/true/yes to log the payload without POSTing
//   JARVIS_DEVICE_ID           device identifier on the Episode (default "openclaw")

const JARVIS_MEMORY_API = process.env.JARVIS_MEMORY_API ?? "http://localhost:3500";
const JARVIS_API_BEARER_TOKEN = process.env.JARVIS_API_BEARER_TOKEN ?? "";
const MAX_MESSAGES = Number.parseInt(process.env.JARVIS_HOOK_MAX_MESSAGES ?? "24", 10);
const DRY_RUN = /^(1|true|yes)$/i.test(process.env.JARVIS_HOOK_DRY_RUN ?? "");
const DEVICE_ID = process.env.JARVIS_DEVICE_ID ?? "openclaw";

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

function buildTaskLine({ trigger, recentUser }) {
  if (recentUser) return summarizeText(recentUser, 400);
  return `OpenClaw handoff (${trigger})`;
}

function buildNotesBlock({ trigger, sessionKey, groupId, confidence, messages, recentUser, recentAssistant }) {
  const transcript = buildTranscript(messages);
  return [
    `[COMPLETION] OpenClaw automatic handoff snapshot (${trigger})`,
    "WHY: Preserve cross-system continuity before compaction or session reset.",
    "IMPACT: Claude and OpenClaw can recover recent state from jarvis-memory.",
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

async function postHandoff(payload) {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (JARVIS_API_BEARER_TOKEN) {
    headers["authorization"] = `Bearer ${JARVIS_API_BEARER_TOKEN}`;
  }
  const response = await fetch(`${JARVIS_MEMORY_API}/api/v2/session/handoff`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(10000),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`v2 handoff failed (${response.status}): ${body}`);
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
  const recentUser = getLastByRole(messages, "user");
  const recentAssistant = getLastByRole(messages, "assistant");
  const notes = buildNotesBlock({ trigger, sessionKey, groupId, confidence, messages, recentUser, recentAssistant });
  const task = buildTaskLine({ trigger, recentUser });

  // Deterministic idempotency key: the server dedupes on this within a 1h
  // window for the same session. Hashing trigger+sessionKey+notes means
  // "same logical event, same transcript snapshot" = no duplicate write.
  const fingerprint = await sha256(`${trigger}\n${sessionKey}\n${notes}`);
  const idempotencyKey = `handoff-enforcer-${fingerprint.slice(0, 32)}`;

  const payload = {
    task,
    group_id: groupId,
    next_steps: [],
    notes,
    device: DEVICE_ID,
    idempotency_key: idempotencyKey,
    session_key: sessionKey,
    source: "hook:jarvis-handoff-enforcer",
  };

  if (DRY_RUN) {
    console.log(`[jarvis-handoff-enforcer] Dry run payload for ${sessionKey}:`);
    console.log(JSON.stringify(payload, null, 2));
    return;
  }

  const result = await postHandoff(payload);
  const snapshotId = result?.snapshot_id ?? "unknown";
  const episodeId = result?.episode_id ?? "unknown";
  const idempotentHit = result?.idempotent_hit ? " (idempotent hit)" : "";
  console.log(
    `[jarvis-handoff-enforcer] Wrote handoff snapshot=${snapshotId} episode=${episodeId} for ${sessionKey} (${groupId})${idempotentHit}.`
  );
};

export default handler;
