// jarvis-start-guard
//
// Fires on `agent:bootstrap` + `message:sent`.
// On bootstrap: queries `/api/v2/handoff/latest` for each configured group_id
// and injects a warning into the agent's bootstrap context if a recent
// handoff was written by ANOTHER system (Claude Code, CLI, etc). This is
// the "did someone else just work on this?" continuity check.
// On message:sent: checks whether the assistant's first reply acknowledged
// the cross-system context — logs a START-GUARD VIOLATION if not.
//
// Configuration via env:
//   JARVIS_MEMORY_API                 base URL (default http://localhost:3500)
//   JARVIS_API_BEARER_TOKEN           bearer token if the API requires auth
//   JARVIS_START_GUARD_GROUP_IDS      comma-separated list of group_ids to watch (overrides default)
//   JARVIS_START_GUARD_LOOKBACK_HOURS default 2
//   JARVIS_START_GUARD_MAX_WARNINGS   max warnings to inject per bootstrap (default 3)
//   JARVIS_START_GUARD_STATE_FILE     local state path (default ~/.openclaw/jarvis-start-guard-state.json)
//   JARVIS_START_GUARD_DRY_RUN        1/true/yes to skip writes

const JARVIS_MEMORY_API = process.env.JARVIS_MEMORY_API ?? "http://localhost:3500";
const JARVIS_API_BEARER_TOKEN = process.env.JARVIS_API_BEARER_TOKEN ?? "";
const LOOKBACK_HOURS = Number.parseFloat(process.env.JARVIS_START_GUARD_LOOKBACK_HOURS ?? "2");
const MAX_WARNING_COUNT = Number.parseInt(process.env.JARVIS_START_GUARD_MAX_WARNINGS ?? "3", 10);
const STATE_FILE = process.env.JARVIS_START_GUARD_STATE_FILE ?? `${process.env.HOME}/.openclaw/jarvis-start-guard-state.json`;
const DRY_RUN = /^(1|true|yes)$/i.test(process.env.JARVIS_START_GUARD_DRY_RUN ?? "");

// Default group_ids to watch for cross-system handoffs. Override via
// JARVIS_START_GUARD_GROUP_IDS="navi,atlas-system,..." or edit this list.
const DEFAULT_WATCH_GROUPS = [
  "navi",
  "catalyst",
  "atlas-system",
  "jarvis",
  "system",
];

function parseGroupIds(): string[] {
  const env = process.env.JARVIS_START_GUARD_GROUP_IDS ?? "";
  const list = env.split(",").map((s) => s.trim()).filter(Boolean);
  return list.length > 0 ? list : DEFAULT_WATCH_GROUPS;
}

function normalizeWhitespace(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function summarizeText(value, max = 220) {
  const clean = normalizeWhitespace(value);
  return clean.length <= max ? clean : `${clean.slice(0, max - 1)}…`;
}

function toMillis(value) {
  const ms = Date.parse(String(value ?? ""));
  return Number.isFinite(ms) ? ms : null;
}

// A handoff is "from another system" iff its source/device indicates some
// non-OpenClaw origin. We skip:
//   - our own hooks (source starting with "hook:jarvis")
//   - device == "openclaw"
// Everything else (Claude Code hooks, CLI writes, REST API, other MCP clients)
// counts as cross-system context that the OpenClaw agent should acknowledge.
function isOtherSystem(handoff): boolean {
  const source = String(handoff?.source ?? "").toLowerCase();
  const device = String(handoff?.device ?? "").toLowerCase();
  if (source.startsWith("hook:jarvis")) return false;
  if (device === "openclaw") return false;
  return true;
}

function extractBootstrapFiles(event) {
  const files = event?.context?.bootstrapFiles;
  return Array.isArray(files) ? files : [];
}

function getSessionKey(event) {
  return String(event?.sessionKey ?? event?.context?.sessionKey ?? "unknown");
}

function getMessageText(event) {
  return normalizeWhitespace(event?.context?.content ?? "");
}

function looksAcknowledged(text, warnings) {
  const clean = normalizeWhitespace(text).toLowerCase();
  if (!clean) return false;
  if (/(claude|openclaw|handoff|left off|pick up|picking up|continuity|recent context)/i.test(clean)) {
    return true;
  }
  return warnings.some((warning) => clean.includes(String(warning.groupId ?? "").toLowerCase()));
}

async function loadState() {
  try {
    const fs = await import("node:fs/promises");
    const raw = await fs.readFile(STATE_FILE, "utf8");
    return JSON.parse(raw);
  } catch {
    return { updatedAt: null, sessions: {} };
  }
}

async function saveState(state) {
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  await fs.mkdir(path.dirname(STATE_FILE), { recursive: true });
  state.updatedAt = new Date().toISOString();
  await fs.writeFile(STATE_FILE, JSON.stringify(state, null, 2));
}

function pruneState(state) {
  const cutoff = Date.now() - 24 * 60 * 60 * 1000;
  for (const [sessionKey, entry] of Object.entries(state.sessions ?? {})) {
    const startedAt = toMillis(entry?.startedAt);
    if (startedAt && startedAt < cutoff) {
      delete state.sessions[sessionKey];
    }
  }
}

async function fetchLatestHandoff(groupId: string) {
  const url = new URL(`${JARVIS_MEMORY_API}/api/v2/handoff/latest`);
  url.searchParams.set("group_id", groupId);
  url.searchParams.set("max_age_hours", String(LOOKBACK_HOURS));

  const headers: Record<string, string> = {};
  if (JARVIS_API_BEARER_TOKEN) {
    headers["authorization"] = `Bearer ${JARVIS_API_BEARER_TOKEN}`;
  }

  try {
    const response = await fetch(url, { headers, signal: AbortSignal.timeout(8000) });
    if (response.status === 404) return null; // no handoff in window — expected
    if (!response.ok) {
      // 5xx / 4xx that isn't 404 means the API is reachable but something's wrong.
      // Silent return-null hides real outages; log so operators see them.
      console.warn(
        `[jarvis-start-guard] /api/v2/handoff/latest returned ${response.status} for group_id=${groupId} — treating as no-handoff, but API may be degraded.`
      );
      return null;
    }
    return await response.json();
  } catch (err) {
    console.warn(
      `[jarvis-start-guard] /api/v2/handoff/latest network error for group_id=${groupId}: ${err instanceof Error ? err.message : String(err)}`
    );
    return null;
  }
}

async function findCrossSystemWarnings() {
  const watchGroups = parseGroupIds();
  const warnings = [];

  for (const groupId of watchGroups) {
    const handoff = await fetchLatestHandoff(groupId);
    if (!handoff) continue;
    if (!isOtherSystem(handoff)) continue;

    warnings.push({
      id: handoff.episode_id ?? handoff.snapshot_id ?? "",
      groupId: handoff.group_id ?? groupId,
      agentId: handoff.source ?? handoff.device ?? "unknown",
      createdAt: handoff.created_at ?? null,
      excerpt: summarizeText(handoff.content ?? "", 220),
    });
  }

  return warnings
    .sort((a, b) => (toMillis(b.createdAt) ?? 0) - (toMillis(a.createdAt) ?? 0))
    .slice(0, MAX_WARNING_COUNT);
}

function buildWarningBlock(warnings, options = {}) {
  const pendingViolation = Boolean((options as any).pendingViolation);
  const lines = [
    pendingViolation ? "## JARVIS START GUARD VIOLATION (auto-injected)" : "## JARVIS START GUARD (auto-injected)",
    pendingViolation
      ? "This session already saw recent cross-system context and still has not acknowledged it. The next substantive reply must acknowledge the handoff before new work."
      : "Recent cross-system context exists. Before starting new work on any matching area below, explicitly acknowledge the handoff and what you found.",
    "",
  ];

  for (const warning of warnings) {
    const agoMinutes = Math.max(1, Math.round((Date.now() - (toMillis(warning.createdAt) ?? Date.now())) / 60000));
    lines.push(`- ${warning.groupId} — ${warning.agentId} — ${agoMinutes}m ago — ${warning.excerpt}`);
  }

  lines.push(
    "",
    pendingViolation
      ? "Required behavior: stop and acknowledge the recent cross-system context in the very next substantive reply before doing new work."
      : "Required behavior: if the user is touching one of these areas, your next substantive reply must acknowledge the recent cross-system context before doing new work.",
    ""
  );

  return lines.join("\n");
}

function injectWarningIntoBootstrap(event, warningBlock) {
  const files = extractBootstrapFiles(event);
  const preferredNames = ["AGENTS.md", "MEMORY.md", "current_focus.md"];
  const target = preferredNames
    .map((name) => files.find((file) => file?.name === name && typeof file?.content === "string"))
    .find(Boolean);

  if (!target) {
    return false;
  }

  if (String(target.content).includes("JARVIS START GUARD (auto-injected)")) {
    return true;
  }

  target.content = `${warningBlock}\n${target.content}`;
  return true;
}

async function handleBootstrap(event) {
  const warnings = await findCrossSystemWarnings();
  if (warnings.length === 0) {
    return;
  }

  const sessionKey = getSessionKey(event);
  const warningSignature = warnings.map((warning) => `${warning.groupId}:${warning.id}`).join("|");
  const state = await loadState();
  pruneState(state);
  const existing = state.sessions?.[sessionKey];
  const pendingViolation = Boolean(existing?.pendingAcknowledgement && !existing?.acknowledged);
  const warningBlock = buildWarningBlock(warnings, { pendingViolation });

  if (existing?.warningSignature === warningSignature) {
    existing.lastBootstrapAt = new Date().toISOString();
    existing.guardInjected = true;
    existing.warnings = warnings;
    existing.bootstrapWarning = warningBlock;
    if (pendingViolation) {
      existing.pendingAlertCount = Number(existing.pendingAlertCount ?? 0) + 1;
      existing.lastPendingAlertAt = new Date().toISOString();
    }
    if (Array.isArray(event?.messages)) {
      event.messages.push(
        pendingViolation
          ? `[jarvis-start-guard] Unacknowledged cross-system context still blocks ${sessionKey}: ${warnings.map((warning) => warning.groupId).join(", ")}.`
          : `[jarvis-start-guard] Recent cross-system context exists for ${warnings.map((warning) => warning.groupId).join(", ")}.`
      );
    }
    const injectedAgain = injectWarningIntoBootstrap(event, warningBlock);
    existing.guardInjected = existing.guardInjected || injectedAgain;

    if (DRY_RUN) {
      console.log(`[jarvis-start-guard] Dry run refresh for ${sessionKey}:`);
      console.log(JSON.stringify(existing, null, 2));
      return;
    }

    await saveState(state);
    return;
  }

  const injected = injectWarningIntoBootstrap(event, warningBlock);
  if (Array.isArray(event?.messages)) {
    event.messages.push(`[jarvis-start-guard] Recent cross-system context exists for ${warnings.map((warning) => warning.groupId).join(", ")}.`);
  }

  state.sessions[sessionKey] = {
    sessionKey,
    startedAt: existing?.startedAt ?? new Date().toISOString(),
    lastBootstrapAt: new Date().toISOString(),
    guardInjected: injected,
    pendingAcknowledgement: existing?.acknowledged ? false : true,
    acknowledged: Boolean(existing?.acknowledged),
    acknowledgedAt: existing?.acknowledgedAt ?? null,
    replyCount: Number(existing?.replyCount ?? 0),
    firstReplyMissingAcknowledgement: Boolean(existing?.firstReplyMissingAcknowledgement),
    pendingAlertCount: Number(existing?.pendingAlertCount ?? 0),
    lastPendingAlertAt: existing?.lastPendingAlertAt ?? null,
    warnings,
    warningSignature,
    bootstrapWarning: warningBlock,
  };

  if (DRY_RUN) {
    console.log(`[jarvis-start-guard] Dry run for ${sessionKey}:`);
    console.log(JSON.stringify(state.sessions[sessionKey], null, 2));
    return;
  }

  await saveState(state);
  console.log(`[jarvis-start-guard] Guard injected for ${sessionKey} (${warnings.map((warning) => warning.groupId).join(", ")}).`);
}

async function handleMessageSent(event) {
  const sessionKey = getSessionKey(event);
  const message = getMessageText(event);
  if (!message) return;

  const state = await loadState();
  const entry = state.sessions?.[sessionKey];
  if (!entry || !entry.pendingAcknowledgement) {
    return;
  }

  entry.replyCount = Number(entry.replyCount ?? 0) + 1;
  entry.lastAssistantReplyAt = new Date().toISOString();
  entry.lastAssistantReplyExcerpt = summarizeText(message, 280);

  if (looksAcknowledged(message, entry.warnings ?? [])) {
    entry.pendingAcknowledgement = false;
    entry.acknowledged = true;
    entry.acknowledgedAt = new Date().toISOString();
    console.log(`[jarvis-start-guard] Acknowledgement detected for ${sessionKey}.`);
  } else if (entry.replyCount === 1) {
    entry.firstReplyMissingAcknowledgement = true;
    entry.pendingAlertCount = Number(entry.pendingAlertCount ?? 0) + 1;
    entry.lastPendingAlertAt = new Date().toISOString();
    console.error(`[jarvis-start-guard] START-GUARD VIOLATION for ${sessionKey}: first assistant reply did not acknowledge recent cross-system context.`);
  }

  if (DRY_RUN) {
    console.log(`[jarvis-start-guard] Dry run update for ${sessionKey}:`);
    console.log(JSON.stringify(entry, null, 2));
    return;
  }

  pruneState(state);
  await saveState(state);
}

const handler = async (event) => {
  const trigger = event?.action ? `${event.type}:${event.action}` : String(event?.type ?? "unknown");

  if (trigger === "agent:bootstrap") {
    await handleBootstrap(event);
    return;
  }

  if (trigger === "message:sent") {
    await handleMessageSent(event);
  }
};

export default handler;
