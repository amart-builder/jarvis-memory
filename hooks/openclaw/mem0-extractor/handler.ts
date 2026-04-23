/**
 * mem0-extractor hook
 * Fires on session:compact:before — extracts durable facts into mem0
 * before OpenClaw compresses the session history.
 *
 * Uses the jarvis-memory REST API at http://localhost:3500
 * Endpoint: POST /api/v1/add  (content: string, user_id: string)
 */

const MEM0_URL = process.env.MEM0_URL ?? "http://localhost:3500";
const MEM0_USER_ID = process.env.MEM0_USER_ID ?? "user";
const MEM0_MAX_MESSAGES = parseInt(process.env.MEM0_MAX_MESSAGES ?? "40", 10);

interface Message {
  role: string;
  content: string | Array<{ type: string; text?: string }>;
}

interface HookEvent {
  type: string;
  action?: string;
  sessionKey?: string;
  session?: {
    messages?: Message[];
    key?: string;
  };
  messages?: string[];
  timestamp?: Date;
}

function extractText(content: string | Array<{ type: string; text?: string }>): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((c) => c.type === "text" && c.text)
      .map((c) => c.text!)
      .join("\n");
  }
  return "";
}

const handler = async (event: HookEvent): Promise<void> => {
  // Only fire on compact:before
  if (event.type !== "session" || event.action !== "compact:before") {
    return;
  }

  const sessionKey = event.sessionKey ?? event.session?.key ?? "unknown";
  const allMessages: Message[] = event.session?.messages ?? [];

  if (allMessages.length === 0) {
    console.log(`[mem0-extractor] No messages in session ${sessionKey}, skipping.`);
    return;
  }

  // Take last N user/assistant turns, build a readable transcript for mem0 to extract from
  const recent = allMessages
    .slice(-MEM0_MAX_MESSAGES)
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => {
      const text = extractText(m.content).trim();
      if (!text) return null;
      const prefix = m.role === "user" ? "User" : "Assistant";
      return `${prefix}: ${text.slice(0, 1000)}`; // cap each turn to 1000 chars
    })
    .filter(Boolean);

  if (recent.length === 0) {
    console.log(`[mem0-extractor] No user/assistant messages to extract from ${sessionKey}.`);
    return;
  }

  const transcript = recent.join("\n\n");

  console.log(
    `[mem0-extractor] Extracting facts from ${recent.length} turns (session: ${sessionKey})`
  );

  try {
    // Health check first
    const healthRes = await fetch(`${MEM0_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!healthRes.ok) {
      console.log(`[mem0-extractor] mem0 health check failed (${healthRes.status}), skipping.`);
      return;
    }

    // POST to Atlas mem0 wrapper: /api/v1/add
    const res = await fetch(`${MEM0_URL}/api/v1/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: transcript,
        user_id: MEM0_USER_ID,
        metadata: {
          source: "hook:session:compact:before",
          session_key: sessionKey,
          extracted_at: new Date().toISOString(),
        },
      }),
      signal: AbortSignal.timeout(45000), // Haiku extraction can take up to 30s on long contexts
    });

    if (!res.ok) {
      const body = await res.text();
      console.log(`[mem0-extractor] mem0 extraction failed (${res.status}): ${body}`);
      return;
    }

    const result = await res.json() as { results?: Array<{ memory: string; event: string }> };
    const stored = (result as { results?: Array<{ memory: string; event: string }> }).results ?? [];
    const added = stored.filter((r) => r.event === "ADD");
    const updated = stored.filter((r) => r.event === "UPDATE");

    console.log(
      `[mem0-extractor] ✓ Done: +${added.length} added, ~${updated.length} updated (${stored.length} total)`
    );

    if (added.length > 0) {
      console.log("[mem0-extractor] New facts stored:");
      added.slice(0, 8).forEach((r) => console.log(`  + ${r.memory}`));
      if (added.length > 8) console.log(`  ... and ${added.length - 8} more`);
    }
  } catch (err: unknown) {
    if (err instanceof Error && err.name === "TimeoutError") {
      console.log(`[mem0-extractor] mem0 timed out — skipping extraction for ${sessionKey}.`);
    } else {
      console.log(`[mem0-extractor] Error: ${err}`);
    }
  }
};

export default handler;
