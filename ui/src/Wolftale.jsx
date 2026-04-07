import { useState, useEffect, useRef, useCallback } from "react";

// ─── Config ──────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const SESSION_KEY = "wolftale_session_id";

const DOMAIN_META = {
  preference:  { color: "#c8f566", label: "preference" },
  identity:    { color: "#6b7fd4", label: "identity"   },
  technical:   { color: "#4db896", label: "technical"  },
  commitment:  { color: "#e8a838", label: "commitment" },
  relational:  { color: "#e87d6b", label: "relational" },
  ephemeral:   { color: "#888",    label: "ephemeral"  },
  other:       { color: "#666",    label: "other"      },
};

const PLACEHOLDERS = [
  "Tell me something about yourself…",
  "What are you working on right now?",
  "Where are you based?",
  "What tools do you use day to day?",
  "What do you prefer when it comes to communication?",
];

// ─── Session ─────────────────────────────────────────────────────────────────

function getOrCreateSession() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

// ─── API ─────────────────────────────────────────────────────────────────────

async function apiChat(sessionId, message) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-ID": sessionId,
    },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || `Server error ${res.status}`);
  }
  return res.json();
}

async function apiMemories(sessionId) {
  const res = await fetch(`${API_BASE}/api/memories`, {
    headers: { "X-Session-ID": sessionId },
  });
  if (!res.ok) throw new Error("Failed to load memories");
  return res.json();
}

async function apiClear(sessionId) {
  const res = await fetch(`${API_BASE}/api/memories`, {
    method: "DELETE",
    headers: { "X-Session-ID": sessionId },
  });
  if (!res.ok) throw new Error("Failed to clear memories");
  return res.json();
}

function exportUrl(sessionId) {
  return `${API_BASE}/api/export?x-session-id=${sessionId}`;
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function DomainPill({ domain, confidence }) {
  const meta = DOMAIN_META[domain] || DOMAIN_META.other;
  return (
    <span style={{
      display: "inline-flex",
      alignItems: "center",
      gap: "5px",
      fontSize: "10px",
      letterSpacing: "0.08em",
      textTransform: "uppercase",
      color: meta.color,
      border: `1px solid ${meta.color}33`,
      borderRadius: "2px",
      padding: "2px 7px",
      fontFamily: "'DM Mono', monospace",
    }}>
      {meta.label}
      {confidence !== undefined && (
        <span style={{ opacity: 0.6 }}>{Math.round(confidence * 100)}%</span>
      )}
    </span>
  );
}

function GateBadge({ decision }) {
  const colors = {
    extract: "#c8f566",
    edge:    "#e8a838",
    skip:    "#444",
  };
  return (
    <span style={{
      fontSize: "9px",
      letterSpacing: "0.12em",
      textTransform: "uppercase",
      color: colors[decision] || "#444",
      fontFamily: "'DM Mono', monospace",
      opacity: 0.8,
    }}>
      gate:{decision}
    </span>
  );
}

function MemoryCard({ claim, isNew = false }) {
  const meta = DOMAIN_META[claim.domain] || DOMAIN_META.other;
  const conf = claim.confidence ?? claim.original_confidence ?? 0;
  const barWidth = Math.max(4, Math.round(conf * 100));

  return (
    <div style={{
      borderLeft: `2px solid ${meta.color}55`,
      padding: "10px 12px",
      background: `${meta.color}06`,
      borderRadius: "0 3px 3px 0",
      animation: isNew ? "slideInClaim 0.4s ease forwards" : "none",
      opacity: isNew ? 0 : 1,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px", marginBottom: "6px" }}>
        <DomainPill domain={claim.domain} confidence={conf} />
        {isNew && (
          <span style={{ fontSize: "9px", color: meta.color, letterSpacing: "0.1em", fontFamily: "'DM Mono', monospace" }}>
            NEW ◆
          </span>
        )}
      </div>
      <div style={{ fontSize: "13px", color: "#c8c5be", lineHeight: 1.5, marginBottom: "6px" }}>
        {claim.claim}
      </div>
      <div style={{ height: "2px", background: "#1a1a1a", borderRadius: "1px", overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${barWidth}%`,
          background: meta.color,
          opacity: 0.5,
          borderRadius: "1px",
          transition: "width 0.6s ease",
        }} />
      </div>
    </div>
  );
}

function ChatMessage({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      gap: "4px",
      animation: "fadeInMsg 0.35s ease forwards",
      opacity: 0,
    }}>
      <div style={{
        maxWidth: "78%",
        background: isUser ? "#1a1a1a" : "transparent",
        border: isUser ? "1px solid #2a2a2a" : "none",
        borderRadius: isUser ? "12px 12px 3px 12px" : "0",
        padding: isUser ? "10px 14px" : "2px 0",
        fontSize: "15px",
        color: isUser ? "#e8e4dc" : "#c0bdb6",
        lineHeight: 1.6,
        fontFamily: "'DM Serif Display', Georgia, serif",
      }}>
        {msg.content}
      </div>
      {msg.gateMeta && (
        <div style={{ display: "flex", gap: "8px", alignItems: "center", paddingLeft: "2px" }}>
          <GateBadge decision={msg.gateMeta.decision} />
          {msg.gateMeta.storeAction && (
            <span style={{ fontSize: "9px", color: "#444", letterSpacing: "0.1em", fontFamily: "'DM Mono', monospace" }}>
              {msg.gateMeta.storeAction}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: "flex", gap: "5px", alignItems: "center", padding: "8px 0" }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: "5px",
          height: "5px",
          borderRadius: "50%",
          background: "#333",
          animation: `typingDot 1.2s ease infinite`,
          animationDelay: `${i * 0.2}s`,
        }} />
      ))}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function Wolftale() {
  const [phase, setPhase]             = useState("idle");   // idle | chat
  const [sessionId]                   = useState(getOrCreateSession);
  const [messages, setMessages]       = useState([]);
  const [memories, setMemories]       = useState([]);
  const [newClaimId, setNewClaimId]   = useState(null);
  const [input, setInput]             = useState("");
  const [isLoading, setIsLoading]     = useState(false);
  const [error, setError]             = useState(null);
  const [showMemories, setShowMemories] = useState(true);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [exported, setExported]       = useState(false);
  const [clearing, setClearing]       = useState(false);
  const [memoryCount, setMemoryCount] = useState(0);

  const chatEndRef  = useRef(null);
  const inputRef    = useRef(null);
  const textareaRef = useRef(null);

  // Rotate placeholder
  useEffect(() => {
    if (phase !== "idle") return;
    const t = setInterval(() => setPlaceholderIdx(i => (i + 1) % PLACEHOLDERS.length), 4000);
    return () => clearInterval(t);
  }, [phase]);

  // Scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Load existing memories on mount
  useEffect(() => {
    apiMemories(sessionId)
      .then(data => {
        setMemories(data.claims || []);
        setMemoryCount(data.count || 0);
        if (data.count > 0) setPhase("chat");
      })
      .catch(() => {});
  }, [sessionId]);

  const handleStart = () => {
    setPhase("chat");
    setTimeout(() => textareaRef.current?.focus(), 100);
  };

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    setInput("");
    setError(null);
    setIsLoading(true);

    // Optimistic user message
    const userMsg = { role: "user", content: text, id: Date.now() };
    setMessages(prev => [...prev, userMsg]);

    try {
      const data = await apiChat(sessionId, text);

      // Assistant message with gate trace
      const assistantMsg = {
        role: "assistant",
        content: data.response,
        id: Date.now() + 1,
        gateMeta: {
          decision: data.gate_decision,
          storeAction: data.store_action,
        },
      };
      setMessages(prev => [...prev, assistantMsg]);

      // Update memory panel
      if (data.extracted && data.claim) {
        setMemories(prev => {
          const exists = prev.find(m => m.id === data.claim.id);
          if (exists) return prev;
          return [data.claim, ...prev];
        });
        setNewClaimId(data.claim.id);
        setTimeout(() => setNewClaimId(null), 3000);
      }
      setMemoryCount(data.memory_count);

    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.filter(m => m.id !== userMsg.id));
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, sessionId]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleExport = () => {
    const link = document.createElement("a");
    link.href = `${API_BASE}/api/export`;
    link.setAttribute("download", "wolftale_memories.json");
    // Fetch with session header instead of query param
    fetch(`${API_BASE}/api/export`, {
      headers: { "X-Session-ID": sessionId },
    })
      .then(r => r.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "wolftale_memories.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setExported(true);
        setTimeout(() => setExported(false), 3000);
      })
      .catch(() => setError("Export failed"));
  };

  const handleClear = async () => {
    if (!window.confirm("Clear all memories for this session? This cannot be undone.")) return;
    setClearing(true);
    try {
      await apiClear(sessionId);
      setMemories([]);
      setMemoryCount(0);
      setMessages([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setClearing(false);
    }
  };

  // ── Idle screen ────────────────────────────────────────────────────────────
  if (phase === "idle") {
    return (
      <div style={s.root}>
        <div style={s.grain} />
        <Header memoryCount={memoryCount} />

        <main style={s.idleMain}>
          <div style={s.idleInner}>
            <div style={s.eyebrow}>Personal Memory Layer · Wolflow</div>
            <h1 style={s.headline}>
              AI memory<br />
              <em style={s.headlineEm}>that's yours.</em>
            </h1>
            <p style={s.subline}>
              Every AI assistant forgets you when the session ends.
              Wolftale remembers — and the memory belongs to you,
              not a platform. Persistent, portable, exportable.
            </p>

            <div style={s.featureRow}>
              {[
                ["◆", "#c8f566", "Extracts claims", "Learns what matters from what you share naturally"],
                ["◆", "#6b7fd4", "Detects conflicts", "Surfaces contradictions rather than silently overwriting"],
                ["◆", "#4db896", "Exports your data", "Download your memory as JSON — it's yours to keep"],
              ].map(([icon, color, title, desc]) => (
                <div key={title} style={s.featureCard}>
                  <span style={{ color, fontSize: "12px" }}>{icon}</span>
                  <div>
                    <div style={{ fontSize: "13px", color: "#e8e4dc", marginBottom: "4px" }}>{title}</div>
                    <div style={{ fontSize: "12px", color: "#555", lineHeight: 1.5 }}>{desc}</div>
                  </div>
                </div>
              ))}
            </div>

            <div style={s.idleInputWrapper}>
              <textarea
                style={s.idleTextarea}
                placeholder={PLACEHOLDERS[placeholderIdx]}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (input.trim()) {
                      setPhase("chat");
                      setTimeout(handleSend, 50);
                    }
                  }
                }}
                rows={3}
              />
              <div style={s.idleInputFooter}>
                <span style={s.idleInputHint}>Press Enter to start · Shift+Enter for newline</span>
                <button
                  style={{
                    ...s.primaryBtn,
                    opacity: input.trim() ? 1 : 0.35,
                    cursor: input.trim() ? "pointer" : "default",
                  }}
                  onClick={() => {
                    if (input.trim()) {
                      setPhase("chat");
                      setTimeout(handleSend, 50);
                    }
                  }}
                >
                  Begin →
                </button>
              </div>
            </div>

            <p style={s.disclaimer}>
              Demo sessions last 7 days. Export your memories anytime — they're yours.
              {" "}<a href="https://github.com/wolflow-ai/wolftale" target="_blank" rel="noopener" style={s.dimLink}>View source →</a>
            </p>
          </div>
        </main>
        <GlobalStyles />
      </div>
    );
  }

  // ── Chat screen ────────────────────────────────────────────────────────────
  return (
    <div style={s.root}>
      <div style={s.grain} />
      <Header memoryCount={memoryCount} onBack={() => setPhase("idle")} />

      <div style={s.chatLayout}>

        {/* ── Left: conversation ── */}
        <div style={s.chatPane}>
          <div style={s.messageList}>
            {messages.length === 0 && (
              <div style={s.emptyChat}>
                <div style={{ fontSize: "28px", color: "#1e1e1e", marginBottom: "12px" }}>◈</div>
                <div style={{ fontSize: "14px", color: "#333", marginBottom: "6px" }}>
                  Your conversation starts here.
                </div>
                <div style={{ fontSize: "12px", color: "#252525", fontStyle: "italic" }}>
                  Share something — a preference, a fact about yourself, a project you're working on.
                  Wolftale will remember what matters.
                </div>
              </div>
            )}

            {messages.map(msg => (
              <ChatMessage key={msg.id} msg={msg} />
            ))}

            {isLoading && <TypingIndicator />}

            {error && (
              <div style={s.errorMsg}>
                ◆ {error}
                <button
                  style={{ marginLeft: "12px", color: "#e85858", background: "none", border: "none", cursor: "pointer", fontSize: "12px" }}
                  onClick={() => setError(null)}
                >
                  dismiss
                </button>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div style={s.inputArea}>
            <textarea
              ref={textareaRef}
              style={s.chatTextarea}
              placeholder="Say something…"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              disabled={isLoading}
            />
            <div style={s.inputRow}>
              <span style={s.inputHint}>Enter to send · Shift+Enter for newline</span>
              <button
                style={{
                  ...s.sendBtn,
                  opacity: input.trim() && !isLoading ? 1 : 0.3,
                  cursor: input.trim() && !isLoading ? "pointer" : "default",
                }}
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
              >
                {isLoading ? "…" : "Send →"}
              </button>
            </div>
          </div>
        </div>

        {/* ── Right: memory panel ── */}
        <div style={s.memoryPane}>
          <div style={s.memoryHeader}>
            <div style={s.memoryTitle}>
              <span style={{ color: "#c8f566", fontSize: "12px" }}>◈</span>
              Memory
              <span style={s.memoryCount}>{memoryCount}</span>
            </div>
            <div style={s.memoryActions}>
              <button
                style={s.iconBtn}
                onClick={handleExport}
                title="Export memories as JSON"
              >
                {exported ? "✓" : "↓"}
              </button>
              <button
                style={{ ...s.iconBtn, color: "#555" }}
                onClick={handleClear}
                disabled={clearing}
                title="Clear all memories"
              >
                ×
              </button>
            </div>
          </div>

          <div style={s.memoryScroll}>
            {memories.length === 0 ? (
              <div style={s.memoryEmpty}>
                <div style={{ fontSize: "11px", color: "#2a2a2a", lineHeight: 1.6 }}>
                  No memories yet. Share something about yourself — a preference, where you're based, what you're working on.
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {memories.map(claim => (
                  <MemoryCard
                    key={claim.id}
                    claim={claim}
                    isNew={claim.id === newClaimId}
                  />
                ))}
              </div>
            )}
          </div>

          <div style={s.memoryFooter}>
            <div style={s.portabilityNote}>
              <span style={{ color: "#c8f566", fontSize: "10px" }}>◆</span>
              {" "}Your data. Export anytime.
            </div>
            <button style={s.exportBtn} onClick={handleExport}>
              {exported ? "Exported ✓" : "Export →"}
            </button>
          </div>
        </div>

      </div>
      <GlobalStyles />
    </div>
  );
}

// ─── Header ──────────────────────────────────────────────────────────────────

function Header({ memoryCount, onBack }) {
  return (
    <header style={s.header}>
      <a href="https://wolflow.ai" style={{ textDecoration: "none" }}>
        <div style={s.logo}>
          <span style={s.logoMark}>◈</span>
          <span style={s.logoText}>Wolftale</span>
        </div>
      </a>
      <nav style={s.nav}>
        {onBack && (
          <button style={s.navBtn} onClick={onBack}>← back</button>
        )}
        <a
          href="https://github.com/wolflow-ai/wolftale"
          target="_blank"
          rel="noopener"
          style={{ ...s.navBtn, textDecoration: "none" }}
        >
          GitHub →
        </a>
        <a
          href="https://clewismessina.com"
          target="_blank"
          rel="noopener"
          style={{ ...s.navBtn, textDecoration: "none", color: "#3a3a3a" }}
        >
          clewismessina.com
        </a>
      </nav>
    </header>
  );
}

// ─── Global styles ────────────────────────────────────────────────────────────

function GlobalStyles() {
  return (
    <style>{`
      * { box-sizing: border-box; }
      body { margin: 0; background: #0b0b0b; }
      textarea { resize: none; }
      textarea::placeholder { color: #2a2a2a; }
      textarea:focus { outline: none; }
      button:focus { outline: none; }
      ::-webkit-scrollbar { width: 3px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }

      @keyframes slideInClaim {
        from { opacity: 0; transform: translateX(10px); }
        to   { opacity: 1; transform: translateX(0); }
      }
      @keyframes fadeInMsg {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      @keyframes typingDot {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.3; }
        30%            { transform: translateY(-4px); opacity: 1; }
      }
    `}</style>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const s = {
  root: {
    minHeight: "100vh",
    background: "#0b0b0b",
    color: "#e8e4dc",
    fontFamily: "'DM Serif Display', Georgia, serif",
    position: "relative",
    overflowX: "hidden",
  },
  grain: {
    position: "fixed",
    inset: 0,
    backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E")`,
    pointerEvents: "none",
    zIndex: 0,
  },

  // Header
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "22px 40px",
    position: "relative",
    zIndex: 10,
    borderBottom: "1px solid #141414",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },
  logoMark: {
    fontSize: "18px",
    color: "#c8f566",
  },
  logoText: {
    fontSize: "16px",
    letterSpacing: "0.06em",
    color: "#e8e4dc",
  },
  nav: {
    display: "flex",
    alignItems: "center",
    gap: "20px",
  },
  navBtn: {
    background: "none",
    border: "none",
    color: "#444",
    fontSize: "12px",
    letterSpacing: "0.04em",
    cursor: "pointer",
    fontFamily: "'DM Serif Display', Georgia, serif",
    padding: 0,
  },

  // Idle
  idleMain: {
    position: "relative",
    zIndex: 1,
    display: "flex",
    justifyContent: "center",
    padding: "72px 24px 80px",
    minHeight: "calc(100vh - 80px)",
  },
  idleInner: {
    maxWidth: "620px",
    width: "100%",
  },
  eyebrow: {
    fontSize: "10px",
    letterSpacing: "0.18em",
    color: "#c8f566",
    textTransform: "uppercase",
    marginBottom: "20px",
    opacity: 0.7,
    fontFamily: "'DM Mono', monospace",
  },
  headline: {
    fontSize: "clamp(44px, 7vw, 68px)",
    fontWeight: "normal",
    lineHeight: 1.05,
    margin: "0 0 20px 0",
    color: "#f0ece4",
    letterSpacing: "-0.02em",
  },
  headlineEm: {
    fontStyle: "italic",
    color: "#6b7fd4",
  },
  subline: {
    fontSize: "16px",
    color: "#666",
    lineHeight: 1.75,
    margin: "0 0 40px 0",
    maxWidth: "480px",
    fontFamily: "'DM Serif Display', Georgia, serif",
    fontWeight: "normal",
  },
  featureRow: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    marginBottom: "40px",
  },
  featureCard: {
    display: "flex",
    gap: "14px",
    alignItems: "flex-start",
    padding: "14px 16px",
    border: "1px solid #181818",
    borderRadius: "3px",
    background: "#0d0d0d",
  },
  idleInputWrapper: {
    background: "#0f0f0f",
    border: "1px solid #222",
    borderRadius: "3px",
    marginBottom: "20px",
  },
  idleTextarea: {
    width: "100%",
    background: "transparent",
    border: "none",
    color: "#e8e4dc",
    fontSize: "15px",
    fontFamily: "'DM Serif Display', Georgia, serif",
    lineHeight: 1.6,
    padding: "20px 20px 12px",
    caretColor: "#c8f566",
  },
  idleInputFooter: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "10px 20px 14px",
    borderTop: "1px solid #181818",
    gap: "12px",
    flexWrap: "wrap",
  },
  idleInputHint: {
    fontSize: "11px",
    color: "#2e2e2e",
    fontFamily: "'DM Mono', monospace",
  },
  primaryBtn: {
    background: "#c8f566",
    color: "#0b0b0b",
    border: "none",
    padding: "10px 22px",
    fontSize: "13px",
    fontFamily: "'DM Serif Display', Georgia, serif",
    letterSpacing: "0.04em",
    cursor: "pointer",
    borderRadius: "2px",
    fontWeight: "bold",
    transition: "opacity 0.2s",
  },
  disclaimer: {
    fontSize: "11px",
    color: "#2a2a2a",
    lineHeight: 1.6,
    fontFamily: "'DM Mono', monospace",
  },
  dimLink: {
    color: "#3a3a3a",
    textDecoration: "none",
  },

  // Chat layout
  chatLayout: {
    display: "grid",
    gridTemplateColumns: "1fr 320px",
    height: "calc(100vh - 72px)",
    position: "relative",
    zIndex: 1,
  },

  // Chat pane
  chatPane: {
    display: "flex",
    flexDirection: "column",
    borderRight: "1px solid #141414",
    overflow: "hidden",
  },
  messageList: {
    flex: 1,
    overflowY: "auto",
    padding: "32px 40px",
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },
  emptyChat: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    minHeight: "200px",
    textAlign: "center",
    padding: "40px",
  },
  inputArea: {
    borderTop: "1px solid #141414",
    padding: "16px 40px 20px",
    background: "#0b0b0b",
  },
  chatTextarea: {
    width: "100%",
    background: "#0f0f0f",
    border: "1px solid #1e1e1e",
    borderRadius: "3px",
    color: "#e8e4dc",
    fontSize: "15px",
    fontFamily: "'DM Serif Display', Georgia, serif",
    lineHeight: 1.6,
    padding: "12px 14px 8px",
    caretColor: "#c8f566",
    marginBottom: "10px",
    transition: "border-color 0.2s",
  },
  inputRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "12px",
  },
  inputHint: {
    fontSize: "10px",
    color: "#252525",
    fontFamily: "'DM Mono', monospace",
  },
  sendBtn: {
    background: "#c8f566",
    color: "#0b0b0b",
    border: "none",
    padding: "8px 18px",
    fontSize: "12px",
    fontFamily: "'DM Serif Display', Georgia, serif",
    letterSpacing: "0.04em",
    cursor: "pointer",
    borderRadius: "2px",
    fontWeight: "bold",
    transition: "opacity 0.15s",
  },

  // Memory pane
  memoryPane: {
    display: "flex",
    flexDirection: "column",
    background: "#090909",
    overflow: "hidden",
  },
  memoryHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "18px 20px 14px",
    borderBottom: "1px solid #141414",
  },
  memoryTitle: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "12px",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    color: "#444",
    fontFamily: "'DM Mono', monospace",
  },
  memoryCount: {
    background: "#1a1a1a",
    color: "#555",
    fontSize: "10px",
    padding: "1px 6px",
    borderRadius: "10px",
    fontFamily: "'DM Mono', monospace",
  },
  memoryActions: {
    display: "flex",
    gap: "4px",
  },
  iconBtn: {
    background: "none",
    border: "1px solid #1e1e1e",
    color: "#c8f566",
    width: "26px",
    height: "26px",
    borderRadius: "2px",
    cursor: "pointer",
    fontSize: "13px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "'DM Mono', monospace",
    transition: "border-color 0.2s",
  },
  memoryScroll: {
    flex: 1,
    overflowY: "auto",
    padding: "16px",
  },
  memoryEmpty: {
    padding: "20px 4px",
  },
  memoryFooter: {
    borderTop: "1px solid #141414",
    padding: "14px 16px",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  portabilityNote: {
    fontSize: "10px",
    color: "#2a2a2a",
    letterSpacing: "0.04em",
    fontFamily: "'DM Mono', monospace",
  },
  exportBtn: {
    background: "none",
    border: "1px solid #1e1e1e",
    color: "#c8f566",
    fontSize: "11px",
    padding: "5px 12px",
    borderRadius: "2px",
    cursor: "pointer",
    fontFamily: "'DM Mono', monospace",
    letterSpacing: "0.06em",
    transition: "border-color 0.2s",
  },

  // Error
  errorMsg: {
    fontSize: "12px",
    color: "#e85858",
    padding: "10px 14px",
    border: "1px solid #e8585822",
    borderRadius: "3px",
    background: "#1a0a0a",
    fontFamily: "'DM Mono', monospace",
  },
};
