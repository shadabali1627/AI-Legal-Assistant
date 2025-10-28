import React, { useState, useEffect } from "react";
import ChatWindow from "./ChatWindow.jsx";
import Sources from "./Sources.jsx";

const API_BASE =
  window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000/api"
    : "https://ai-legal-assistant-aqt4.onrender.com/api";


export default function App() {
  const [messages, setMessages] = useState([
    {
      who: "bot",
      text:
        "Hi! Ask me about Pakistani landmark cases, citations, principles, or benches. I’ll answer using the indexed cases and cite sources like [1], [2].",
    },
  ]);
  const [status, setStatus] = useState("idle");
  const [mode, setMode] = useState("");
  const [sources, setSources] = useState([]);
  const [input, setInput] = useState("");

  useEffect(() => {
    async function checkBackend() {
      setStatus("checking backend…");
      try {
        const res = await fetch(API_BASE.replace(/\/api$/, "/"));
        setStatus(res.ok ? "ready" : "backend not ready");
      } catch {
        setStatus("backend not reachable");
      }
    }
    checkBackend();
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim()) return;
    const q = input.trim();
    setMessages((prev) => [...prev, { who: "user", text: q }, { who: "bot", text: "Thinking…" }]);
    setInput("");
    setStatus("contacting backend…");

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 5 }),
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { who: "bot", text: data.answer || "No answer available." },
      ]);
      setSources(data.sources || []);
      setMode(data.mode || "");
      setStatus("done");
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { who: "bot", text: "Error contacting server. See console for details." },
      ]);
      setStatus("error");
    }
  }

  return (
    <>
      <header>
        <div className="brand">
          <img src="/assets/logo.png" alt="logo" className="logo" onError={(e) => (e.target.style.display = "none")} />
          <div>
            <h1>AI Legal Assistant</h1>
            <p className="tagline">Pakistan case-law RAG chatbot (Gemini + FAISS)</p>
          </div>
        </div>
      </header>

      <main>
        <section className="chat">
          <ChatWindow messages={messages} />
          <form className="chat-form" onSubmit={handleSubmit}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g., What principle did Shehla Zia v. WAPDA establish?"
              required
            />
            <button type="submit">Ask</button>
          </form>
          <div className="pill-row">
            {mode && <span className="mode-pill">Generator: {mode}</span>}
            <span className="status-pill">{status}</span>
          </div>
        </section>

        <aside className="sources">
          <h2>Sources</h2>
          <Sources sources={sources} />
        </aside>
      </main>

      <footer>
        Frontend: React • Backend: FastAPI • Embeddings: Gemini / Sentence-Transformers • Vector DB: FAISS
      </footer>
    </>
  );
}
