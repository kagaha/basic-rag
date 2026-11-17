import { useEffect, useRef, useState } from "react";
import { askQuestion } from "../api.js";
import ModelChips from "./ModelChips.jsx";
import { AlertIcon, SendIcon, SparkIcon } from "./icons.jsx";

function Sources({ sources }) {
  if (!sources?.length) return null;
  return (
    <details className="sources">
      <summary>
        {sources.length} source passage{sources.length === 1 ? "" : "s"} used
      </summary>
      <ol>
        {sources.map((s, i) => (
          <li key={i} className="source">
            <div className="source-head">
              <span className="source-name">{s.source}</span>
              <span className="score" title="Cosine similarity to the question">
                match {Math.round(s.score * 100)}%
              </span>
            </div>
            <p>{s.content}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}

function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="bubble bubble-user">{message.text}</div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="msg">
        <div className="alert" role="alert">
          <AlertIcon width={18} height={18} />
          <span>{message.text}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="msg">
      <div className="avatar" aria-hidden="true">
        <SparkIcon width={16} height={16} />
      </div>
      <div className="bubble bubble-ai">
        {message.rewritten && (
          <p className="rewritten">
            Searched for: <em>{message.rewritten}</em>
          </p>
        )}
        <p className="answer">{message.text}</p>
        <Sources sources={message.sources} />
        {message.model && (
          <p className="byline">Answered by {message.model}</p>
        )}
      </div>
    </div>
  );
}

export default function Chat({ apiKey, session, config }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const thread = useRef(null);

  useEffect(() => {
    const el = thread.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function send() {
    const q = question.trim();
    if (!q || busy) return;

    setQuestion("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);

    try {
      const res = await askQuestion({
        apiKey: apiKey.trim(),
        sessionId: session.session_id,
        question: q,
      });
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: res.answer,
          sources: res.sources,
          rewritten: res.rewritten_question,
          model: res.llm_model,
        },
      ]);
    } catch (err) {
      setMessages((m) => [...m, { role: "error", text: err.message }]);
    } finally {
      setBusy(false);
    }
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  return (
    <section className="card">
      <div className="step-head">
        <span className="step-badge">2</span>
        <div>
          <h2>Ask a question</h2>
          <p className="muted">
            Answers come only from your documents. Follow-up questions remember
            the conversation.
          </p>
        </div>
      </div>

      <ModelChips config={config} className="chips-inline" />

      <div className="thread" ref={thread} aria-live="polite">
        {messages.length === 0 && !busy && (
          <div className="empty">
            <SparkIcon width={22} height={22} />
            <p>Ask anything about the documents you uploaded.</p>
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={i} message={m} />
        ))}
        {busy && (
          <div className="msg">
            <div className="avatar" aria-hidden="true">
              <SparkIcon width={16} height={16} />
            </div>
            <div className="bubble bubble-ai typing" aria-label="Thinking">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </div>

      <div className="composer">
        <textarea
          className="textarea composer-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type your question and press Enter…"
          rows={2}
          maxLength={2000}
          disabled={busy}
        />
        <button
          type="button"
          className="btn btn-primary"
          onClick={send}
          disabled={busy || !question.trim()}
        >
          Ask <SendIcon width={17} height={17} />
        </button>
      </div>
    </section>
  );
}
