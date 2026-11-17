import { useCallback, useEffect, useState } from "react";
import { endSession, getConfig } from "./api.js";
import Chat from "./components/Chat.jsx";
import DocumentsStep from "./components/DocumentsStep.jsx";
import ModelChips from "./components/ModelChips.jsx";
import { LockIcon } from "./components/icons.jsx";

export default function App() {
  const [config, setConfig] = useState(null);
  const [offline, setOffline] = useState(false);
  // The API key lives only in React state: no localStorage, no URL.
  const [apiKey, setApiKey] = useState("");
  const [session, setSession] = useState(null);

  const loadConfig = useCallback(() => {
    setOffline(false);
    getConfig()
      .then(setConfig)
      .catch(() => setOffline(true));
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  function handleReset() {
    if (session) endSession(session.session_id).catch(() => {});
    setSession(null);
  }

  return (
    <div className="app">
      <div className="bg" aria-hidden="true" />

      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="22" height="22" fill="none">
              <path
                d="M9 11h14M9 16h14M9 21h8"
                stroke="white"
                strokeWidth="2.4"
                strokeLinecap="round"
              />
            </svg>
          </span>
          <span className="brand-name">Basic RAG</span>
        </div>
        <ModelChips config={config} offline={offline} onRetry={loadConfig} />
      </header>

      <main className="container">
        <section className="hero">
          <p className="eyebrow">Retrieval-Augmented Generation</p>
          <h1>
            Ask questions. Get answers
            <br />
            <span className="gradient-text">grounded in your documents.</span>
          </h1>
          <p className="lede">
            Add your own text documents, then ask in plain English. Relevant
            passages are retrieved from a vector database and handed to the LLM,
            so every answer is based on what you provided.
          </p>
        </section>

        <DocumentsStep
          apiKey={apiKey}
          onApiKeyChange={setApiKey}
          session={session}
          onUploaded={setSession}
          onReset={handleReset}
        />

        {session ? (
          <Chat
            key={session.session_id}
            apiKey={apiKey}
            session={session}
            config={config}
          />
        ) : (
          <section className="card card-locked">
            <div className="step-head">
              <span className="step-badge step-locked">
                <LockIcon width={15} height={15} />
              </span>
              <div>
                <h2>Ask a question</h2>
                <p className="muted">
                  Upload your documents first to unlock the question box.
                </p>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        Built with React, FastAPI, LangChain, ChromaDB and Gemini.
      </footer>
    </div>
  );
}
