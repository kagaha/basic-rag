import { ChipIcon, LayersIcon } from "./icons.jsx";

// Shows which embedding model and LLM the backend is using.
export default function ModelChips({ config, offline, onRetry, className = "" }) {
  if (offline) {
    return (
      <div className={`chips ${className}`}>
        <span className="chip chip-offline">
          <span className="dot dot-red" /> Backend offline
        </span>
        <button type="button" className="chip chip-button" onClick={onRetry}>
          Retry
        </button>
      </div>
    );
  }

  if (!config) {
    return (
      <div className={`chips ${className}`}>
        <span className="chip chip-muted">Connecting to backend…</span>
      </div>
    );
  }

  return (
    <div className={`chips ${className}`}>
      <span className="chip" title="Model that turns text into vectors">
        <LayersIcon width={15} height={15} />
        <span className="chip-label">Embedding</span>
        <strong>{config.embedding_model}</strong>
      </span>
      <span className="chip" title="Model that writes the final answer">
        <ChipIcon width={15} height={15} />
        <span className="chip-label">LLM</span>
        <strong>{config.llm_model}</strong>
      </span>
    </div>
  );
}
