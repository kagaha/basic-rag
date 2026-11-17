import { useRef, useState } from "react";
import { uploadDocuments } from "../api.js";
import ApiKeyField from "./ApiKeyField.jsx";
import {
  AlertIcon,
  CheckIcon,
  FileIcon,
  UploadIcon,
  XIcon,
} from "./icons.jsx";

const MAX_FILE_BYTES = 5 * 1024 * 1024;
let nextId = 1;

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

export default function DocumentsStep({
  apiKey,
  onApiKeyChange,
  session,
  onUploaded,
  onReset,
}) {
  const [tab, setTab] = useState("upload");
  const [docs, setDocs] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const fileInput = useRef(null);

  function addFiles(fileList) {
    const accepted = [];
    const skipped = [];

    for (const file of Array.from(fileList)) {
      if (!file.name.toLowerCase().endsWith(".txt")) {
        skipped.push(`${file.name} (only .txt files)`);
      } else if (file.size === 0) {
        skipped.push(`${file.name} (empty)`);
      } else if (file.size > MAX_FILE_BYTES) {
        skipped.push(`${file.name} (over 5 MB)`);
      } else {
        accepted.push({
          id: nextId++,
          kind: "file",
          name: file.name,
          size: file.size,
          file,
        });
      }
    }

    setDocs((prev) => {
      const seen = new Set(prev.map((d) => `${d.kind}:${d.name}:${d.size}`));
      return [
        ...prev,
        ...accepted.filter((d) => !seen.has(`${d.kind}:${d.name}:${d.size}`)),
      ];
    });
    setNotice(skipped.length ? `Skipped: ${skipped.join(", ")}` : "");
    setError("");
  }

  function addPastedText() {
    const content = text.trim();
    if (!content) return;
    const pastedCount = docs.filter((d) => d.kind === "text").length;
    setDocs((prev) => [
      ...prev,
      {
        id: nextId++,
        kind: "text",
        name: title.trim() || `Pasted text ${pastedCount + 1}`,
        size: new Blob([content]).size,
        content,
      },
    ]);
    setTitle("");
    setText("");
    setNotice("");
    setError("");
  }

  function removeDoc(id) {
    setDocs((prev) => prev.filter((d) => d.id !== id));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setUploading(true);
    try {
      const result = await uploadDocuments({
        apiKey: apiKey.trim(),
        files: docs.filter((d) => d.kind === "file").map((d) => d.file),
        texts: docs
          .filter((d) => d.kind === "text")
          .map((d) => ({ name: d.name, content: d.content })),
      });
      onUploaded(result);
      setDocs([]);
      setNotice("");
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  const canSubmit = apiKey.trim().length > 0 && docs.length > 0 && !uploading;

  // ---- Success state ------------------------------------------------------
  if (session) {
    return (
      <section className="card card-success">
        <div className="step-head">
          <span className="step-badge step-done">
            <CheckIcon width={16} height={16} />
          </span>
          <div>
            <h2>Documents uploaded successfully</h2>
            <p className="muted">
              {plural(session.documents.length, "document")} indexed as{" "}
              {plural(session.chunks, "chunk")} ·{" "}
              {session.characters.toLocaleString()} characters. You can now ask
              questions below.
            </p>
          </div>
          <button type="button" className="btn btn-ghost" onClick={onReset}>
            Use different documents
          </button>
        </div>
        <ul className="doc-chips">
          {session.documents.map((d, i) => (
            <li key={`${d.name}-${i}`} className="doc-chip">
              <FileIcon width={15} height={15} />
              <span>{d.name}</span>
            </li>
          ))}
        </ul>
      </section>
    );
  }

  // ---- Upload form --------------------------------------------------------
  return (
    <section className="card">
      <div className="step-head">
        <span className="step-badge">1</span>
        <div>
          <h2>Add your API key and documents</h2>
          <p className="muted">
            Plain-text (.txt) documents only. Upload files, or paste the full
            text of one or more documents.
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} noValidate>
        <ApiKeyField
          value={apiKey}
          onChange={onApiKeyChange}
          disabled={uploading}
        />

        <div className="field">
          <span className="label">Documents</span>

          <div className="tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === "upload"}
              className={`tab ${tab === "upload" ? "tab-active" : ""}`}
              onClick={() => setTab("upload")}
            >
              Upload .txt files
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "paste"}
              className={`tab ${tab === "paste" ? "tab-active" : ""}`}
              onClick={() => setTab("paste")}
            >
              Paste text
            </button>
          </div>

          {tab === "upload" ? (
            <div
              className={`dropzone ${dragging ? "dropzone-active" : ""}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                addFiles(e.dataTransfer.files);
              }}
              onClick={() => fileInput.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  fileInput.current?.click();
                }
              }}
            >
              <UploadIcon width={26} height={26} />
              <p>
                <strong>Drop .txt files here</strong> or click to browse
              </p>
              <p className="hint">Up to 5 MB per file. You can add several.</p>
              <input
                ref={fileInput}
                type="file"
                accept=".txt,text/plain"
                multiple
                hidden
                onChange={(e) => {
                  addFiles(e.target.files);
                  e.target.value = "";
                }}
              />
            </div>
          ) : (
            <div className="paste">
              <input
                className="text-input"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Document name (optional)"
                maxLength={80}
              />
              <textarea
                className="textarea"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste the entire text of a document here…"
                rows={8}
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={addPastedText}
                disabled={!text.trim()}
              >
                Add as document
              </button>
            </div>
          )}

          {notice && <p className="notice">{notice}</p>}

          {docs.length > 0 && (
            <ul className="doc-list">
              {docs.map((d) => (
                <li key={d.id} className="doc-row">
                  <FileIcon width={18} height={18} />
                  <span className="doc-name">{d.name}</span>
                  <span className="doc-meta">
                    {d.kind === "text" ? "pasted · " : ""}
                    {formatSize(d.size)}
                  </span>
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={() => removeDoc(d.id)}
                    aria-label={`Remove ${d.name}`}
                    disabled={uploading}
                  >
                    <XIcon width={16} height={16} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {error && (
          <div className="alert" role="alert">
            <AlertIcon width={18} height={18} />
            <span>{error}</span>
          </div>
        )}

        <div className="actions">
          <button type="submit" className="btn btn-primary" disabled={!canSubmit}>
            {uploading ? (
              <>
                <span className="spinner" /> Indexing documents…
              </>
            ) : (
              <>
                <UploadIcon width={17} height={17} /> Upload documents
              </>
            )}
          </button>
          {uploading && (
            <span className="muted small">
              The first upload can take a little longer while the embedding model
              loads.
            </span>
          )}
        </div>
      </form>
    </section>
  );
}
