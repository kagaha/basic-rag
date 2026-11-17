// Thin client for the FastAPI backend.
// In development Vite proxies /api to http://localhost:8000.
// For a deployed build, set VITE_API_BASE to the backend URL.
const BASE = import.meta.env.VITE_API_BASE ?? "";

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, options);
  } catch {
    throw new Error(
      "Cannot reach the backend. Make sure it is running (uvicorn backend.main:app --port 8000)."
    );
  }

  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : "";
    } catch {
      // body was not JSON
    }
    throw new Error(detail || `Request failed (${response.status}).`);
  }

  return response.json();
}

export function getConfig() {
  return request("/api/config");
}

// The API key travels in a header and is never stored by the frontend.
export function uploadDocuments({ apiKey, files, texts }) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  texts.forEach((t) => {
    form.append("texts", t.content);
    form.append("text_names", t.name);
  });
  return request("/api/documents", {
    method: "POST",
    headers: { "X-Api-Key": apiKey },
    body: form,
  });
}

export function askQuestion({ apiKey, sessionId, question }) {
  return request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Api-Key": apiKey },
    body: JSON.stringify({ session_id: sessionId, question }),
  });
}

export function endSession(sessionId) {
  return request(`/api/session/${sessionId}`, { method: "DELETE" });
}
