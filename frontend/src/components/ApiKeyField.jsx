import { LockIcon } from "./icons.jsx";

// Masked input: the key is never shown on screen, never put in the URL and
// never saved to localStorage. It lives in React state until the tab closes.
export default function ApiKeyField({ value, onChange, disabled }) {
  return (
    <div className="field">
      <label htmlFor="api-key">Gemini API key</label>
      <div className="input-wrap">
        <LockIcon className="input-icon" />
        <input
          id="api-key"
          name="gemini-api-key"
          type="password"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Paste your API key"
          autoComplete="new-password"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          data-1p-ignore="true"
          data-lpignore="true"
          disabled={disabled}
        />
      </div>
      <p className="hint">
        Hidden as you type and kept in memory only. It is sent with your requests to
        the backend and Gemini, and is never stored.
      </p>
    </div>
  );
}
