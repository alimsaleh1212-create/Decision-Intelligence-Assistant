import { type KeyboardEvent, useState } from "react";

interface Props {
  onSubmit: (query: string) => void;
  isLoading: boolean;
}

const MAX_LEN = 2000;

export function QueryInput({ onSubmit, isLoading }: Props) {
  const [text, setText] = useState("");

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canSubmit) {
      e.preventDefault();
      onSubmit(text.trim());
    }
  };

  const canSubmit = text.trim().length > 0 && !isLoading;

  return (
    <div className="query-section">
      <div className="query-label">⌘ Query</div>
      <textarea
        className="query-textarea"
        placeholder="Describe the support issue or paste a ticket…"
        value={text}
        onChange={(e) => setText(e.target.value.slice(0, MAX_LEN))}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
        rows={4}
      />
      <div className="query-actions">
        <span className="query-char-count">
          {text.length} / {MAX_LEN}  ·  Ctrl+Enter to submit
        </span>
        <button
          className="btn-submit"
          onClick={() => onSubmit(text.trim())}
          disabled={!canSubmit}
        >
          {isLoading && <span className="spinner" />}
          {isLoading ? "Analysing" : "Analyse"}
        </button>
      </div>
    </div>
  );
}
