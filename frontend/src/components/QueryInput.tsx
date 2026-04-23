import { type KeyboardEvent, useState } from "react";

interface Props {
  onSubmit: (query: string, brand?: string) => void;
  isLoading: boolean;
  brands: string[];
}

const MAX_LEN = 2000;

export function QueryInput({ onSubmit, isLoading, brands }: Props) {
  const [text, setText] = useState("");
  const [brand, setBrand] = useState("");

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canSubmit) {
      e.preventDefault();
      onSubmit(text.trim(), brand || undefined);
    }
  };

  const canSubmit = text.trim().length > 0 && !isLoading;

  return (
    <div className="query-section">
      <div className="query-label">Query</div>

      {/* Brand filter row */}
      <div className="query-brand-row">
        <span className="query-brand-label">Brand</span>
        <div className="brand-select-wrapper">
          <select
            className="brand-select"
            value={brand}
            onChange={(e) => setBrand(e.target.value)}
            disabled={isLoading || brands.length === 0}
          >
            <option value="">All brands</option>
            {brands.map((b) => (
              <option key={b} value={b}>
                {b}
              </option>
            ))}
          </select>
        </div>
        {brand && (
          <button
            className="brand-active-badge"
            onClick={() => setBrand("")}
            title="Clear brand filter"
          >
            {brand}
            <span className="brand-active-badge-x">✕</span>
          </button>
        )}
      </div>

      <textarea
        className="query-textarea"
        placeholder="Describe the support issue or paste a ticket…"
        value={text}
        onChange={(e) => setText(e.target.value.slice(0, MAX_LEN))}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
        rows={5}
      />

      <div className="query-actions">
        <span className="query-char-count">
          {text.length} / {MAX_LEN} · Ctrl+Enter
        </span>
        <button
          className="btn-submit"
          onClick={() => onSubmit(text.trim(), brand || undefined)}
          disabled={!canSubmit}
        >
          {isLoading && <span className="spinner" />}
          {isLoading ? "Analysing" : "Analyse"}
        </button>
      </div>
    </div>
  );
}
