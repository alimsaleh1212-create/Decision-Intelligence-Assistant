import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import { AnswerPanel } from "./components/AnswerPanel";
import { ComparisonTable } from "./components/ComparisonTable";
import { QueryInput } from "./components/QueryInput";
import { SourcePanel } from "./components/SourcePanel";
import type { HealthResponse, PriorityResponse, QueryResponse } from "./types";

interface AppState {
  queryResult: QueryResponse | null;
  mlResult: PriorityResponse | null;
  llmResult: PriorityResponse | null;
  isLoading: boolean;
  error: string | null;
}

function StatusDot({ label, reachable }: { label: string; reachable: boolean | null }) {
  const cls = reachable === null ? "" : reachable ? "ok" : "error";
  return <span className={`status-dot ${cls}`}>{label}</span>;
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [brands, setBrands] = useState<string[]>([]);
  const [state, setState] = useState<AppState>({
    queryResult: null,
    mlResult: null,
    llmResult: null,
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.brands().then((r) => setBrands(r.brands)).catch(() => setBrands([]));
  }, []);

  const handleQuery = useCallback(async (query: string, brand?: string) => {
    if (!query) return;

    setState((s) => ({ ...s, isLoading: true, error: null }));

    try {
      const [queryRes, mlRes, llmRes] = await Promise.all([
        api.query(query, brand),
        api.priorityML(query),
        api.priorityLLM(query),
      ]);

      setState({
        queryResult: queryRes,
        mlResult: mlRes,
        llmResult: llmRes,
        isLoading: false,
        error: null,
      });
    } catch (err) {
      setState((s) => ({
        ...s,
        isLoading: false,
        error: err instanceof Error ? err.message : "Unknown error",
      }));
    }
  }, []);

  const { queryResult, mlResult, llmResult, isLoading, error } = state;

  return (
    <div className="app-shell">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="app-header-brand">
          <span className="app-header-logo">DIA</span>
          <div className="app-header-divider" />
          <span className="app-header-sub">Decision Intelligence Assistant</span>
        </div>
        <div className="status-dots">
          <StatusDot label="OLLAMA" reachable={health?.ollama.reachable ?? null} />
          <StatusDot label="QDRANT" reachable={health?.qdrant.reachable ?? null} />
          {health?.gemini_fallback_configured && (
            <StatusDot label="GEMINI" reachable={true} />
          )}
        </div>
      </header>

      {/* ── Body ────────────────────────────────────────────────────────── */}
      <div className="app-body">
        {/* Left: query + sources */}
        <div className="left-panel">
          <QueryInput onSubmit={handleQuery} isLoading={isLoading} brands={brands} />
          <SourcePanel tickets={queryResult?.retrieved_tickets ?? []} />
        </div>

        {/* Right: answers + comparison */}
        <div className="right-panel">
          {error && (
            <div className="error-banner">
              ⚠ {error}
            </div>
          )}

          {!queryResult && !isLoading && !error ? (
            <div className="empty-state">
              <div className="empty-state-orb">
                <div className="empty-state-glyph">⌬</div>
              </div>
              <div className="empty-state-title">Ready to analyse</div>
              <div className="empty-state-hint">
                Type a support ticket on the left. The system retrieves similar
                cases, generates RAG and non-RAG answers, and compares ML vs
                LLM priority prediction.
              </div>
            </div>
          ) : (
            <>
              <AnswerPanel
                ragAnswer={queryResult?.rag_answer ?? null}
                nonRagAnswer={queryResult?.non_rag_answer ?? null}
                isLoading={isLoading}
              />
              <ComparisonTable
                mlResult={mlResult}
                llmResult={llmResult}
                isLoading={isLoading}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
