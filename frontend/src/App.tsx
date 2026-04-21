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
  const [state, setState] = useState<AppState>({
    queryResult: null,
    mlResult: null,
    llmResult: null,
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const handleQuery = useCallback(async (query: string) => {
    if (!query) return;

    setState((s) => ({ ...s, isLoading: true, error: null }));

    try {
      // Fire query + both priority predictions concurrently
      const [queryRes, mlRes, llmRes] = await Promise.all([
        api.query(query),
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
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="app-header">
        <div className="app-header-brand">
          <span className="app-header-logo">DIA</span>
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

      {/* ── Body ───────────────────────────────────────────────────────── */}
      <div className="app-body">
        {/* Left: query + sources */}
        <div className="left-panel">
          <QueryInput onSubmit={handleQuery} isLoading={isLoading} />
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
              <div className="empty-state-glyph">⌬</div>
              <div className="empty-state-title">Ready to analyse</div>
              <div className="empty-state-hint">
                Type a support ticket on the left. The system will retrieve
                similar cases, generate RAG and non-RAG answers, and compare
                ML vs LLM priority prediction.
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
