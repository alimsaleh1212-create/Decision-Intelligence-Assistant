import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import { AnswerPanel } from "./components/AnswerPanel";
import { ComparisonTable } from "./components/ComparisonTable";
import { ObservabilityPage } from "./components/ObservabilityPage";
import { QueryInput } from "./components/QueryInput";
import { SourcePanel } from "./components/SourcePanel";
import type { HealthResponse, PriorityResponse, QueryResponse } from "./types";

type Tab = "intelligence" | "observability";

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
  const [activeTab, setActiveTab] = useState<Tab>("intelligence");
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

  const handleQuery = useCallback(async (query: string, brand?: string, threshold?: number) => {
    if (!query) return;

    setState((s) => ({ ...s, isLoading: true, error: null }));

    try {
      const [queryRes, mlRes, llmRes] = await Promise.all([
        api.query(query, brand, threshold),
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

      // Fire-and-forget: record observation for the observability page.
      api.recordObservation({
        query,
        brand: brand ?? null,
        rag_score_threshold: threshold ?? null,
        rag_answer: queryRes.rag_answer,
        non_rag_answer: queryRes.non_rag_answer,
        retrieved_tickets_count: queryRes.retrieved_tickets.length,
        ml: {
          label: mlRes.label,
          confidence: mlRes.confidence,
          latency_ms: mlRes.latency_ms,
          provider: mlRes.provider,
          cost_usd: mlRes.cost_usd,
        },
        llm: {
          label: llmRes.label,
          confidence: llmRes.confidence,
          latency_ms: llmRes.latency_ms,
          provider: llmRes.provider,
          cost_usd: llmRes.cost_usd,
        },
      }).catch(() => {
        // Observability is non-critical — silently ignore failures.
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
          {health?.gemini_configured && (
            <StatusDot label="GEMINI" reachable={true} />
          )}
        </div>
      </header>

      {/* ── Tab nav ─────────────────────────────────────────────────────── */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === "intelligence" ? "active" : ""}`}
          onClick={() => setActiveTab("intelligence")}
        >
          Intelligence
        </button>
        <button
          className={`tab-btn ${activeTab === "observability" ? "active" : ""}`}
          onClick={() => setActiveTab("observability")}
        >
          Observability
        </button>
      </nav>

      {/* ── Intelligence tab ────────────────────────────────────────────── */}
      {activeTab === "intelligence" && (
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
      )}

      {/* ── Observability tab ────────────────────────────────────────────── */}
      {activeTab === "observability" && <ObservabilityPage />}
    </div>
  );
}
