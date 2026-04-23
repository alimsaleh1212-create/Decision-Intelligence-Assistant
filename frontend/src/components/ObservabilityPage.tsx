import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { LogsResponse, MetricsResponse, ObservationRecord } from "../types";

// ── Metric card ─────────────────────────────────────────────────────────────

type AccentColor = "gold" | "teal" | "red" | "blue";

interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  accent: AccentColor;
  delay?: number;
}

function MetricCard({ label, value, sub, accent, delay = 0 }: MetricCardProps) {
  return (
    <div
      className={`obs-metric-card obs-metric-card--${accent}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={`obs-metric-accent obs-metric-accent--${accent}`} />
      <div className="obs-metric-label">{label}</div>
      <div className={`obs-metric-value obs-metric-value--${accent}`}>{value}</div>
      {sub && <div className="obs-metric-sub">{sub}</div>}
    </div>
  );
}

// ── Label chip ───────────────────────────────────────────────────────────────

function LabelPill({ label }: { label: string }) {
  return (
    <span className={`label-chip ${label === "urgent" ? "urgent" : "normal"}`}>
      {label}
    </span>
  );
}

// ── Trace drawer ─────────────────────────────────────────────────────────────

interface TraceDrawerProps {
  record: ObservationRecord | null;
  onClose: () => void;
}

function TraceDrawer({ record, onClose }: TraceDrawerProps) {
  const drawerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!record) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [record, onClose]);

  if (!record) return null;

  const ts = new Date(record.timestamp).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const fmtLatency = (ms: number) =>
    ms < 1 ? "<1 ms" : ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`;

  const fmtCost = (usd: number) =>
    usd === 0 ? "Free" : usd < 0.001 ? `$${usd.toFixed(6)}` : `$${usd.toFixed(4)}`;

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <div className="trace-drawer" ref={drawerRef}>
        {/* Header */}
        <div className="drawer-header">
          <div className="drawer-header-left">
            <span className="drawer-eyebrow">Trace</span>
            <span className="drawer-title">Request Detail</span>
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="drawer-body">
          {/* Query + meta */}
          <section className="drawer-section">
            <div className="drawer-section-label">Query</div>
            <div className="drawer-query-text">{record.query}</div>
            <div className="drawer-meta-row">
              <span className="drawer-ts">{ts}</span>
              {record.brand && <span className="drawer-badge">{record.brand}</span>}
              {record.rag_score_threshold != null && (
                <span className="drawer-badge">
                  threshold {record.rag_score_threshold.toFixed(2)}
                </span>
              )}
              <span className="drawer-badge">{record.retrieved_tickets_count} tickets</span>
            </div>
          </section>

          {/* Predictors side-by-side */}
          <section className="drawer-section">
            <div className="drawer-section-label">Predictors</div>
            <div className="drawer-predictors">
              {[
                { name: "ML Classifier", snap: record.ml, accent: "teal" as const },
                { name: "LLM Zero-shot", snap: record.llm, accent: "gold" as const },
              ].map(({ name, snap, accent }) => (
                <div className={`drawer-predictor-card drawer-predictor-card--${accent}`} key={name}>
                  <div className="drawer-predictor-header">
                    <span className="drawer-predictor-name">{name}</span>
                    <span className="drawer-predictor-provider">{snap.provider}</span>
                  </div>
                  <div className="drawer-predictor-stats">
                    <LabelPill label={snap.label} />
                    <div className="drawer-stat-group">
                      <span className="drawer-stat-label">conf</span>
                      <span className="drawer-stat-val">
                        {snap.confidence != null
                          ? `${(snap.confidence * 100).toFixed(0)}%`
                          : "n/a"}
                      </span>
                    </div>
                    <div className="drawer-stat-group">
                      <span className="drawer-stat-label">latency</span>
                      <span className="drawer-stat-val">{fmtLatency(snap.latency_ms)}</span>
                    </div>
                    <div className="drawer-stat-group">
                      <span className="drawer-stat-label">cost</span>
                      <span className={`drawer-stat-val ${snap.cost_usd > 0 ? "drawer-stat-cost" : ""}`}>
                        {fmtCost(snap.cost_usd)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* RAG answer */}
          <section className="drawer-section drawer-section--rag">
            <div className="drawer-section-label">
              <span className="drawer-label-dot drawer-label-dot--teal" />
              RAG Answer
            </div>
            <div className="drawer-answer-text">{record.rag_answer}</div>
          </section>

          {/* Non-RAG answer */}
          <section className="drawer-section drawer-section--nonrag">
            <div className="drawer-section-label">
              <span className="drawer-label-dot drawer-label-dot--blue" />
              Non-RAG Answer
            </div>
            <div className="drawer-answer-text">{record.non_rag_answer}</div>
          </section>
        </div>
      </div>
    </>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────

export function ObservabilityPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [logs, setLogs] = useState<LogsResponse | null>(null);
  const [selected, setSelected] = useState<ObservationRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [m, l] = await Promise.all([api.getMetrics(), api.getLogs(100)]);
      setMetrics(m);
      setLogs(l);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load observability data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 15_000);
    return () => clearInterval(id);
  }, [refresh]);

  const fmtLatency = (ms: number) =>
    ms < 1 ? "<1 ms" : ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(1)} s`;

  const fmtCost = (usd: number) =>
    usd === 0 ? "$0.00" : usd < 0.001 ? `$${usd.toFixed(6)}` : `$${usd.toFixed(4)}`;

  return (
    <div className="obs-page">
      {/* ── Metrics bar ─────────────────────────────────────────────────── */}
      <div className="obs-metrics-bar">
        <MetricCard
          label="Total Queries"
          value={metrics ? String(metrics.total_queries) : "—"}
          sub="all time"
          accent="gold"
          delay={0}
        />
        <MetricCard
          label="Avg LLM Latency"
          value={metrics ? fmtLatency(metrics.avg_llm_latency_ms) : "—"}
          sub={metrics ? `ML avg: ${fmtLatency(metrics.avg_ml_latency_ms)}` : undefined}
          accent="teal"
          delay={60}
        />
        <MetricCard
          label="Total Cost"
          value={metrics ? fmtCost(metrics.total_cost_usd) : "—"}
          sub="Gemini usage"
          accent="red"
          delay={120}
        />
        <MetricCard
          label="LLM Urgent Rate"
          value={metrics ? `${(metrics.urgent_rate * 100).toFixed(0)}%` : "—"}
          sub={metrics ? `ML: ${(metrics.ml_urgent_rate * 100).toFixed(0)}%` : undefined}
          accent="blue"
          delay={180}
        />
      </div>

      {/* ── Controls ────────────────────────────────────────────────────── */}
      <div className="obs-controls">
        <div className="obs-section-heading">
          <span className="obs-section-title">Recent Requests</span>
          {logs && (
            <span className="obs-section-count">{logs.total} total</span>
          )}
        </div>
        <button className="obs-refresh-btn" onClick={refresh} disabled={loading}>
          {loading ? (
            <span className="obs-refresh-spinner" />
          ) : (
            "↺ Refresh"
          )}
        </button>
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}

      {/* ── Logs table ──────────────────────────────────────────────────── */}
      {logs && logs.records.length === 0 ? (
        <div className="obs-empty">
          <div className="obs-empty-glyph">◈</div>
          <div>No observations yet</div>
          <div className="obs-empty-sub">Submit a query on the Intelligence tab to start recording.</div>
        </div>
      ) : (
        <div className="obs-table-wrap">
          <table className="obs-table">
            <thead>
              <tr>
                <th></th>
                <th>Time</th>
                <th>Query</th>
                <th>ML</th>
                <th>LLM</th>
                <th>Latency</th>
                <th>Cost</th>
                <th>Tickets</th>
              </tr>
            </thead>
            <tbody>
              {(logs?.records ?? []).map((rec, i) => (
                <tr
                  key={rec.id}
                  className={`obs-row obs-row--${rec.llm.label}`}
                  onClick={() => setSelected(rec)}
                  style={{ animationDelay: `${i * 20}ms` }}
                >
                  <td className="obs-cell-indicator">
                    <span className={`obs-indicator obs-indicator--${rec.llm.label}`} />
                  </td>
                  <td className="obs-cell-ts">
                    {new Date(rec.timestamp).toLocaleTimeString(undefined, {
                      hour: "2-digit",
                      minute: "2-digit",
                      second: "2-digit",
                    })}
                  </td>
                  <td className="obs-cell-query">
                    <span className="obs-query-truncate">{rec.query}</span>
                  </td>
                  <td><LabelPill label={rec.ml.label} /></td>
                  <td><LabelPill label={rec.llm.label} /></td>
                  <td className="obs-cell-mono obs-cell-latency">
                    {fmtLatency(rec.llm.latency_ms)}
                  </td>
                  <td className="obs-cell-mono">
                    {rec.llm.cost_usd === 0 ? (
                      <span className="cost-free">Free</span>
                    ) : (
                      <span className="cost-value">{fmtCost(rec.llm.cost_usd)}</span>
                    )}
                  </td>
                  <td className="obs-cell-mono obs-cell-dim">{rec.retrieved_tickets_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Trace drawer ────────────────────────────────────────────────── */}
      <TraceDrawer record={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
