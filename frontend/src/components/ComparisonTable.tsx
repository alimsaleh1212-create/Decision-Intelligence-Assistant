import type { PriorityResponse } from "../types";

interface Props {
  mlResult: PriorityResponse | null;
  llmResult: PriorityResponse | null;
  isLoading: boolean;
}

function latencyClass(ms: number): string {
  if (ms < 50) return "fast";
  if (ms < 1000) return "slow";
  return "vslow";
}

function LabelChip({ label }: { label: "urgent" | "normal" }) {
  return <span className={`label-chip ${label}`}>{label}</span>;
}

function ConfidenceBar({ value, label }: { value: number | null; label: "urgent" | "normal" }) {
  if (value === null) {
    return (
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-muted)" }}>
        n/a
      </span>
    );
  }
  return (
    <div className="confidence-bar">
      <div className="confidence-track">
        <div
          className={`confidence-fill ${label}`}
          style={{ width: `${(value * 100).toFixed(0)}%` }}
        />
      </div>
      <span className="confidence-val">{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

interface RowProps {
  name: string;
  sub: string;
  result: PriorityResponse | null;
  isLoading: boolean;
}

function PredictorRow({ name, sub, result, isLoading }: RowProps) {
  const skeletonCell = (
    <div
      className="skeleton-line"
      style={{ width: "70%", height: 10, display: "inline-block" }}
    />
  );

  if (isLoading || !result) {
    return (
      <tr>
        <td>
          <div className="predictor-name">{name}</div>
          <div className="predictor-sub">{sub}</div>
        </td>
        <td>{isLoading ? skeletonCell : "—"}</td>
        <td>{isLoading ? skeletonCell : "—"}</td>
        <td>{isLoading ? skeletonCell : "—"}</td>
        <td>{isLoading ? skeletonCell : "—"}</td>
      </tr>
    );
  }

  return (
    <tr>
      <td>
        <div className="predictor-name">{name}</div>
        <div className="predictor-sub">{result.provider}</div>
      </td>
      <td>
        <LabelChip label={result.label} />
      </td>
      <td>
        <ConfidenceBar value={result.confidence} label={result.label} />
      </td>
      <td>
        <span className={`metric-val ${latencyClass(result.latency_ms)}`}>
          {result.latency_ms < 1
            ? "<1 ms"
            : result.latency_ms < 1000
            ? `${result.latency_ms.toFixed(0)} ms`
            : `${(result.latency_ms / 1000).toFixed(1)} s`}
        </span>
      </td>
      <td>
        <span className="cost-free">$0.00</span>
      </td>
    </tr>
  );
}

export function ComparisonTable({ mlResult, llmResult, isLoading }: Props) {
  return (
    <div className="comparison-panel">
      <div className="comparison-header">
        <span className="comparison-title">Priority Predictor Comparison</span>
        <span className="comparison-subtitle">ML classifier vs LLM zero-shot</span>
      </div>
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Predictor</th>
            <th>Label</th>
            <th>Confidence</th>
            <th>Latency</th>
            <th>Cost / call</th>
          </tr>
        </thead>
        <tbody>
          <PredictorRow
            name="ML Classifier"
            sub="scikit-learn · engineered features"
            result={mlResult}
            isLoading={isLoading}
          />
          <PredictorRow
            name="LLM Zero-shot"
            sub="Ollama / Gemini fallback"
            result={llmResult}
            isLoading={isLoading}
          />
        </tbody>
      </table>
    </div>
  );
}
