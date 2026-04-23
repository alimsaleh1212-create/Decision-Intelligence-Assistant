/** TypeScript interfaces mirroring the backend Pydantic schemas. */

export interface RetrievedTicket {
  text: string;
  score: number;
  brand: string;
}

export interface QueryResponse {
  query: string;
  rag_answer: string;
  non_rag_answer: string;
  retrieved_tickets: RetrievedTicket[];
}

export interface PriorityResponse {
  label: "urgent" | "normal";
  confidence: number | null;
  latency_ms: number;
  provider: string;
  cost_usd: number;
}

export interface HealthResponse {
  status: string;
  ollama: { reachable: boolean; detail: string };
  qdrant: { reachable: boolean; detail: string };
  gemini_configured: boolean;
}

export interface BrandsResponse {
  brands: string[];
}

export interface PredictorSnapshot {
  label: "urgent" | "normal";
  confidence: number | null;
  latency_ms: number;
  provider: string;
  cost_usd: number;
}

export interface ObservationRecord {
  id: string;
  timestamp: string;
  query: string;
  brand: string | null;
  rag_score_threshold: number | null;
  rag_answer: string;
  non_rag_answer: string;
  retrieved_tickets_count: number;
  ml: PredictorSnapshot;
  llm: PredictorSnapshot;
}

export interface RecordRequest {
  query: string;
  brand?: string | null;
  rag_score_threshold?: number | null;
  rag_answer: string;
  non_rag_answer: string;
  retrieved_tickets_count: number;
  ml: PredictorSnapshot;
  llm: PredictorSnapshot;
}

export interface LogsResponse {
  records: ObservationRecord[];
  total: number;
}

export interface MetricsResponse {
  total_queries: number;
  avg_llm_latency_ms: number;
  avg_ml_latency_ms: number;
  total_cost_usd: number;
  urgent_rate: number;
  ml_urgent_rate: number;
}
