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
  gemini_fallback_configured: boolean;
}

export interface BrandsResponse {
  brands: string[];
}
