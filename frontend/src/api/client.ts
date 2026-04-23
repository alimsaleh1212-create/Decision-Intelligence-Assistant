/**
 * Typed fetch wrappers for all backend endpoints.
 *
 * BASE_URL resolves to the Vite proxy in dev (empty string = same origin),
 * or the injected VITE_API_BASE_URL in production builds.
 */

import type {
  BrandsResponse,
  HealthResponse,
  LogsResponse,
  MetricsResponse,
  ObservationRecord,
  PriorityResponse,
  QueryResponse,
  RecordRequest,
} from "../types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }

  return response.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: (): Promise<HealthResponse> =>
    get<HealthResponse>("/api/health"),

  brands: (): Promise<BrandsResponse> =>
    get<BrandsResponse>("/api/brands"),

  query: (query: string, brand?: string, ragScoreThreshold?: number): Promise<QueryResponse> =>
    post<QueryResponse>("/api/query", {
      query,
      brand: brand || null,
      rag_score_threshold: ragScoreThreshold ?? null,
    }),

  priorityML: (text: string): Promise<PriorityResponse> =>
    post<PriorityResponse>("/api/priority/ml", { text }),

  priorityLLM: (text: string): Promise<PriorityResponse> =>
    post<PriorityResponse>("/api/priority/llm", { text }),

  recordObservation: (req: RecordRequest): Promise<ObservationRecord> =>
    post<ObservationRecord>("/api/observability/record", req),

  getLogs: (limit = 100): Promise<LogsResponse> =>
    get<LogsResponse>(`/api/observability/logs?limit=${limit}`),

  getMetrics: (): Promise<MetricsResponse> =>
    get<MetricsResponse>("/api/observability/metrics"),
};
