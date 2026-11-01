import type { ActiveScenario, ComparePayload, ExampleIndex, ExamplePayload, ModelMeta, PredictionResult, RoadInputRow, Scenario, SchedulerConfig, Solution, VillageInputRow } from './types';

const configuredBase = process.env.NEXT_PUBLIC_API_BASE_URL;
const isLocalBrowser = typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname);
const API_BASE = (configuredBase || (isLocalBrowser ? 'http://127.0.0.1:8000' : '')).replace(/\/$/, '');
type ApiEnvelope<T> = { ok: true; data: T } | { ok: false; error: string };

async function request<T>(action: string, options: { method?: 'GET' | 'POST'; params?: Record<string, string | number | undefined>; body?: unknown } = {}): Promise<T> {
  const search = new URLSearchParams({ action });
  Object.entries(options.params ?? {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') search.set(key, String(value));
  });
  const response = await fetch(`${API_BASE}/api/index?${search.toString()}`, {
    method: options.method ?? 'GET',
    headers: options.body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    cache: 'no-store',
  });
  let payload: ApiEnvelope<T>;
  try {
    payload = (await response.json()) as ApiEnvelope<T>;
  } catch {
    throw new Error(`后端返回了非 JSON 响应（HTTP ${response.status}）`);
  }
  if (!response.ok || !payload.ok) throw new Error(payload.ok ? `HTTP ${response.status}` : payload.error);
  return payload.data;
}

export const api = {
  health: () => request<{ status: string }>('health'),
  meta: () => request<ModelMeta>('meta'),
  examples: () => request<ExampleIndex>('examples'),
  example: (date: string, group: string) => request<ExamplePayload>('example', { params: { date, group } }),
  compare: (date: string) => request<ComparePayload>('compare', { params: { date } }),
  build: (payload: { villages: VillageInputRow[]; roads: RoadInputRow[]; config: SchedulerConfig; weather: string; input_mode: string; predict_missing: boolean; overwrite_existing?: boolean; source?: string; label?: string }) => request<ActiveScenario>('build', { method: 'POST', body: payload }),
  predict: (payload: { villages: VillageInputRow[]; roads: RoadInputRow[]; weather: string; source?: string; overwrite_existing?: boolean }) => request<PredictionResult>('predict', { method: 'POST', body: payload }),
  schedule: (payload: { scenario: Scenario; config: SchedulerConfig; mode: string; group?: string; date?: string }) => request<Solution>('schedule', { method: 'POST', body: payload }),
  explain: (payload: { scenario: Scenario; solution: Solution; group: string; question: string }) => request<{ mode: 'ai' | 'mock'; status: string; answer: string; fallback_reason?: string | null; user_message?: string | null }>('explain', { method: 'POST', body: payload }),
  importExcel: (payload: { data_base64: string; filename?: string; predict_missing?: boolean; config?: SchedulerConfig; weather?: string }) => request<ActiveScenario>('import-excel', { method: 'POST', body: payload }),
};

export async function downloadTemplate(): Promise<void> {
  const response = await fetch(`${API_BASE}/api/index?action=template`);
  if (!response.ok) throw new Error(`模板下载失败（HTTP ${response.status}）`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'scenario_template.xlsx';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export { API_BASE };
