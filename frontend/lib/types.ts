export type Point = { x: number; y: number };

export interface Village {
  id: number;
  name: string;
  x: number;
  y: number;
  population?: number | null;
  disaster_level?: number | null;
  is_priority?: number;
  distance_to_center?: number;
  road_accessibility?: number | null;
  weather_code?: number;
  hist_demand_mean?: number | null;
  demand?: number;
}

export interface Road {
  id: number;
  road_id?: string;
  from: number;
  to: number;
  length: number;
  base_speed?: number;
  base_speed_is_fallback?: boolean;
  terrain?: string;
  weather?: string;
  weather_code?: number;
  geology_risk?: number | null;
  weather_sensitivity?: number;
  hist_break_count?: number | null;
  risk_score: number;
  adjusted_speed?: number;
}

export interface SchedulerConfig {
  num_vehicles: number;
  vehicle_capacity: number;
  vehicle_time_limit: number;
  risk_cost_alpha: number;
  speed_penalty: number;
  time_limit: number;
  max_states: number;
}

export interface Scenario {
  date?: string;
  label?: string;
  weather?: string;
  source?: 'example' | 'manual' | 'excel';
  input_mode?: 'minimal' | 'full';
  scenario_hash?: string;
  villages: Record<string, Village>;
  roads: Record<string, Road>;
  demands: Record<string, number>;
  risk_scores: Record<string, number>;
  adjusted_speed: Record<string, number>;
  priority_villages: number[];
  isolated_villages: number[];
  config?: SchedulerConfig | null;
  metadata?: Record<string, unknown>;
}

export interface TrajectoryFrame {
  time: number;
  location: number;
  load: number;
  delivered: Record<string, number>;
  action: 'depart' | 'deliver' | 'move' | 'reload' | string;
}

export interface DeliveryRow { demand: number; delivered: number; satisfied_pct: number; }

export interface SolutionMetrics {
  slowest_vehicle_time?: number;
  total_distance?: number;
  priority_villages_satisfied?: number;
  total_exposed_risk?: number;
  high_risk_edge_visits?: number;
  total_demand_satisfied_rate?: number;
  unserved_demand_total?: number;
  [key: string]: number | string | undefined;
}

export interface Solution {
  scenario_date?: string;
  ablation_mode?: string;
  config_snapshot?: Partial<SchedulerConfig>;
  vehicle_trajectories: Record<string, TrajectoryFrame[]>;
  village_delivery_summary: Record<string, DeliveryRow>;
  metrics: SolutionMetrics;
  solver_status: string;
  solve_time_actual?: number;
  generated_at?: string;
  scenario_hash?: string;
  group?: string;
  scenario_source?: string;
  scenario_config?: SchedulerConfig;
  summary?: PlanSummary;
}

export interface PlanSummary {
  scenario_date?: string;
  scenario_label?: string;
  weather?: string;
  group?: string;
  group_label?: string;
  solver_status?: string;
  solve_time_actual?: number;
  metrics: SolutionMetrics;
  priority_villages: Array<{ id: number; name: string; demand: number; delivered: number }>;
  top_risk_roads: Array<{ road_id: string; from: number; to: number; risk_score: number }>;
  isolated_villages: number[];
  village_count: number;
  road_count: number;
  vehicle_count: number;
  villages: Array<Record<string, unknown>>;
  roads: Array<Record<string, unknown>>;
  vehicles: Array<Record<string, unknown>>;
}

export interface ExampleMeta { date: string; label: string; weather: string; modes: string[]; }
export interface ExampleIndex { default_date: string; default_group: string; scenarios: ExampleMeta[]; }
export interface ExamplePayload { date: string; group: string; scenario: Scenario; solution: Solution; summary: PlanSummary; layout: Record<string, Point>; }

export interface ActiveScenario {
  scenario: Scenario;
  source: 'manual' | 'excel' | 'example';
  label: string;
  scenario_hash: string;
  config_schema_version: number;
  created_at: string;
  example_date?: string | null;
  config: SchedulerConfig;
}

export interface PredictionStatus {
  row: number;
  name: string;
  field: string;
  status: 'kept_user' | 'predicted' | 'pending' | 'error';
  value: number | null;
  errors: string[];
}

export interface VillageInputRow {
  id?: number;
  name: string;
  x: number;
  y: number;
  demand?: number | null;
  priority: number;
  population?: number | null;
  disaster_level?: number | null;
  road_accessibility?: number | null;
  hist_demand_mean?: number | null;
}

export interface RoadInputRow {
  from: number;
  to: number;
  distance: number;
  risk?: number | null;
  base_speed?: number | null;
  geology_risk?: number | null;
  hist_break_count?: number | null;
}

export interface PredictionResult {
  villages: VillageInputRow[];
  roads: RoadInputRow[];
  village_status: PredictionStatus[];
  road_status: PredictionStatus[];
  prediction_metadata: Record<string, unknown>;
}

export interface CompareGroup { group: string; label: string; solution: Solution; summary: PlanSummary; }
export interface ComparePayload { date: string; scenario: Scenario; groups: CompareGroup[]; layout: Record<string, Point>; }

export interface ModelMeta {
  demand: { model_type: string; test_metrics: { MAE: number; RMSE: number; ['R²']: number }; trained_at: string };
  risk: { model_type: string; test_metrics: { accuracy: number; f1: number; auc: number }; trained_at: string };
  llm_configured: boolean;
  llm_provider?: string;
  llm_model?: string;
  suggested_questions: string[];
}

export interface LlmAnswer {
  mode: 'ai' | 'mock';
  status: string;
  answer: string;
  fallback_reason?: string | null;
  user_message?: string | null;
  error_type?: string | null;
}

export interface LlmTestResult {
  ok: boolean;
  provider: string;
  model: string;
  error_type?: string | null;
  message: string;
}
