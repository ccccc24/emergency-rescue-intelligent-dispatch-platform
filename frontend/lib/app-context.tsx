'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { api } from './api';
import type { ActiveScenario, ExampleIndex, ExampleMeta, PlanSummary, Point, Scenario, SchedulerConfig, Solution } from './types';

const MODE_BY_GROUP: Record<string, string> = { A: 'hist_demand', B: 'no_risk_adjust', C: 'full', user: 'full' };
const DEFAULT_CONFIG: SchedulerConfig = { num_vehicles: 3, vehicle_capacity: 100, vehicle_time_limit: 100, risk_cost_alpha: 0.2, speed_penalty: 0.1, time_limit: 10, max_states: 20000 };

function layoutFromScenario(scenario: Scenario): Record<string, Point> {
  const layout: Record<string, Point> = { '0': { x: 0, y: 0 } };
  Object.values(scenario.villages ?? {}).forEach((village) => { layout[String(village.id)] = { x: Number(village.x), y: Number(village.y) }; });
  return layout;
}

interface AppContextValue {
  ready: boolean; loading: boolean; error: string | null; examples: ExampleIndex | null; exampleList: ExampleMeta[];
  selectedDate: string; selectedGroup: string; scenario: Scenario | null; solution: Solution | null; summary: PlanSummary | null;
  layout: Record<string, Point>; config: SchedulerConfig; source: string; isUserScenario: boolean;
  loadExample: (date: string, group?: string) => Promise<void>;
  runCurrentSchedule: (override?: { scenario?: Scenario; config?: SchedulerConfig; group?: string }) => Promise<Solution>;
  applyActiveScenario: (active: ActiveScenario, result?: Solution | null) => void;
  setConfig: (config: SchedulerConfig) => void;
  setErrorMessage: (message: string | null) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [examples, setExamples] = useState<ExampleIndex | null>(null);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedGroup, setSelectedGroup] = useState('C');
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [solution, setSolution] = useState<Solution | null>(null);
  const [summary, setSummary] = useState<PlanSummary | null>(null);
  const [layout, setLayout] = useState<Record<string, Point>>({});
  const [config, setConfigState] = useState<SchedulerConfig>(DEFAULT_CONFIG);
  const [source, setSource] = useState('example');

  const loadExample = useCallback(async (date: string, group = 'C') => {
    setLoading(true); setError(null);
    try {
      const payload = await api.example(date, group);
      setSelectedDate(date); setSelectedGroup(group); setScenario(payload.scenario); setSolution(payload.solution);
      setSummary(payload.summary); setLayout(payload.layout); setSource('example');
      setConfigState({ ...DEFAULT_CONFIG, ...(payload.scenario.config ?? payload.solution.scenario_config ?? {}) });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally { setLoading(false); }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const index = await api.examples();
        if (cancelled) return;
        setExamples(index);
        await loadExample(index.default_date, index.default_group);
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      } finally {
        if (!cancelled) { setLoading(false); setReady(true); }
      }
    })();
    return () => { cancelled = true; };
  }, [loadExample]);

  const applyActiveScenario = useCallback((active: ActiveScenario, result?: Solution | null) => {
    setScenario(active.scenario); setSource(active.source); setConfigState({ ...DEFAULT_CONFIG, ...active.config });
    setLayout(layoutFromScenario(active.scenario)); setSolution(result ?? null); setSummary(result?.summary ?? null);
    setSelectedDate(active.example_date ?? ''); setSelectedGroup(active.source === 'example' && active.example_date ? (result?.group ?? selectedGroup) : 'user');
  }, [selectedGroup]);

  const runCurrentSchedule = useCallback(async (override?: { scenario?: Scenario; config?: SchedulerConfig; group?: string }) => {
    const targetScenario = override?.scenario ?? scenario;
    if (!targetScenario) throw new Error('当前没有可调度场景');
    const targetGroup = override?.group ?? (source === 'example' ? selectedGroup : 'user');
    const targetConfig = override?.config ?? config;
    setLoading(true); setError(null);
    try {
      const result = await api.schedule({ scenario: targetScenario, config: targetConfig, mode: MODE_BY_GROUP[targetGroup] ?? 'full', group: targetGroup });
      setSolution(result); setSummary(result.summary ?? null); setSelectedGroup(targetGroup); return result;
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : String(reason); setError(message); throw reason;
    } finally { setLoading(false); }
  }, [config, scenario, selectedGroup, source]);

  const exampleList = examples?.scenarios ?? [];
  const value = useMemo<AppContextValue>(() => ({
    ready, loading, error, examples, exampleList, selectedDate, selectedGroup, scenario, solution, summary, layout, config, source,
    isUserScenario: source !== 'example', loadExample, runCurrentSchedule, applyActiveScenario, setConfig: setConfigState, setErrorMessage: setError,
  }), [applyActiveScenario, config, error, exampleList, examples, layout, loadExample, loading, ready, runCurrentSchedule, scenario, selectedDate, selectedGroup, solution, source, summary]);

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const value = useContext(AppContext);
  if (!value) throw new Error('useApp must be used inside AppProvider');
  return value;
}

export { DEFAULT_CONFIG, MODE_BY_GROUP };
