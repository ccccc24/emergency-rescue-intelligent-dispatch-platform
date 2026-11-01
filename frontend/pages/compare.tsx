'use client';

import { useEffect, useMemo, useState } from 'react';
import { NetworkMap } from '../components/NetworkMap';
import { PageHeader } from '../components/PageHeader';
import { EmptyPanel, LoadingPanel } from '../components/States';
import { api } from '../lib/api';
import { useApp } from '../lib/app-context';
import type { ComparePayload } from '../lib/types';

export default function ComparePage() {
  const { examples, exampleList, selectedDate, loadExample } = useApp();
  const [compareDate, setCompareDate] = useState(selectedDate || ''); const [payload, setPayload] = useState<ComparePayload | null>(null); const [focusGroup, setFocusGroup] = useState('C'); const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  useEffect(() => { if (!compareDate && examples?.default_date) setCompareDate(examples.default_date); }, [compareDate, examples]);
  useEffect(() => { if (!compareDate) return; let cancelled = false; setLoading(true); setError(''); api.compare(compareDate).then((result) => { if (!cancelled) setPayload(result); }).catch((reason) => { if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason)); }).finally(() => { if (!cancelled) setLoading(false); }); return () => { cancelled = true; }; }, [compareDate]);
  const groups = payload?.groups ?? []; const focused = groups.find((item) => item.group === focusGroup) ?? groups[groups.length - 1];
  const riskReduction = useMemo(() => { const baseline = groups.find((item) => item.group === 'B')?.summary.metrics.total_exposed_risk; const full = groups.find((item) => item.group === 'C')?.summary.metrics.total_exposed_risk; return baseline === undefined || full === undefined || Number(baseline) === 0 ? null : (Number(baseline) - Number(full)) / Number(baseline) * 100; }, [groups]);
  return (
    <>
      <PageHeader kicker="ABLATION BENCHMARK" title="方案对比" description="对比历史需求基线、无风险修正与完整 ML + 风险输入三组方案。" actions={<button className="button primary" disabled={!focused || !payload} onClick={() => payload && focused && loadExample(payload.date, focused.group)}>设为当前方案</button>} />
      <section className="control-strip"><label><span>对比场景</span><select value={compareDate} onChange={(e) => setCompareDate(e.target.value)}>{exampleList.map((e) => <option key={e.date} value={e.date}>{e.label} · {e.date}</option>)}</select></label><div className="segmented">{groups.map((g) => <button key={g.group} className={focusGroup === g.group ? 'active' : ''} onClick={() => setFocusGroup(g.group)}>{g.group} · {g.label}</button>)}</div><div className="control-note"><strong>{riskReduction === null ? '—' : `${riskReduction.toFixed(1)}%`}</strong><span>完整方案相对无风险修正的暴露风险变化</span></div></section>
      {loading ? <LoadingPanel label="正在读取三组实验方案…" /> : null}{error ? <div className="inline-error">{error}</div> : null}{!loading && !payload ? <EmptyPanel title="没有可对比的数据" detail="请选择一个示例场景。" /> : null}
      {payload ? <>
        <section className="comparison-cards">{groups.map((g) => <button key={g.group} className={`comparison-card ${focusGroup === g.group ? 'active' : ''}`} onClick={() => setFocusGroup(g.group)}><span className="comparison-code">{g.group}</span><div><strong>{g.label}</strong><small>{g.solution.solver_status} · {Number(g.summary.solve_time_actual ?? g.solution.solve_time_actual ?? 0).toFixed(2)}s</small></div><div className="comparison-score">{g.summary.metrics.total_demand_satisfied_rate === undefined ? '—' : `${(Number(g.summary.metrics.total_demand_satisfied_rate) * 100).toFixed(1)}%`}</div></button>)}</section>
        <section className="compare-layout"><div className="panel"><div className="panel-heading"><div><span className="eyebrow">METRIC MATRIX</span><h2>核心指标</h2></div><span className="muted">风险与时间越低越好</span></div><div className="table-wrap"><table className="comparison-table"><thead><tr><th>指标</th>{groups.map((g) => <th key={g.group}>{g.group}</th>)}</tr></thead><tbody><MetricRow label="需求满足率" values={groups.map((g) => Number(g.summary.metrics.total_demand_satisfied_rate ?? 0) * 100)} format={(v) => `${v.toFixed(1)}%`} best="max" /><MetricRow label="最慢车辆时间" values={groups.map((g) => Number(g.summary.metrics.slowest_vehicle_time ?? 0))} format={(v) => `${v.toFixed(2)}h`} best="min" /><MetricRow label="总暴露风险" values={groups.map((g) => Number(g.summary.metrics.total_exposed_risk ?? 0))} format={(v) => v.toFixed(1)} best="min" /><MetricRow label="高风险边访问" values={groups.map((g) => Number(g.summary.metrics.high_risk_edge_visits ?? 0))} format={(v) => v.toFixed(0)} best="min" /><MetricRow label="优先村完成数" values={groups.map((g) => Number(g.summary.metrics.priority_villages_satisfied ?? 0))} format={(v) => v.toFixed(0)} best="max" /></tbody></table></div></div>
          <div className="panel compare-chart-panel"><div className="panel-heading"><div><span className="eyebrow">RISK EXPOSURE</span><h2>风险暴露对比</h2></div></div><div className="bar-chart">{groups.map((g) => { const value = Number(g.summary.metrics.total_exposed_risk ?? 0); const max = Math.max(...groups.map((item) => Number(item.summary.metrics.total_exposed_risk ?? 0)), 1); return <div className="bar-row" key={g.group}><div><span>{g.group}</span><small>{g.label}</small></div><div className="bar-track"><i className={`bar-${g.group.toLowerCase()}`} style={{ width: `${Math.max(value / max * 100, 2)}%` }} /></div><strong>{value.toFixed(1)}</strong></div>; })}</div><div className="chart-footnote">C 组同时使用 ML 需求预测和风险修正；B 组关闭风险速度修正，用其作为风险规避效果的对照。</div></div>
        </section>
        <section className="panel compare-map-panel"><div className="panel-heading"><div><span className="eyebrow">SELECTED PLAN</span><h2>{focused?.group} · {focused?.label}</h2></div><span className="badge badge-soft">离线路网</span></div>{focused ? <NetworkMap scenario={payload.scenario} solution={focused.solution} layout={payload.layout} compact /> : null}</section>
      </> : null}
    </>
  );
}
function MetricRow({ label, values, format, best }: { label: string; values: number[]; format: (value: number) => string; best: 'min' | 'max' }) {
  const target = best === 'max' ? Math.max(...values) : Math.min(...values);
  return <tr><td>{label}</td>{values.map((value, index) => <td key={index} className={value === target ? 'metric-best' : ''}>{format(value)}</td>)}</tr>;
}
