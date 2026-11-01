'use client';

import { useMemo, useState } from 'react';
import { KpiCard } from '../components/KpiCard';
import { NetworkMap } from '../components/NetworkMap';
import { PageHeader } from '../components/PageHeader';
import { EmptyPanel, LoadingPanel } from '../components/States';
import { useApp } from '../lib/app-context';

export default function OverviewPage() {
  const { scenario, solution, summary, layout, loading, exampleList, selectedDate, selectedGroup, isUserScenario, loadExample, runCurrentSchedule } = useApp();
  const [actionError, setActionError] = useState('');
  const rows = useMemo(() => {
    if (!scenario) return [];
    return Object.values(scenario.villages).map((village) => {
      const delivery = solution?.village_delivery_summary?.[String(village.id)];
      const demand = Number(delivery?.demand ?? scenario.demands?.[String(village.id)] ?? village.demand ?? 0);
      const delivered = Number(delivery?.delivered ?? 0);
      return { id: village.id, name: village.name, disaster: Number(village.disaster_level ?? 0), demand, delivered, rate: demand > 0 ? Math.min(delivered, demand) / demand : null, priority: scenario.priority_villages.includes(Number(village.id)) };
    });
  }, [scenario, solution]);
  const metrics = summary?.metrics ?? solution?.metrics ?? {};
  const satisfaction = Number(metrics.total_demand_satisfied_rate);
  const slowest = Number(metrics.slowest_vehicle_time);
  const risk = Number(metrics.total_exposed_risk);
  const prioritySatisfied = Number(metrics.priority_villages_satisfied ?? 0);
  const priorityTotal = scenario?.priority_villages.length ?? 0;
  async function rerun() { setActionError(''); try { await runCurrentSchedule(); } catch (reason) { setActionError(reason instanceof Error ? reason.message : String(reason)); } }
  return (
    <>
      <PageHeader kicker="SITUATION OVERVIEW" title="态势总览" description="把 ML 预测结果、道路风险与多车调度方案放在同一张离线作战图上。" actions={<button className="button primary" onClick={rerun} disabled={loading || !scenario}>{loading ? '计算中…' : '重新计算当前方案'}</button>} />
      <section className="control-strip">
        <label><span>示例场景</span><select value={isUserScenario ? '' : selectedDate} onChange={(event) => event.target.value && loadExample(event.target.value, selectedGroup || 'C')}>{isUserScenario ? <option value="">当前用户场景</option> : null}{exampleList.map((example) => <option key={example.date} value={example.date}>{example.label} · {example.date}</option>)}</select></label>
        <div className="segmented">{['A', 'B', 'C'].map((group) => <button key={group} className={!isUserScenario && selectedGroup === group ? 'active' : ''} disabled={isUserScenario || loading} onClick={() => loadExample(selectedDate, group)}>方案 {group}</button>)}</div>
        <div className="control-note"><strong>{scenario?.weather || '—'}</strong><span>{scenario ? `${Object.keys(scenario.villages).length} 个村 · ${Object.keys(scenario.roads).length} 条航路` : '等待场景'}</span></div>
      </section>
      {actionError ? <div className="inline-error">{actionError}</div> : null}
      {loading && !solution ? <LoadingPanel label="正在载入示例方案…" /> : null}
      {scenario ? <>
        <section className="kpi-grid five">
          <KpiCard label="求解状态" value={solution?.solver_status || '未计算'} hint={`${Number(solution?.solve_time_actual ?? summary?.solve_time_actual ?? 0).toFixed(2)} 秒`} tone={solution?.solver_status === 'success' ? 'green' : 'amber'} icon="✓" />
          <KpiCard label="需求满足率" value={Number.isFinite(satisfaction) ? `${(satisfaction * 100).toFixed(1)}%` : '—'} hint={`未送达 ${Number(metrics.unserved_demand_total ?? 0).toFixed(1)} 箱`} tone="blue" icon="▣" />
          <KpiCard label="最慢车辆时间" value={Number.isFinite(slowest) ? `${slowest.toFixed(2)} h` : '—'} hint="所有车辆完成任务的瓶颈" tone="default" icon="◷" />
          <KpiCard label="暴露风险" value={Number.isFinite(risk) ? risk.toFixed(1) : '—'} hint={`高风险边访问 ${Number(metrics.high_risk_edge_visits ?? 0)} 次`} tone="red" icon="△" />
          <KpiCard label="优先村" value={`${prioritySatisfied} / ${priorityTotal}`} hint="按需求完成状态统计" tone="amber" icon="★" />
        </section>
        <section className="overview-grid">
          <div className="panel map-panel"><div className="panel-heading"><div><span className="eyebrow">OFFLINE NETWORK</span><h2>风险着色线路图</h2></div><span className="muted">节点为本地坐标，不依赖在线地图</span></div><NetworkMap scenario={scenario} solution={solution} layout={layout} /></div>
          <div className="panel demand-panel"><div className="panel-heading"><div><span className="eyebrow">DEMAND BOARD</span><h2>村庄需求与送达</h2></div></div><div className="table-wrap"><table><thead><tr><th>村庄</th><th>灾情</th><th>需求</th><th>已送达</th><th>满足率</th></tr></thead><tbody>{rows.map((row) => <tr key={row.id} className={row.priority ? 'priority-row' : ''}><td><span className="village-name">{row.priority ? '★ ' : ''}{row.name}</span></td><td><span className={`level level-${Math.min(row.disaster, 3)}`}>L{row.disaster}</span></td><td>{row.demand.toFixed(1)}</td><td>{row.delivered.toFixed(1)}</td><td>{row.rate === null ? '—' : `${(row.rate * 100).toFixed(1)}%`}</td></tr>)}</tbody></table></div></div>
        </section>
      </> : !loading ? <EmptyPanel title="尚未载入场景" detail="请在后端可用后刷新页面，或前往“场景与需求”创建自定义场景。" /> : null}
    </>
  );
}
