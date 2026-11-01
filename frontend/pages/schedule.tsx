'use client';

import { useEffect, useMemo, useState } from 'react';
import { KpiCard } from '../components/KpiCard';
import { NetworkMap } from '../components/NetworkMap';
import { PageHeader } from '../components/PageHeader';
import { EmptyPanel } from '../components/States';
import { useApp } from '../lib/app-context';

const VEHICLE_COLORS = ['#2563eb', '#e85d34', '#7c3aed', '#0891b2', '#be123c'];
export default function SchedulePage() {
  const { scenario, solution, summary, layout, loading, runCurrentSchedule } = useApp();
  const [playing, setPlaying] = useState(false); const [time, setTime] = useState(0); const [speed, setSpeed] = useState(1); const [error, setError] = useState('');
  const maxTime = useMemo(() => { const values = Object.values(solution?.vehicle_trajectories ?? {}).flat().map((frame) => Number(frame.time)); return values.length ? Math.max(...values) : 0; }, [solution]);
  useEffect(() => { setTime(0); setPlaying(false); }, [solution?.generated_at, solution?.scenario_hash]);
  useEffect(() => { if (!playing || maxTime <= 0) return; const timer = window.setInterval(() => setTime((current) => { const next = current + maxTime * 0.012 * speed; if (next >= maxTime) { setPlaying(false); return maxTime; } return next; }), 60); return () => window.clearInterval(timer); }, [maxTime, playing, speed]);
  async function calculate() { setError(''); try { await runCurrentSchedule(); } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } }
  if (!scenario) return <EmptyPanel title="尚未选择场景" detail="请先在“场景与需求”中创建场景。" />;
  const metrics = summary?.metrics ?? solution?.metrics ?? {}; const satisfaction = Number(metrics.total_demand_satisfied_rate);
  return (
    <>
      <PageHeader kicker="HEURISTIC SCHEDULER" title="智能调度" description="A* 风格的多车启发式搜索；时间轴可回放每辆车的移动、补给与投送轨迹。" actions={<button className="button primary" onClick={calculate} disabled={loading}>{loading ? '搜索中…' : solution ? '重新计算方案' : '开始调度'}</button>} />
      {error ? <div className="inline-error">{error}</div> : null}
      {!solution ? <EmptyPanel title="当前场景还没有调度结果" detail="点击右上角“开始调度”。最大状态数和搜索时间都有限制，超时会返回当前可行解。" /> : <>
        <section className="kpi-grid four"><KpiCard label="求解状态" value={solution.solver_status} hint="探索状态与耗时记录在结果中" tone={solution.solver_status === 'success' ? 'green' : 'amber'} /><KpiCard label="任务完成率" value={Number.isFinite(satisfaction) ? `${(satisfaction * 100).toFixed(1)}%` : '—'} hint={`未送达 ${Number(metrics.unserved_demand_total ?? 0).toFixed(1)} 箱`} tone="blue" /><KpiCard label="最慢车辆" value={`${Number(metrics.slowest_vehicle_time ?? 0).toFixed(2)} h`} hint={`总里程 ${Number(metrics.total_distance ?? 0).toFixed(1)}`} /><KpiCard label="风险评估" value={Number(metrics.total_exposed_risk ?? 0).toFixed(1)} hint={`高风险航路 ${Number(metrics.high_risk_edge_visits ?? 0)} 次`} tone="red" /></section>
        <section className="schedule-layout"><div className="panel map-panel schedule-map"><div className="timeline-controls"><button className="play-button" onClick={() => { if (time >= maxTime) setTime(0); setPlaying((value) => !value); }}>{playing ? 'Ⅱ' : '▶'}</button><div className="timeline-track"><input type="range" min="0" max={maxTime || 1} step={Math.max(maxTime / 500, 0.001)} value={time} onChange={(event) => { setPlaying(false); setTime(Number(event.target.value)); }} /><div className="timeline-labels"><span>0.00 h</span><strong>{time.toFixed(2)} / {maxTime.toFixed(2)} h</strong><span>{maxTime.toFixed(2)} h</span></div></div><label className="speed-control"><span>速度</span><select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label></div><NetworkMap scenario={scenario} solution={solution} layout={layout} currentTime={time} /></div>
          <div className="vehicle-stack">{Object.entries(solution.vehicle_trajectories).map(([vehicleId, frames], index) => { const route = frames.map((frame) => Number(frame.location)); const uniqueRoute = route.filter((location, i) => i === 0 || location !== route[i - 1]); const last = frames[frames.length - 1]; return <div className="panel vehicle-card" key={vehicleId}><div className="vehicle-title"><span className="vehicle-dot" style={{ background: VEHICLE_COLORS[index % VEHICLE_COLORS.length] }} /><div><strong>{vehicleId.replace('vehicle_', '车辆 ')}</strong><small>{frames.length} 个轨迹点</small></div><span className="badge badge-soft">{Number(last?.time ?? 0).toFixed(2)} h</span></div><div className="route-strip">{uniqueRoute.map((location, i) => <span key={`${location}-${i}`} className={location === 0 ? 'depot' : ''}>{location === 0 ? 'H' : scenario.villages[String(location)]?.name || location}</span>)}</div><div className="vehicle-metrics"><div><span>最终载荷</span><strong>{Number(last?.load ?? 0).toFixed(1)}</strong></div><div><span>最后动作</span><strong>{last?.action || '—'}</strong></div><div><span>已投送村</span><strong>{Object.keys(last?.delivered ?? {}).length}</strong></div></div></div>; })}</div>
        </section>
        <section className="panel delivery-timeline"><div className="panel-heading"><div><span className="eyebrow">DELIVERY SUMMARY</span><h2>村庄投送结果</h2></div></div><div className="delivery-grid">{Object.values(scenario.villages).map((village) => { const row = solution.village_delivery_summary[String(village.id)]; const demand = Number(row?.demand ?? scenario.demands[String(village.id)] ?? 0); const delivered = Number(row?.delivered ?? 0); const rate = demand > 0 ? Math.min(delivered / demand, 1) : 0; return <div key={village.id} className="delivery-card"><div><strong>{village.name}</strong><span>{delivered.toFixed(1)} / {demand.toFixed(1)} 箱</span></div><div className="progress"><i style={{ width: `${rate * 100}%` }} /></div><small>{(rate * 100).toFixed(1)}%</small></div>; })}</div></section>
      </>}
    </>
  );
}
