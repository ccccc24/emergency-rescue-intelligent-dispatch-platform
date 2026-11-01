'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/router';
import { PageHeader } from '../components/PageHeader';
import { api, downloadTemplate } from '../lib/api';
import { DEFAULT_CONFIG, useApp } from '../lib/app-context';
import type { ModelMeta, PredictionStatus, RoadInputRow, Scenario, VillageInputRow } from '../lib/types';

interface VillageEditor { id: number; name: string; x: string; y: string; demand: string; priority: boolean; population: string; disaster_level: string; road_accessibility: string; hist_demand_mean: string; }
interface RoadEditor { from: string; to: string; distance: string; risk: string; base_speed: string; geology_risk: string; hist_break_count: string; }

const STARTER_VILLAGES: VillageEditor[] = [
  { id: 1, name: '东岭村', x: '10', y: '0', demand: '80', priority: true, population: '1200', disaster_level: '2', road_accessibility: '0.85', hist_demand_mean: '72' },
  { id: 2, name: '河西村', x: '0', y: '12', demand: '55', priority: false, population: '900', disaster_level: '1', road_accessibility: '0.92', hist_demand_mean: '48' },
  { id: 3, name: '南坡村', x: '-12', y: '-5', demand: '45', priority: false, population: '700', disaster_level: '1', road_accessibility: '0.78', hist_demand_mean: '40' },
];
const STARTER_ROADS: RoadEditor[] = [
  { from: '0', to: '1', distance: '12', risk: '0.2', base_speed: '60', geology_risk: '0.1', hist_break_count: '0' },
  { from: '0', to: '2', distance: '14', risk: '0.3', base_speed: '55', geology_risk: '0.2', hist_break_count: '1' },
  { from: '0', to: '3', distance: '13', risk: '0.25', base_speed: '50', geology_risk: '0.15', hist_break_count: '0' },
  { from: '1', to: '2', distance: '9', risk: '0.35', base_speed: '45', geology_risk: '0.3', hist_break_count: '1' },
];
const optionalNumber = (value: string) => value.trim() === '' ? null : (Number.isFinite(Number(value)) ? Number(value) : null);

function scenarioToVillages(scenario: Scenario | null): VillageEditor[] {
  if (!scenario) return STARTER_VILLAGES;
  return Object.values(scenario.villages).map((village) => ({ id: Number(village.id), name: String(village.name), x: String(village.x ?? 0), y: String(village.y ?? 0), demand: String(scenario.demands?.[String(village.id)] ?? village.demand ?? ''), priority: scenario.priority_villages.includes(Number(village.id)), population: village.population == null ? '' : String(village.population), disaster_level: village.disaster_level == null ? '' : String(village.disaster_level), road_accessibility: village.road_accessibility == null ? '' : String(village.road_accessibility), hist_demand_mean: village.hist_demand_mean == null ? '' : String(village.hist_demand_mean) }));
}
function scenarioToRoads(scenario: Scenario | null): RoadEditor[] {
  if (!scenario) return STARTER_ROADS;
  return Object.values(scenario.roads).map((road) => ({ from: String(road.from), to: String(road.to), distance: String(road.length ?? 0), risk: String(road.risk_score ?? ''), base_speed: road.base_speed == null ? '' : String(road.base_speed), geology_risk: road.geology_risk == null ? '' : String(road.geology_risk), hist_break_count: road.hist_break_count == null ? '' : String(road.hist_break_count) }));
}

export default function ScenarioPage() {
  const router = useRouter();
  const { scenario, config, setConfig, applyActiveScenario, loadExample, exampleList, selectedDate, isUserScenario } = useApp();
  const [villages, setVillages] = useState<VillageEditor[]>(STARTER_VILLAGES);
  const [roads, setRoads] = useState<RoadEditor[]>(STARTER_ROADS);
  const [weather, setWeather] = useState('晴');
  const [inputMode, setInputMode] = useState('full');
  const [useMl, setUseMl] = useState(true);
  const [overwrite, setOverwrite] = useState(false);
  const [label, setLabel] = useState('自定义救援场景');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [predictionStatus, setPredictionStatus] = useState<PredictionStatus[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setVillages(scenarioToVillages(scenario)); setRoads(scenarioToRoads(scenario)); setWeather(scenario?.weather || '晴'); setInputMode(scenario?.input_mode || 'full'); setLabel(scenario?.label || '自定义救援场景');
    if (scenario) setConfig({ ...DEFAULT_CONFIG, ...(scenario.config ?? {}) });
  }, [scenario, setConfig]);
  useEffect(() => { api.meta().then(setMeta).catch(() => undefined); }, []);

  const villagePayload = useMemo<VillageInputRow[]>(() => villages.map((v) => ({ id: Number(v.id), name: v.name.trim(), x: Number(v.x), y: Number(v.y), demand: optionalNumber(v.demand), priority: v.priority ? 1 : 0, population: optionalNumber(v.population), disaster_level: optionalNumber(v.disaster_level), road_accessibility: optionalNumber(v.road_accessibility), hist_demand_mean: optionalNumber(v.hist_demand_mean) })), [villages]);
  const roadPayload = useMemo<RoadInputRow[]>(() => roads.map((r) => ({ from: Number(r.from), to: Number(r.to), distance: Number(r.distance), risk: optionalNumber(r.risk), base_speed: optionalNumber(r.base_speed), geology_risk: optionalNumber(r.geology_risk), hist_break_count: optionalNumber(r.hist_break_count) })), [roads]);
  const updateVillage = (index: number, field: keyof VillageEditor, value: string | boolean) => setVillages((current) => current.map((row, i) => i === index ? { ...row, [field]: value } : row));
  const updateRoad = (index: number, field: keyof RoadEditor, value: string) => setRoads((current) => current.map((row, i) => i === index ? { ...row, [field]: value } : row));
  const addVillage = () => { const nextId = Math.max(0, ...villages.map((item) => item.id)) + 1; setVillages((current) => [...current, { id: nextId, name: `新村 ${nextId}`, x: '0', y: '0', demand: '', priority: false, population: '', disaster_level: '1', road_accessibility: '', hist_demand_mean: '' }]); };
  const addRoad = () => setRoads((current) => [...current, { from: '0', to: String(villages[0]?.id ?? 1), distance: '10', risk: '', base_speed: '50', geology_risk: '', hist_break_count: '' }]);

  async function handlePredict() {
    setLoading(true); setError(''); setMessage('');
    try {
      const result = await api.predict({ villages: villagePayload, roads: roadPayload, weather, source: 'manual', overwrite_existing: overwrite });
      setVillages((current) => current.map((row, i) => result.villages[i] ? { ...row, demand: String(result.villages[i].demand ?? '') } : row));
      setRoads((current) => current.map((row, i) => result.roads[i] ? { ...row, risk: String(result.roads[i].risk ?? '') } : row));
      setPredictionStatus([...result.village_status, ...result.road_status]); setMessage(`ML 预测完成：补全 ${Number(result.prediction_metadata.rows_predicted ?? 0)} 个字段。`);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setLoading(false); }
  }

  async function handleApply(thenSchedule: boolean) {
    setLoading(true); setError(''); setMessage('');
    try {
      const active = await api.build({ villages: villagePayload, roads: roadPayload, config, weather, input_mode: inputMode, predict_missing: useMl, overwrite_existing: overwrite, source: 'manual', label });
      if (thenSchedule) {
        setMessage('场景构建成功，正在执行调度搜索…');
        const result = await api.schedule({ scenario: active.scenario, config: active.config, mode: 'full', group: 'user' });
        applyActiveScenario(active, result); router.push('/schedule');
      } else { applyActiveScenario(active, null); setMessage('场景已应用。前往“智能调度”页面后可以重新计算。'); }
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setLoading(false); }
  }

  async function handleExcel(file: File) {
    setLoading(true); setError(''); setMessage('');
    try {
      const bytes = new Uint8Array(await file.arrayBuffer()); let binary = ''; for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
      const active = await api.importExcel({ data_base64: btoa(binary), filename: file.name, predict_missing: useMl, config, weather });
      applyActiveScenario(active, null); setMessage(`已导入 ${file.name}。请检查数据后执行调度。`);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setLoading(false); if (fileInputRef.current) fileInputRef.current.value = ''; }
  }

  return (
    <>
      <PageHeader kicker="SCENARIO & DEMAND" title="场景与需求" description="编辑村庄与航路；缺失的 demand/risk 可交给随机森林预测模型补全，再转换为调度器输入。" actions={<button className="button ghost" onClick={() => downloadTemplate()}>下载 Excel 模板</button>} />
      <section className="control-strip">
        <label><span>载入示例</span><select value={isUserScenario ? '' : selectedDate} onChange={(event) => event.target.value && loadExample(event.target.value, 'C')}>{isUserScenario ? <option value="">当前为用户场景</option> : null}{exampleList.map((e) => <option key={e.date} value={e.date}>{e.label} · {e.date}</option>)}</select></label>
        <label><span>场景名称</span><input value={label} onChange={(event) => setLabel(event.target.value)} /></label>
        <label><span>天气</span><select value={weather} onChange={(event) => setWeather(event.target.value)}><option value="晴">晴</option><option value="雨">雨</option><option value="暴风">暴风</option></select></label>
        <label><span>输入模式</span><select value={inputMode} onChange={(event) => setInputMode(event.target.value)}><option value="full">完整特征 + ML</option><option value="minimal">直接输入 demand/risk</option></select></label>
      </section>
      {message ? <div className="inline-success">{message}</div> : null}{error ? <div className="inline-error">{error}</div> : null}
      <section className="editor-layout">
        <div className="panel editor-main">
          <div className="panel-heading"><div><span className="eyebrow">VILLAGES</span><h2>村庄需求表</h2></div><button className="button small" onClick={addVillage}>+ 添加村庄</button></div>
          <div className="table-wrap editor-table"><table><thead><tr><th>ID</th><th>名称</th><th>X</th><th>Y</th><th>需求</th><th>优先</th><th>人口</th><th>灾情</th><th>可达性</th><th>历史均值</th><th /></tr></thead><tbody>{villages.map((v, i) => <tr key={v.id}><td><span className="id-chip">{v.id}</span></td><td><input value={v.name} onChange={(e) => updateVillage(i, 'name', e.target.value)} /></td><td><input type="number" value={v.x} onChange={(e) => updateVillage(i, 'x', e.target.value)} /></td><td><input type="number" value={v.y} onChange={(e) => updateVillage(i, 'y', e.target.value)} /></td><td><input type="number" placeholder="ML" value={v.demand} onChange={(e) => updateVillage(i, 'demand', e.target.value)} /></td><td><input type="checkbox" checked={v.priority} onChange={(e) => updateVillage(i, 'priority', e.target.checked)} /></td><td><input type="number" value={v.population} onChange={(e) => updateVillage(i, 'population', e.target.value)} /></td><td><input type="number" value={v.disaster_level} onChange={(e) => updateVillage(i, 'disaster_level', e.target.value)} /></td><td><input type="number" step="0.01" value={v.road_accessibility} onChange={(e) => updateVillage(i, 'road_accessibility', e.target.value)} /></td><td><input type="number" value={v.hist_demand_mean} onChange={(e) => updateVillage(i, 'hist_demand_mean', e.target.value)} /></td><td><button className="icon-button danger" disabled={villages.length <= 2} onClick={() => setVillages((rows) => rows.filter((_, j) => j !== i))}>×</button></td></tr>)}</tbody></table></div>
          <div className="panel-heading road-heading"><div><span className="eyebrow">ROUTES</span><h2>航路风险表</h2></div><button className="button small" onClick={addRoad}>+ 添加航路</button></div>
          <div className="table-wrap editor-table"><table><thead><tr><th>起点</th><th>终点</th><th>距离</th><th>风险</th><th>基础速度</th><th>地质风险</th><th>历史中断</th><th /></tr></thead><tbody>{roads.map((r, i) => <tr key={`${i}-${r.from}-${r.to}`}><td><select value={r.from} onChange={(e) => updateRoad(i, 'from', e.target.value)}><option value="0">救援中心</option>{villages.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}</select></td><td><select value={r.to} onChange={(e) => updateRoad(i, 'to', e.target.value)}><option value="0">救援中心</option>{villages.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}</select></td><td><input type="number" value={r.distance} onChange={(e) => updateRoad(i, 'distance', e.target.value)} /></td><td><input type="number" step="0.01" placeholder="ML" value={r.risk} onChange={(e) => updateRoad(i, 'risk', e.target.value)} /></td><td><input type="number" value={r.base_speed} onChange={(e) => updateRoad(i, 'base_speed', e.target.value)} /></td><td><input type="number" step="0.01" value={r.geology_risk} onChange={(e) => updateRoad(i, 'geology_risk', e.target.value)} /></td><td><input type="number" value={r.hist_break_count} onChange={(e) => updateRoad(i, 'hist_break_count', e.target.value)} /></td><td><button className="icon-button danger" disabled={roads.length <= 1} onClick={() => setRoads((rows) => rows.filter((_, j) => j !== i))}>×</button></td></tr>)}</tbody></table></div>
        </div>
        <aside className="editor-side">
          <div className="panel"><div className="panel-heading"><div><span className="eyebrow">ML BRIDGE</span><h2>预测与校验</h2></div></div><label className="switch-row"><input type="checkbox" checked={useMl} onChange={(e) => setUseMl(e.target.checked)} /><span>构建时自动补全缺失 demand/risk</span></label><label className="switch-row"><input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} /><span>覆盖已有数值</span></label><button className="button secondary full" onClick={handlePredict} disabled={loading}>先执行 ML 预测</button><div className="model-mini-grid"><div><span>需求模型</span><strong>RF · MAE {meta ? Number(meta.demand.test_metrics.MAE).toFixed(2) : '—'}</strong></div><div><span>风险模型</span><strong>RF · AUC {meta ? Number(meta.risk.test_metrics.auc).toFixed(3) : '—'}</strong></div></div>{predictionStatus.length ? <div className="prediction-list">{predictionStatus.slice(0, 8).map((item, i) => <div key={`${item.field}-${item.row}-${i}`}><span>{item.name} · {item.field}</span><strong className={`prediction-${item.status}`}>{item.status === 'predicted' ? Number(item.value).toFixed(2) : item.status === 'kept_user' ? '保留输入' : item.status}</strong></div>)}</div> : null}</div>
          <div className="panel"><div className="panel-heading"><div><span className="eyebrow">PLANNER CONFIG</span><h2>调度参数</h2></div></div><div className="field-grid"><label><span>车辆数</span><input type="number" min="1" max="8" value={config.num_vehicles} onChange={(e) => setConfig({ ...config, num_vehicles: Number(e.target.value) })} /></label><label><span>载重</span><input type="number" value={config.vehicle_capacity} onChange={(e) => setConfig({ ...config, vehicle_capacity: Number(e.target.value) })} /></label><label><span>时间上限</span><input type="number" value={config.vehicle_time_limit} onChange={(e) => setConfig({ ...config, vehicle_time_limit: Number(e.target.value) })} /></label><label><span>风险权重</span><input type="number" step="0.05" min="0" max="0.5" value={config.risk_cost_alpha} onChange={(e) => setConfig({ ...config, risk_cost_alpha: Number(e.target.value) })} /></label><label><span>搜索秒数</span><input type="number" min="5" max="35" value={config.time_limit} onChange={(e) => setConfig({ ...config, time_limit: Number(e.target.value) })} /></label><label><span>状态上限</span><input type="number" min="1000" max="50000" step="1000" value={config.max_states} onChange={(e) => setConfig({ ...config, max_states: Number(e.target.value) })} /></label></div><button className="button primary full" onClick={() => handleApply(true)} disabled={loading}>{loading ? '处理中…' : '应用并执行调度'}</button><button className="button ghost full" onClick={() => handleApply(false)} disabled={loading}>仅应用场景</button></div>
          <div className="panel excel-panel"><div className="panel-heading"><div><span className="eyebrow">EXCEL</span><h2>批量导入</h2></div></div><input ref={fileInputRef} type="file" accept=".xlsx" onChange={(e) => { const file = e.target.files?.[0]; if (file) void handleExcel(file); }} /><p>支持 `villages`、`roads`、`config` 三个工作表。</p></div>
        </aside>
      </section>
    </>
  );
}
