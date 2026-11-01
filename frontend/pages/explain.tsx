'use client';

import { useEffect, useState } from 'react';
import { KpiCard } from '../components/KpiCard';
import { PageHeader } from '../components/PageHeader';
import { EmptyPanel } from '../components/States';
import { api } from '../lib/api';
import { useApp } from '../lib/app-context';
import type { ModelMeta } from '../lib/types';

const FALLBACK_QUESTIONS = ['为什么这些村庄被列为优先村？', '当前方案的主要风险是什么？', '如果高风险道路中断，应如何调整？', '用一句话总结当前方案。'];
export default function ExplainPage() {
  const { scenario, solution, summary, selectedGroup, isUserScenario } = useApp();
  const [meta, setMeta] = useState<ModelMeta | null>(null); const [question, setQuestion] = useState(FALLBACK_QUESTIONS[0]); const [answer, setAnswer] = useState<{ mode: string; status: string; answer: string; fallback_reason?: string | null; user_message?: string | null } | null>(null); const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  useEffect(() => { api.meta().then(setMeta).catch(() => undefined); }, []);
  async function ask(nextQuestion?: string) { if (!scenario || !solution) return; const actualQuestion = nextQuestion ?? question; setQuestion(actualQuestion); setLoading(true); setError(''); try { const result = await api.explain({ scenario, solution, group: isUserScenario ? 'user' : selectedGroup, question: actualQuestion }); setAnswer(result); } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); } finally { setLoading(false); } }
  const questions = meta?.suggested_questions?.length ? meta.suggested_questions : FALLBACK_QUESTIONS;
  if (!scenario || !solution) return <EmptyPanel title="还没有可解释的调度方案" detail="请先载入示例方案，或在“场景与需求”中运行一次调度。" />;
  const metrics = summary?.metrics ?? solution.metrics;
  return (
    <>
      <PageHeader kicker="OPTIONAL AI LAYER" title="AI 决策解释" description="LLM 只读取结构化方案摘要，不参与 A* 搜索循环。未配置 Key 时自动使用规则化 mock 回答。" actions={<span className={`badge ${meta?.llm_configured ? 'badge-success' : 'badge-warning'}`}>{meta?.llm_configured ? 'LLM 已配置' : 'Mock 模式'}</span>} />
      <section className="kpi-grid four"><KpiCard label="求解状态" value={solution.solver_status} hint={isUserScenario ? '用户场景' : `示例方案 ${selectedGroup}`} tone={solution.solver_status === 'success' ? 'green' : 'amber'} /><KpiCard label="需求满足率" value={metrics.total_demand_satisfied_rate === undefined ? '—' : `${(Number(metrics.total_demand_satisfied_rate) * 100).toFixed(1)}%`} hint={`未送达 ${Number(metrics.unserved_demand_total ?? 0).toFixed(1)} 箱`} tone="blue" /><KpiCard label="最慢时间" value={`${Number(metrics.slowest_vehicle_time ?? 0).toFixed(2)} h`} hint="用于解释任务瓶颈" /><KpiCard label="暴露风险" value={Number(metrics.total_exposed_risk ?? 0).toFixed(1)} hint={`高风险边 ${Number(metrics.high_risk_edge_visits ?? 0)} 次`} tone="red" /></section>
      <section className="explain-layout"><div className="panel ask-panel"><div className="panel-heading"><div><span className="eyebrow">ASK THE PLAN</span><h2>询问调度方案</h2></div></div><div className="question-chips">{questions.map((item) => <button key={item} className={question === item ? 'active' : ''} onClick={() => ask(item)}>{item}</button>)}</div><div className="ask-box"><textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={5} placeholder="例如：为什么优先安排 4 号村？" /><button className="button primary" onClick={() => ask()} disabled={loading}>{loading ? '分析中…' : '提交问题'}</button></div>{error ? <div className="inline-error">{error}</div> : null}<div className="explain-boundary"><strong>边界声明</strong><p>AI 只能解释当前结构化结果，不会修改调度结果，也不会在每个搜索状态调用模型。</p></div></div>
        <div className="panel answer-panel"><div className="panel-heading"><div><span className="eyebrow">RESPONSE</span><h2>解释结果</h2></div>{answer ? <span className={`badge ${answer.mode === 'ai' ? 'badge-success' : 'badge-warning'}`}>{answer.mode === 'ai' ? 'AI' : '规则兜底'}</span> : null}</div>{answer ? <div className="answer-content">{answer.user_message ? <div className="notice-line">{answer.user_message}</div> : null}<div className="answer-text">{answer.answer}</div>{answer.fallback_reason ? <details><summary>技术降级原因</summary><p>{answer.fallback_reason}</p></details> : null}</div> : <div className="answer-placeholder"><span>✦</span><h3>方案摘要已准备</h3><p>选择一个问题，让解释层把当前指标、优先村与高风险航路组织成可读说明。</p></div>}</div>
      </section>
      <section className="panel summary-strip"><div className="panel-heading"><div><span className="eyebrow">PLAN SNAPSHOT</span><h2>结构化方案摘要</h2></div><span className="muted">{summary?.scenario_label || scenario.label} · {summary?.group_label || selectedGroup}</span></div><div className="snapshot-grid"><div><span>优先村</span><strong>{summary?.priority_villages?.map((item) => item.name).join('、') || '无'}</strong></div><div><span>最高风险航路</span><strong>{summary?.top_risk_roads?.[0] ? `${summary.top_risk_roads[0].road_id} · ${(summary.top_risk_roads[0].risk_score * 100).toFixed(1)}%` : '无'}</strong></div><div><span>车辆数</span><strong>{summary?.vehicle_count ?? Object.keys(solution.vehicle_trajectories).length}</strong></div><div><span>搜索耗时</span><strong>{Number(solution.solve_time_actual ?? summary?.solve_time_actual ?? 0).toFixed(2)} s</strong></div></div></section>
    </>
  );
}
