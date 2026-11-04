'use client';

import { useEffect, useMemo, useState } from 'react';
import { KpiCard } from '../components/KpiCard';
import { PageHeader } from '../components/PageHeader';
import { EmptyPanel } from '../components/States';
import { api } from '../lib/api';
import { useApp } from '../lib/app-context';
import type { LlmAnswer, ModelMeta } from '../lib/types';

const FALLBACK_QUESTIONS = [
  '为什么这些村庄被列为优先村？',
  '当前方案最需要关注什么？',
  '如果高风险道路中断，应该怎样调整？',
  '用一句话总结当前调度方案。',
];
const API_KEY_STORAGE = 'chiyuan.deepseek.api-key';
type KeyStatus = 'idle' | 'saved' | 'valid' | 'error';

export default function ExplainPage() {
  const { scenario, solution, summary, selectedGroup, isUserScenario } = useApp();
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [question, setQuestion] = useState(FALLBACK_QUESTIONS[0]);
  const [answer, setAnswer] = useState<LlmAnswer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [testingKey, setTestingKey] = useState(false);
  const [keyStatus, setKeyStatus] = useState<KeyStatus>('idle');
  const [keyMessage, setKeyMessage] = useState('');

  useEffect(() => {
    api.meta().then(setMeta).catch(() => undefined);
    const savedKey = window.sessionStorage.getItem(API_KEY_STORAGE) || '';
    if (savedKey) {
      setApiKey(savedKey);
      setKeyStatus('saved');
      setKeyMessage('已从当前浏览器会话恢复 DeepSeek API Key。');
    }
  }, []);

  const questions = useMemo(
    () => meta?.suggested_questions?.length ? meta.suggested_questions : FALLBACK_QUESTIONS,
    [meta],
  );
  const modelLabel = meta?.llm_model || 'deepseek-flash';
  const hasLlmAccess = Boolean(apiKey.trim() || meta?.llm_configured);

  async function testDeepSeekKey() {
    const key = apiKey.trim();
    if (!key) {
      setKeyStatus('error');
      setKeyMessage('请先输入 DeepSeek API Key。');
      return;
    }
    setTestingKey(true);
    setKeyStatus('idle');
    setKeyMessage('正在连接 DeepSeek 并验证模型...');
    try {
      const result = await api.testLlm(key);
      if (!result.ok) {
        setKeyStatus('error');
        setKeyMessage(result.message);
        return;
      }
      window.sessionStorage.setItem(API_KEY_STORAGE, key);
      setKeyStatus('valid');
      setKeyMessage(`${result.provider} / ${result.model} 连接成功，可以开始提问。`);
    } catch (reason) {
      setKeyStatus('error');
      setKeyMessage(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setTestingKey(false);
    }
  }

  function clearDeepSeekKey() {
    window.sessionStorage.removeItem(API_KEY_STORAGE);
    setApiKey('');
    setKeyStatus('idle');
    setKeyMessage('已清除当前浏览器会话中的 API Key。');
  }

  async function ask(nextQuestion?: string) {
    if (!scenario || !solution) return;
    const key = apiKey.trim();
    if (!key && !meta?.llm_configured) {
      setError('请先填写并测试 DeepSeek API Key，再提交问题。');
      return;
    }
    const actualQuestion = nextQuestion ?? question;
    setQuestion(actualQuestion);
    setLoading(true);
    setError('');
    try {
      const result = await api.explain(
        {
          scenario,
          solution,
          group: isUserScenario ? 'user' : selectedGroup,
          question: actualQuestion,
        },
        key || undefined,
      );
      setAnswer(result);
      if (result.mode !== 'ai' && result.error_type?.startsWith('http_')) {
        setKeyStatus('error');
        setKeyMessage(result.user_message || 'DeepSeek Key 验证失败，请重新测试。');
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLoading(false);
    }
  }

  if (!scenario || !solution) {
    return <EmptyPanel title="还没有可解释的调度方案" detail="请先在示例场景或场景需求页完成一次调度。" />;
  }

  const metrics = summary?.metrics ?? solution.metrics;
  return (
    <>
      <PageHeader
        kicker="OPTIONAL AI LAYER"
        title="AI 决策解释"
        description="使用 DeepSeek 解释当前调度方案。需配置 Key 进行 AI 问答。"
        actions={(
          <span className={`badge ${hasLlmAccess ? 'badge-success' : 'badge-warning'}`}>
            {apiKey.trim() ? '个人 DeepSeek Key 已就绪' : meta?.llm_configured ? 'DeepSeek 服务端已配置' : '需要 DeepSeek Key'}
          </span>
        )}
      />

      <section className="panel llm-key-panel">
        <div className="llm-key-copy">
          <span className="eyebrow">DEEPSEEK CONNECTION</span>
          <h2>配置你的 DeepSeek API Key</h2>
          <p>
            本平台使用 DeepSeek 模型 <code>{modelLabel}</code>。需先填写 Key，
            填写后可先执行连接测试，再让 AI 解释当前方案。
          </p>
          <a href="https://platform.deepseek.com/api_keys" target="_blank" rel="noreferrer">
            前往 DeepSeek 开放平台创建 API Key ↗
          </a>
        </div>
        <div className="llm-key-controls">
          <div className="llm-key-row">
            <input
              className="llm-key-input"
              type={showApiKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(event) => {
                setApiKey(event.target.value);
                setKeyStatus('idle');
                setKeyMessage('');
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') testDeepSeekKey();
              }}
              placeholder="sk-...（仅填写 DeepSeek API Key）"
              autoComplete="new-password"
              spellCheck={false}
              aria-label="DeepSeek API Key"
            />
            <button className="button ghost small" type="button" onClick={() => setShowApiKey((value) => !value)}>
              {showApiKey ? '隐藏' : '显示'}
            </button>
            <button className="button secondary small" type="button" onClick={testDeepSeekKey} disabled={testingKey}>
              {testingKey ? '测试中...' : '保存并测试'}
            </button>
            <button className="button ghost small" type="button" onClick={clearDeepSeekKey} disabled={!apiKey}>
              清除
            </button>
          </div>
          {keyMessage ? (
            <div className={`llm-key-message ${keyStatus === 'valid' ? 'success' : keyStatus === 'error' ? 'error' : ''}`}>
              {keyMessage}
            </div>
          ) : null}
          <p className="llm-key-help">
            Key 只保存在当前浏览器标签页的 sessionStorage，关闭标签页后自动清除；
            请求时通过 HTTPS 发送到本项目后端并转发至 DeepSeek，不会写入代码仓库或持久化数据库。
          </p>
        </div>
      </section>

      <section className="kpi-grid four">
        <KpiCard label="求解状态" value={solution.solver_status} hint={isUserScenario ? '用户场景' : `示例场景 ${selectedGroup}`} tone={solution.solver_status === 'success' ? 'green' : 'amber'} />
        <KpiCard label="需求满足率" value={metrics.total_demand_satisfied_rate === undefined ? '—' : `${(Number(metrics.total_demand_satisfied_rate) * 100).toFixed(1)}%`} hint={`未送达 ${Number(metrics.unserved_demand_total ?? 0).toFixed(1)} 箱`} tone="blue" />
        <KpiCard label="最长时效" value={`${Number(metrics.slowest_vehicle_time ?? 0).toFixed(2)} h`} hint="用于解释整体时效瓶颈" />
        <KpiCard label="暴露风险" value={Number(metrics.total_exposed_risk ?? 0).toFixed(1)} hint={`高风险边 ${Number(metrics.high_risk_edge_visits ?? 0)} 次`} tone="red" />
      </section>

      <section className="explain-layout">
        <div className="panel ask-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">ASK THE PLAN</span><h2>询问调度方案</h2></div>
          </div>
          <div className="question-chips">
            {questions.map((item) => (
              <button key={item} className={question === item ? 'active' : ''} onClick={() => ask(item)}>
                {item}
              </button>
            ))}
          </div>
          <div className="ask-box">
            <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={5} placeholder="例如：为什么优先安排 4 号村？" />
            <button className="button primary" onClick={() => ask()} disabled={loading || !hasLlmAccess}>
              {loading ? '分析中...' : hasLlmAccess ? '提交问题' : '请先配置 DeepSeek Key'}
            </button>
          </div>
          {error ? <div className="inline-error">{error}</div> : null}
          <div className="explain-boundary">
            <strong>边界限制</strong>
            <p>AI 只能解释当前结构化方案，不能修改调度结果，也不能替代每个车辆状态背后的模型与算法。</p>
          </div>
        </div>

        <div className="panel answer-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">RESPONSE</span><h2>解释结果</h2></div>
            {answer ? <span className={`badge ${answer.mode === 'ai' ? 'badge-success' : 'badge-warning'}`}>{answer.mode === 'ai' ? 'DeepSeek AI' : '规则兜底'}</span> : null}
          </div>
          {answer ? (
            <div className="answer-content">
              {answer.user_message ? <div className="notice-line">{answer.user_message}</div> : null}
              <div className="answer-text">{answer.answer}</div>
              {answer.fallback_reason ? <details><summary>兜底原因</summary><p>{answer.fallback_reason}</p></details> : null}
            </div>
          ) : (
            <div className="answer-placeholder">
              <span>AI</span>
              <h3>先配置 Key，再解释方案</h3>
              <p>连接测试通过后选择一个问题，DeepSeek 会把当前指标、优先村和高风险道路组织成可读解释。</p>
            </div>
          )}
        </div>
      </section>

      <section className="panel summary-strip">
        <div className="panel-heading">
          <div><span className="eyebrow">PLAN SNAPSHOT</span><h2>结构化方案摘要</h2></div>
          <span className="muted">{summary?.scenario_label || scenario.label} · {summary?.group_label || selectedGroup}</span>
        </div>
        <div className="snapshot-grid">
          <div><span>优先村</span><strong>{summary?.priority_villages?.map((item) => item.name).join('、') || '无'}</strong></div>
          <div><span>最高风险道路</span><strong>{summary?.top_risk_roads?.[0] ? `${summary.top_risk_roads[0].road_id} · ${(summary.top_risk_roads[0].risk_score * 100).toFixed(1)}%` : '无'}</strong></div>
          <div><span>车辆数</span><strong>{summary?.vehicle_count ?? Object.keys(solution.vehicle_trajectories).length}</strong></div>
          <div><span>求解耗时</span><strong>{Number(solution.solve_time_actual ?? summary?.solve_time_actual ?? 0).toFixed(2)} s</strong></div>
        </div>
      </section>
    </>
  );
}
