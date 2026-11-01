'use client';

import Link from 'next/link';
import { useRouter } from 'next/router';
import type { ReactNode } from 'react';
import { useApp } from '../lib/app-context';

const NAV = [
  { href: '/', label: '态势总览', eyebrow: 'OVERVIEW', glyph: '◉' },
  { href: '/scenario', label: '数据输入', eyebrow: 'INPUT', glyph: '◇' },
  { href: '/schedule', label: '智能调度', eyebrow: 'PLANNER', glyph: '↗' },
  { href: '/compare', label: '方案对比', eyebrow: 'BENCHMARK', glyph: '▥' },
  { href: '/explain', label: 'AI 决策解释', eyebrow: 'EXPLAIN', glyph: '✦' },
];

export function Shell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = router.pathname;
  const { ready, loading, error, scenario, source, selectedDate, selectedGroup } = useApp();
  const sourceLabel = source === 'example' ? '示例数据' : source === 'excel' ? 'Excel 导入' : '用户场景';
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block"><div className="brand-mark">驰</div><div><div className="brand-name">驰援调度</div><div className="brand-sub">CHIYUAN DISPATCH</div></div></div>
        <div className="sidebar-caption">救援控制台 / WORKSPACE</div>
        <nav className="nav-list" aria-label="主导航">
          {NAV.map((item) => (
            <Link key={item.href} href={item.href} className={`nav-item ${pathname === item.href ? 'active' : ''}`}>
              <span className="nav-glyph">{item.glyph}</span><span className="nav-copy"><span>{item.label}</span><small>{item.eyebrow}</small></span>
            </Link>
          ))}
        </nav>
        <div className="sidebar-foot"><span className={`status-dot ${ready && !error ? 'online' : ''}`} /><div><strong>{ready ? (error ? '后端连接异常' : 'FastAPI 已连接') : '正在连接后端'}</strong><span>{selectedDate ? `${selectedDate} · 方案 ${selectedGroup}` : '等待场景载入'}</span></div></div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="topbar-context"><span className="topbar-label">当前任务</span><strong>{scenario?.label || '正在加载场景'}</strong><span className="badge badge-soft">{sourceLabel}</span></div>
          <div className="topbar-status">{loading ? <span className="loading-pill"><i /> 计算中</span> : null}<span className="badge badge-success">A* 风格启发式</span></div>
        </header>
        {error ? <div className="global-error"><strong>请求失败：</strong>{error}</div> : null}
        <main className="page-content">{children}</main>
      </div>
    </div>
  );
}
