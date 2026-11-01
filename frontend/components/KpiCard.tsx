import type { ReactNode } from 'react';
export function KpiCard({ label, value, hint, tone = 'default', icon }: { label: string; value: ReactNode; hint?: string; tone?: 'default' | 'blue' | 'green' | 'amber' | 'red'; icon?: string }) {
  return <div className={`kpi-card tone-${tone}`}><div className="kpi-top"><span>{label}</span>{icon ? <span className="kpi-icon">{icon}</span> : null}</div><div className="kpi-value">{value}</div>{hint ? <div className="kpi-hint">{hint}</div> : null}</div>;
}
