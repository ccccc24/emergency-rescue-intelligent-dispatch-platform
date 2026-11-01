import type { ReactNode } from 'react';
export function PageHeader({ kicker, title, description, actions }: { kicker: string; title: string; description: string; actions?: ReactNode }) {
  return <div className="page-header"><div><div className="page-kicker">{kicker}</div><h1>{title}</h1><p>{description}</p></div>{actions ? <div className="page-actions">{actions}</div> : null}</div>;
}
