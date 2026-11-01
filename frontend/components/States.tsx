export function LoadingPanel({ label = '正在计算，请稍候…' }: { label?: string }) {
  return <div className="loading-panel"><span className="spinner" /><div><strong>{label}</strong><p>后端正在执行数据处理或启发式搜索。</p></div></div>;
}
export function EmptyPanel({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-panel"><span>◇</span><h3>{title}</h3><p>{detail}</p></div>;
}
