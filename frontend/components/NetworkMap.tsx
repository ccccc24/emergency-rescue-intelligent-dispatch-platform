'use client';

import { useMemo } from 'react';
import type { Point, Scenario, Solution, TrajectoryFrame, Village } from '../lib/types';

const WIDTH = 900;
const HEIGHT = 540;
const PAD = 56;
const VEHICLE_COLORS = ['#2563eb', '#e85d34', '#7c3aed', '#0891b2', '#be123c'];
const riskColor = (risk: number) => risk >= 0.65 ? '#dc4c4c' : risk >= 0.35 ? '#e9a23b' : '#4ca66b';
const riskWidth = (risk: number) => risk >= 0.65 ? 5 : risk >= 0.35 ? 3.6 : 2.4;

export function NetworkMap({ scenario, solution, layout, currentTime, showRoutes = true, compact = false }: { scenario: Scenario; solution: Solution | null; layout: Record<string, Point>; currentTime?: number; showRoutes?: boolean; compact?: boolean }) {
  const projection = useMemo(() => {
    const points: Point[] = [{ x: 0, y: 0 }];
    const addPoint = (point?: { x?: unknown; y?: unknown }) => {
      const x = Number(point?.x);
      const y = Number(point?.y);
      if (Number.isFinite(x) && Number.isFinite(y)) points.push({ x, y });
    };

    Object.values(layout ?? {}).forEach(addPoint);
    Object.values(scenario.villages ?? {}).forEach(addPoint);
    const minX = Math.min(...points.map((point) => point.x));
    const maxX = Math.max(...points.map((point) => point.x));
    const minY = Math.min(...points.map((point) => point.y));
    const maxY = Math.max(...points.map((point) => point.y));
    const rangeX = Math.max(maxX - minX, 1);
    const rangeY = Math.max(maxY - minY, 1);
    const scale = Math.min((WIDTH - PAD * 2) / rangeX, (HEIGHT - PAD * 2) / rangeY);
    const offsetX = (WIDTH - rangeX * scale) / 2;
    const offsetY = (HEIGHT - rangeY * scale) / 2;
    const project = (point: Point) => ({ x: offsetX + (point.x - minX) * scale, y: HEIGHT - (offsetY + (point.y - minY) * scale) });
    const nodePoint = (nodeId: string | number): Point => {
      const key = String(nodeId);
    
      const explicit = layout?.[key];
      if (explicit) {
        const x = Number(explicit.x);
        const y = Number(explicit.y);
        if (Number.isFinite(x) && Number.isFinite(y)) return project({ x, y });
      }
    
      const village = scenario.villages?.[key];
      if (village) {
        const x = Number(village.x);
        const y = Number(village.y);
        if (Number.isFinite(x) && Number.isFinite(y)) return project({ x, y });
      }
    
      return project({ x: 0, y: 0 });
    };
    return { nodePoint };
  }, [layout, scenario]);

  const trajectories = solution?.vehicle_trajectories ?? {};
  const vehicleMarkers = useMemo(() => {
    if (currentTime === undefined || !solution) return [];
    return Object.entries(trajectories).map(([vehicleId, frames], index) => {
      const marker = interpolateVehicle(frames, currentTime);
      return marker ? { vehicleId, point: projection.nodePoint(marker.location), color: VEHICLE_COLORS[index % VEHICLE_COLORS.length] } : null;
    }).filter(Boolean) as Array<{ vehicleId: string; point: Point; color: string }>;
  }, [currentTime, projection, solution, trajectories]);

  return (
    <div className={`network-map ${compact ? 'compact' : ''}`}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="离线救援路网地图">
        <defs><filter id="node-shadow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="0" dy="3" stdDeviation="3" floodColor="#17344f" floodOpacity="0.18" /></filter></defs>
        <rect x="0" y="0" width={WIDTH} height={HEIGHT} rx="22" fill="#f8fbfe" />
        <g opacity="0.33">{Array.from({ length: 11 }).map((_, index) => <line key={`v-${index}`} x1={index * 90} y1="0" x2={index * 90} y2={HEIGHT} stroke="#dce8f2" />)}{Array.from({ length: 7 }).map((_, index) => <line key={`h-${index}`} x1="0" y1={index * 90} x2={WIDTH} y2={index * 90} stroke="#dce8f2" />)}</g>
        {Object.values(scenario.roads ?? {}).map((road) => {
          const start = projection.nodePoint(road.from);
          const end = projection.nodePoint(road.to);
          return <g key={`${road.from}-${road.to}-${road.id}`}><line x1={start.x} y1={start.y} x2={end.x} y2={end.y} stroke="#fff" strokeWidth={riskWidth(road.risk_score) + 4} strokeLinecap="round" /><line x1={start.x} y1={start.y} x2={end.x} y2={end.y} stroke={riskColor(road.risk_score)} strokeWidth={riskWidth(road.risk_score)} strokeLinecap="round" opacity="0.82" /></g>;
        })}
        {showRoutes ? Object.entries(trajectories).map(([vehicleId, frames], index) => {
          if (!frames?.length) return null;
          const path = frames.map((frame, pointIndex) => { const point = projection.nodePoint(frame.location); return `${pointIndex === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`; }).join(' ');
          return <path key={vehicleId} d={path} fill="none" stroke={VEHICLE_COLORS[index % VEHICLE_COLORS.length]} strokeWidth="3.2" strokeDasharray="8 7" strokeLinecap="round" strokeLinejoin="round" opacity="0.78" />;
        }) : null}
        {Object.values(scenario.villages ?? {}).map((village: Village) => {
          const point = projection.nodePoint(village.id);
          const priority = scenario.priority_villages?.includes(Number(village.id));
          const demand = Number(scenario.demands?.[String(village.id)] ?? village.demand ?? 0);
          return <g key={village.id} filter="url(#node-shadow)">{priority ? <circle cx={point.x} cy={point.y} r="18" fill="#fff3d7" stroke="#e9a23b" strokeWidth="2" /> : null}<circle cx={point.x} cy={point.y} r="10" fill={priority ? '#f59d28' : '#2f78c4'} stroke="#fff" strokeWidth="3" /><text x={point.x} y={point.y + 31} textAnchor="middle" className="map-label">{village.name}</text><text x={point.x} y={point.y + 46} textAnchor="middle" className="map-demand">{demand.toFixed(1)} 箱</text></g>;
        })}
        <g filter="url(#node-shadow)"><rect x={projection.nodePoint(0).x - 18} y={projection.nodePoint(0).y - 18} width="36" height="36" rx="10" fill="#143d68" /><text x={projection.nodePoint(0).x} y={projection.nodePoint(0).y + 6} textAnchor="middle" fill="#fff" fontSize="16" fontWeight="800">H</text><text x={projection.nodePoint(0).x} y={projection.nodePoint(0).y + 35} textAnchor="middle" className="map-label">救援中心</text></g>
        {vehicleMarkers.map((marker) => <g key={marker.vehicleId} filter="url(#node-shadow)"><circle cx={marker.point.x} cy={marker.point.y} r="14" fill={marker.color} stroke="#fff" strokeWidth="3" /><text x={marker.point.x} y={marker.point.y + 4} textAnchor="middle" fill="#fff" fontSize="10" fontWeight="800">{marker.vehicleId.replace('vehicle_', 'V')}</text></g>)}
      </svg>
      <div className="map-legend"><span><i style={{ background: '#4ca66b' }} />低风险</span><span><i style={{ background: '#e9a23b' }} />中风险</span><span><i style={{ background: '#dc4c4c' }} />高风险</span><span><i className="priority-dot" />优先村</span></div>
    </div>
  );
}

function interpolateVehicle(frames: TrajectoryFrame[], time: number): { location: number } | null {
  if (!frames.length) return null;
  if (time <= Number(frames[0].time)) return { location: Number(frames[0].location) };
  for (let index = 0; index < frames.length - 1; index += 1) {
    const current = frames[index]; const next = frames[index + 1];
    if (time >= Number(current.time) && time <= Number(next.time)) {
      const fraction = next.time === current.time ? 1 : (time - current.time) / (next.time - current.time);
      return { location: fraction >= 0.5 ? Number(next.location) : Number(current.location) };
    }
  }
  return { location: Number(frames[frames.length - 1].location) };
}
