import type { AppProps } from 'next/app';
import Head from 'next/head';
import { AppProvider } from '../lib/app-context';
import { Shell } from '../components/Shell';
import '../styles/globals.css';

export default function MyApp({ Component, pageProps }: AppProps) {
  return <><Head><title>驰援 · 驰援应急救援智能调度平台</title><meta name="description" content="需求预测、风险评估与 A* 风格启发式多车救援调度平台" /><meta name="viewport" content="width=device-width, initial-scale=1" /></Head><AppProvider><Shell><Component {...pageProps} /></Shell></AppProvider></>;
}
