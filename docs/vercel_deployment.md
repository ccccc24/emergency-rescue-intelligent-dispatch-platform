# Vercel 部署说明

本项目采用同一仓库、两个 Vercel Project：

- `api/index.py`：FastAPI 后端，部署为 Vercel Python Function。
- `frontend/`：Next.js + React 前端，部署为 Vercel Next.js Project。

## 本地运行

后端：

```powershell
pip install -r requirements-dev.txt
python -m uvicorn api.index:app --reload --host 127.0.0.1 --port 8000
```

健康检查：

```text
http://127.0.0.1:8000/api/index?action=health
```

前端：

```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local
npm run dev
```

打开 `http://localhost:3000`。`.env.local` 内容：

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## 部署 FastAPI 后端

1. 在 Vercel 导入仓库，新建 Project。
2. Root Directory 保持仓库根目录。
3. Framework Preset 选择 `Other`。
4. Build Command 留空。
5. 部署后访问 `https://<后端项目>.vercel.app/api/index?action=health`。

可选环境变量：

```text
DEEPSEEK_API_KEY=...
LLM_BASE_URL=https://api.deepseek.com/chat/completions
LLM_MODEL=deepseek-flash
CORS_ORIGINS=https://<前端项目>.vercel.app
```

不配置 LLM Key 时自动使用规则 mock 模式。

`vercel.json` 已把 `app/common/`、`configs/`、`data/`、`models/`、`outputs/`、`scripts/`、`src/` 和 Excel 模板包含进 Function。最大执行时间 60 秒，内存 1024 MB。

## 部署 Next.js 前端

1. 再次导入同一仓库，新建第二个 Project。
2. Root Directory 选择 `frontend`。
3. Framework Preset 选择 `Next.js`。
4. 添加环境变量：

```text
NEXT_PUBLIC_API_BASE_URL=https://<后端项目>.vercel.app
```

5. Deploy。末尾不要加 `/`。

建议把自定义域名绑定到前端项目；后端可以继续使用自己的 `vercel.app` 域名。

## 为什么使用单端点 action 分发

统一入口为 `/api/index`，通过 `action` 区分操作：

```text
/api/index?action=examples
/api/index?action=example&date=2025-08-29&group=C
/api/index?action=schedule
```

这样 Vercel 只需要识别一个明确的 ASGI 入口：

```python
app = FastAPI()
```

可以让 Vercel 明确识别 FastAPI ASGI 入口。

## 用户在页面配置 DeepSeek API Key

即使服务端没有配置 `DEEPSEEK_API_KEY`，用户也可以在前端“AI 决策解释”页面填写自己的 Key：

1. 前往 DeepSeek 开放平台创建 API Key：<https://platform.deepseek.com/api_keys>
2. 在页面中粘贴 Key，点击“保存并测试”。
3. 测试通过后提交解释问题。

安全约束：

- Key 仅保存在当前浏览器标签页的 `sessionStorage`，关闭标签页后自动清除。
- 请求通过 `X-DeepSeek-API-Key` 请求头发送给本项目后端，再由后端转发给 DeepSeek。
- 后端不会持久化或主动写入日志；请勿把 Key 写入仓库或 `.env.example`。
- 用户自己的 Key 固定使用 DeepSeek 官方端点和 `deepseek-flash` 模型。
