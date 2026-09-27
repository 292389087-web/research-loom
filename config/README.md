> 文档状态：当前。配置说明不声明本机是否已经填写密钥。

# DeepSeek 配置

打开同目录下的 `deepseek.local.toml`，只填写：

```toml
api_key = "你的真实密钥"
```

其余内容已设置：官方地址 `https://api.deepseek.com`、模型 `deepseek-flash`（当前对应 V4.1 Flash）、思考模式开启、强度 high、每次最多 32768 个输出 token、每项任务累计预留上限 131072。预留上限不是实际消耗；思考和最终回答共享输出预算。没有密钥时 API 保持不可用，仍可用本地模型。

文件使用 UTF-8 编码，`config/deepseek.local.toml` 已被 `.gitignore` 精确排除，同时保留 `config/*.local.toml` 通配保护；不要提交填好密钥的文件。若该文件曾被 Git 跟踪，新增 `.gitignore` 不会自动移除历史记录，需要轮换密钥并执行 `git rm --cached config/deepseek.local.toml` 后再检查提交历史。`deepseek.example.toml` 是可分享的空白模板。环境变量 `LOOM_REMOTE_*` 的优先级高于文件；环境中若已有旧配置，需要清除或更新。`LOOM_REMOTE_CONFIG` 可指定其他配置文件路径。

填写后重启 API 和模型 worker，新配置才生效。在项目目录的两个终端分别启动：

```powershell
python start.py --api-only --workers 2
python -m loom.worker_main --supervise
```

已有进程先停止，避免同端口重复启动。训练调度器不调用 API，不必为填写密钥单独重启。若使用内嵌 worker 的 `python start.py`，只重启这一项即可。

`/api/models` 的 `remote_configured` 只表示配置齐全，**不表示密钥已经通过服务商验证**。首次提交获许可的任务才会真正调用服务。论文、任务及图片的外发许可仍要在页面明确设置；填写密钥不会自动授权资料外发。

思考输出只记录服务商返回的 token 用量，不将思考正文作为最终回答。每阶段保存思考配置和有效输出上限，修改模型、端点或思考设置后需新建任务，避免旧任务恢复时混用配置。`deepseek-flash` 是服务商模型别名，后端未来可能更新，平台另记录响应中的模型字段。

已核实的官方说明（2026-09-15）：

- [V4.1 Flash 发布说明及模型名](https://api-docs.deepseek.com/news/news260910/)
- [思考模式与官方地址示例](https://api-docs.deepseek.com/guides/thinking_mode/)
- [Chat Completions 参数](https://api-docs.deepseek.com/api/create-chat-completion/)

仓库不记录本机是否已配置真实密钥。真实账户调用是否完成，只能由对应日期的脱敏调用账本和验证报告证明；配置文件存在或 `/api/models` 显示齐全都不能替代真实调用验证。
