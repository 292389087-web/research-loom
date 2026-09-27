> 文档状态：当前。历史性能数字只对应各自带日期的验证快照。

# Research Loom 执行可靠性与运行说明

## 运行形态

便捷启动仍可使用 `python start.py`，一个 API 进程内嵌一个 worker。要验证 API / 执行隔离，先停止旧服务，在两个终端分别运行：

```powershell
cd research-loom
python start.py --api-only --workers 2
```

```powershell
cd research-loom
python -m loom.worker_main --supervise
```

两个终端必须使用同一个 `LOOM_DATA_DIR`。API 进程不加载 GPU 模型；独立执行进程复用本地 Qwen3-VL 与 BGE-M3 权重。GPU worker 只启动一个监督器，避免多份常驻模型争抢显存。SQLite 的租约约束防止重复提交，不能使两个已加载模型共享一份显存。

监督器只管理自己创建的 worker 子进程，子进程退出后才重新启动；最多重启 5 次并退避。无进度 600 秒或超过任务总期限时，独立 worker 停止续租并退出，释放阻塞计算资源。嵌入模式只停止续租，不强制结束整个 API 服务。监督器本身退出、断电、磁盘损坏或系统时间异常，不在本机应用的自动修复保证内。

## 持久执行合同

- 提交：`Idempotency-Key` 可选，1–128 字符，作用域为当前本地数据库。相同键与相同请求返回原 run_id；内容冲突返回 409。键与任务、首个事件原子保存，当前不自动过期。若以后加入多用户，必须改为身份/项目内作用域。
- 入队：队列容量检查和 INSERT 在 `BEGIN IMMEDIATE` 中完成，不依赖 Python 线程锁。PDF 解析另外按 paper_id 去重；文件内容哈希不等于任意研究请求幂等。
- 派发：持久任务表 + 0.5 秒轮询，一个执行槽。租约默认 30 秒，约每 10 秒续约。过期接管递增 epoch；旧执行者的 checkpoint、证据、事件和终态写入在事务内检查执行身份。
- 取消：取消请求持久接收，任务在检查边界停止。终态提交会检查已接收取消，阻止迟到结果转为成功。已经保存的研究资料不会回滚删除。
- 结果：结果和终态事件在同一数据库事务提交；模拟事件写入失败时，结果提交一并回滚。SSE 是持久事件的读取视图，支持 Last-Event-ID 续读。
- 重试：TimeoutError/ConnectionError 属于有界暂时错误，保存指数退避的绝对 not_before。Schema 错误只做一次结构修复，失败不自动无限重试。恢复默认最多 2 次，任务期限默认 1 小时。
- 模型调用：总预算默认 8 次，开始调用前即计入并保存；人工重试也不重置这个预算。人工重试会显式重设任务期限与恢复额度；预算耗尽需新建更小任务。

正常退出会保留已完成阶段。真实崩溃可能留下 `started` 尝试；恢复时标为 `unknown` 并保留 usage=NULL。NULL 意味未知，不是免费或零 token。模型回答已经落盘但尚未完成 Schema 检查时，恢复从记录的回答继续；已提交阶段不重新采样。

本系统没有声称模型调用 exactly-once。租约丢失后旧计算可能短暂继续，保护的是已提交状态；单 GPU 部署通过一个监督器控制实际进程替换。多机外部服务还需要服务端请求身份、查询与幂等合同。

## 逐步追溯

- `GET /api/runs/{id}/trace`：任务、epoch、恢复次数、绝对期限、模型尝试、输入哈希和完整持久事件。
- `GET /api/runs/{id}/trace?include_payloads=true`：再包含 SYSTEM、Prompt、Schema、生成参数、原始回答和使用量。输入图像保存在 `data/model-inputs/`，按图像内容哈希命名。
- `GET /api/metrics`：当前 API 进程请求计数、最近 1000 条请求的 P95、任务状态与模型尝试数量。
- `GET /api/status`：API 配置和数据库中当前执行身份。独立 worker 模式下，API 的内存 model_loaded 不代表 GPU worker 的模型状态，更不能把 API 存活等同于推理有进展。
- 页面“查看执行记录”中新增 Trace JSON 链接；原始 Prompt 不放在普通日志中。

新版本不能补造旧任务未保存的调用证据。trace 会明确说明没有账本的可能原因，不能把旧任务的零条记录解释成零次模型调用。

持久账本目前不自动归档，原始输入输出也可能包含未公开论文内容。保留策略、容量告警和跨项目权限属于后续工作；不要将完整 trace 任意公开。这里保存显式输入输出、动作与状态，没有要求或记录模型隐藏思维链。

## 入口与安全范围

本机默认监听 127.0.0.1，允许 localhost / loopback Host，同源写请求；响应带随机 X-Request-ID、nosniff、禁止页面被框架嵌入及 CSP。JSON 请求体限制 64 KiB，上传另有 40 MiB 文件限制，分块请求也计数；入口按块计数并使用 1 MiB 内存阈值的临时缓冲，超过阈值自动落盘，再分块交给 multipart 解析和业务层写入。这是内存有界的磁盘缓冲上传，不是网络到最终哈希文件的零拷贝直通。每进程最多 64 个活动请求、16 个 SSE、2 个上传，超限返回 429。慢请求体接收有 30 秒单次等待限制。

可选 `LOOM_API_TOKEN` 对所有 `/api/` 请求启用 Bearer 校验。示例：

```powershell
$env:LOOM_API_TOKEN = "自行生成的长随机值"
python start.py --api-only --workers 2
```

启用后，调用方必须传 `Authorization: Bearer ...`。当前 Web 页面没有登录流程或给 EventSource 注入此请求头的实现，因此此模式用于受控 API 客户端；默认本地页面模式不设置 token。该 token 是单用户访问凭据，不是多租户权限系统。跨机器暴露还需要 HTTPS、认证授权和部署入口治理，不能直接把 host 改为 0.0.0.0 就当作安全上线。

没有开放模型 SQL、shell 或任意 Python 执行工具。论文内容及模型输出按文本转义展示；Host/Origin/CSP 与无危险工具并不能证明模型抵御所有提示注入。文件系统账本保护也不等于对拥有本机写权限的人提供防篡改能力。

## 复跑实验

```powershell
python -X utf8 -m pytest -q --tb=short --junitxml=data/reliability-tests-20260914.xml
python -X utf8 scripts/reliability_lab.py
python -X utf8 scripts/smoke_local.py --mode question
python -X utf8 scripts/check_ui.py
python -X utf8 scripts/check_result_ui.py
```

前两条验证软件机制与独立实验服务，不调用 GPU。故障模型位于 tests/，每次实际调用返回不同决定；故障注入没有生产 HTTP 开关。强制退出只发生于专门创建的测试子进程。第三条才运行真实本地模型，后两条使用本机 Edge 做页面回归。

压力脚本使用独立 `data/reliability/时间戳/` 数据库和随机 loopback 端口，创建两个测试 API 进程；退出时只终止自己创建的服务树。结果包含环境、源文件哈希、请求状态分布、成功请求 QPS、P50/P95；不把快速拒绝请求计成成功吞吐。

## 升级和回退

本次数据库迁移增加租约/期限列、submissions 和 model_attempts 表，不重写历史研究结果。迁移前应停止旧版 API/worker；旧程序不理解新执行约束，不能与新版同时写同一库。原数据库备份应使用 SQLite backup API 或停机一致性复制。

当前尚未提供反向迁移。若需要回退，停止新进程并恢复迁移前的完整备份；不要只删除新表再继续运行。已有备份与实验结果位于 data/。
