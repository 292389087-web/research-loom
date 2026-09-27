> 文档状态：历史。2026-09-19 批量实验管理验证；此前同日 91 项工作区测试是改版前快照。

# 批量实验验收

本次验证使用真实 Python 子进程和小型可信测试训练入口，不使用真实研究数据训练，不调用 DeepSeek、Qwen 或 Codex 模型。合成指标不作为真实实验成绩。

## 实测内容

- 全量 pytest：99 项通过；原有 91 项与新增 8 项批量测试，包含参数化用例。
- 矩阵：验证 4×4 展开 16 个不重复组合；手工组合保序；重复配置、过大矩阵、越界值和模型控制器混入均拒绝。
- 真子进程：2 个配置×2 个 seed 各自生成配置和结果 manifest，共 4 个独立 Attempt；模型调用账本为零。
- 注入训练报错：分别验证继续及暂停策略，失败记录保留；恢复后下一组成功，最终标记 completed_with_failures。
- 强制结束训练子进程：新调度器标记原 Attempt 为结果未知的失败，执行剩余配置，不重复原身份或静默重跑。
- 暂停正在运行的 seed：当前 seed 完成后保持下一 seed 排队；恢复后完成剩余项。
- 预算到期：暂停后仍保留原 deadline，过期恢复被拒绝，调度器标记 budget_exhausted。
- API 与 04：改动后的配置不能复用旧确认哈希；冻结后需单独启动；结果按配置编号和参数进入 04；批次不能请求模型调参建议。
- Edge 无头界面回归：4 个配置/12 次训练预览、确认后修改使确认失效、仅创建 draft、旧建议按钮消失、CSV/演示/图表/导出/数字核查、桌面与移动宽度无横向溢出，页面脚本错误为零。

## 命令与证据

在 `research-loom` 下执行：

```powershell
python -X utf8 -m pytest -q --junitxml=data/evals/batch-experiments-20260919/junit.xml
python scripts/check_docs_consistency.py
```

界面验证需 Playwright 和本机 Edge，先在单独终端启动隔离 API：

```powershell
$env:LOOM_DATA_DIR = "$PWD\data\evals\batch-experiments-20260919\ui-runtime"
$env:LOOM_REMOTE_CONFIG = "$PWD\data\evals\batch-experiments-20260919\no-remote.toml"
python start.py --api-only --port 8011
```

另一个终端：

```powershell
python scripts/check_batch_workspace_ui.py --url http://127.0.0.1:8011 --data-dir "$PWD\data\evals\batch-experiments-20260919\ui-runtime"
```

脚本向所选目录写合成记录，不能指向正式实验目录。`data/evals/batch-experiments-20260919/` 保存 JUnit、环境、源码 SHA-256、截图与界面报告。`before/` 是本地变更前备份，不应直接发布。前一版本的证据目录不覆盖。

## 没有据此声称的能力

没有重跑数小时的真实训练组合实验，没有测量调参收益或 GPU 利用率提升，没有证明所有操作系统断电场景，没有实现 checkpoint 续训或多机接管。支持的是单主机批次恢复，调度程序与可信训练工程仍需本地运行。具体限制与失败语义见 [操作说明](../BATCH_EXPERIMENTS.md)。

本机测试存在一条 Starlette/httpx 弃用警告；未为此升级依赖。全量测试耗时以 JUnit 原始记录为准。
