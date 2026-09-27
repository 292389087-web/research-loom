> 已被同日后续批量实验改版替代。当前 03 见 [批量实验管理](../BATCH_EXPERIMENTS.md)，当前验收见 [批量实验验收](BATCH_VALIDATION_20260919.md)。

> 文档状态：历史验证记录。2026-09-19，Windows / Python 3.12.7。本次验证不使用真实训练、外部 API 或模型推理；训练执行边界使用小型测试子进程。

# 03/04 工作区验收

实现和操作说明见 [工作区 v4](RESEARCH_WORKSPACE_V4.md)。本次将 03 定位为准备、审核和执行一轮实验，将 04 定位为独立选择数据并形成可追溯分析材料。

## 测试结果

| 验证 | 实测结果 | 范围 |
|---|---|---|
| `python -X utf8 -m pytest -q --junitxml=data/evals/workspace-redesign-20260919/junit.xml` | 91 passed，38.22 秒 | 原有回归与新增设计、数据、CPU 演示契约 |
| Edge 无头浏览器回归 | 通过，页面脚本错误 0 | 1440×1050 桌面、390×844 移动窗口，无横向溢出 |
| CPU 演示隔离导入 | 通过，包含在上述 91 项中 | 阻断 GPU/训练库导入后创建 API、分析合成数据、拒绝模型与训练 |

存在 1 条已安装 Starlette/httpx 的弃用警告；本次没有升级依赖。测试数量不是语义质量得分。

新增契约覆盖：奖励系数与超参数越组修改阻断、协议和种子锁定、审核建议约束、上下限校验、无排序指标时运行测试子进程、多源数据快照、CSV 非有限值/重复/单位冲突拒绝、跨协议禁止直接排名和降幅核查、奖励定义变化阻止 reward 对比、失败 seed 排除和快照不可变、分析调用账本、含糊问题澄清、外发许可、初稿版本绑定、导出和 SVG 转义。

界面回归覆盖：选择奖励调整后锁定超参数与训练天数、修改差异可见、创建待启动计划；04 独立加载演示数据、选择基准、生成汇总表与 SVG、下载 CSV、核查 5% 降幅、上传模板 CSV 并分析；桌面和移动布局检查。回归使用专用适配器，入口若被意外执行会立即报错，不会启动真实训练。

## 原始证据和复验

目录：`data/evals/workspace-redesign-20260919/`。

- `junit.xml`：91 项结果及执行时间。
- `ui/result.json`：界面检查结果。
- `ui/03-grouped-design.png`、`ui/04-demo-analysis.png`：桌面截图。
- `ui/lab-mobile.png`、`ui/bi-mobile.png`：移动窗口截图。
- `ui/demo-comparison.csv`：浏览器实际下载的对照表。
- `environment.json`、`source-sha256.json`：本次依赖环境和涉及文件的 SHA-256，不含密钥。

在 `research-loom` 下启动隔离服务（使用单独终端）：

```powershell
$env:LOOM_DATA_DIR = "$PWD\data\evals\workspace-redesign-20260919\ui-runtime"
$env:LOOM_REMOTE_CONFIG = "$PWD\data\evals\workspace-redesign-20260919\no-remote.toml"
python start.py --api-only --port 8011
```

另开终端，在相同项目目录运行（额外需要 Playwright 包和本机 Edge）：

```powershell
python scripts/check_research_workspace_ui.py --url http://127.0.0.1:8011 --data-dir "$PWD\data\evals\workspace-redesign-20260919\ui-runtime"
python scripts/check_docs_consistency.py
```

回归脚本会向指定隔离目录新增合成实验记录；不要把 `--data-dir` 指向个人正式实验目录。报告路径固定在上述证据目录，重跑会更新截图和界面报告。

## 结论边界

- 合成数据中的 5% 降幅只验证计算与展示，不能作为算法改进成果。
- 自然语言路径以受控模型替身验证计划、账本和结果，未在本轮评价 DeepSeek/Qwen 的实际语义准确率。
- 统计目前提供均值、样本标准差和数量，不自动做显著性检验、因果推断或置信区间。
- CSV 的口径来自用户声明；平台能检查一致性，不能证明原始实验真实或单位填报正确。
- CPU 模式验证了禁用相关库时的运行路径，未执行空白系统重装。
- 当前仍为本地主机部署，未据此声称多机高可用或大规模推理吞吐。
