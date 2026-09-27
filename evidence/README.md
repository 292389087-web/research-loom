# evidence：可公开核对的工程证据

本目录是从作者本机完整验证记录中挑选并脱敏的**小型证据包**，随仓库公开发布，用于核对项目文档中引用的关键数字。

原始记录（完整逐日数据、训练产物、研究资料）保留在作者本机 `data/` 目录，不随仓库提供。所有本机绝对路径、私有适配器标识与研究内容细节均已排除。

## 目录结构

| 路径 | 内容 | 生成方式 |
|---|---|---|
| `regression/full-regression.xml` | 全量回归，**123 passed** | `python -X utf8 -m pytest -q --junitxml=evidence/regression/full-regression.xml` |
| `regression/hardening.xml` | 工程补强测试，**21 passed** | `python -X utf8 -m pytest tests/test_engineering_hardening.py -q --junitxml=...` |
| `regression/initial-regression.xml` | 补强前的**初始失败记录**（保留原貌，不重写） | 同上 |
| `regression/receipt-source-check.xml` | “同时篡改逐日文件及磁盘 manifest”的独立性复验 | 见 `docs/ENGINEERING_HARDENING_20260922.md` 第 3 节 |
| `real-run/real-validation.json` | 一次真实 10 天 CUDA 训练回归的数值核对 | 由本机 `data/evals/engineering-hardening-20260922/` 复制 |
| `real-run/reward-equivalence.json` | 501 组奖励诊断等价性隔离检查 | 同上 |
| `ui/02-draft-and-citations.png` | “02 初稿与引用”界面（引用候选与数值核查） | `scripts/check_*_ui.py`（Playwright + Edge） |
| `ui/03-experiment-lab.png` | “03 自动实验”界面（创建实验计划） | 同上 |
| `ui/04-data-agent.png` | “04 实验 Data Agent”界面（查询实验数据） | 同上 |
| `ui/04-data-agent-mobile.png` | 移动窗口布局检查 | 同上 |
| `report-20260922.md` | 2026-09-22 工程补强与真实验证的脱敏摘要 | 本机报告的手工脱敏版 |

## 使用说明

- 回归 XML 是测试环境的运行记录；自行复跑请使用仓库根目录命令（见 `docs/REPRODUCIBILITY.md`）。
- 真实训练回归使用作者本机登记的**私有训练工程与数据**，本目录只公开流程性核对结果（状态、误差量级、显存峰值），不包含训练数据、模型产物与研究内容。
- UI 截图对应 2026-09 工作区版本，使用公开论文与工程测试数据演示，不含研究资料内容。
- 数字引用规则见 `docs/REPRODUCIBILITY.md`：“README 与对外材料中的数字必须能定位到报告文件、测试命令、环境和代码版本”。

## 为什么不是全部证据

本机 `data/evals/` 中还有更完整的验证记录（并发压测、界面回归录像、真实训练逐日数据与产物清单等）。其中相当一部分依赖私有输入或包含未发表研究内容，按项目既有的“可复验性分层”原则（`docs/REPRODUCIBILITY.md` 三层证据），只公开可独立核对的部分；其余以“作者本机留存、未随公开包提供”标注。
