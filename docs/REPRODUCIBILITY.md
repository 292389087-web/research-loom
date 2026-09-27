> 文档状态：当前。区分源码可运行、工程结论可复验和科研结果可复现。

# 交付与复验

## 2026-09-22 五项工程补强

功能、边界和升级方式见 [工程补强说明](ENGINEERING_HARDENING_20260922.md)。本机证据入口为 `data/evals/engineering-hardening-20260922/REPORT.md`，包括并发幂等、依赖预检、产物复核/篡改故障测试、参数筛选和逐日分析契约、501 组奖励等价性检查，以及新完成的 1 次 10 天 CUDA 工程回归。它与 9 月 21 日的优化实验分开保存，不替换历史失败或历史快照。

轻量回归：`python -X utf8 -m pytest tests/test_engineering_hardening.py -q`。真实训练脚本 `data/evals/engineering-hardening-20260922/verify_real.py` 依赖本机登记的完整训练工程、私有数据及 GPU，不能当作公开仓库的零配置测试。该脚本以固定幂等键复用已经创建的验证批次，不隐式重跑完成项。浏览器交互和另一套训练工程本轮暂缓。

## 2026-09-21 真实训练与 Data Agent 闭环

作者本机保留完整验收记录（含测试脚本、冻结计划、训练产物副本、修复前失败、04 查询与模型账本、数值核查、曲线及问题清单）；`artifact-sha256.json` 校验导出的训练产物。详细验收记录未随公开仓库提供。真实数据和模型依赖仍需原环境提供，未宣称公开仓库可独立复现实验效果。

本轮的另一套训练工程没有完整跑通，网页交互也没有完成新的浏览器验证。修复前的失败记录没有改成成功；相同种子的新训练用于验证修复只影响产物验收。复验前阅读 `TEST_PLAN.md`，不要直接重复执行会创建长时间 GPU 任务的测试脚本。

## 2026-09-19 工作区与批量实验验证

当前 03 见 [批量实验管理](BATCH_EXPERIMENTS.md)，当前实测结果见 [批量验收](archive/BATCH_VALIDATION_20260919.md)。同日改版前的 [工作区验收](archive/WORKSPACE_VALIDATION_20260919.md) 为历史 91 passed，另有隔离数据目录下的 Edge 桌面/移动界面回归。证据目录为 `data/evals/workspace-redesign-20260919/`，包括 JUnit、截图、界面报告和本轮环境/源码哈希。

纯 CPU 演示安装 `requirements-demo.txt` 后运行 `python start.py --demo`。该入口使用 `data/demo`，禁止模型任务和真实训练。演示数据明确标为合成，不能作为真实训练效果证据。已用独立子进程阻断 torch、transformers、sentence_transformers、optuna、psutil 的导入，验证页面/API 和确定性分析仍可用；没有声称完成全新机器的依赖安装验证。

当前批量证据独立保存在 `data/evals/batch-experiments-20260919/`，不覆盖前一版本。

## 三层证据

1. **软件可运行**：固定 Python/CUDA 组合，安装依赖，运行契约测试和样例流程。
2. **工程结论可复验**：故障注入、幂等并发、背压和 HTTP 压测均保留机器可读报告、环境及代码哈希。
3. **真实模型/实验可复现**：额外需要输入材料、模型版本、训练适配器、数据集许可、随机种子、配置和产物 manifest。无法公开的资料只提供哈希与获取条件，不把“作者本机存在”描述为公开可复现。

## 已验证环境

历史环境快照为 `docs/environment.json`；本次环境与源码哈希在 `data/evals/patch-adoption-20260918/`。本机 Python 为 3.12.7，已补录 optuna 与 psutil 的实际版本。推荐 Python 3.12；先安装与机器 CUDA 匹配的 PyTorch，再使用已验证约束：

```powershell
python -m pip install -r requirements.txt -c constraints-validated.txt
python -X utf8 -m pytest -q
```

`constraints-validated.txt` 是直接依赖的已运行版本约束，不是包含全部传递依赖和平台 wheel 哈希的 lockfile，也不是本次在空白环境安装成功的证明。升级依赖后必须重新执行测试和真实模型冒烟，并生成新环境快照。本次不引入未经验证的 Docker 或 CI 成功声明。

## 构建可公开证据包

`data/` 可以保留在个人电脑而不进入 Git，但对外引用的关键数字应能由小型脱敏证据包核对（本仓库 `evidence/` 即为此类材料）。先复制并修改 `config/evidence-pack.example.json`，只列出允许公开的报告，再运行：

```powershell
python scripts/build_evidence_pack.py `
  --spec config/evidence-pack.example.json `
  --output dist/evidence-pack
python scripts/verify_evidence_pack.py dist/evidence-pack
```

构建器只复制显式白名单，生成文件大小和 SHA-256 manifest，并拒绝常见密钥文件和疑似 API Key。它不能自动判断论文、日志或路径是否涉及隐私；发布前仍必须人工检查。真实 API Key、原始未发表论文、完整 Prompt/回答和私有数据不得进入公开证据包。

默认 example 白名单只包含环境及说明文档，不扫描整个 `data/evals`，以免带出原始回答或未发表研究内容。要打包真实验证，请逐文件添加确认过的报告。校验器拒绝空清单、越界、重复条目、未登记文件和哈希不匹配；哈希校验只能证明相对于 manifest 的内容一致性，不能证明结果科学正确或发布者身份。工具不会自动脱敏，生成包也不等于已经公开发布。

## 数字使用规则

- README 与对外材料中的数字必须能定位到报告文件、测试命令、环境和代码版本。
- 历史报告只证明当时版本；当前代码变化后不得默认继承旧结论。
- 单篇真实冒烟证明链路连通，不代表跨领域质量。
- 元数据 API QPS 不代表模型推理 QPS；拒绝响应不计入成功吞吐。
- 缺失原始产物时明确写“作者本机留存、未随公开包提供”，不伪造复验状态。
