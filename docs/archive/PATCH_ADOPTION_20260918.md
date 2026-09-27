> 文档状态：历史修改的验收记录。日期 2026-09-18；不替代历史模型/训练报告。

# 外部审查建议采纳记录

## 判断

采纳“工程可靠性证据较充分，而语义质量评测、交付可复验性和文档一致性需要补强”的判断。原合并方案要求可追溯生成、质量评测及性能证据；项目后续转向科研场景后，可靠执行和人工审核已经形成主线，语义质量仍不能由工程测试代替。本项目定位用于说明可靠 Agent 与本地全栈交付，算法研究证据由独立的研究工作承担，避免为了覆盖所有技术名词而新增浅层功能。

审查方针对复制包的结论不能照搬到本机：本机 Python 为 3.12.7，存在 `data/` 和作者本机登记的完整训练工程，可以运行完整回归。对外复制包遗漏证据的问题仍需通过显式证据包解决。

## 已采纳与修订

| 项目 | 处理 |
|---|---|
| 当前状态/复验/语义评测三份文档 | 放入项目 docs，作为当前说明入口 |
| README 与历史文档冲突 | 统一人工审核实验定位，给旧设计与验收加历史标记 |
| 旧限制提示 | 改为已存在修复记录，但必须核对本轮 manifest；不回写旧建议和旧实验 |
| 硬编码工作负载事实 | 4211 任务/每分钟一个仅描述旧基线，不能推定所有新数据集相同 |
| 建议可追踪性 | 笔记本附每轮源码哈希；新建议 Prompt 标为 v3，不宣称自动完成哈希与修复版本的认证 |
| 上传整包内存缓存 | 使用 1 MiB spooled buffer，超阈值前先切磁盘，64 KiB 回放；I/O 走线程池，失败路径关闭文件、释放容量 |
| 语义计分框架 | 采纳 JSONL 与确定性指标；补严格形状校验、空样本语义、额外任务标记、示例/来源 metadata 区分 |
| 证据包 | 采纳 SHA-256 manifest；收窄默认白名单，补 JSON/Bearer 密钥识别、其他 local 配置排除与副本扫描 |
| 证据校验 | 补缺失、篡改、额外文件、空清单、越界、重复路径和符号链接检查 |
| 依赖约束 | 从真实环境补录 optuna 5.0.0、psutil 5.9.0；明确不是完整 lockfile 或全新环境安装证明 |
| 总结文档路径 | 更新外部审查目录中的总结副本，不在项目根目录制造重复副本 |

未直接采纳原补丁中 `data/evals/**/*.json` 的默认打包规则，因为本机该目录含原始模型输入/回答和实验资料。工具只检查常见密钥形态，不自动进行科研隐私脱敏。真实密钥未读取到输出、未修改；若此前带密钥的复制包已经交给他人或上传，应由持有人在供应商控制台轮换，`.gitignore` 不能撤回已经交付的内容。

## 验收

原补丁新增的 6 项测试先单独通过；补充边界测试后，全量测试结果为 **74 passed, 1 warning，39.36 秒**。警告为 Starlette TestClient 对 httpx 的弃用提示，未为消除警告擅自升级模型环境。

```powershell
cd research-loom
python -X utf8 -m pytest -q --junitxml=data/evals/patch-adoption-20260918/junit.xml
python -X utf8 scripts/check_docs_consistency.py
python -X utf8 scripts/evaluate_semantic_quality.py --example --gold evals/semantic/gold.example.jsonl --predictions evals/semantic/predictions.example.jsonl --output data/evals/patch-adoption-20260918/semantic-example.json
python -X utf8 scripts/build_evidence_pack.py --spec config/evidence-pack.adoption-20260918.json --output dist/evidence-pack-20260918
python -X utf8 scripts/verify_evidence_pack.py dist/evidence-pack-20260918
```

构建命令要求输出目录不存在或为空；再次复验请换一个新的输出目录名，不覆盖旧证据。原始 JUnit 留在本机，包内使用移除主机名和绝对工作区路径后的副本。来源 manifest 对应本次源码和文档；包级 manifest 对交付文件逐一校验。

本次没有调用收费 API、没有运行真实 GPU 推理、没有重新开展真实训练，因此不更新此前真实运行的耗时、显存或算法成绩。现有历史真实产物仍留在本机；本次包只支持检查新工具与回归结果，不声称重新验证了全部历史数字。

## 尚未完成

- 跨领域公开论文冻结集、真实主张标注、双人独立复核；
- 从实际自由生成结果到 gold claim 的人工对齐与质量对照；
- 全新机器安装复验、完整传递依赖 lockfile 和 CI；
- 真实模型私有资产对外交付的许可与获取说明。

计分框架和证据工具解决的是测量与交付基础，不自动完成这些数据、人工工作和跨机器实验。原修改前文件备份与原补丁应用清单保存在作者本机。
