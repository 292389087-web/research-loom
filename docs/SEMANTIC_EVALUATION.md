> 文档状态：当前。评测框架已实现；跨领域冻结评测集与正式人工复核结果尚未完成。

# 语义质量评测协议

自动化契约测试证明 Schema、幂等、恢复和证据身份，不证明模型结论在语义上正确。本项目将质量证据分为四层：结构合法、来源存在、语义支持、科研可用。前两层可以确定性自动检查；后两层必须依赖冻结人工标注与研究者复核。

## 数据冻结

- 公开论文按任务划分为开发集和测试集；最终报告前冻结测试集。
- `gold` 记录任务级正确证据，以及每个主张的 `supported / unsupported / insufficient` 标签和对应证据。
- `predictions` 记录检索排序、系统主张标签和引用；可附两名评审者的 `accepted / needs_revision` 判断。
- 调 Prompt、检索参数和模型只看开发集；不得根据测试集错误修改答案后继续报告同一版本结果。
- 自动生成的标注必须人工复核；LLM-as-a-Judge 不能作为唯一质量证据。

## 确定性指标

`loom/evaluation.py` 与 `scripts/evaluate_semantic_quality.py` 计算：

- 检索：Recall@1/5/10、MRR、nDCG@1/5/10；
- 主张：标签准确率、Macro-F1、错误支持率、证据不足时的拒答 precision/recall；
- 引用：主张—证据链接的 precision/recall；
- 人工复核：接受率与两名评审者的 Cohen's kappa；
- 覆盖：缺失任务和额外任务，不静默丢弃失败样本。

示例数据只验证格式和脚本，不是项目质量结果：

```powershell
python scripts/evaluate_semantic_quality.py `
  --example `
  --gold evals/semantic/gold.example.jsonl `
  --predictions evals/semantic/predictions.example.jsonl `
  --output data/evals/semantic/example-report.json
```

## 正式报告最低要求

正式结果必须同时保存数据版本、模型与 Prompt 版本、检索配置、生成时间、输入文件 SHA-256 和原始逐题预测。建议至少比较 BM25、BM25+BGE-M3、混合检索+复核三个版本，并报告失败样例。没有完成冻结评测前，只能说“评测框架已实现、完成单篇真实冒烟”，不能写跨领域准确率或质量提升百分比。

## 计分口径与输入边界

本框架的主张分类任务使用预先固定的 `claim_id`。若评价自由生成的 finding，须先由人工对齐主张与证据并处理额外主张；不能只给两段无关文本分配同一 ID 就当成语义一致。引用 precision/recall 是人工标注的主张—证据链接集合匹配，包含可能用作反证的链接，不等于仅统计正向支持引用的精确率。

- Macro-F1 固定对三个标签求平均，无样本的类别按 F1=0；没有任何 gold 主张时报告 null。
- 错误支持率分母为预测 `supported` 的主张数；分子为 gold 并非 supported 的这些主张，不能解释为全部自由文本的幻觉率。
- 检索指标只对 gold 中有相关证据的任务计算；缺失预测按空检索/缺失标签计入。
- 额外任务显示在覆盖报告中，并使 `eligible_for_comparison=false`，不能隐藏它们后宣称正式对照有效。
- 人工接受率按投票数统计；kappa 的两个数组位置必须由同两名固定评审者提供。全都投同一标签导致分母为零时返回 null。
- 文本字段、人工标签和数据冻结由输入提供者负责，脚本不能证明标注由人完成。

CLI 必须显式选择 `--example` 或 `--metadata 文件.json`。metadata 至少包含 `dataset_version`、`split`、`model_version`、`prompt_version`、`retrieval_config`、`generated_at`、`annotation_protocol`；它只记录来源声明，不自动认证质量。输出包含输入哈希、评测器代码哈希和执行时间。
