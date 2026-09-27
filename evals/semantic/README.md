# 语义评测输入

> 这里的 `*.example.jsonl` 仅用于验证评测脚本格式，不是项目质量结果，不得作为对外指标引用。

- `gold` 由人工在冻结数据集上标注：任务级正确证据、主张标签和主张对应证据。
- `predictions` 由固定代码、模型、Prompt 和检索配置生成；可附两名评审者的 `human_verdicts`。
- 标签只有 `supported`、`unsupported`、`insufficient`。证据不足必须使用 `insufficient`，不能强行二分类。
- 调参只使用开发集；最终测试集在报告前冻结，不边调 Prompt 边修改答案。

示例命令：

```powershell
python scripts/evaluate_semantic_quality.py `
  --example `
  --gold evals/semantic/gold.example.jsonl `
  --predictions evals/semantic/predictions.example.jsonl `
  --output data/evals/semantic/example-report.json
```

主张指标针对预先固定的 claim_id。自由生成内容需额外人工对齐和标注；示例中的人工标签是格式样例，不是真实双人评测。完整指标分母及正式报告 metadata 要求见 `docs/SEMANTIC_EVALUATION.md`。
