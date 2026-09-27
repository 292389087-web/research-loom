def markdown_report(run):
    result = run.get("result") or {}
    lines = ["# 研途 · 研究记录", "", f"任务：{run['id']}", f"状态：{run['status']}", "", "## 研究问题", result.get("question") or "论文理解与实验规划", "", "## 证据范围", result.get("scope", {}).get("novelty", "本地论文"), ""]
    for finding in result.get("findings", []):
        lines += [f"### {finding['id']} · {finding['category']}", finding["statement"], "", f"来源校验：{finding['source_status']}；判断类型：{finding['basis']}"]
        for citation in finding["citations"]:
            lines += [f"- {citation['evidence_id']}，PDF 第 {citation.get('page')} 页：{citation['quote']}"]
        if finding.get("research_implication"):
            lines += ["", "**研究启发（模型建议）**", finding["research_implication"]]
        if finding.get("model_review"):
            lines += ["", f"模型复核（非独立验证）：{finding['model_review']['status']} — {finding['model_review']['reason']}"]
        lines += finding.get("review_reasons", [])
        lines.append("")
    for direction in result.get("directions", []):
        lines += [f"## 候选方向：{direction['title']}", "新颖性尚未验证；以下为研究假设。", "", direction["motivation"], direction["change"], ""]
        for key, value in direction["experiment"].items():
            if key not in {"hypothesis", "setup", "baselines", "metrics", "ablations", "falsification", "feasibility"}:
                continue
            lines += [f"**{key}**", "; ".join(value) if isinstance(value, list) else value, ""]
        lines += ["**待查文献检索式**", *[f"- {q}" for q in direction["novelty_queries"]], "", "**主要风险**", *[f"- {r}" for r in direction["risks"]], ""]
        if direction.get("model_review"):
            check = direction["model_review"]
            lines += ["**方案复核（模型辅助，非保证）**", check["recommendation"], *[f"- {c}" for c in check["concerns"]], "先验证：" + check["next_check"], ""]
    lines += ["## 需要进一步确认", *[f"- {q}" for q in result.get("open_questions", [])], "", "## 数据与运行版本", f"Prompt：{result.get('prompt_version', '')}", f"证据快照：{result.get('evidence_snapshot', '')}", "", "原文摘录匹配仅确认引用文本存在，不等于语义已经独立核实。"]
    if run.get("reviews"):
        lines += ["", "## 人工复核记录"]
        for review in run["reviews"]:
            lines += [f"- {review['item_id']} · {review['verdict']} · {review['created']}：{review['note']}"]
    return "\n".join(lines)
