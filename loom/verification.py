"""证据存在/原文摘录校验与语义判断分开；原文匹配不等于语义蕴含。"""
import re


def normalized(text):
    text = re.sub(r"(?<=[a-zA-Z])-\s*\n\s*(?=[a-zA-Z])", "", text)
    return re.sub(r"\s+", "", text).lower().replace("−", "-").replace("–", "-")


def verify_findings(findings, evidence):
    known = {ev["id"]: ev for ev in evidence}
    output = []
    for i, finding in enumerate(findings):
        item = dict(finding)
        item["id"] = f"f{i+1}"
        checked = []
        for citation in item.get("citations", []):
            ev = known.get(citation["evidence_id"])
            exact = bool(ev and normalized(citation["quote"]) in normalized(ev["content"]) and len(normalized(citation["quote"])) >= 8)
            checked.append({**citation, "quote_matches": exact, "kind": ev["kind"] if ev else "missing", "paper_id": ev["paper_id"] if ev else None, "page": ev["page"] if ev else None})
        item["citations"] = checked
        all_match = bool(checked) and all(c["quote_matches"] for c in checked)
        if not all_match:
            item["source_status"] = "needs_review"
        elif any(c["kind"] == "visual" for c in checked):
            item["source_status"] = "visual_interpretation"
        else:
            item["source_status"] = "quote_matched"
        item["semantic_status"] = "inference" if item.get("basis") == "model_inference" else "not_independently_verified"
        item["review_reasons"] = []
        if re.search(r"未(?:明确|提供|说明|报告|进行)|没有(?:提供|评估|实验)|未见", item["statement"]):
            item["review_reasons"].append("涉及缺失性判断：当前片段不足以证明整篇论文未做或未报告，请核查全文")
            item["basis"] = "model_inference"
            item["semantic_status"] = "inference"
        if re.search(r"适合.*(?:GPU|算力|周内)|保证|创新性地|足以证明", item["statement"]):
            item["review_reasons"].append("涉及资源适配或较强结论，原文摘录匹配不能验证此判断")
        if any(c["kind"] == "visual" for c in checked):
            item["basis"] = "model_inference"
            item["semantic_status"] = "inference"
        output.append(item)
    return output


def verify_directions(directions, evidence):
    known = {e["id"] for e in evidence}
    output = []
    for i, direction in enumerate(directions):
        item = dict(direction)
        item["id"] = f"d{i+1}"
        item["novelty_status"] = "not_verified"
        item["basis"] = "research_hypothesis"
        item["source_status"] = "references_exist" if item["evidence_ids"] and set(item["evidence_ids"]) <= known else "needs_review"
        exp = dict(item["experiment"])
        item["experiment"] = exp
        relation = exp.get("relation")
        if relation:
            exp["model_draft_hypothesis"] = exp["hypothesis"]
            exp["model_draft_falsification"] = exp["falsification"]
            hypothesis, falsification = compile_relation(relation)
            exp["hypothesis"], exp["falsification"] = hypothesis, falsification
            exp["relation_status"] = "compiled"
        item["missing_experiment_fields"] = [k for k in ("hypothesis", "setup", "baselines", "metrics", "ablations", "falsification", "feasibility") if not exp.get(k)]
        output.append(item)
    return output


def compile_relation(relation):
    """从明确的实验关系生成相互对应的假设/否定条件，不依赖模型二次心算。"""
    candidate, reference, metric = relation["candidate"], relation["reference"], relation["metric"]
    lower = relation["metric_direction"] == "lower"
    comparator = "低于" if lower else "高于"
    prefix = f"在{relation['condition']}条件下，"
    protocol = "差异阈值/非劣效界限、数据划分、资源预算及重复运行方案应在正式测试前根据预实验确定并冻结。"
    expectation = relation["expectation"]
    if expectation == "better":
        return prefix + f"{candidate}的{metric}{comparator}{reference}。", f"若{candidate}的{metric}未达到相对{reference}预先约定的改善标准，则该假设未获支持。" + protocol
    if expectation == "not_worse":
        return prefix + f"{candidate}的{metric}不劣于{reference}（该指标{'越低越好' if lower else '越高越好'}）。", f"若{candidate}的{metric}劣于{reference}且超出事先约定的非劣效界限，则否定该假设。" + protocol
    if expectation == "different":
        return prefix + f"{candidate}与{reference}在{metric}上存在有实际意义的差异。", f"若{candidate}与{reference}的{metric}差异未达到预先约定的判定标准，则差异假设未获支持。" + protocol
    return prefix + f"探索{candidate}与{reference}在{metric}上的关系，不预设优劣。", "此项是探索性研究，当前没有可否定的方向性假设；先用预实验形成假设，再另设冻结测试集验证。"
