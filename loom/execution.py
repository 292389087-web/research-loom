"""模型步骤的持久执行边界。已落盘回答可回放；未记录回答只能标记未知。"""
import hashlib
import json
import io
from .store import dump, now, uid
from .gpu import ResourceBusy


class StageExecutor:
    def __init__(self, store, settings, model, fault=None):
        self.store, self.settings, self.model = store, settings, model
        self.fault = fault or (lambda point: None)

    def generate(self, run_id, cp, stage, system, instruction, schema, version,
                 emit, cancelled, image=None, max_tokens=1900):
        if stage in cp:
            emit("stage_replayed", {"stage":stage,"message":f"{stage}复用已提交结果，不重新调用模型"})
            return cp[stage]
        prompt = instruction + "\n输出必须符合此 JSON Schema（只输出 JSON 对象，不要复制 Schema）：\n" + json.dumps(schema.model_json_schema(),ensure_ascii=False)
        image_meta = None
        if image is not None:
            buffer = io.BytesIO()
            image.save(buffer,format="PNG")
            blob = buffer.getvalue()
            sha = hashlib.sha256(blob).hexdigest()
            artifacts = self.store.root / "model-inputs"
            artifacts.mkdir(exist_ok=True)
            target = artifacts / (sha+".png")
            if not target.exists():
                temporary = artifacts / (uid("incoming")+".png")
                temporary.write_bytes(blob)
                temporary.replace(target)
            image_meta = {"sha256":sha,"file":f"model-inputs/{sha}.png","size":list(image.size)}
        for repair in range(2):
            cancelled()
            selected,decision=(self.model,{"backend":"test_or_legacy"})
            if hasattr(self.model,"prepare"):
                selected,decision=self.model.prepare(run_id,cp,stage,prompt,image,max_tokens)
            call_max_tokens=max(max_tokens,decision.get("output_token_floor",0))
            payload = {"system":system,"prompt":prompt,"schema":schema.model_json_schema(),"prompt_version":version,
                       "image":image_meta,"max_tokens":call_max_tokens,"model":type(self.model).__name__,
                       "model_path":None if decision["backend"]=="api" else str(self.settings.model_path),"do_sample":False,
                       "max_image_pixels":self.settings.max_image_pixels,"routing":decision}
            input_hash = hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            prior = self.store.query("SELECT * FROM model_attempts WHERE run_id=? AND stage=? AND input_hash=? AND (status='responded' OR (status='invalid' AND epoch=?)) ORDER BY created DESC LIMIT 1",(run_id,stage,input_hash,self.store.local.lease[2]))
            if prior:
                attempt = prior[0]
                attempt_id, text = attempt["id"], attempt["raw_output"]
                usage = json.loads(attempt["usage"]) if attempt["usage"] else None
                emit("response_replayed",{"message":f"{stage}复用已记录模型回答","stage":stage,"attempt_id":attempt_id})
            else:
                if cp["calls"] >= self.settings.max_model_calls:
                    raise RuntimeError("任务累计模型调用预算耗尽；请缩小范围或新建任务")
                if decision["backend"]=="api":
                    if cp.get("remote_calls",0)>=self.settings.remote_call_limit or cp.get("remote_output_reserved",0)+call_max_tokens>self.settings.remote_output_budget:
                        raise ValueError("远程调用预算耗尽，不再外发")
                    cp["remote_calls"]=cp.get("remote_calls",0)+1
                    cp["remote_output_reserved"]=cp.get("remote_output_reserved",0)+call_max_tokens
                attempt_id = uid("ma")
                cp["calls"] += 1
                with self.store.db() as db:
                    # 上次调用在记录回答前中断，费用/输出未知，不能记为零。
                    db.execute("UPDATE model_attempts SET status='unknown',error=?,updated=? WHERE run_id=? AND stage=? AND status='started'",("回答未可靠记录，不能证明模型未执行",now(),run_id,stage))
                    epoch = self.store.local.lease[2]
                    db.execute("INSERT INTO model_attempts(id,run_id,epoch,stage,input_hash,input,status,created,updated) VALUES (?,?,?,?,?,?,?,?,?)",(attempt_id,run_id,epoch,stage,input_hash,dump(payload),"started",now(),now()))
                    db.execute("UPDATE runs SET checkpoint=?,updated=? WHERE id=?",(dump(cp),now(),run_id))
                    self.store._event(db,run_id,"model_started",{"message":f"{stage}：{decision['backend']} 模型处理中","stage":stage,"attempt_id":attempt_id,"call":cp["calls"],"budget":self.settings.max_model_calls,"epoch":epoch,"input_hash":input_hash})
                def interrupted():
                    try:
                        cancelled()
                        return False
                    except InterruptedError:
                        return True
                try:
                    text, usage = selected.generate(system,prompt,image=image,max_tokens=call_max_tokens,cancelled=interrupted)
                    self.fault("response_before_record")
                except ResourceBusy:
                    cp["calls"]-=1
                    with self.store.db() as db:
                        db.execute("UPDATE model_attempts SET status='not_executed',updated=? WHERE id=?",(now(),attempt_id))
                        db.execute("UPDATE runs SET checkpoint=?,updated=? WHERE id=?",(dump(cp),now(),run_id))
                    raise
                except Exception as exc:
                    self.store.execute("UPDATE model_attempts SET status='unknown',error=?,updated=? WHERE id=?",(f"{type(exc).__name__}: {str(exc)[:1000]}",now(),attempt_id))
                    raise
                with self.store.db() as db:
                    db.execute("UPDATE model_attempts SET status='responded',raw_output=?,usage=?,updated=? WHERE id=?",(text,dump(usage),now(),attempt_id))
                    cp["usage"].append({"stage":stage,"attempt_id":attempt_id,**usage})
                    db.execute("UPDATE runs SET checkpoint=?,updated=? WHERE id=?",(dump(cp),now(),run_id))
                self.fault("response_recorded")
            try:
                value = text.strip()
                if value.startswith("```"):
                    value = value.split("\n",1)[-1].rsplit("```",1)[0].strip()
                result = schema.model_validate(json.loads(value)).model_dump()
            except (ValueError,TypeError) as exc:
                self.store.execute("UPDATE model_attempts SET status='invalid',error=?,updated=? WHERE id=?",(str(exc)[:1500],now(),attempt_id))
                if repair:
                    raise ValueError(f"{stage}结构校验失败，未将无效结果标记为成功：{str(exc)[:250]}") from exc
                emit("schema_retry",{"message":"结构化输出未通过校验，尝试一次修复","stage":stage,"attempt_id":attempt_id})
                prompt += "\n上次输出无效。请缩短内容并修复结构。错误："+str(exc)[:1200]+"\n上次输出："+text[:6500]
                continue
            cancelled()
            cp[stage] = result
            with self.store.db() as db:
                db.execute("UPDATE model_attempts SET status='accepted',updated=? WHERE id=?",(now(),attempt_id))
                db.execute("UPDATE runs SET checkpoint=?,updated=? WHERE id=?",(dump(cp),now(),run_id))
                self.store._event(db,run_id,"model_completed",{"message":f"{stage}完成","stage":stage,"attempt_id":attempt_id,**(usage or {})})
            self.fault("stage_committed")
            return result
