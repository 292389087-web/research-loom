"""资料权限先于复杂度和资源路由；远程后端不自动开启。"""

import base64
import io
import json
import hashlib
import traceback
import time
from urllib.parse import urlsplit
import httpx
from .gpu import gpu_slot, gpu_busy, ResourceBusy


class RemoteModel:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport

    @property
    def configured(self):
        return bool(
            self.settings.remote_url
            and self.settings.remote_model
            and self.settings.remote_key
        )

    def request_options(self):
        if urlsplit(self.settings.remote_url).hostname != "api.deepseek.com":
            return {}
        thinking = self.settings.remote_thinking
        effort = self.settings.remote_reasoning_effort
        if thinking not in {"enabled", "disabled"} or effort not in {
            "low",
            "high",
            "max",
        }:
            raise ValueError("DeepSeek thinking 或 reasoning_effort 配置无效")
        options = {
            "thinking": {"type": thinking},
            "response_format": {"type": "json_object"},
        }
        if thinking == "enabled":
            options["reasoning_effort"] = effort
        return options

    def generate(
        self, system, prompt, image=None, max_tokens=1800, cancelled=lambda: False
    ):
        if not self.configured:
            raise ValueError("增强 API 尚未配置")
        url = self.settings.remote_url.rstrip("/")
        parsed = urlsplit(url)
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or (
                parsed.scheme != "https"
                and not (
                    parsed.scheme == "http"
                    and parsed.hostname in {"127.0.0.1", "localhost"}
                )
            )
        ):
            raise ValueError(
                "API 地址必须是 HTTPS，或本机测试地址，且不能包含凭据/查询参数"
            )
        if cancelled():
            raise InterruptedError("任务取消")
        content = prompt
        if image is not None:
            if not self.settings.remote_vision:
                raise ValueError("当前 API 未声明支持图片")
            image = image.convert("RGB")
            image.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(buffer.getvalue()).decode()
                    },
                },
            ]
        start = time.monotonic()
        try:
            with httpx.Client(
                timeout=self.settings.generation_timeout,
                transport=self.transport,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                response = client.post(
                    url + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.settings.remote_key},
                    json={
                        "model": self.settings.remote_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": content},
                        ],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        **self.request_options(),
                    },
                )
        except httpx.TimeoutException as exc:
            raise TimeoutError("增强 API 超时；服务端是否执行及费用可能未知") from exc
        except httpx.HTTPError as exc:
            raise ConnectionError("增强 API 连接失败") from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise ConnectionError(f"增强 API 暂时失败，HTTP {response.status_code}")
        if response.status_code != 200:
            raise ValueError(
                f"增强 API 拒绝请求，HTTP {response.status_code}，请检查配置"
            )
        if len(response.content) > 2 * 1024 * 1024:
            raise ValueError("增强 API 响应超过限制")
        body = response.json()
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("增强 API 返回格式不兼容") from exc
        if not isinstance(text, str):
            raise ValueError("增强 API 未返回文本回答")
        if cancelled():
            raise InterruptedError("任务取消")
        usage = body.get("usage") or {}
        return text, {
            "seconds": round(time.monotonic() - start, 3),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "provider": "remote",
            "model": self.settings.remote_model,
            "provider_request_id": body.get("id"),
            "cost_status": "not_priced",
            "usage_status": "reported" if usage else "unknown",
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "response_model": body.get("model"),
            "finish_reason": body["choices"][0].get("finish_reason"),
        }


class RoutedModel:
    def __init__(self, router, decision):
        self.router, self.decision = router, decision

    def generate(self, *args, **kwargs):
        if self.decision["backend"] == "api":
            return self.router.remote.generate(*args, **kwargs)
        with gpu_slot(self.router.store.root):
            result_usage = None
            try:
                text, usage = self.router.local.generate(*args, **kwargs)
                result_usage = {
                    **usage,
                    "provider": "local",
                    "model": "Qwen3-VL-4B-Instruct",
                }
                return text, result_usage
            except BaseException as exc:
                traceback.clear_frames(exc.__traceback__)
                raise
            finally:
                # 释放权重后才释放互斥锁，否则训练可能拿到锁但没有足够显存。
                if hasattr(self.router.local, "unload"):
                    released = self.router.local.unload()
                    if result_usage is not None:
                        result_usage["after_unload"] = released


class ModelRouter:
    def __init__(self, store, settings, local, remote=None):
        self.store, self.settings, self.local = store, settings, local
        self.remote = remote or RemoteModel(settings)

    def permitted(self, request, image):
        if not request.get("allow_remote"):
            return False
        if image and not request.get("allow_remote_images"):
            return False
        if request.get('dataset_id'):
            from .research_data import dataset
            data = dataset(self.store, request['dataset_id'])
            if data['kind']=='studies':
                if not all(self.permitted({'study_id':sid,'allow_remote':True},False) for sid in data['study_ids']):
                    return False
            elif not data.get('allow_remote'):
                return False
        if request.get("study_id"):
            sid, visited_studies = request["study_id"], set()
            while sid:
                if sid in visited_studies:
                    return False
                visited_studies.add(sid)
                rows = self.store.query("SELECT spec FROM studies WHERE id=?", (sid,))
                if not rows:
                    return False
                spec = json.loads(rows[0]["spec"])
                if not spec.get("allow_remote", False):
                    return False
                sid = spec.get("parent_study_id")
        ids = request.get("paper_ids", [])
        for pid in ids:
            rows = self.store.query(
                "SELECT data FROM bibliography WHERE paper_id=?", (pid,)
            )
            if not rows or not json.loads(rows[0]["data"]).get("allow_remote", False):
                return False
        parent = request.get("parent_run_id")
        visited = set()
        while parent:
            if parent in visited:
                return False
            visited.add(parent)
            previous = self.store.run(parent)["request"]
            if not previous.get("allow_remote"):
                return False
            parent = previous.get("parent_run_id")
        return True

    def choose(self, request, stage, prompt, image=False, busy=None):
        permitted = self.permitted(request, image)
        policy = request.get("model_policy", "auto")
        busy = gpu_busy(self.store.root) if busy is None else busy
        capable = self.remote.configured and (not image or self.settings.remote_vision)
        complex_task = not image and (
            stage in {"研究方案", "方案可行性复核", "分析计划", "实验建议"}
            or len(request.get("paper_ids", [])) >= 3
            or len(prompt) > 16000
        )
        if policy == "api":
            if not permitted:
                raise ValueError("任务、文献或历史对话禁止外发，不能强制使用 API")
            if not capable:
                raise ValueError("增强 API 未配置或缺少图片能力")
            return {
                "backend": "api",
                "reason": "用户选择增强模式",
                "model": self.settings.remote_model,
                "policy_version": "hybrid-v1",
            }
        if policy == "auto" and permitted and capable and (busy or complex_task):
            return {
                "backend": "api",
                "reason": "GPU 正忙" if busy else "复杂分析阶段",
                "model": self.settings.remote_model,
                "policy_version": "hybrid-v1",
            }
        if busy:
            raise ResourceBusy("仅本地任务或 API 不可用：等待 GPU，不外发资料")
        return {
            "backend": "local",
            "reason": "资料仅本地" if not permitted else "本地优先或增强不可用",
            "model": "Qwen3-VL-4B-Instruct",
            "policy_version": "hybrid-v1",
        }

    def prepare(self, run_id, cp, stage, prompt, image, max_tokens):
        request = self.store.run(run_id)["request"]
        decisions = cp.setdefault("routing", {})
        decision = decisions.get(stage)
        if decision is None:
            decision = self.choose(request, stage, prompt, image is not None)
            if decision["backend"] == "api":
                decision["request_options"] = self.remote.request_options()
                decision["output_token_floor"] = (
                    self.settings.remote_thinking_tokens
                    if decision["request_options"].get("thinking", {}).get("type")
                    == "enabled"
                    else 0
                )
                decision["endpoint_sha256"] = hashlib.sha256(
                    self.settings.remote_url.rstrip("/").encode()
                ).hexdigest()
            decisions[stage] = decision
            self.store.update_run(run_id, checkpoint=cp)
            self.store.event(
                run_id,
                "model_routed",
                {
                    "stage": stage,
                    **decision,
                    "message": f"{stage} → {decision['backend']}：{decision['reason']}",
                },
            )
        if decision["backend"] == "api":
            if decision.get("request_options", {}) != self.remote.request_options():
                raise ValueError("API 推理配置已变更，请新建任务避免混用")
            floor = (
                self.settings.remote_thinking_tokens
                if self.remote.request_options().get("thinking", {}).get("type")
                == "enabled"
                else 0
            )
            if decision.get("output_token_floor", 0) != floor:
                raise ValueError("API 输出预算配置已变更，请新建任务")
            if not self.permitted(request, image is not None):
                raise ValueError("资料外发许可已撤回")
            if decision["model"] != self.settings.remote_model:
                raise ValueError("API 模型已变更，请新建任务避免混用")
            if (
                decision.get("endpoint_sha256")
                != hashlib.sha256(
                    self.settings.remote_url.rstrip("/").encode()
                ).hexdigest()
            ):
                raise ValueError("API 地址已变更，请新建任务")
        return RoutedModel(self, decision), decision
