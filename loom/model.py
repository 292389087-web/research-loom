"""本地 Qwen3-VL，单 GPU 串行推理、可取消、硬截止；不使用云 API。"""
import os
import time
import threading

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class LocalModel:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.processor = None
        self.lock = threading.Lock()
        self.load_seconds = None
        self.current_generation = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        path = self.settings.model_path
        if path is None:
            raise RuntimeError("未找到本地 Qwen3-VL 权重，请设置 LOOM_MODEL_PATH")
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA；本版本需要本地 NVIDIA GPU 推理")
        started = time.monotonic()
        processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(path), local_files_only=True, dtype=torch.float16, device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        ).eval()
        self.processor, self.model = processor, model
        self.load_seconds = round(time.monotonic() - started, 2)

    def unload(self):
        import gc
        with self.lock:
            self.model=self.processor=None
            self.current_generation=None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                return {"allocated_bytes":torch.cuda.memory_allocated(),"reserved_bytes":torch.cuda.memory_reserved()}
            return {"allocated_bytes":0,"reserved_bytes":0}

    def generate(self, system, prompt, image=None, max_tokens=1800, cancelled=lambda: False):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled() or time.monotonic() >= deadline

        with self.lock:
            self.load()
            if cancelled():
                raise InterruptedError("用户取消任务")
            started = time.monotonic()
            deadline = started + self.settings.generation_timeout
            content = [{"type": "text", "text": prompt}]
            images = None
            if image is not None:
                image = image.convert("RGB")
                ratio = min(1, (self.settings.max_image_pixels / (image.width * image.height)) ** .5)
                if ratio < 1:
                    from PIL import Image
                    image = image.resize((max(28, int(image.width*ratio)), max(28, int(image.height*ratio))), Image.Resampling.LANCZOS)
                content.insert(0, {"type": "image", "image": image})
                images = [image]
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[template], images=images, return_tensors="pt", padding=True).to(self.model.device)
            input_tokens = int(inputs.input_ids.shape[1])
            if input_tokens > 5500:
                raise ValueError("本地推理上下文超过 5500 token 上限，请减少论文范围")
            torch.cuda.reset_peak_memory_stats()
            self.current_generation = {"input_tokens": input_tokens, "max_output_tokens": max_tokens, "started": time.time(), "attention": self.model.config.text_config._attn_implementation}
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=max_tokens, logits_to_keep=1,
                    stopping_criteria=StoppingCriteriaList([Stop()]),
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            new_tokens = output[:, input_tokens:]
            text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            usage = {"input_tokens": input_tokens, "output_tokens": int(new_tokens.shape[1]), "seconds": round(time.monotonic()-started, 3), "peak_gpu_gb": round(torch.cuda.max_memory_allocated()/1024**3, 3)}
            del inputs, output, new_tokens
            self.current_generation = None
            if cancelled():
                raise InterruptedError("用户取消任务")
            if time.monotonic() >= deadline:
                raise TimeoutError("本地模型生成超时，任务可重试")
            return text, usage
