from queue import Empty
import sys
import threading

from ...backends import parse_device_map
from .base import BaseBackend


class TransformersBackend(BaseBackend):
    name = "transformers"

    @classmethod
    def from_args(cls, args):
        return cls(args.model, args.dtype, args.require_gpu, args.device_map, args.prefill_chunk_size)

    def generation_options(self, messages, max_tokens):
        return {"inputs": self.prepare(messages, max_tokens)}

    def __init__(self, model_id, dtype, require_gpu=False, device_map=None, prefill_chunk_size=None):
        self.model_id, self.dtype = model_id, dtype
        self.require_gpu = require_gpu
        self.device_map = parse_device_map(device_map)
        self.prefill_chunk_size = prefill_chunk_size
        self.failure = None
        self.device = None
        self.devices = []
        self.lock = threading.Lock()

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_cuda = torch.cuda.is_available()
        if self.require_gpu and not use_cuda:
            raise RuntimeError("CUDA is required but unavailable")
        if self.prefill_chunk_size:
            from transformers import GenerationConfig

            if not hasattr(GenerationConfig(), "prefill_chunk_size"):
                raise RuntimeError("installed Transformers does not support chunked prefill")

        dtypes = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
        options = {"torch_dtype": dtypes[self.dtype]}
        if self.device_map is not None:
            count = torch.cuda.device_count()
            if count < 2:
                raise RuntimeError("multi-GPU placement requires at least two CUDA GPUs")
            options["device_map"] = self.device_map
            if self.device_map == "balanced":
                options["max_memory"] = {
                    index: max(0, torch.cuda.mem_get_info(index)[0] - 1024**3)
                    for index in range(count)
                }
            elif max(self.device_map.values()) >= count:
                raise RuntimeError(f"device map contains a GPU index outside the {count} visible CUDA GPUs")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **options)
        if self.device_map is not None:
            placements = {f"cuda:{where}" if isinstance(where, int) else str(where)
                          for where in self.model.hf_device_map.values()}
            if len(placements) < 2 or any(not where.startswith("cuda:") for where in placements):
                raise RuntimeError(f"placement needs weights on multiple CUDA GPUs: {sorted(placements)}")
            self.devices = sorted(placements)
        else:
            self.model.to("cuda" if use_cuda else "cpu")
            self.devices = [str(self.model.device)]
        self.model.eval()
        self.device = str(self.model.get_input_embeddings().weight.device)
        self.dtype = str(self.model.dtype)

    def prepare(self, messages, max_tokens):
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.get_input_embeddings().weight.device)
        context = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(context, int) and inputs["input_ids"].shape[-1] + max_tokens > context:
            raise ValueError(f"prompt plus completion exceeds model context ({context} tokens)")
        return inputs

    def generate(self, messages, max_tokens, temperature, top_p, *, generation=None, inputs=None):
        from transformers import TextIteratorStreamer

        if inputs is None:
            inputs = self.prepare(messages, max_tokens)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=60)
        kw = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer, do_sample=temperature > 0)
        if self.prefill_chunk_size and inputs["input_ids"].shape[-1] > self.prefill_chunk_size:
            kw["prefill_chunk_size"] = self.prefill_chunk_size
        if temperature > 0:
            kw.update(temperature=temperature, top_p=top_p)
        if generation:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class Cancelled(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return generation.should_stop()

            kw.update(stopping_criteria=StoppingCriteriaList([Cancelled()]), return_dict_in_generate=False)
        if not self.lock.acquire(timeout=1):
            raise RuntimeError("generation worker busy")
        errors = []

        def generate():
            try:
                output = self.model.generate(**kw)
                if generation:
                    prompt_tokens = inputs["input_ids"].shape[-1]
                    completion_tokens = output.shape[-1] - prompt_tokens
                    eos = self.model.generation_config.eos_token_id
                    eos_ids = eos if isinstance(eos, (list, tuple)) else (eos,)
                    if completion_tokens >= max_tokens and int(output[0, -1]) not in eos_ids:
                        generation.finish_reason = "length"
                    generation.usage = dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                            total_tokens=prompt_tokens + completion_tokens)
            except Exception as exc:
                errors.append(exc)
                streamer.end()
            finally:
                self.lock.release()

        worker = threading.Thread(target=generate, daemon=True)
        try:
            worker.start()
        except Exception:
            self.lock.release()
            raise
        try:
            yield from streamer
        except Empty:
            self.failure = "generation worker timed out; restart server"
            raise TimeoutError(self.failure)
        finally:
            if generation and worker.is_alive():
                # A closed generator or a stalled streamer must release the worker.
                if sys.exc_info()[0] is not None:
                    generation.cancelled.set()
                worker.join(timeout=2)
                generation.worker_stopped = not worker.is_alive()
                if not generation.worker_stopped:
                    self.failure = "generation worker did not stop; restart server"
            else:
                worker.join(timeout=1)
        if errors:
            raise errors[0]
