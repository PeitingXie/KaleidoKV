import json
import os
import time
from pathlib import Path

from . import admission

OUTPUT_ENV = "KALEIDOKV_OUTPUT"
SGLANG_VERSION = "0.5.4"


def install_hooks():
    from importlib.metadata import version

    if version("sglang") != SGLANG_VERSION:
        raise RuntimeError(f"KaleidoKV patches the scheduler of sglang=={SGLANG_VERSION}")
    if os.environ.get(OUTPUT_ENV):
        record_first_tokens(Path(os.environ[OUTPUT_ENV]))
    if os.environ.get(admission.CONFIG_ENV):
        admission.install()


def record_first_tokens(output):
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler, "_kaleidokv_ttft", False):
        return
    Scheduler._kaleidokv_ttft = True
    process_prefill = Scheduler.process_batch_result_prefill
    flush_cache = Scheduler.flush_cache

    def on_prefill_result(self, batch, result, *args, **kwargs):
        out = process_prefill(self, batch, result, *args, **kwargs)
        first_tokens = getattr(self, "_first_tokens", None)
        if self.tp_rank == 0 and first_tokens is not None:
            now = time.time()
            for req in batch.reqs:
                if req.output_ids and req.rid not in first_tokens:
                    first_tokens[req.rid] = now
        return out

    def on_flush(self, *args, **kwargs):
        result = flush_cache(self, *args, **kwargs)
        if result:
            self._flushes = getattr(self, "_flushes", 0) + 1
            if self._flushes == 1:
                self._first_tokens = {}
            elif self.tp_rank == 0 and getattr(self, "_first_tokens", None) is not None:
                (output / "first_tokens.json").write_text(json.dumps(self._first_tokens))
                self._first_tokens = None
        return result

    Scheduler.process_batch_result_prefill = on_prefill_result
    Scheduler.flush_cache = on_flush


def kv_capacity(model_path, tp_size, kv_gib):
    config = json.loads((Path(model_path) / "config.json").read_text())
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    head_dim = config.get("head_dim") or config["hidden_size"] // heads
    row_bytes = config["num_key_value_heads"] // tp_size * head_dim * 2
    token_bytes = 2 * layers * row_bytes
    pointer_bytes = 48 * layers
    metadata_reserve = 96 * 2**20
    return (kv_gib * 2**30 - metadata_reserve - pointer_bytes) // token_bytes - 1


def generate(
    plan,
    model_path,
    stop_token_ids,
    output,
    tp_size=2,
    kv_gib=4,
    max_new_tokens=128,
    context_length=8192,
):
    from sglang.srt.entrypoints.engine import Engine
    from transformers import AutoTokenizer

    output = Path(output)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    engine = Engine(
        model_path=model_path,
        dtype="bfloat16",
        tp_size=tp_size,
        context_length=context_length,
        mem_fraction_static=0.9,
        max_total_tokens=kv_capacity(model_path, tp_size, kv_gib),
        chunked_prefill_size=2048,
        max_prefill_tokens=2048,
        max_running_requests=256,
        attention_backend="flashinfer",
        page_size=1,
        skip_tokenizer_init=True,
        schedule_policy="fcfs",
        random_seed=42,
        cuda_graph_max_bs=16,
        enable_mixed_chunk=False,
        log_level="warning",
    )
    try:
        sampling = dict(temperature=0, max_new_tokens=max_new_tokens, stop_token_ids=stop_token_ids)
        engine.generate(
            input_ids=plan["input_ids"][:2], sampling_params=dict(sampling, max_new_tokens=2)
        )
        flush(engine)
        started = time.perf_counter()
        submitted = time.time()
        responses = engine.generate(input_ids=plan["input_ids"], sampling_params=sampling)
        texts = [tokenizer.decode(r["output_ids"], skip_special_tokens=True) for r in responses]
        seconds = time.perf_counter() - started
        flush(engine)
    finally:
        engine.shutdown()

    first_tokens = json.loads((output / "first_tokens.json").read_text())
    answers = []
    for request_id, response, text in zip(plan["ids"], responses, texts):
        meta = response["meta_info"]
        answers.append(
            dict(
                id=request_id,
                text=text,
                output_tokens=len(response["output_ids"]),
                cached_tokens=meta.get("cached_tokens", 0),
                ttft=first_tokens[meta["id"]] - submitted,
            )
        )
    return answers, seconds


def flush(engine):
    for _ in range(30):
        result = engine.flush_cache()
        if getattr(result, "success", result is True):
            return
        time.sleep(0.1)
    raise RuntimeError("SGLang did not flush its cache")
