import json
import os
from collections import Counter
from pathlib import Path

from .prefix_index import fingerprint

CONFIG_ENV = "KALEIDOKV_ADMISSION"

_state = None


def representatives(index):
    first_seen = {}

    def shared_path(q):
        return tuple(tuple(edge) for edge in index["paths"][q] if index["counts"][edge[0]] > 1)

    for q, digest in enumerate(index["fingerprints"]):
        if digest not in first_seen:
            first_seen[digest] = q
            continue
        other = first_seen[digest]
        if index["lengths"][q] != index["lengths"][other] or shared_path(q) != shared_path(other):
            raise ValueError("Identical prompts must share the same prefix path")
    return first_seen


class AdmissionState:
    def __init__(self, scheduler, config):
        self.scheduler = scheduler
        self.index = json.loads(Path(config["index"]).read_text())
        self.output = Path(config["output"])
        self.window = config.get("window", 64)
        self.min_shared_tokens = config.get("min_shared_tokens", 32)
        self.claim_chunked = config.get("claim_chunked", True)
        self.by_fingerprint = representatives(self.index)
        self.claims = {}
        self.counters = Counter()

    def shared_segments(self, q):
        for node, start, end in self.index["paths"][q]:
            if self.index["counts"][node] > 1:
                yield node, start, end

    def request_id(self, req):
        if not hasattr(req, "_kaleidokv_q"):
            req._kaleidokv_q = self.by_fingerprint.get(fingerprint(req.origin_input_ids))
        return req._kaleidokv_q

    def cached_length(self, req):
        from sglang.srt.mem_cache.radix_cache import RadixKey

        cache = self.scheduler.tree_cache
        tokens, extra = req.origin_input_ids, req.extra_key
        limit, pos, node = len(tokens) - 1, 0, cache.root_node
        while pos < limit:
            child = node.children.get(cache.get_child_key_fn(RadixKey([tokens[pos]], extra)))
            if child is None or child.value is None:
                break
            edge = child.key.token_ids
            count = min(len(edge), limit - pos)
            if child.key.extra_key == extra and edge == tokens[pos : pos + count]:
                matched = len(edge)
            else:
                matched = cache.key_match_fn(child.key, RadixKey(tokens[pos:limit], extra))
            pos += matched
            if matched < len(edge):
                break
            node = child
        return pos

    def offer(self, offered, q, cached):
        last = self.index["lengths"][q] - 1
        for node, start, end in self.shared_segments(q):
            reach = min(end, cached + self.scheduler.chunked_prefill_size, last)
            if reach > max(start, cached):
                offered[node] = max(offered.get(node, 0), reach)

    def split_waiting(self, waiting):
        offered, eligible, held = {}, [], []
        chunked = getattr(self.scheduler, "chunked_req", None)
        if self.claim_chunked and chunked is not None:
            q = self.request_id(chunked)
            if q is not None:
                done = len(chunked.prefix_indices)
                for node, start, end in self.shared_segments(q):
                    if end > max(start, done):
                        offered[node] = max(offered.get(node, 0), end)
        for req in waiting[: self.window]:
            q = self.request_id(req)
            if q is None:
                eligible.append(req)
                continue
            cached = self.cached_length(req)
            if self.must_wait(req, q, cached, offered):
                held.append(req)
                self.counters["held"] += 1
                continue
            eligible.append(req)
            self.offer(offered, q, cached)
        return eligible + waiting[self.window :], held

    def must_wait(self, req, q, cached, offered):
        for node, start, end in self.shared_segments(q):
            in_flight = [
                reach for owner, reach in self.claims.get(node, {}).items() if owner != req.rid
            ]
            ready_at = max(in_flight + [offered.get(node, 0)])
            if min(end, ready_at) - max(start, cached) >= self.min_shared_tokens:
                return True
        return False

    def record_extend(self, batch):
        if not batch.forward_mode.is_extend():
            return
        for req, done, count in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
            q = self.request_id(req)
            if q is None:
                continue
            reach = min(done + count, len(req.origin_input_ids))
            for node, start, end in self.shared_segments(q):
                if min(end, reach) > max(start, done):
                    self.claims.setdefault(node, {})[req.rid] = min(end, reach)

    def release(self, req, finished):
        q = self.request_id(req)
        if q is None:
            return
        cached = min(len(req.origin_input_ids), len(req.fill_ids))
        for node, _, _ in self.shared_segments(q):
            owners = self.claims.get(node)
            if owners and req.rid in owners and (finished or owners[req.rid] <= cached):
                del owners[req.rid]
                if not owners:
                    self.claims.pop(node)

    def save_counters(self):
        if self.scheduler.tp_rank == 0:
            (self.output / "admission_counters.json").write_text(
                json.dumps(self.counters, indent=2)
            )


def install():
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.radix_cache import RadixCache

    if getattr(Scheduler, "_kaleidokv_admission", False):
        return
    Scheduler._kaleidokv_admission = True
    config = json.loads(Path(os.environ[CONFIG_ENV]).read_text())

    scheduler_init = Scheduler.__init__

    def init(self, *args, **kwargs):
        global _state
        scheduler_init(self, *args, **kwargs)
        if self.tree_cache.page_size != 1 or self.enable_hierarchical_cache:
            raise RuntimeError("Prefix-ready admission needs page size 1 and no hierarchical cache")
        _state = AdmissionState(self, config)

    get_new_batch_prefill = Scheduler.get_new_batch_prefill

    def admit(self):
        if _state is None:
            return get_new_batch_prefill(self)
        admission_closed = (
            self.chunked_req is None
            and not self.try_preemption
            and not self.grammar_queue
            and (self.running_batch.batch_is_full or not self.waiting_queue)
        )
        if admission_closed:
            return get_new_batch_prefill(self)
        waiting = self.waiting_queue
        eligible, held = _state.split_waiting(waiting)
        self.waiting_queue = eligible
        try:
            batch = get_new_batch_prefill(self)
        finally:
            remaining = set(self.waiting_queue) | set(held)
            before = set(waiting)
            arrived = [req for req in self.waiting_queue if req not in before]
            self.waiting_queue = [req for req in waiting if req in remaining] + arrived
        if batch is None and held:
            _state.counters["fallbacks"] += 1
            batch = get_new_batch_prefill(self)
        return batch

    run_batch = Scheduler.run_batch

    def run(self, batch):
        if _state is not None:
            _state.record_extend(batch)
        return run_batch(self, batch)

    Scheduler.__init__ = init
    Scheduler.get_new_batch_prefill = admit
    Scheduler.run_batch = run

    for name in ("cache_unfinished_req", "cache_finished_req"):
        original = getattr(RadixCache, name)

        def cache_req(
            self, req, *args, _original=original, _finished=name == "cache_finished_req", **kwargs
        ):
            result = _original(self, req, *args, **kwargs)
            if _state is not None and self is _state.scheduler.tree_cache:
                _state.release(req, _finished)
            return result

        setattr(RadixCache, name, cache_req)

    cache_reset = RadixCache.reset

    def reset(self):
        result = cache_reset(self)
        if _state is not None and self is _state.scheduler.tree_cache:
            _state.claims.clear()
            _state.save_counters()
        return result

    RadixCache.reset = reset
