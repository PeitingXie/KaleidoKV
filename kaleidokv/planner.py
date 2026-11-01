import time

from .layout import layout_weight, plan_layout
from .prefix_index import build, summarize_prompts
from .workload import Workload

LAYOUTS = ("kaleidokv", "original")
ORDERS = ("cost", "dfs", "arrival")


def plan(data, tokenizer, layout="kaleidokv", order="cost", check_tokens=True):
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    started = time.perf_counter()
    workload = Workload(data, tokenizer)
    search = {}
    if layout == "original":
        permutations = workload.original
    else:
        permutations, search = plan_layout(workload.original, workload.lengths)
    prompts = [workload.render(q, p) for q, p in enumerate(permutations)]
    summary = summarize_prompts(prompts, "cost" if order == "cost" else "first")
    request_order = list(range(len(prompts))) if order == "arrival" else summary["order"]
    ordered = [prompts[q] for q in request_order]
    index = build(ordered)
    if index["ideal_P"] != summary["ideal_P"]:
        raise RuntimeError("The prefix index disagrees with the planned prefill")
    planning_seconds = time.perf_counter() - started
    if check_tokens:
        workload.validate(permutations)
    return dict(
        ids=[workload.rows[q]["id"] for q in request_order],
        input_ids=ordered,
        request_order=request_order,
        permutations=[list(p) for p in permutations],
        planned_prefill=summary["ideal_P"],
        evidence_weight=layout_weight(permutations, workload.lengths),
        prompt_tokens=sum(map(len, prompts)),
        index=index,
        search=search,
        planning_seconds=planning_seconds,
    )
