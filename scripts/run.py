import argparse
import json
import os
import statistics
from pathlib import Path

from kaleidokv import admission, engine, plan, scoring
from kaleidokv.models import TEMPLATES, load_workload

if os.environ.get(engine.OUTPUT_ENV):
    engine.install_hooks()


def main():
    parser = argparse.ArgumentParser(description="Plan a workload and run it on SGLang.")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", choices=sorted(TEMPLATES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--kv-gib", type=int, default=4, help="KV cache budget per GPU in GiB")
    parser.add_argument("--layout", choices=["kaleidokv", "original"], default="kaleidokv")
    parser.add_argument("--order", choices=["cost", "dfs", "arrival"], default="cost")
    parser.add_argument(
        "--no-admission", action="store_true", help="turn off prefix-ready admission"
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pack, tokenizer, template = load_workload(args.workload, args.model, args.template)
    style = TEMPLATES[template]
    contract = pack["input"]["contract"]
    result = plan(pack["input"], tokenizer, args.layout, args.order)
    (output / "index.json").write_text(json.dumps(result["index"]))

    os.environ[engine.OUTPUT_ENV] = str(output)
    if not args.no_admission:
        config = output / "admission.json"
        config.write_text(json.dumps(dict(index=str(output / "index.json"), output=str(output))))
        os.environ[admission.CONFIG_ENV] = str(config)
    engine.install_hooks()
    answers, generation_seconds = engine.generate(
        result,
        args.model,
        style["stop"],
        output,
        tp_size=args.tp_size,
        kv_gib=args.kv_gib,
        max_new_tokens=contract["max_output_tokens"],
        context_length=contract["context_tokens"],
    )

    total_seconds = result["planning_seconds"] + generation_seconds
    summary = dict(
        workload=args.workload,
        model=args.model,
        layout=args.layout,
        order=args.order,
        admission=not args.no_admission,
        requests=len(answers),
        planning_seconds=result["planning_seconds"],
        generation_seconds=generation_seconds,
        end_to_end_seconds=total_seconds,
        throughput=len(answers) / total_seconds,
        mean_ttft_seconds=result["planning_seconds"] + statistics.mean(a["ttft"] for a in answers),
        planned_prefill=result["planned_prefill"],
        executed_prefill=result["prompt_tokens"] - sum(a["cached_tokens"] for a in answers),
        quality=scoring.score(pack["gold"], answers),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(output / "answers.jsonl", "w") as f:
        for answer in answers:
            f.write(json.dumps(answer) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
