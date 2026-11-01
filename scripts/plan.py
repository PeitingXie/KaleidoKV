import argparse
import gzip
import json

from kaleidokv import plan
from kaleidokv.models import TEMPLATES, load_workload


def main():
    parser = argparse.ArgumentParser(
        description="Plan evidence orders and the submission order on the CPU."
    )
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", required=True, help="model folder, only its tokenizer is read")
    parser.add_argument("--template", choices=sorted(TEMPLATES))
    parser.add_argument("--layout", choices=["kaleidokv", "original"], default="kaleidokv")
    parser.add_argument("--order", choices=["cost", "dfs", "arrival"], default="cost")
    parser.add_argument("--output", help="write the plan to this .json.gz file")
    args = parser.parse_args()

    pack, tokenizer, _ = load_workload(args.workload, args.model, args.template)
    original = plan(pack["input"], tokenizer, layout="original", order="arrival")
    result = plan(pack["input"], tokenizer, args.layout, args.order)
    print(f"requests          {len(result['ids'])}")
    print(f"prompt tokens     {result['prompt_tokens']}")
    print(f"retrieval order   {original['planned_prefill']} prefill tokens with prefix caching")
    print(f"{args.layout:<17} {result['planned_prefill']} prefill tokens")
    print(f"planning time     {result['planning_seconds']:.1f} s")
    if args.output:
        with gzip.open(args.output, "wt") as f:
            json.dump({k: v for k, v in result.items() if k != "search"}, f)


if __name__ == "__main__":
    main()
