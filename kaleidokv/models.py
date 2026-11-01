import json
from pathlib import Path

CHATML_SYSTEM = "<|im_start|>system\n"
CHATML_USER = "<|im_end|>\n<|im_start|>user\n"
CHATML_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"

LLAMA3_SYSTEM = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
LLAMA3_USER = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
LLAMA3_ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

TEMPLATES = {
    "qwen2.5": dict(
        system=CHATML_SYSTEM,
        user=CHATML_USER,
        assistant=CHATML_ASSISTANT,
        strip_system=False,
        stop=[151645, 151643],
    ),
    "qwen3": dict(
        system=CHATML_SYSTEM,
        user=CHATML_USER,
        assistant=CHATML_ASSISTANT + "<think>\n\n</think>\n\n",
        strip_system=False,
        stop=[151645, 151643],
    ),
    "llama3": dict(
        system=LLAMA3_SYSTEM,
        user=LLAMA3_USER,
        assistant=LLAMA3_ASSISTANT,
        strip_system=True,
        stop=[128001, 128009],
    ),
    "llama3.1": dict(
        system=LLAMA3_SYSTEM + "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
        user=LLAMA3_USER,
        assistant=LLAMA3_ASSISTANT,
        strip_system=True,
        stop=[128001, 128008, 128009],
    ),
}


def detect_template(model_path):
    name = Path(model_path).name.lower()
    if "llama-3.1" in name or "llama3.1" in name:
        return "llama3.1"
    if "llama-3" in name or "llama3" in name:
        return "llama3"
    if "qwen3" in name:
        return "qwen3"
    if "qwen2.5" in name:
        return "qwen2.5"
    raise ValueError(f"Cannot tell the chat template of {model_path}, pass --template")


def apply_template(pack, template):
    style = TEMPLATES[template]
    data = dict(pack["input"])
    header = data["header"]
    if not header.startswith(CHATML_SYSTEM) or header.count(CHATML_USER) != 1:
        raise ValueError("Workload headers are written in ChatML")
    system, user = header[len(CHATML_SYSTEM) :].split(CHATML_USER)
    if style["strip_system"]:
        system = system.strip()
    data["header"] = style["system"] + system + style["user"] + user
    requests = []
    for request in data["requests"]:
        suffix = request.get("suffix", request["question"] + CHATML_ASSISTANT)
        if not suffix.endswith(CHATML_ASSISTANT):
            raise ValueError("Request suffixes are written in ChatML")
        requests.append(dict(request, suffix=suffix[: -len(CHATML_ASSISTANT)] + style["assistant"]))
    data["requests"] = requests
    return dict(pack, input=data)


def load_workload(path, model, template=None):
    from tokenizers import Tokenizer

    template = template or detect_template(model)
    with open(path) as f:
        pack = apply_template(json.load(f), template)
    tokenizer = Tokenizer.from_file(str(Path(model) / "tokenizer.json"))
    return pack, tokenizer, template
