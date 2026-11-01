from types import SimpleNamespace


class Workload:

    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.header = data["header"]
        self.rows = data["requests"]
        self.texts = [m["text"] for m in data["materials"]]
        if not self.rows or not self.texts:
            raise ValueError("Requests and materials must be nonempty")
        for field, rows in [("request", self.rows), ("material", data["materials"])]:
            ids = [r["id"] for r in rows]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate {field} IDs")
        self.system = tuple(self.encode(self.header))
        self.blocks = [tuple(ids) for ids in self.encode_many(self.texts)]
        self.lengths = tuple(map(len, self.blocks))
        self.suffixes = [
            r.get("suffix", r["question"] + "<|im_end|>\n<|im_start|>assistant\n")
            for r in self.rows
        ]
        suffix_tokens = self.encode_many(self.suffixes)
        self.original = [tuple(r["materials"]) for r in self.rows]
        self.requests = []
        for q, (row, materials, suffix) in enumerate(zip(self.rows, self.original, self.suffixes)):
            if not materials or len(set(materials)) != len(materials):
                raise ValueError(f"Request {row['id']}: evidence must be nonempty and unique")
            if any((type(m) is not int or not 0 <= m < len(self.texts) for m in materials)):
                raise ValueError(f"Request {row['id']}: invalid material index")
            tokens = tuple(suffix_tokens[q])
            if not tokens:
                raise ValueError("Each request must end with a nonempty query suffix")
            self.requests.append(SimpleNamespace(query_tokens=tokens))

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_many(self, texts):
        batch = getattr(self.tokenizer, "encode_batch_fast", self.tokenizer.encode_batch)
        if not self.word_marker_prefix():
            return [e.ids for e in batch(list(texts), add_special_tokens=False)]
        nl = self.encode(chr(10))
        out = []
        for e in batch([chr(10) + t for t in texts], add_special_tokens=False):
            if e.ids[: len(nl)] != nl:
                raise ValueError(
                    "newline-conditioned encoding did not start with the newline tokens"
                )
            out.append(e.ids[len(nl) :])
        return out

    def word_marker_prefix(self):
        if not hasattr(self, "_word_marker"):
            import json

            pre = json.loads(self.tokenizer.to_str()).get("pre_tokenizer") or {}
            parts = pre.get("pretokenizers", [pre]) if pre.get("type") == "Sequence" else [pre]
            self._word_marker = any(
                (
                    x.get("type") == "Metaspace" and x.get("prepend_scheme", "always") != "never"
                    for x in parts
                )
            )
        return self._word_marker

    def render(self, q, permutation):
        if sorted(permutation) != sorted(self.original[q]):
            raise ValueError("A layout must keep the evidence of every request")
        return (
            list(self.system)
            + [t for m in permutation for t in self.blocks[m]]
            + list(self.requests[q].query_tokens)
        )

    def validate(self, permutations):
        prompts = [self.render(q, p) for q, p in enumerate(permutations)]
        texts = [
            self.header + "".join((self.texts[m] for m in p)) + self.suffixes[q]
            for q, p in enumerate(permutations)
        ]
        encode_batch = getattr(self.tokenizer, "encode_batch_fast", self.tokenizer.encode_batch)
        complete = encode_batch(texts, add_special_tokens=False)
        for q, (expected, actual) in enumerate(zip(prompts, complete)):
            if expected != actual.ids:
                raise ValueError(
                    f"Request {self.rows[q]['id']}: tokenization crosses a block boundary"
                )
        return prompts
