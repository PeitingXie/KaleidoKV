import argparse
import collections
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

RAGEVAL_COMMIT = "c44aa98aea18362eb5e3751f11e13dbc8f6cab83"
RAW_URL = f"https://raw.githubusercontent.com/OpenBMB/RAGEval/{RAGEVAL_COMMIT}/dragonball_dataset/"
FILES = {
    "dragonball_docs.jsonl": "7d99b3eeb4d93e87094b9351fe360199c3a787ea10b84ccd80844529a5e9933a",
    "dragonball_queries.jsonl": "183f028236203834be023d509cb7d7a84761f1a2259b7c956cb676f194ffc0c1",
}
HEADER = (
    "<|im_start|>system\nAnswer the question using the supplied document excerpts. Some excerpts may be "
    "unrelated to the question, and their display order carries no meaning. If the excerpts do not contain "
    "the answer, say that it cannot be answered. Answer concisely.<|im_end|>\n<|im_start|>user\n"
)
WORD = re.compile(r"\w+")
K1, B = 1.5, 0.75


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = folder / name
        if not path.exists() or sha256(path) != expected:
            print(f"downloading {name}")
            urllib.request.urlretrieve(RAW_URL + name, path)
        if sha256(path) != expected:
            raise RuntimeError(f"{name} does not match the pinned RAGEval release")
    return folder / "dragonball_docs.jsonl", folder / "dragonball_queries.jsonl"


def words(text):
    return WORD.findall(text.casefold())


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def chunk_documents(docs, chunk_words):
    chunks = collections.defaultdict(list)
    doc_chunks = collections.defaultdict(list)
    for doc in sorted(docs, key=lambda d: (d["domain"], d["doc_id"])):
        domain = doc["domain"]
        title = doc.get("company_name") or ""
        paragraphs = [p.strip() for p in doc["content"].split("\n") if p.strip()]
        current = []
        for paragraph in paragraphs + [None]:
            if paragraph is not None:
                current.append(paragraph)
                if sum(len(p.split()) for p in current) < chunk_words:
                    continue
            if current:
                chunks[domain].append((doc["doc_id"], title, "\n".join(current)))
                doc_chunks[(domain, doc["doc_id"])].append(len(chunks[domain]) - 1)
                current = []
    return chunks, doc_chunks


class BM25:
    def __init__(self, texts):
        tokens = [words(text) for text in texts]
        average = sum(map(len, tokens)) / len(tokens)
        self.postings = collections.defaultdict(list)
        for i, terms in enumerate(tokens):
            for term, count in collections.Counter(terms).items():
                self.postings[term].append((i, count))
        n = len(tokens)
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }
        self.norm = [K1 * (1 - B + B * len(terms) / average) for terms in tokens]

    def top(self, query, k):
        scores = collections.defaultdict(float)
        for term in set(words(query)):
            for i, count in self.postings.get(term, ()):
                scores[i] += self.idf[term] * count * (K1 + 1) / (count + self.norm[i])
        return sorted(scores, key=lambda i: (-scores[i], i))[:k]


def build(docs_path, queries_path, top_k=4, chunk_words=380):
    docs = [d for d in read_jsonl(docs_path) if d["language"] == "en"]
    queries = [q for q in read_jsonl(queries_path) if q["language"] == "en"]
    chunks, doc_chunks = chunk_documents(docs, chunk_words)
    search = {
        domain: BM25([f"{title}\n{text}" for _, title, text in rows])
        for domain, rows in chunks.items()
    }
    material_of, materials, requests, gold = {}, [], [], []
    for query in queries:
        domain = query["domain"]
        text = query["query"]["content"]
        top = search[domain].top(text, top_k)
        if len(top) < top_k:
            continue
        evidence = []
        for i in top:
            if (domain, i) not in material_of:
                doc_id, title, chunk = chunks[domain][i]
                key = hashlib.sha1(f"{domain}\x00{doc_id}\x00{chunk}".encode()).hexdigest()[:16]
                material_id = "dragonball:" + key
                material_of[(domain, i)] = len(materials)
                materials.append(
                    dict(
                        id=material_id,
                        text=f'<excerpt id="{material_id}">\nSource: {title or domain}\n{chunk}\n</excerpt>\n',
                    )
                )
            evidence.append(material_of[(domain, i)])
        request_id = f'{domain}_{query["query"]["query_id"]}'
        requests.append(
            dict(
                id=request_id,
                question="Question: " + text.strip(),
                materials=evidence,
                domain=domain,
                query_type=query["query"]["query_type"],
            )
        )
        truth = query["ground_truth"]
        gold.append(
            dict(
                id=request_id,
                answer=truth["content"],
                aliases=[],
                keypoints=truth.get("keypoints", []),
            )
        )
    data = dict(
        header=HEADER,
        materials=materials,
        requests=requests,
        contract=dict(context_tokens=8192, max_output_tokens=128, display_order_mutable=True),
        workload_provenance=dict(
            recipe=f"dragonball_en_chunk{chunk_words}w_bm25_top{top_k}",
            dataset="DragonBall (RAGEval, Zhu et al., 2024), English queries and documents",
        ),
    )
    return dict(input=data, gold=gold, metric="qa")


def main():
    parser = argparse.ArgumentParser(description="Build the DragonBall workload used in the paper.")
    parser.add_argument("--raw", default="data/raw", help="folder for the RAGEval files")
    parser.add_argument("--output", default="data/dragonball.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-words", type=int, default=380)
    args = parser.parse_args()
    docs, queries = download(args.raw)
    pack = build(docs, queries, args.top_k, args.chunk_words)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pack, f)
    data = pack["input"]
    print(
        f'{args.output}: {len(data["requests"])} requests, {len(data["materials"])} evidence chunks'
    )
    print(f"sha256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
