import re
import string


def normalize(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(gold, answers):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    references = {row["id"]: row for row in gold}
    exact, rouge = [], []
    for answer in answers:
        reference = references[answer["id"]]
        candidates = [reference["answer"]] + list(reference.get("aliases") or [])
        prediction = normalize(answer["text"])
        exact.append(max(int(normalize(c) == prediction) for c in candidates))
        rouge.append(scorer.score(str(reference["answer"]), answer["text"])["rougeL"].fmeasure)
    count = len(answers)
    return dict(
        requests=count, rouge_l=100 * sum(rouge) / count, exact_match=100 * sum(exact) / count
    )
