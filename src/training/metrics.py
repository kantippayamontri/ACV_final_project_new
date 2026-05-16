"""Training metrics: BLEU and ROUGE-L."""
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer


def compute_metrics(predictions: list[str], references: list[list[str]]):
    if not predictions:
        raise ValueError("Cannot compute metrics on empty predictions")

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    bleu = corpus_bleu(predictions, references).score

    rouge_l_sum = 0.0
    for pred, ref_list in zip(predictions, references):
        scores = scorer.score(pred, ref_list[0])
        rouge_l_sum += scores["rougeL"].fmeasure
    rouge_l = (rouge_l_sum / len(predictions)) * 100.0

    return {"BLEU": bleu, "ROUGE-L": rouge_l}
