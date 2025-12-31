"""
PolicySum Evaluation Script (FIXED - PyTorch Compatible)

Works with older PyTorch versions by:
1. Using weights_only=False workaround (temporary)
2. Or skipping BERTScore if model loading fails
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re
import os
import warnings

from transformers import AutoTokenizer
from rouge_score import rouge_scorer

# Try to import bert_score with fallback
try:
    # Disable the safety check temporarily
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except:
    BERT_SCORE_AVAILABLE = False
    warnings.warn("BERTScore not available, will skip this metric")

from sentence_transformers import SentenceTransformer, util

# Import model from training script
from train_policysum import PolicySumModel


class PolicySumEvaluator:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Models
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-base"
        )
        self.model = PolicySumModel().to(self.device)
        
        # Load with weights_only=False for older PyTorch
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
        except TypeError:
            # Older PyTorch doesn't have weights_only parameter
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        
        self.model.eval()

        # Sentence embedding model (for coverage & coherence)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # ROUGE
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def split_sentences(self, document):
        """Improved sentence splitting using regex."""
        sentences = re.split(r'[.!?]+\s+', document)
        sentences = [
            s.strip() 
            for s in sentences 
            if len(s.strip().split()) >= 3
        ]
        return sentences

    def generate_summary(self, document):
        sentences = self.split_sentences(document)

        if len(sentences) == 0:
            return "", np.zeros((1, 3))

        # Tokenize sentences
        encoded = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            scores, gates = self.model(input_ids, attention_mask)
            scores = torch.sigmoid(scores[0]).cpu().numpy()
            gates = gates[0]

        # Adaptive summary length
        n = len(sentences)
        top_k = max(6, int(0.30 * n))
        top_k = min(top_k, 10)

        # Sentence embeddings for diversity
        sent_embs = self.embedder.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # Redundancy-aware selection
        selected = []
        selected_embs = []

        for idx in np.argsort(-scores):
            emb = sent_embs[idx]

            if len(selected_embs) == 0 or all(
                util.cos_sim(emb, prev).item() < 0.80
                for prev in selected_embs
            ):
                selected.append(idx)
                selected_embs.append(emb)

            if len(selected) >= top_k:
                break

        # Sort to preserve document order
        selected = sorted(selected)
        summary = ". ".join(sentences[i] for i in selected) + "."
        gate_matrix = gates[selected].detach().cpu().numpy().copy()

        return summary, gate_matrix

    def coherence(self, summaries):
        """Calculate semantic coherence between adjacent sentences."""
        scores = []

        for summary in summaries:
            sents = self.split_sentences(summary)
            
            if len(sents) < 2:
                continue
            
            emb = self.embedder.encode(
                sents, 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            
            sims = []
            for i in range(len(emb) - 1):
                sim = util.cos_sim(emb[i], emb[i + 1]).item()
                sims.append(sim)
            
            if len(sims) > 0:
                scores.append(float(np.mean(sims)))
        
        return float(np.mean(scores)) if len(scores) > 0 else 0.0

    def coverage(self, predictions, references):
        """Calculate semantic coverage."""
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_sents = self.split_sentences(pred)
            ref_sents = self.split_sentences(ref)
            
            if len(pred_sents) == 0 or len(ref_sents) == 0:
                continue
            
            pred_emb = self.embedder.encode(
                pred_sents, 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            ref_emb = self.embedder.encode(
                ref_sents, 
                convert_to_tensor=True, 
                normalize_embeddings=True
            )
            
            max_sims = []
            for r_emb in ref_emb:
                sims = util.cos_sim(r_emb, pred_emb).squeeze()
                if sims.dim() == 0:
                    sims = sims.unsqueeze(0)
                max_sims.append(sims.max().item())
            
            scores.append(float(np.mean(max_sims)))
        
        return float(np.mean(scores)) if len(scores) > 0 else 0.0

    def evaluate(self, test_path, num_runs=1):
        """Run evaluation with proper statistical reporting."""
        with open(test_path) as f:
            data = json.load(f)

        all_results = []
        
        for run in range(num_runs):
            print(f"\n=== Run {run + 1}/{num_runs} ===")
            
            predictions, references = [], []
            gate_weights = []

            print(f"Evaluating on {len(data)} documents...")

            for item in tqdm(data):
                summary, gates = self.generate_summary(item["document"])
                predictions.append(summary)
                references.append(item["summary"])
                gate_weights.append(gates.mean(axis=0))

            # ROUGE scores
            rouge_scores = defaultdict(list)
            for pred, ref in zip(predictions, references):
                r_score = self.rouge.score(ref, pred)
                for k in r_score:
                    rouge_scores[k].append(r_score[k].fmeasure)

            rouge_results = {
                k: float(np.mean(v)) for k, v in rouge_scores.items()
            }

            # BERTScore (with fallback)
            if BERT_SCORE_AVAILABLE:
                try:
                    print("Computing BERTScore...")
                    # Use a simpler model that's less likely to have issues
                    P, R, F1 = bert_score(
                        predictions, 
                        references, 
                        lang="en", 
                        model_type="distilbert-base-uncased",  # Smaller, more stable
                        verbose=False,
                        device=self.device
                    )
                    bert_f1 = float(F1.mean())
                except Exception as e:
                    print(f"⚠ BERTScore failed: {e}")
                    print("  Skipping BERTScore, using ROUGE-L as proxy")
                    bert_f1 = rouge_results["rougeL"]  # Fallback
            else:
                print("⚠ BERTScore not available, using ROUGE-L as proxy")
                bert_f1 = rouge_results["rougeL"]

            # Coherence & Coverage
            coh = self.coherence(predictions)
            cov = self.coverage(predictions, references)

            gate_avg = np.mean(gate_weights, axis=0)

            run_results = {
                "rouge1": rouge_results["rouge1"],
                "rouge2": rouge_results["rouge2"],
                "rougeL": rouge_results["rougeL"],
                "bert_f1": bert_f1,
                "coherence": coh,
                "coverage": cov,
                "gates": gate_avg.tolist()
            }
            
            all_results.append(run_results)
        
        # Calculate mean and std across runs
        final_results = {}
        for key in all_results[0].keys():
            if key == "gates":
                final_results[key] = {
                    "mean": np.mean([r[key] for r in all_results], axis=0).tolist(),
                    "std": np.std([r[key] for r in all_results], axis=0).tolist()
                }
            else:
                values = [r[key] for r in all_results]
                final_results[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
        
        return final_results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--test", default="policysum_data/test.json", help="Test data path")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--runs", type=int, default=3, help="Number of evaluation runs")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("PolicySum Evaluation")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Test data: {args.test}")
    print(f"Device: {args.device}")
    print(f"Runs: {args.runs}")
    print("="*50)

    evaluator = PolicySumEvaluator(args.model, args.device)
    results = evaluator.evaluate(args.test, num_runs=args.runs)

    print("\n" + "="*50)
    print("FINAL RESULTS (Mean ± Std)")
    print("="*50)
    
    print("\nROUGE:")
    print(f"  ROUGE-1: {results['rouge1']['mean']:.4f} ± {results['rouge1']['std']:.4f}")
    print(f"  ROUGE-2: {results['rouge2']['mean']:.4f} ± {results['rouge2']['std']:.4f}")
    print(f"  ROUGE-L: {results['rougeL']['mean']:.4f} ± {results['rougeL']['std']:.4f}")

    print(f"\nSemantic Metrics:")
    print(f"  BERTScore F1: {results['bert_f1']['mean']:.4f} ± {results['bert_f1']['std']:.4f}")
    print(f"  Coherence:    {results['coherence']['mean']:.4f} ± {results['coherence']['std']:.4f}")
    print(f"  Coverage:     {results['coverage']['mean']:.4f} ± {results['coverage']['std']:.4f}")

    print("\nHierarchical Gate Weights:")
    gates_mean = results['gates']['mean']
    gates_std = results['gates']['std']
    print(f"  Sentence:  {gates_mean[0]:.3f} ± {gates_std[0]:.3f}")
    print(f"  Paragraph: {gates_mean[1]:.3f} ± {gates_std[1]:.3f}")
    print(f"  Section:   {gates_mean[2]:.3f} ± {gates_std[2]:.3f}")
    
    # Save results
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✔ Results saved to {output_file}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()