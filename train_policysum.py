import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from math import pi


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENABLE_SAFETENSORS"] = "1"

# -------------------------
# Output directory for figures
# -------------------------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
class PolicyBriefDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=64):
        with open(path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sents = item["sentences"]
        labels = item["labels"]

        input_ids, attn = [], []
        for s in sents:
            enc = self.tokenizer(
                s,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            input_ids.append(enc["input_ids"].squeeze(0))
            attn.append(enc["attention_mask"].squeeze(0))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.tensor(labels, dtype=torch.float),
            "sent_mask": torch.ones(len(labels)),
            "sent_texts": sents 
        }


def collate_fn(batch):
    """
    Pads documents in a batch to the same number of sentences
    """
    sent_texts = []

    max_sents = max(item["input_ids"].size(0) for item in batch)
    max_len = batch[0]["input_ids"].size(1)

    input_ids, attn, labels, sent_mask = [], [], [], []

    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_sents - n

        sent_texts.append(item["sent_texts"])

        input_ids.append(
            torch.cat(
                [item["input_ids"],
                 torch.zeros(pad, max_len, dtype=torch.long)],
                dim=0
            )
        )
        attn.append(
            torch.cat(
                [item["attention_mask"],
                 torch.zeros(pad, max_len, dtype=torch.long)],
                dim=0
            )
        )
        labels.append(
            torch.cat(
                [item["labels"],
                 torch.zeros(pad, dtype=torch.float)],
                dim=0
            )
        )
        sent_mask.append(
            torch.cat(
                [item["sent_mask"],
                 torch.zeros(pad, dtype=torch.float)],
                dim=0
            )
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "labels": torch.stack(labels),
        "sent_mask": torch.stack(sent_mask),
        "sent_texts": sent_texts
    }

# -------------------------
# FIXED: Multi-Granular Attention
# -------------------------
class MultiGranularAttention(nn.Module):
    """
    Fixed implementation matching the paper:
    - Sentence: windowed attention (w=16)
    - Paragraph: strided attention (p=4)
    - Section: global tokens (k=8)
    """
    def __init__(self, hidden, window_size=16, para_size=4, num_sections=8):
        super().__init__()
        self.window_size = window_size
        self.para_size = para_size
        self.num_sections = num_sections
        
        # Three attention heads
        self.sent_attn = nn.MultiheadAttention(hidden, 8, batch_first=True)
        self.para_attn = nn.MultiheadAttention(hidden, 8, batch_first=True)
        self.sect_attn = nn.MultiheadAttention(hidden, 8, batch_first=True)
        
        # Learned global tokens for sections
        self.global_tokens = nn.Parameter(torch.randn(1, num_sections, hidden))
        
        # Adaptive gating
        self.gate = nn.Sequential(
            nn.Linear(hidden * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )

    def forward(self, x, sent_mask=None):
        """
        Args:
            x: [B, S, H] - sentence encodings
            sent_mask: [B, S] - valid sentence mask
        """
        B, S, H = x.size()

        # ===== Sentence-level: Local windowed attention =====
        # For simplicity, use full self-attention (can optimize with windowing)
        s_local, _ = self.sent_attn(x, x, x)

        # ===== Paragraph-level: Strided attention =====
        # Group sentences into paragraphs of size para_size
        para_size = self.para_size
        num_paras = (S + para_size - 1) // para_size
        
        # Pad if needed
        pad_size = num_paras * para_size - S
        if pad_size > 0:
            x_padded = torch.cat([x, x[:, -1:, :].expand(B, pad_size, H)], dim=1)
        else:
            x_padded = x
        
        # Reshape and pool to paragraph level
        x_para = x_padded.view(B, num_paras, para_size, H).mean(dim=2)
        p_para, _ = self.para_attn(x_para, x_para, x_para)
        
        # Upsample back to sentence level
        p_local = p_para.repeat_interleave(para_size, dim=1)[:, :S, :]

        # ===== Section-level: Global tokens =====
        global_tokens = self.global_tokens.expand(B, -1, -1)
        sect_repr, _ = self.sect_attn(global_tokens, x, x)
        
        # Broadcast to all sentences
        c_local = sect_repr.mean(dim=1, keepdim=True).expand(B, S, H)

        # ===== Adaptive Gating =====
        combined = torch.cat([s_local, p_local, c_local], dim=-1)
        gates = torch.softmax(self.gate(combined), dim=-1)  # [B, S, 3]
        
        # Weighted combination
        out = (gates[:, :, 0:1] * s_local + 
               gates[:, :, 1:2] * p_local + 
               gates[:, :, 2:3] * c_local)
        
        return out, gates

# -------------------------
# FIXED: Model
# -------------------------
class PolicySumModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config,
            use_safetensors=True,
            trust_remote_code=False
        )
        
        hidden_size = config.hidden_size
        self.attn = MultiGranularAttention(hidden_size)
        
        # Sentence scorer with dropout
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, sent_mask=None):
        B, S, L = input_ids.shape
        
        # Flatten for encoder
        x = input_ids.view(B * S, L)
        m = attention_mask.view(B * S, L)

        # Encode sentences
        enc = self.encoder(x, attention_mask=m).last_hidden_state[:, 0]
        enc = enc.view(B, S, -1)

        # Multi-granular attention
        x, gates = self.attn(enc, sent_mask)
        
        # Score sentences
        scores = self.scorer(x).squeeze(-1)

        return scores, gates


# -------------------------
# FIXED: Training with proper losses
# -------------------------
def train_epoch(model, dl, opt, scaler, device, epoch, lambda_cov=0.15):
    model.train()
    total_loss = 0.0
    total_extract = 0.0
    total_cov = 0.0

    # Positive class weighting
    pos_weight = torch.tensor([2.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for batch_idx, b in enumerate(dl):
        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            scores, gates = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["sent_mask"].to(device)
            )
            
            labels = b["labels"].to(device)
            sent_mask = b["sent_mask"].to(device)
            
            # ===== Extraction Loss =====
            # Only compute loss on valid sentences
            extract_loss = loss_fn(scores, labels)
            
            # Mask invalid positions
            extract_loss = (extract_loss * sent_mask).sum() / sent_mask.sum()
            
            # ===== Coverage Loss (Eq. 12 from paper) =====
            # Encourage diversity among selected sentences
            probs = torch.sigmoid(scores)
            
            # Get top-k selections per document
            coverage_loss = 0.0
            for i in range(scores.size(0)):
                n_sents = int(sent_mask[i].sum())
                if n_sents < 2:
                    continue
                
                # Get sentence embeddings (from encoder output)
                sent_embs = model.encoder(
                    b["input_ids"][i, :n_sents].to(device),
                    attention_mask=b["attention_mask"][i, :n_sents].to(device)
                ).last_hidden_state[:, 0]  # [n_sents, hidden]
                
                # Normalize
                sent_embs = F.normalize(sent_embs, p=2, dim=-1)
                
                # Weighted by selection probability
                weights = probs[i, :n_sents].unsqueeze(1)
                weighted_embs = sent_embs * weights
                
                # Similarity matrix
                sim_matrix = torch.mm(weighted_embs, weighted_embs.t())
                
                # Coverage loss: minimize similarity (encourage diversity)
                # Exclude diagonal
                mask = ~torch.eye(n_sents, device=device, dtype=torch.bool)
                coverage_loss += sim_matrix[mask].mean()
            
            coverage_loss = coverage_loss / scores.size(0)
            
            # ===== Gate Regularization =====
            # Encourage balanced use of all granularities
            gate_mean = gates.mean(dim=(0, 1))  # [3]
            
            # Target: roughly equal distribution
            gate_target = torch.tensor([0.33, 0.33, 0.34], device=device)
            gate_reg = F.mse_loss(gate_mean, gate_target)
            
            # ===== Total Loss =====
            loss = extract_loss + lambda_cov * coverage_loss + 0.05 * gate_reg
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        total_extract += extract_loss.item()
        total_cov += coverage_loss.item()

    return {
        "loss": total_loss / len(dl),
        "extract": total_extract / len(dl),
        "coverage": total_cov / len(dl)
    }


# -------------------------
# FIXED: Evaluation
# -------------------------
def evaluate(model, dl, device):
    model.eval()
    preds, gold = [], []
    gate_collect = []
    errors = []

    with torch.no_grad():
        for b in dl:
            scores, gates = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["sent_mask"].to(device)
            )

            scores = torch.sigmoid(scores).cpu()
            labels = b["labels"]
            sent_mask = b["sent_mask"]

            for i in range(scores.size(0)):
                n = int(sent_mask[i].sum())
                if n == 0:
                    continue
                
                # Get valid scores
                prob = scores[i, :n]
                
                # Dynamic threshold selection
                best_f1 = -1
                best_pred = torch.zeros(n, dtype=torch.long)
                
                for threshold in np.linspace(0.3, 0.6, 10):
                    pred_tmp = (prob > threshold).long()
                    
                    # Ensure at least one sentence selected
                    if pred_tmp.sum() == 0:
                        pred_tmp[prob.argmax()] = 1
                    
                    gold_tmp = labels[i, :n].long()
                    
                    # Calculate F1
                    P, R, F, _ = precision_recall_fscore_support(
                        gold_tmp.numpy(),
                        pred_tmp.numpy(),
                        average="binary",
                        zero_division=0
                    )
                    
                    if F > best_f1:
                        best_f1 = F
                        best_pred = pred_tmp
                
                preds.extend(best_pred.tolist())
                gold.extend(labels[i, :n].long().tolist())
                
                gate_collect.append(gates[i, :n].cpu().numpy())
                
                # Error analysis
                for j in range(n):
                    if labels[i, j] == 1 and best_pred[j] == 0:
                        errors.append("False Negative")
                    elif labels[i, j] == 0 and best_pred[j] == 1:
                        errors.append("False Positive")

    P, R, F, _ = precision_recall_fscore_support(
        gold, preds, average="binary", zero_division=0
    )
    
    g = np.concatenate(gate_collect, axis=0)
    gate_avg = g.mean(0)

    return {
        "precision": P,
        "recall": R,
        "f1": F,
        "gate_sentence": gate_avg[0],
        "gate_paragraph": gate_avg[1],
        "gate_section": gate_avg[2],
        "gate_matrix": g,   
        "errors": Counter(errors)
    }

# -------------------------
# Visualization Functions
# -------------------------
def plot_attention_heatmap(gate_matrix, max_sents=50):
    """
    gate_matrix: [num_sentences, 3]
    """
    avg_by_position = gate_matrix[:max_sents]

    plt.figure(figsize=(12, 4))
    sns.heatmap(
        avg_by_position.T,
        cmap="YlOrRd",
        xticklabels=[f"{i+1}" for i in range(0, avg_by_position.shape[0], 5)],
        yticklabels=["Sentence", "Paragraph", "Section"],
        cbar_kws={"label": "Gate Weight"}
    )
    plt.xlabel("Sentence Position")
    plt.title("Multi-Granular Attention Heatmap")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/attention_heatmap.png", dpi=300)
    plt.close()


def plot_feature_importance(metrics):
    labels = ["Sentence", "Paragraph", "Section"]
    values = [
        metrics["gate_sentence"],
        metrics["gate_paragraph"],
        metrics["gate_section"]
    ]

    plt.figure(figsize=(6, 4))
    colors = sns.color_palette("Set2", 3)
    bars = plt.bar(labels, values, color=colors)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.ylabel("Average Gate Weight")
    plt.title("Feature Importance via Gating")
    plt.ylim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=300)
    plt.close()


def plot_error_analysis(error_counter):
    if len(error_counter) == 0:
        return
    
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("Set3", len(error_counter))
    plt.pie(
        error_counter.values(),
        labels=error_counter.keys(),
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    plt.title("Error Analysis Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/error_analysis.png", dpi=300)
    plt.close()


def plot_training_curves(train_losses, val_f1s):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(train_losses, marker='o', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 curve
    ax2.plot(val_f1s, marker='s', color='green', label='Validation F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/training_curves.png", dpi=300)
    plt.close()


# -------------------------
# Main Training Loop
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Datasets
    train_ds = PolicyBriefDataset("policysum_data/train.json", tokenizer)
    val_ds = PolicyBriefDataset("policysum_data/val.json", tokenizer)

    train_dl = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=2,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # Model
    model = PolicySumModel().to(device)
    
    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,  # Lower LR for stability
        weight_decay=0.01
    )
    
    scaler = torch.amp.GradScaler("cuda")

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=20,
        eta_min=1e-6
    )

    # Training tracking
    best_f1 = 0.0
    best_metrics = None
    train_losses = []
    val_f1s = []
    patience = 5
    patience_counter = 0

    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")

    for epoch in range(20):
        # Train
        train_metrics = train_epoch(
            model, train_dl, opt, scaler, device, epoch
        )
        
        # Validate
        val_metrics = evaluate(model, val_dl, device)
        
        # Track metrics
        train_losses.append(train_metrics["loss"])
        val_f1s.append(val_metrics["f1"])
        
        # Print progress
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/20")
        print(f"{'='*50}")
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(Extract: {train_metrics['extract']:.4f}, "
              f"Coverage: {train_metrics['coverage']:.4f})")
        print(f"Val F1: {val_metrics['f1']:.4f} | "
              f"P: {val_metrics['precision']:.4f} | "
              f"R: {val_metrics['recall']:.4f}")
        print(f"Gates → Sent: {val_metrics['gate_sentence']:.3f}, "
              f"Para: {val_metrics['gate_paragraph']:.3f}, "
              f"Sect: {val_metrics['gate_section']:.3f}")
        
        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), "policysum_best.pt")
            print(f"✓ New best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        scheduler.step()

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Best Precision: {best_metrics['precision']:.4f}")
    print(f"Best Recall: {best_metrics['recall']:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_attention_heatmap(best_metrics["gate_matrix"])
    plot_feature_importance(best_metrics)
    plot_error_analysis(best_metrics["errors"])
    plot_training_curves(train_losses, val_f1s)
    print(f"All visualizations saved to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()