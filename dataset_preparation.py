"""
Real Dataset Preparation for PolicySum
Downloads and processes GovReport dataset for policy brief summarization

Available Datasets:
1. GovReport: Government reports with summaries (19K documents)
2. Sci2Pol-Corpus: Science-to-policy briefs (639 pairs)
3. Multi-LexSum: Legal case summaries (9,280 cases)

We'll use GovReport as it's the most suitable for policy briefs
"""

import json
import os
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import spacy
import numpy as np


# Download spacy model if not present
# python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")

def download_govreport():
    """Download GovReport dataset from HuggingFace"""
    print("Downloading GovReport dataset...")
    dataset = load_dataset("ccdv/govreport-summarization")
    return dataset

def prepare_extractive_labels(document, summary):
    doc_sents = [sent.text.strip() for sent in nlp(document).sents]
    summary_lower = summary.lower()
    summary_words = set(summary_lower.split())

    overlaps = []

    # 🔹 Step 1: compute overlap scores
    for sent in doc_sents:
        sent_words = set(sent.lower().split())
        overlap = len(sent_words & summary_words) / max(len(sent_words), 1)
        overlaps.append(overlap)

    # 🔹 Step 2: threshold-based labels (LOWERED threshold)
    labels = [1 if o > 0.5 else 0 for o in overlaps]

    # 🔹 Step 3: force top-2 oracle sentences (VERY IMPORTANT)
    if len(overlaps) >= 2:
        top2 = np.argsort(overlaps)[-2:]
        for idx in top2:
            labels[idx] = 1

    return doc_sents, labels


def process_govreport_to_policysum_format(dataset_split, max_samples=1000):
    """
    Convert GovReport to PolicySum format with extractive labels
    """
    processed_data = []
    
    print(f"Processing {max_samples} samples from {dataset_split} split...")
    
    for idx, item in enumerate(tqdm(dataset_split)):
        if idx >= max_samples:
            break
            
        document = item['report']
        summary = item['summary']
        
        # Skip very long documents (memory constraints)
        if len(document.split()) > 8000:
            continue
        
        # Extract sentences and create labels
        sentences, labels = prepare_extractive_labels(document, summary)
        
        # Only keep first 30 sentences for training efficiency
        sentences = sentences[:30]
        labels = labels[:30]
        
        # Skip if no positive labels
        if sum(labels) == 0:
            continue
        
        processed_data.append({
            'id': f'govreport_{idx}',
            'document': document,
            'summary': summary,
            'sentences': sentences,
            'labels': labels,
            'num_sentences': len(sentences),
            'num_positive': sum(labels)
        })
    
    return processed_data

def create_policysum_dataset():
    """Main function to create PolicySum dataset from GovReport"""
    
    # Download dataset
    dataset = download_govreport()
    
    # Process splits
    print("\nProcessing training split...")
    train_data = process_govreport_to_policysum_format(
        dataset['train'], 
        max_samples=800
    )
    
    print("\nProcessing validation split...")
    val_data = process_govreport_to_policysum_format(
        dataset['validation'], 
        max_samples=100
    )
    
    print("\nProcessing test split...")
    test_data = process_govreport_to_policysum_format(
        dataset['test'], 
        max_samples=100
    )
    
    # Save to JSON
    os.makedirs('policysum_data', exist_ok=True)
    
    with open('policysum_data/train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('policysum_data/val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open('policysum_data/test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Creation Complete!")
    print("="*60)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\nAverage sentences per document: {sum(d['num_sentences'] for d in train_data)/len(train_data):.1f}")
    print(f"Average positive labels per document: {sum(d['num_positive'] for d in train_data)/len(train_data):.1f}")
    print(f"\nData saved to: policysum_data/")
    
    # Save dataset statistics
    stats = {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'avg_sentences': sum(d['num_sentences'] for d in train_data)/len(train_data),
        'avg_positive_labels': sum(d['num_positive'] for d in train_data)/len(train_data),
        'source': 'GovReport (ccdv/govreport-summarization)'
    }
    
    with open('policysum_data/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    return train_data, val_data, test_data

def create_simplified_training_format():
    """
    Create simplified format compatible with the training code
    """
    print("\nCreating simplified training format...")
    
    # Load processed data
    with open('policysum_data/train.json', 'r') as f:
        data = json.load(f)
    
    simplified_data = []
    for item in data:
        # Join sentences back for simpler format
        doc_text = ' '.join(item['sentences'][:20])  # Limit to 20 sentences
        
        simplified_data.append({
            'document': doc_text,
            'summary': item['summary']
        })
    
    # Save simplified format
    with open('policysum_data/train_simple.json', 'w') as f:
        json.dump(simplified_data, f, indent=2)
    
    print(f"Simplified training data saved: policysum_data/train_simple.json")

if __name__ == '__main__':
    print("="*60)
    print("PolicySum Dataset Preparation")
    print("="*60)
    print("\nThis script will:")
    print("1. Download GovReport dataset (19K government reports)")
    print("2. Process into extractive summarization format")
    print("3. Create train/val/test splits")
    print("4. Save in PolicySum-compatible format")
    print("\nEstimated time: 10-15 minutes")
    print("="*60)
    
    input("\nPress Enter to begin...")
    
    # Create dataset
    train_data, val_data, test_data = create_policysum_dataset()
    
    # Create simplified format for training
    create_simplified_training_format()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Use 'policysum_data/train_simple.json' for training")
    print("2. Modify the training script to use this data path")
    print("3. Run training: python train_policysum.py")
    print("\nDataset Details:")
    print("- Source: GovReport (Government Research Reports)")
    print("- Format: Extractive labels for sentence selection")
    print("- Ready for PolicySum model training")
    print("="*60)