import sys
import codecs
import os
from datasets import load_dataset, concatenate_datasets
import numpy as np
import torch

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForSeq2SeqLM,
    set_seed, 
    T5Tokenizer
)

# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

# Set seed for reproducibility
set_seed(42)

# Load and preprocess data
ds = load_dataset("TUKE-KEMT/hate_speech_slovak")
label_0 = ds['train'].filter(lambda example: example['label'] == 0)
label_1 = ds['train'].filter(lambda example: example['label'] == 1)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("TUKE-KEMT/slovak-t5-base", use_fast=False, legacy=False)
model = AutoModelForSequenceClassification.from_pretrained("TUKE-KEMT/slovak-t5-base", num_labels=2)

# Data augmentation (synonym replacement)
def augment_text(text):
    # Placeholder for augmentation logic
    return text

# Enhanced preprocessing
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ").strip()
    return augment_text(text)

# Tokenization function with improved preprocessing
def tokenize(batch):
    batch["text"] = [clean_text(t) for t in batch["text"]]
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# Create stratified few-shot splits
def create_stratified_split(label_0, label_1, n_samples, seed=42):
    few_shot_0 = label_0.shuffle(seed=seed).select(range(n_samples))
    few_shot_1 = label_1.shuffle(seed=seed).select(range(n_samples))
    return concatenate_datasets([few_shot_0, few_shot_1]).shuffle(seed=seed)

# Prepare datasets
def prepare_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset.rename_column("label", "labels")

train_dataset = prepare_dataset(create_stratified_split(label_0, label_1, 60))
val_dataset = prepare_dataset(create_stratified_split(label_0, label_1, 20, seed=43))
test_dataset = prepare_dataset(create_stratified_split(label_0, label_1, 80, seed=44))

# Training arguments with enhanced settings
training_args = TrainingArguments(
    output_dir="./hate_speech_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_steps=200,
    weight_decay=0.05,
    gradient_accumulation_steps=4,
    lr_scheduler_type="cosine",
)

# Custom metrics
def compute_metrics(pred):
    logits = pred.predictions[0]
    preds = logits.argmax(-1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train and evaluate
trainer.train()
trainer.evaluate(eval_dataset=val_dataset)

# Save the enhanced model
trainer.save_model("./hate_speech_model/enhanced_model")
