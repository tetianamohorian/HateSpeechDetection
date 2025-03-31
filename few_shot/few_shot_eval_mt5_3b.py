import sys
import codecs
from datasets import load_dataset, concatenate_datasets
import numpy as np
import torch

from sklearn.metrics import precision_recall_fscore_support,precision_recall_curve
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    AutoModelForSeq2SeqLM,
    set_seed, 
    T5Tokenizer
)

# Set seed for reproducibility
set_seed(42)

# Load and preprocess data
ds = load_dataset("TUKE-KEMT/hate_speech_slovak")
label_0 = ds['train'].filter(lambda example: example['label'] == 0)
label_1 = ds['train'].filter(lambda example: example['label'] == 1)

# Create stratified few-shot splits
def create_stratified_split(label_0, label_1, n_samples, seed=42):
    few_shot_0 = label_0.shuffle(seed=seed).select(range(n_samples))
    few_shot_1 = label_1.shuffle(seed=seed).select(range(n_samples))
    return concatenate_datasets([few_shot_0, few_shot_1]).shuffle(seed=seed)

# Create train/val/test splits
train_dataset = create_stratified_split(label_0, label_1, n_samples=40)
val_dataset = create_stratified_split(label_0, label_1, n_samples=10, seed=43)
test_dataset = create_stratified_split(label_0, label_1, n_samples=50, seed=44)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/mt5-3B-mmarco-en-pt", force_download=True)
model = AutoModelForSequenceClassification.from_pretrained("unicamp-dl/mt5-3B-mmarco-en-pt", num_labels=2)



# Tokenization function with padding
def tokenize(batch):
    return tokenizer(
    batch["text"],
    padding="max_length",
    truncation=True,
    max_length=256
    )


# Prepare datasets
def prepare_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(dataset[0])
    return dataset.rename_column("label", "labels")

train_dataset = prepare_dataset(train_dataset)
val_dataset = prepare_dataset(val_dataset)
test_dataset = prepare_dataset(test_dataset)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training arguments with improved settings
training_args = TrainingArguments(
    output_dir="./hate_speech_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,  # Adjust as needed
    num_train_epochs=7,  # Increased epochs for better training
    eval_strategy="epoch",  # Use "epoch" for both strategies
    save_strategy="epoch",  # Matching the evaluation strategy
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_steps=100,  # Increased warmup steps
    weight_decay=0.01,
    report_to="none",
    seed=42,
    logging_steps=10,
    gradient_accumulation_steps=2,  # For more effective training on larger datasets
    lr_scheduler_type="cosine",  # Using cosine scheduler for learning rate
    logging_dir='./logs',
)

# Custom metrics computation
def compute_metrics(pred):
    logits = pred.predictions[0]  # Ensure only the logits are used
    preds = logits.argmax(-1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='binary'
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Initialize trainer with validation data and metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Evaluate on test set
def find_optimal_threshold(trainer, dataset):
    # Get predictions
    predictions = trainer.predict(dataset)
    
    # Extract logits (the first element in the predictions tuple)
    logits = predictions.predictions  # This is likely a tuple (logits, other_info)

    # If logits is a tuple, extract only the logits
    if isinstance(logits, tuple):
        logits = logits[0]  # Extract the logits from the tuple
    
    # Check the shape of logits to debug the issue
    print(f"Logits shape: {logits.shape}")
    
    # Ensure logits has the shape (batch_size, 2) for binary classification
    if logits.shape[-1] != 2:
        logits = logits[:, :2]  # Take only the first two columns (logits for the two classes)
        print(f"Logits shape after slicing: {logits.shape}")

    # Convert logits to tensor if necessary
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)  # Convert logits to a tensor if needed
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get probabilities for the positive class (label=1)
    positive_probs = probs[:, 1].numpy()  # The probabilities for the positive class (label=1)
    
    # Get true labels from predictions
    true_labels = predictions.label_ids
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(true_labels, positive_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find the optimal threshold based on F1-score
    optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last threshold (it is always 1)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, precisions[optimal_idx], recalls[optimal_idx], f1_scores[optimal_idx]


def evaluate_with_threshold(trainer, dataset, threshold=0.5):
    predictions = trainer.predict(dataset)
    
    # Ensure that logits are properly reshaped to (batch_size, 2) before applying softmax
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]  # Extract logits if it's a tuple

    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get predicted labels based on the threshold for the positive class (label=1)
    predicted_labels = (probs[:, 1] > threshold).numpy().astype(int)
    
    true_labels = predictions.label_ids
    
    # Calculate precision-recall-fscore
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, 
        predicted_labels, 
        average='binary',
        zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Example usage:
# Find the optimal threshold using validation data
print("\nFinding optimal threshold...")
optimal_threshold, best_precision, best_recall, best_f1 = find_optimal_threshold(trainer, val_dataset)
print(f"Optimal threshold: {optimal_threshold:.4f}")

# Evaluate with optimal threshold
print("\nEvaluating with optimal threshold:")
optimized_results = evaluate_with_threshold(trainer, test_dataset, threshold=optimal_threshold)
print(f"Precision: {optimized_results['precision']:.4f}")
print(f"Recall: {optimized_results['recall']:.4f}")
print(f"F1: {optimized_results['f1']:.4f}")


# Save the model
#trainer.save_model("./hate_speech_model/best_model")