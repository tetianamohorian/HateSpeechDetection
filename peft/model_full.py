import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from peft import get_peft_model, LoraConfig, TaskType

# Set seed for reproducibility
set_seed(42)

# Load dataset
ds = load_dataset("TUKE-KEMT/hate_speech_slovak")
train_dataset = ds['train']
train_test_split = ds['train'].train_test_split(test_size=0.2, seed=42)
val_dataset = train_test_split['test']
train_dataset = train_test_split['train']
test_dataset = ds['test']

# Load tokenizer and base model
model_name = "ApoTro/slovak-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Apply LoRA tuning
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

def tokenize(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

def prepare_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset.rename_column("label", "labels")

train_dataset = prepare_dataset(train_dataset)
val_dataset = prepare_dataset(val_dataset)
test_dataset = prepare_dataset(test_dataset)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./hate_speech_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    num_train_epochs=7,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_steps=100,
    weight_decay=0.01,
    report_to="none",
    seed=42,
    logging_steps=10,
    gradient_accumulation_steps=2,
    lr_scheduler_type="cosine",
    logging_dir='./logs',
)

def compute_metrics(pred):
    logits = pred.predictions[0]
    preds = logits.argmax(-1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./hate_speech_model/best_model")

# Reload fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./hate_speech_model/best_model")
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

def find_optimal_threshold(trainer, dataset):
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = torch.tensor(logits)
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    positive_probs = probs[:, 1].numpy()
    true_labels = predictions.label_ids
    
    precisions, recalls, thresholds = precision_recall_curve(true_labels, positive_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, precisions[optimal_idx], recalls[optimal_idx], f1_scores[optimal_idx]

def evaluate_with_threshold(trainer, dataset, threshold=0.5):
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = torch.tensor(logits)
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_labels = (probs[:, 1] > threshold).numpy().astype(int)
    true_labels = predictions.label_ids
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary', zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

print("\n🔍 Finding optimal threshold...")
optimal_threshold, best_precision, best_recall, best_f1 = find_optimal_threshold(trainer, val_dataset)
print(f"✅ Optimal threshold: {optimal_threshold:.4f}")

print("\n📊 Evaluating on the test set with the optimal threshold:")
optimized_results = evaluate_with_threshold(trainer, test_dataset, threshold=optimal_threshold)
print(f"🎯 Precision: {optimized_results['precision']:.4f}")
print(f"🎯 Recall: {optimized_results['recall']:.4f}")
print(f"🎯 F1-score: {optimized_results['f1']:.4f}")
