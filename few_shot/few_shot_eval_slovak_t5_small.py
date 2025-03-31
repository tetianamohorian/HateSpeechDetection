import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

# 🔹 1. Set seed for reproducibility
set_seed(42)

# 🔹 2. Load dataset
ds = load_dataset("TUKE-KEMT/hate_speech_slovak")
label_0 = ds['train'].filter(lambda example: example['label'] == 0)
label_1 = ds['train'].filter(lambda example: example['label'] == 1)

# Function to create stratified splits
def create_stratified_split(label_0, label_1, n_samples, seed=42):
    few_shot_0 = label_0.shuffle(seed=seed).select(range(n_samples))
    few_shot_1 = label_1.shuffle(seed=seed).select(range(n_samples))
    return concatenate_datasets([few_shot_0, few_shot_1]).shuffle(seed=seed)

# Create train, validation, and test splits
train_dataset = create_stratified_split(label_0, label_1, n_samples=40)
val_dataset = create_stratified_split(label_0, label_1, n_samples=10, seed=43)
test_dataset = create_stratified_split(label_0, label_1, n_samples=50, seed=44)

# 🔹 3. Load tokenizer and model
model_name = "ApoTro/slovak-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

# Function to prepare datasets
def prepare_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset.rename_column("label", "labels")

# Apply preparation to datasets
train_dataset = prepare_dataset(train_dataset)
val_dataset = prepare_dataset(val_dataset)
test_dataset = prepare_dataset(test_dataset)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔹 4. Define training arguments
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

# 🔹 5. Define evaluation metrics
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

# 🔹 6. Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 🔹 7. Train the model
trainer.train()

# 🔹 8. Save the trained model
trainer.save_model("./hate_speech_model/best_model")

# 🔹 9. Load the trained model before testing
model = AutoModelForSequenceClassification.from_pretrained("./hate_speech_model/best_model")
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 🔹 10. Function to find the optimal threshold
def find_optimal_threshold(trainer, dataset):
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    
    # Ensure logits are properly shaped
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = torch.tensor(logits)
    
    # Apply softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)
    positive_probs = probs[:, 1].numpy()
    true_labels = predictions.label_ids
    
    # Compute Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(true_labels, positive_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find the best threshold based on F1-score
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, precisions[optimal_idx], recalls[optimal_idx], f1_scores[optimal_idx]

# 🔹 11. Function to evaluate the model using a custom threshold
def evaluate_with_threshold(trainer, dataset, threshold=0.5):
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = torch.tensor(logits)
    
    # Apply softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_labels = (probs[:, 1] > threshold).numpy().astype(int)
    true_labels = predictions.label_ids
    
    # Compute Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='binary', zero_division=0
    )
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 🔹 12. Find the optimal threshold using validation set
print("\n🔍 Finding optimal threshold...")
optimal_threshold, best_precision, best_recall, best_f1 = find_optimal_threshold(trainer, val_dataset)
print(f"✅ Optimal threshold: {optimal_threshold:.4f}")

# 🔹 13. Final evaluation on the test set using the best threshold
print("\n📊 Evaluating on the test set with the optimal threshold:")
optimized_results = evaluate_with_threshold(trainer, test_dataset, threshold=optimal_threshold)
print(f"🎯 Precision: {optimized_results['precision']:.4f}")
print(f"🎯 Recall: {optimized_results['recall']:.4f}")
print(f"🎯 F1-score: {optimized_results['f1']:.4f}")


def classify_text(text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()
    return "🛑 Hate Speech" if pred == 0 else "✅ Not Hate Speech"

# Testing examples
test_texts = [
    "Toto je uplne normalny text bez nenávisti.",
    "Nenavidim ťa a všetkých ako ty!",
    "Zamalicko",
    "Všetci ľudia tejto skupiny sú strašní a mali by byť vyhodení!"
]

print("\n🔍 Testing custom inputs:")
for text in test_texts:
    result = classify_text(text, model, tokenizer, device)
    print(f"📝 Text: {text}\n➡️ Prediction: {result}\n")

