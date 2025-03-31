import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import precision_recall_fscore_support
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
label_0 = ds['train'].filter(lambda example: example['label'] == 0)
label_1 = ds['train'].filter(lambda example: example['label'] == 1)

def create_stratified_split(label_0, label_1, n_samples, seed=42):
    few_shot_0 = label_0.shuffle(seed=seed).select(range(n_samples))
    few_shot_1 = label_1.shuffle(seed=seed).select(range(n_samples))
    return concatenate_datasets([few_shot_0, few_shot_1]).shuffle(seed=seed)

train_dataset = create_stratified_split(label_0, label_1, n_samples=40)
val_dataset = create_stratified_split(label_0, label_1, n_samples=10, seed=43)
test_dataset = create_stratified_split(label_0, label_1, n_samples=50, seed=44)

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

# Evaluate model
results = trainer.evaluate(test_dataset)
print("\n📊 Evaluation results on test set:")
print(f"🎯 Precision: {results['eval_precision']:.4f}")
print(f"🎯 Recall: {results['eval_recall']:.4f}")
print(f"🎯 F1-score: {results['eval_f1']:.4f}")

# Function for text classification
def classify_text(text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()
    return "🛑 Hate Speech" if pred == 1 else "✅ Not Hate Speech"

# Testing examples
test_texts = [
    "Toto je úplne normálny text bez nenávisti.",
    "Nenávidím ťa a všetkých ako ty!",
    "Každý má právo na svoj názor, ale musíme byť rešpektujúci.",
    "Všetci ľudia tejto skupiny sú strašní a mali by byť vyhodení!"
]

print("\n🔍 Testing custom inputs:")
for text in test_texts:
    result = classify_text(text, model, tokenizer, device)
    print(f"📝 Text: {text}\n➡️ Prediction: {result}\n")
