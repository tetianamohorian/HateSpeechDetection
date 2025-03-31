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

# Устанавливаем seed для воспроизводимости
set_seed(42)

# Загружаем датасет
ds = load_dataset("TUKE-KEMT/hate_speech_slovak")
label_0 = ds['train'].filter(lambda example: example['label'] == 0)
label_1 = ds['train'].filter(lambda example: example['label'] == 1)

# Функция для стратифицированного разбиения
def create_stratified_split(label_0, label_1, n_samples, seed=42):
    few_shot_0 = label_0.shuffle(seed=seed).select(range(n_samples))
    few_shot_1 = label_1.shuffle(seed=seed).select(range(n_samples))
    return concatenate_datasets([few_shot_0, few_shot_1]).shuffle(seed=seed)

# Создаём обучающий и валидационный наборы
train_dataset = create_stratified_split(label_0, label_1, n_samples=40)
val_dataset = create_stratified_split(label_0, label_1, n_samples=10, seed=43)

# Загружаем токенизатор и модель
model_name = "ApoTro/slovak-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Функция токенизации
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# Подготовка датасетов
def prepare_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset.rename_column("label", "labels")

train_dataset = prepare_dataset(train_dataset)
val_dataset = prepare_dataset(val_dataset)

# Перемещаем модель на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Определение параметров обучения
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

# Функция вычисления метрик
def compute_metrics(pred):
    logits = pred.predictions[0]
    preds = logits.argmax(-1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Создаём тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Обучаем модель
trainer.train()

# Сохраняем обученную модель
trainer.save_model("./hate_speech_model/best_model")
print("✅ Модель успешно обучена и сохранена.")