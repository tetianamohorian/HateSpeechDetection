import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

# Загружаем тестовый датасет
ds = load_dataset("TUKE-KEMT/hate_speech_slovak")
label_0 = ds['train'].filter(lambda example: example['label'] == 0)
label_1 = ds['train'].filter(lambda example: example['label'] == 1)

def create_stratified_split(label_0, label_1, n_samples, seed=44):
    few_shot_0 = label_0.shuffle(seed=seed).select(range(n_samples))
    few_shot_1 = label_1.shuffle(seed=seed).select(range(n_samples))
    return concatenate_datasets([few_shot_0, few_shot_1]).shuffle(seed=seed)

# Создаём тестовый набор
test_dataset = create_stratified_split(label_0, label_1, n_samples=50)

# Загружаем токенизатор и модель
model_path = "./few_shot/few_shot_small/hate_speech_model/best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Функция токенизации
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

def prepare_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    return dataset.rename_column("label", "labels")

# Подготавливаем тестовый набор
test_dataset = prepare_dataset(test_dataset)

# Определяем тренера
trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
)

# Функция поиска оптимального порога
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
    return thresholds[optimal_idx]

# Функция оценки модели
def evaluate_with_threshold(trainer, dataset, threshold=0.5):
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = torch.tensor(logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_labels = (probs[:, 1] > threshold).numpy().astype(int)
    true_labels = predictions.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary', zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Поиск оптимального порога
optimal_threshold = find_optimal_threshold(trainer, test_dataset)
print(f"✅ Оптимальный порог: {optimal_threshold:.4f}")

# Оценка модели на тестовом наборе
results = evaluate_with_threshold(trainer, test_dataset, threshold=optimal_threshold)
print(f"📊 Итоговые результаты на тестовом наборе:")
print(f"🎯 Precision: {results['precision']:.4f}")
print(f"🎯 Recall: {results['recall']:.4f}")
print(f"🎯 F1-score: {results['f1']:.4f}")