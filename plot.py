import matplotlib.pyplot as plt

# Данные для графика
models = ["mt5_small", "mt5_base", "mt5_large", "slovak-t5-base", "slovak_t5_base_small", "SlovakBERT"]
precision = [0.4923, 0.4533, 0.4949, 0.5000, 0.5385, 0.60]
recall = [0.6400, 0.6800, 0.9800, 0.6600, 0.9800, 0.7200]
f1_score = [0.5565, 0.5440, 0.6577, 0.5690, 0.6950, 0.6545]

# Построение графика
plt.figure(figsize=(10, 6))

plt.plot(models, precision, marker='o', label="Presnosť (Precision)", color="orange")
plt.plot(models, recall, marker='o', label="Citlivosť (Recall)", color="red")
plt.plot(models, f1_score, marker='o', label="F1-Score", color="purple")

# Настройки графика
plt.title("Porovnanie modelov pre rozpoznávanie nenávistnej reči", fontsize=14)
plt.xlabel("Modely", fontsize=12)
plt.ylabel("Hodnota metrík", fontsize=12)
plt.xticks(rotation=15)
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="best", fontsize=10)

# Отображение графика
plt.tight_layout()
plt.show()
