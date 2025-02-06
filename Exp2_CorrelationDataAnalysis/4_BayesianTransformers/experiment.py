import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import sys
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')

from bayeformers import to_bayesian
import bayeformers.nn as bnn

# 1. Загрузка и подготовка данных
try:
    data = pd.read_csv('D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp2_CorrelationDataAnalysis\\data\\nifty_500.csv', sep=',')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])
y = data['Last Traded Price']

# Преобразование категориальных переменных в числовые (one-hot encoding)
X = pd.get_dummies(X)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Определение параметров эксперимента
layer_options = [1, 2, 4, 6, 8, 10]      # Количество слоев
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Размер тестовой выборки
epochs = 100  # Число эпох обучения

# Результаты эксперимента
results = []

# 3. Класс модели с трансформером
class DynamicTransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, num_heads=4, hidden_dim=128, num_encoder_layers=4):
        super(DynamicTransformerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Преобразование входных данных в нужную форму для трансформера
        self.linear_in = nn.Linear(input_size, hidden_dim)
        
        # Слой трансформера
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_encoder_layers)
        
        # Полносвязный слой для предсказания
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Преобразуем данные: линейное преобразование и изменение размерности для трансформера
        x = self.linear_in(x)
        x = x.unsqueeze(1)  # Получаем форму (batch_size, seq_len=1, hidden_dim)
        
        # Применяем трансформер
        x = self.transformer_encoder(x)
        
        # Усредняем по последовательности (seq_len)
        x = x.mean(dim=1)
        
        # Выходной прогноз
        x = self.fc(x)
        return x

# 4. Запуск эксперимента с трансформером и преобразованием в байесовскую модель
for num_layers in layer_options:
    for test_size in test_size_options:
        # Определение размера обучающей выборки
        train_size = int(len(X_scaled) * (1 - test_size))
        
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        if len(X_train) < train_size:
            print(f"Пропуск эксперимента: train_size ({train_size}) превышает доступное число данных ({len(X_train)})")
            continue

        # Подготовка данных
        X_train_sub = torch.tensor(X_train[:train_size], dtype=torch.float32)
        y_train_sub = torch.tensor(y_train.values[:train_size], dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Инициализация исходной модели с трансформером
        input_size = X_train.shape[1]
        model = DynamicTransformerModel(input_size, num_layers)
        
        # Преобразование модели в байесовскую с помощью BayeFormers
        bayesian_model = to_bayesian(model, delta=0.05, freeze=True)

        # Настройка оптимизатора
        optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=0.001)

        # Обучение модели
        for epoch in range(epochs):
            bayesian_model.train()
            optimizer.zero_grad()
            predictions = bayesian_model(X_train_sub)
            loss = F.mse_loss(predictions, y_train_sub)
            loss.backward()
            optimizer.step()

        # Оценка модели
        bayesian_model.eval()
        with torch.no_grad():
            test_predictions = bayesian_model(X_test_tensor)
            test_mae = mean_absolute_error(y_test, test_predictions.numpy())
            test_rmse = mean_squared_error(y_test, test_predictions.numpy(), squared=False)

        # Сохранение результата эксперимента
        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        })

        print(f"Слои: {num_layers}, Тестовая выборка: {test_size}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")

# 5. Создание таблицы результатов и сохранение в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Exp1_ModelComparison\\2_BayesianNN\\experiment_results.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results.csv'.")
