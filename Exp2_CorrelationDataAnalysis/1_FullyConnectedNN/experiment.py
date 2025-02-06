import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

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
layer_options = [1, 2, 4, 6, 8, 10]  # Количество слоев
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Размер тестовой выборки
epochs = 100  # Число эпох обучения

# Результаты эксперимента
results = []

# 3. Класс модели
class DynamicRegressionModel(nn.Module):
    def __init__(self, input_size, num_layers):
        super(DynamicRegressionModel, self).__init__()
        layers = []
        current_size = input_size

        # Добавляем скрытые слои
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, max(current_size // 2, 1)))  # Минимум 1 нейрон
            layers.append(nn.ReLU())
            current_size = max(current_size // 2, 1)
        
        # Добавляем выходной слой
        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 4. Запуск эксперимента
# Вместо фиксированного train_size можно вычислять его на основе test_size
for num_layers in layer_options:
    for test_size in test_size_options:
        # Динамически вычисляем размер обучающей выборки
        train_size = int(len(X_scaled) * (1 - test_size))
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        if len(X_train) < train_size:
            print(f"Пропуск эксперимента: train_size ({train_size}) превышает доступное число данных ({len(X_train)})")
            continue

        # Подготовка данных
        X_train_sub = torch.tensor(X_train[:train_size], dtype=torch.float32)
        y_train_sub = torch.tensor(y_train.values[:train_size], dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Инициализация модели
        input_size = X_train.shape[1]
        model = DynamicRegressionModel(input_size, num_layers)

        # Настройка параметров обучения
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Обучение модели
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train_sub)
            loss = criterion(predictions, y_train_sub)
            loss.backward()
            optimizer.step()

        # Оценка модели
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            test_mae = mean_absolute_error(y_test, test_predictions.numpy())
            test_variance = np.var(test_predictions.numpy())

        # Сохранение результата
        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'mae': test_mae,
            'variance': test_variance
        })

        print(f"Слои: {num_layers}, Тестовая выборка: {test_size}, MAE: {test_mae:.4f}, Variance: {test_variance:.4f}")

# 5. Создание таблицы результатов
results_df = pd.DataFrame(results)

# Сохранение таблицы в CSV
results_df.to_csv('Exp2_CorrelationDataAnalysis\\1_FullyConnectedNN\\experiment_results.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results.csv'.")
