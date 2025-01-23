import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

# Получение параметров с консоли
num_layers = int(input("Введите количество скрытых слоёв (например, 3): "))
test_size = float(input("Введите размер тестовой выборки (например, 0.2 для 20%): "))
train_size = int(input("Введите размер обучающей выборки (например, 100): "))  # Один фиксированный размер обучающей выборки

# 1. Загрузка и подготовка данных
try:
    data = pd.read_csv('D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp1_ModelComparison\\data\\student-mat.csv', sep=';')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['G3'])  # Используем G3 как целевую переменную
y = data['G3']

# Преобразование категориальных переменных в числовые (one-hot encoding)
X = pd.get_dummies(X)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки с фиксированным random_state
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
print(f"Данные успешно разделены! Размер тестовой выборки: {test_size * 100:.1f}%")

# Преобразуем в тензоры
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. Построение полносвязной нейронной сети с динамическими слоями
class DynamicRegressionModel(nn.Module):
    def __init__(self, input_size, num_layers):
        super(DynamicRegressionModel, self).__init__()
        layers = []
        current_size = input_size

        # Добавляем скрытые слои
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, current_size // 2))
            layers.append(nn.ReLU())
            current_size //= 2  # Уменьшаем размер слоя в 2 раза
        
        # Добавляем выходной слой
        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Инициализация модели
input_size = X_train.shape[1]

# 3. Настройка параметров обучения
criterion = nn.MSELoss()  # Функция потерь (среднеквадратичная ошибка)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# 4. Создание обучающей выборки с фиксированным размером
X_train_sub = torch.tensor(X_train[:train_size], dtype=torch.float32)
y_train_sub = torch.tensor(y_train.values[:train_size], dtype=torch.float32).view(-1, 1)

# Инициализация модели для текущего эксперимента
model = DynamicRegressionModel(input_size, num_layers)

# 5. Обучение модели
print("Начало обучения...")
try:
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_sub)
        loss = criterion(predictions, y_train_sub)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
except Exception as e:
    print(f"Ошибка во время обучения: {e}")
    exit()

# 6. Оценка модели на фиксированной тестовой выборке
print("\nОценка модели на тестовых данных...")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    
    # Вычисление MAE (Mean Absolute Error)
    test_mae = mean_absolute_error(y_test, test_predictions.numpy())

    # Вычисление дисперсии (Variance of the predictions)
    test_variance = np.var(test_predictions.numpy())  # Дисперсия предсказанных значений

# Вывод результатов
print(f"\nTest MAE: {test_mae:.4f}")
print(f"Test Variance: {test_variance:.4f}")
