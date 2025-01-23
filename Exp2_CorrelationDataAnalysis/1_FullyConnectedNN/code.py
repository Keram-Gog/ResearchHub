import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Получение параметров с консоли
num_layers = int(input("Введите количество скрытых слоёв (например, 5): "))
train_size = int(input("Введите размер тренировочной выборки (например, 100): "))
test_size = int(input("Введите размер тестовой выборки (например, 100): "))

# 1. Загрузка и подготовка данных
data = pd.read_csv('nifty_500.csv', sep=',')

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])
y = data['Last Traded Price']

# Преобразование категориальных переменных в числовые (one-hot encoding)
X = pd.get_dummies(X)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Фиксируем random_state для того, чтобы тестовые данные были одинаковыми
random_state = 42

# Разделим данные, используя фиксированный random_state, чтобы тестовые данные всегда были одинаковыми
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=train_size + test_size, test_size=test_size, random_state=random_state)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. Построение полносвязной нейронной сети с динамическим количеством слоев
class RegressionModel(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(RegressionModel, self).__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, current_dim // 2))
            layers.append(nn.ReLU())
            current_dim //= 2  # Уменьшаем размер слоя в 2 раза
        layers.append(nn.Linear(current_dim, 1))  # Выходной слой
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Создание модели
input_dim = X_train_tensor.shape[1]
model = RegressionModel(input_dim, num_layers)

# 3. Настройка параметров обучения
criterion = nn.MSELoss()  # Функция потерь (среднеквадратичная ошибка)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
batch_size = 32  # Использование мини-батчей

# Обучение модели
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))  # Перемешивание данных
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()  # Обнуляем градиенты
        predictions = model(batch_X)  # Прогноз
        loss = criterion(predictions, batch_y)  # Вычисляем потери
        loss.backward()  # Обратное распространение ошибки
        optimizer.step()  # Обновление весов
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(X_train_tensor):.4f}")

# 4. Оценка модели на тестовой выборке
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)  # MSE на тесте
    test_rmse = mean_squared_error(y_test, test_predictions.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, test_predictions.numpy())

    # Дисперсия предсказаний
    test_variance = test_predictions.var().item()

print(f"\nTest Loss (MSE): {test_loss.item():.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Variance: {test_variance:.4f}")
