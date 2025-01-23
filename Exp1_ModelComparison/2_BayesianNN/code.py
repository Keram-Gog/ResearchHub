import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from bayeformers import to_bayesian
import bayeformers.nn as bnn

# Получение параметров с консоли
num_layers = int(input("Введите количество скрытых слоёв (например, 3): "))
test_size = float(input("Введите размер тестовой выборки (например, 0.2 для 20%): "))
train_size = int(input("Введите размер обучающей выборки (например, 100): "))  # Один фиксированный размер обучающей выборки

# 1. Загрузка и подготовка данных
data = pd.read_csv('student-mat.csv', sep=';')

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

# Преобразуем в тензоры
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. Построение частотной модели с динамическим количеством слоев
class RegressionModel(nn.Module):
    def __init__(self, input_size, num_layers):
        super(RegressionModel, self).__init__()
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

# 3. Создание обучающей выборки с фиксированным размером
X_train_sub = torch.tensor(X_train[:train_size], dtype=torch.float32)
y_train_sub = torch.tensor(y_train.values[:train_size], dtype=torch.float32).view(-1, 1)

# Инициализация частотной модели с динамическим количеством слоев
model = RegressionModel(X_train_sub.shape[1], num_layers)

# 4. Настройка параметров обучения для частотной модели
criterion = nn.MSELoss()  # Функция потерь (среднеквадратичная ошибка)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# Обучение частотной модели
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Обнуляем градиенты
    predictions = model(X_train_sub)  # Прогноз на обучающих данных
    loss = criterion(predictions, y_train_sub)  # Вычисляем потери
    loss.backward()  # Обратное распространение ошибки
    optimizer.step()  # Обновление весов
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 5. Преобразование частотной модели в байесовскую
bayesian_model = to_bayesian(model, delta=0.05, freeze=True)

# 6. Настройка параметров обучения для байесовской модели
samples = 10  # Количество образцов для байесовской модели
batch_size = X_train_sub.shape[0]
output_dim = 1  # Размер выходного слоя
predictions = torch.zeros(samples, batch_size, output_dim)
log_prior = torch.zeros(samples, batch_size)
log_variational_posterior = torch.zeros(samples, batch_size)

# Обучение байесовской модели
for epoch in range(epochs):
    bayesian_model.train()
    for s in range(samples):
        optimizer.zero_grad()  # Обнуляем градиенты
        predictions[s] = bayesian_model(X_train_sub)  # Прогноз
        log_prior[s] = bayesian_model.log_prior()  # Логарифм априорного распределения
        log_variational_posterior[s] = bayesian_model.log_variational_posterior()  # Логарифм вариационного постериора

# 7. Оценка байесовской модели на тестовой выборке
bayesian_model.eval()
with torch.no_grad():
    test_predictions = bayesian_model(X_test_tensor)
    test_rmse = mean_squared_error(y_test, test_predictions.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, test_predictions.numpy())

# Вывод результатов
print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
