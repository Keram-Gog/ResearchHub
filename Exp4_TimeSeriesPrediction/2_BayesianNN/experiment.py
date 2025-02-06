import sys
# Добавляем путь к BayeFormers (убедитесь, что путь указан правильно)
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Импорт функции для преобразования модели в байесовскую
from bayeformers import to_bayesian

# Ввод параметров с консоли
num_hidden_layers = int(input("Введите количество скрытых слоев: "))
test_size_ratio = float(input("Введите долю данных для теста (0.1 - 0.5): "))

# Загрузка данных
data = pd.read_csv('D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp4_TimeSeriesPrediction\\data\\Microsoft_Stock.csv', sep=',')  # Замените на путь к вашему датасету

# Определяем признаки
input_features = ['Open', 'High', 'Low', 'Volume']
columns_to_predict = ['Close']

# Нормализация данных
scaler = MinMaxScaler()
data[input_features + columns_to_predict] = scaler.fit_transform(data[input_features + columns_to_predict])

# Создание пропусков (искусственно удаляем данные)
def create_missing_data(data, missing_percentage=0.1):
    missing_days = int(len(data) * missing_percentage)
    missing_indices = np.random.choice(data.index, size=missing_days, replace=False)
    data.loc[missing_indices, input_features + columns_to_predict] = np.nan
    return data

data_with_missing = create_missing_data(data.copy(), missing_percentage=0.2)
# Заполняем пропуски методом "forward fill"
data_with_missing.fillna(method='ffill', inplace=True)

# Формирование последовательностей
sequence_length = 30

def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

X, y = create_sequences(data_with_missing, input_features, columns_to_predict, sequence_length)

# Фиксированное разделение данных с использованием random seed
def fixed_split_data(data, test_size_ratio, random_seed=42):
    np.random.seed(random_seed)
    total_size = len(data)
    test_size = int(total_size * test_size_ratio)
    all_indices = np.arange(total_size)
    test_indices = np.random.choice(all_indices, size=test_size, replace=False)
    train_indices = np.setdiff1d(all_indices, test_indices)
    
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    
    return train_data, test_data

# Разделение данных с фиксированными тестовыми данными
train_data, test_data = fixed_split_data(data_with_missing, test_size_ratio)

X_train, y_train = create_sequences(train_data, input_features, columns_to_predict, sequence_length)
X_test, y_test = create_sequences(test_data, input_features, columns_to_predict, sequence_length)

# Подготовка данных: преобразуем последовательности в тензоры
X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Определение модели
class BayesianRegressionModelWithVariance(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_hidden_layers):
        super(BayesianRegressionModelWithVariance, self).__init__()
        layers = [nn.Linear(input_size, hidden_layer_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(nn.ReLU())
        # Выходной слой выдаёт два значения: предсказанное значение и логарифм дисперсии
        layers.append(nn.Linear(hidden_layer_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        output = self.network(x)
        mean = output[:, 0]
        log_variance = output[:, 1]
        # Преобразуем логарифм дисперсии в саму дисперсию
        variance = torch.exp(log_variance)
        return mean, variance

# Инициализация модели
input_size = X_train_tensor.shape[1]
hidden_layer_size = 64
model = BayesianRegressionModelWithVariance(input_size, hidden_layer_size, num_hidden_layers)

# Преобразование модели в байесовскую
bayesian_model = to_bayesian(model, delta=0.05, freeze=True)

# Настройка оптимизатора
optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=0.001)
epochs = 100

# Обучение модели
print("\nНачало обучения...")
for epoch in range(epochs):
    bayesian_model.train()
    optimizer.zero_grad()
    mean, variance = bayesian_model(X_train_tensor)
    # Негативное логарифмическое правдоподобие (NLL) как функция потерь
    nll_loss = 0.5 * torch.mean(variance + (y_train_tensor - mean) ** 2 / variance)
    nll_loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {nll_loss.item():.4f}")

# Оценка модели
print("\nОценка модели...")
bayesian_model.eval()
with torch.no_grad():
    test_mean, test_variance = bayesian_model(X_test_tensor)
    test_rmse = mean_squared_error(y_test, test_mean.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, test_mean.numpy())
    test_dispersion = test_variance.mean().item()

print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Dispersion (Mean Variance): {test_dispersion:.4f}")
