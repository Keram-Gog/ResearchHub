import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from bayeformers import to_bayesian
import bayeformers.nn as bnn

# Загрузка данных
data = pd.read_csv("D:\\source\\нир вр ряды\\data\\Microsoft_Stock.csv", sep=',')  # Замените на путь к вашему датасету

# Предположим, что вы хотите предсказать Close на основе Open, High, Low и Volume
input_features = ['Open', 'High', 'Low', 'Volume']
columns_to_predict = ['Close']

# Нормализация данных
scaler = MinMaxScaler()
data[input_features + columns_to_predict] = scaler.fit_transform(data[input_features + columns_to_predict])

# Функция для создания пропусков в данных
def create_missing_data(data, missing_percentage=0.1):
    missing_days = int(len(data) * missing_percentage)
    missing_indices = np.random.choice(data.index, size=missing_days, replace=False)
    data.loc[missing_indices, input_features + columns_to_predict] = np.nan
    return data

# Применяем прореживание с разным процентом пропусков
data_with_missing = create_missing_data(data.copy(), missing_percentage=0.2)  # Пример с 20% пропусков

# Заполнение пропусков с помощью метода forward-fill (используем значения предыдущих дней)
data_with_missing.fillna(method='ffill', inplace=True)

# Формирование временных шагов
sequence_length = 30  # Длина временной последовательности для модели

def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

X, y = create_sequences(data_with_missing, input_features, columns_to_predict, sequence_length)

# Методическое разделение данных на обучающую и тестовую выборки
train_size = int(len(data_with_missing) * 0.8)  # 80% для обучения
train_data, test_data = data_with_missing[:train_size], data_with_missing[train_size:]

X_train, y_train = create_sequences(train_data, input_features, columns_to_predict, sequence_length)
X_test, y_test = create_sequences(test_data, input_features, columns_to_predict, sequence_length)

# Преобразуем данные в 2D для подачи в модель
X_train_2D = X_train.reshape(X_train.shape[0], -1)
X_test_2D = X_test.reshape(X_test.shape[0], -1)

# Преобразуем в тензоры для PyTorch
X_train_tensor = torch.tensor(X_train_2D, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_2D, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Байесовская модель с одним слоем (среднее и дисперсия)
class BayesianRegressionModelWithVariance(nn.Module):
    def __init__(self):
        super(BayesianRegressionModelWithVariance, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(X_train_tensor.shape[1], 64),  # Один скрытый слой
            nn.ReLU(),
            nn.Linear(64, 2)  # Выходной слой (среднее и дисперсия)
        )

    def forward(self, x):
        output = self.network(x)
        mean = output[:, 0]  # Среднее значение
        log_variance = output[:, 1]  # Логарифм дисперсии для стабилизации
        variance = torch.exp(log_variance)  # Дисперсия
        return mean, variance

# Инициализация модели
model = BayesianRegressionModelWithVariance()

# Преобразование модели в байесовскую с использованием bayeformers
bayesian_model = to_bayesian(model, delta=0.05, freeze=True)

# Настройка оптимизатора
optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=0.001)
epochs = 100

# Обучение байесовской модели
for epoch in range(epochs):
    bayesian_model.train()
    optimizer.zero_grad()
    mean, variance = bayesian_model(X_train_tensor)  # Прогнозируем среднее и дисперсию
    nll_loss = 0.5 * torch.mean(variance + (y_train_tensor - mean) ** 2 / variance)  # Байесовская ошибка с учетом дисперсии
    loss = nll_loss  # Потери для оптимизации
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Оценка модели на тестовой выборке
bayesian_model.eval()
with torch.no_grad():
    test_mean, test_variance = bayesian_model(X_test_tensor)
    test_rmse = mean_squared_error(y_test, test_mean.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, test_mean.numpy())
    test_dispersion = test_variance.mean().item()

print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Dispersion (Mean Variance): {test_dispersion:.4f}")
