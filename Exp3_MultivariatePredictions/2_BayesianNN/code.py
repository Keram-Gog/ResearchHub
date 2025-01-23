import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from bayeformers import to_bayesian

# Установка фиксированного random seed
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Получение параметров от пользователя
train_size = int(input("Введите размер обучающей выборки (например, 100): "))  # Размер обучающей выборки
test_size = 100  # Размер тестовой выборки фиксирован
sequence_length = int(input("Введите длину временной последовательности (например, 30): "))
epochs = int(input("Введите количество эпох для обучения (например, 100): "))

# Загрузка данных
data = pd.read_csv("D:\\питон\\MO\\1\\этот\\global_mean_sea_level_1993-2024.csv", sep=',')

# Выбор параметров для прогнозирования и входных данных
columns_to_predict = ['GMSLNoGIA', 'SmoothedGMSLWithGIA', 'SmoothedGMSLWithGIASigremoved']
input_features = [
    'YearPlusFraction', 'NumberOfObservations', 'NumberOfWeightedObservations', 
    'StdDevGMSLNoGIA', 'StdDevGMSLWithGIA', 
    'AltimeterType', 'MergedFileCycle', 'SmoothedGMSLNoGia', 'SmoothedGMSLNoGIASigremoved'
]

# Нормализация данных
scaler = MinMaxScaler()
data[input_features + columns_to_predict] = scaler.fit_transform(data[input_features + columns_to_predict])

# Формирование временных шагов
def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

X, y = create_sequences(data, input_features, columns_to_predict, sequence_length)

# Фиксируем тестовую выборку
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_test, _, y_test, _ = train_test_split(X_temp, y_temp, test_size=test_size / (1 - 0.2), random_state=random_seed)

# Преобразование в 2D для полносвязной сети
X_train_2D = X_train.reshape(X_train.shape[0], -1)
X_test_2D = X_test.reshape(X_test.shape[0], -1)

# Преобразование в тензоры для PyTorch
X_train_tensor = torch.tensor(X_train_2D, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_2D, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Полносвязная сеть с несколькими слоями
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Инициализация полносвязной сети
input_dim = X_train_tensor.shape[1]
output_dim = y_train_tensor.shape[1]
model = FullyConnectedNetwork(input_dim, output_dim)

# Преобразование полносвязной сети в байесовскую
bayesian_model = to_bayesian(model, delta=0.05, freeze=True)

# Настройка параметров обучения
optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=0.001)

# Обучение байесовской модели
for epoch in range(epochs):
    bayesian_model.train()
    optimizer.zero_grad()
    predictions = bayesian_model(X_train_tensor)
    loss = F.mse_loss(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Оценка модели на тестовой выборке
bayesian_model.eval()
with torch.no_grad():
    test_predictions = bayesian_model(X_test_tensor)
    test_rmse = mean_squared_error(y_test, test_predictions.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, test_predictions.numpy())

print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
