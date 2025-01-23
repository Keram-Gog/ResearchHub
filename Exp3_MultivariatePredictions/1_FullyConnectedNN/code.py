import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

# Установка фиксированного random seed
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Получение параметров от пользователя
train_size = int(input("Введите размер обучающей выборки (например, 100): "))  # Только размер трейна
test_size = 100  # Размер тестовой выборки фиксирован
sequence_length = int(input("Введите длину временной последовательности (например, 30): "))
num_layers = int(input("Введите количество скрытых слоёв (например, 4): "))

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

# Преобразование данных в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(X_train.shape[0], -1)  # В 2D для полносвязной сети
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(X_test.shape[0], -1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Создание полносвязной нейронной сети с динамическим числом слоёв
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size, num_layers):
        super(FullyConnectedNN, self).__init__()
        layers = []
        current_size = input_dim

        # Добавляем скрытые слои
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_layer_size))
            layers.append(nn.ReLU())
            current_size = hidden_layer_size

        # Выходной слой
        layers.append(nn.Linear(current_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Инициализация модели, функции потерь и оптимизатора
input_dim = X_train_tensor.shape[1]
output_dim = y_train_tensor.shape[1]
hidden_layer_size = 256  # Фиксированный размер скрытых слоёв
model = FullyConnectedNN(input_dim, output_dim, hidden_layer_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Оценка модели на тестовых данных
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_predictions_np = test_predictions.numpy()
    test_loss = mean_squared_error(y_test, test_predictions_np)
    test_mae = mean_absolute_error(y_test, test_predictions_np)

    # Оценка дисперсии ошибок
    errors = y_test - test_predictions_np
    overall_mae = np.mean(np.abs(errors))
    overall_variance = np.mean(np.var(errors, axis=0))

print(f"\nTest MAE: {test_mae:.4f}")
print(f"Test MSE: {test_loss:.4f}")
print(f"Overall MAE (mean across all parameters): {overall_mae:.4f}")
print(f"Overall Variance (mean across all parameters): {overall_variance:.4f}")
