import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
sequence_length = 30

def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

X, y = create_sequences(data, input_features, columns_to_predict, sequence_length)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразуем в тензоры для PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Модель с трансформером
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, num_heads=8, num_layers=8, d_model=64):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # Линейный слой для преобразования входных данных в d_model
        self.input_layer = nn.Linear(input_dim, d_model)

        # Слои трансформера (многоголовое внимание + нормализация)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=256, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Глобальное усреднение по временным шагам
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Линейный слой для предсказания
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Преобразование входных данных в d_model
        x = self.input_layer(x)

        # Преобразование последовательности через слои трансформера
        for layer in self.encoder_layers:
            x = layer(x)

        # Нормализация
        x = self.encoder_norm(x)

        # Усреднение по временным шагам
        x = x.permute(0, 2, 1)  # Для работы с AdaptiveAvgPool1d
        x = self.global_avg_pool(x).squeeze(-1)

        # Выходной слой
        x = self.output_layer(x)
        return x

# Параметры модели
input_dim = X_train_tensor.shape[2]
output_dim = y_train_tensor.shape[1]
seq_len = X_train_tensor.shape[1]

model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)

# Настройка параметров обучения
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 50
batch_size = 16

# Обучение модели
model.train()
for epoch in range(epochs):
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(X_train_tensor):.4f}")

# Оценка модели на тестовой выборке
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_rmse = mean_squared_error(y_test, test_predictions.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, test_predictions.numpy())

print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
