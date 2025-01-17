import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. Построение модели на основе трансформера
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)  # Преобразуем вход в нужное измерение
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))  # Добавляем позиционное кодирование
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Выходной слой для регрессии
        )

    def forward(self, x):
        x = self.input_projection(x) + self.positional_encoding  # Применяем линейное преобразование и добавляем позиционное кодирование
        x = x.transpose(0, 1)  # Трансформеры принимают вход в виде (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Усредняем по временной оси (после трансформера)
        x = self.output_layer(x)
        return x

# Инициализация модели
model = TransformerModel(input_dim=X_train_tensor.shape[1])

# 3. Настройка параметров обучения
criterion = nn.MSELoss()  # Функция потерь (среднеквадратичная ошибка)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# 4. Обучение модели
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Обнуляем градиенты
    predictions = model(X_train_tensor)  # Прогноз на обучающих данных
    loss = criterion(predictions, y_train_tensor)  # Вычисляем потери
    loss.backward()  # Обратное распространение ошибки
    optimizer.step()  # Обновление весов
    
    # Печать результата каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 5. Оценка модели на тестовой выборке
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    test_rmse = mean_squared_error(y_test, test_predictions.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, test_predictions.numpy())

print(f"\nTest Loss (MSE): {test_loss.item():.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
