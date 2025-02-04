import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Загрузка и подготовка данных
data = pd.read_csv('nifty_500.csv', sep=',')

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])
y = data['Last Traded Price']

# Преобразование категориальных переменных в числовые (one-hot encoding)
X = pd.get_dummies(X)

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=42)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 3. Определение модели с трансформером
class TransformerRegression(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, num_layers, dropout):
        super(TransformerRegression, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_size)  # Эмбеддинги признаков
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=embed_size * 4,
            dropout=dropout,
            batch_first=True,  # Позволяет использовать (batch, seq_len, feature)
        )
        self.fc_out = nn.Linear(embed_size, 1)  # Выходной слой для регрессии

    def forward(self, x):
        x = self.embedding(x)  # Преобразуем вход в эмбеддинги
        x = x.unsqueeze(1)     # Добавляем размер для последовательности: (batch, seq_len=1, feature)
        x = self.transformer(x, x)  # Трансформер-энкодер
        x = x.mean(dim=1)      # Усреднение по seq_len
        return self.fc_out(x)

# 4. Гиперпараметры модели
input_dim = X_train_tensor.shape[1]
embed_size = 64
num_heads = 4
num_layers = 12
dropout = 0.1

model = TransformerRegression(input_dim, embed_size, num_heads, num_layers, dropout)

# 5. Настройка оптимизации
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 6. Использование CUDA, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = X_train_tensor.to(device), X_test_tensor.to(device), y_train_tensor.to(device), y_test_tensor.to(device)

# 7. Обучение модели
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = loss_fn(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 8. Оценка модели
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).squeeze()  # Убираем лишнюю размерность
    test_mae = mean_absolute_error(y_test, test_predictions.cpu().numpy())
    test_mse = mean_squared_error(y_test, test_predictions.cpu().numpy())
    test_r2 = r2_score(y_test, test_predictions.cpu().numpy())
    
    # Вычисляем дисперсию (variance) предсказаний
    predictions_variance = np.var(test_predictions.cpu().numpy())
    
    print(f"\nTest MAE: {test_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R2 Score: {test_r2:.4f}")
    print(f"Predictions Variance: {predictions_variance:.4f}")
