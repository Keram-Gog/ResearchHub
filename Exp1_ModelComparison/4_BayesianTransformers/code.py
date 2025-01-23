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
num_layers = int(input("Введите количество скрытых слоёв (например, 2): "))
test_size = float(input("Введите размер тестовой выборки (например, 0.2 для 20%): "))

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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. Построение частотной модели с трансформером
class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerRegressionModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_projection(x) + self.positional_encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Усредняем по временной оси
        return self.output_layer(x)

# Инициализация частотной модели
model = TransformerRegressionModel(input_dim=X_train_tensor.shape[1], num_layers=num_layers)

# 3. Настройка параметров обучения для частотной модели
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# Обучение частотной модели
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 4. Преобразование частотной модели в байесовскую
bayesian_model = to_bayesian(model, delta=0.05, freeze=True)

# 5. Настройка параметров обучения для байесовской модели
samples = 10
batch_size = X_train_tensor.shape[0]
output_dim = 1
predictions = torch.zeros(samples, batch_size, output_dim)
log_prior = torch.zeros(samples, batch_size)
log_variational_posterior = torch.zeros(samples, batch_size)

# Обучение байесовской модели
for epoch in range(epochs):
    bayesian_model.train()
    for s in range(samples):
        optimizer.zero_grad()
        predictions[s] = bayesian_model(X_train_tensor)
        log_prior[s] = bayesian_model.log_prior()
        log_variational_posterior[s] = bayesian_model.log_variational_posterior()

# 6. Оценка байесовской модели на тестовой выборке
bayesian_model.eval()
with torch.no_grad():
    test_predictions = bayesian_model(X_test_tensor)
    # Средние предсказания по всем образцам
    mean_predictions = test_predictions.mean(dim=0)
    test_rmse = mean_squared_error(y_test, mean_predictions.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, mean_predictions.numpy())

print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
