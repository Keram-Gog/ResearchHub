import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Получаем параметры с консоли
num_layers = int(input("Введите количество скрытых слоев (например, 12): "))
test_size = float(input("Введите размер тестовой выборки (например, 0.1 для 10%): "))

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

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. Определение модели с динамическим количеством слоев
class RegressionModel(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(RegressionModel, self).__init__()
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, current_dim // 2))
            layers.append(nn.ReLU())
            current_dim //= 2  # Уменьшаем размер слоя в 2 раза
        layers.append(nn.Linear(current_dim, 1))  # Выходной слой
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Инициализация модели с динамическим количеством слоев
input_dim = X_train_tensor.shape[1]
model = RegressionModel(input_dim, num_layers)

# 3. Настройка обучения
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 4. Обучение
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

# 5. Оценка модели на тестовой выборке
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    
    # Дисперсия предсказаний
    test_variance = test_predictions.var().item()

    # MAE
    test_mae = mean_absolute_error(y_test, test_predictions.numpy())

    print(f"\nTest Variance: {test_variance:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
