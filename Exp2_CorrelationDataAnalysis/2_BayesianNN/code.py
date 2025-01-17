import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Преобразуем в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 2. Определение модели с 12 слоями
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        # 12 слоев
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 64)
        self.layer7 = nn.Linear(64, 64)
        self.layer8 = nn.Linear(64, 32)
        self.layer9 = nn.Linear(32, 32)
        self.layer10 = nn.Linear(32, 32)
        self.layer11 = nn.Linear(32, 16)
        self.layer12 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))  # Применяем ReLU после каждого слоя
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        x = F.relu(self.layer10(x))
        x = F.relu(self.layer11(x))
        x = self.layer12(x)  # Выходной слой (без активации для регрессии)
        return x

# Инициализация модели
model = RegressionModel(X_train_tensor.shape[1])

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
