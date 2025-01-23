import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

# Загрузка данных
data = pd.read_csv("D:/source/нир вр ряды/data/Microsoft_Stock.csv", sep=',')  # Замените на путь к вашему датасету

# Предположим, что вы хотите предсказать Close на основе Open, High, Low и Volume
input_features = ['Open', 'High', 'Low', 'Volume']
columns_to_predict = ['Close']

# Нормализация данных
scaler = MinMaxScaler()
data[input_features + columns_to_predict] = scaler.fit_transform(data[input_features + columns_to_predict])

# Функция для прореживания данных (удаление данных за случайные дни в каждом месяце)
def drop_random_days(data, drop_percentage):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    for month in data['Month'].unique():
        month_data = data[data['Month'] == month]
        drop_count = int(len(month_data) * drop_percentage)
        drop_indices = np.random.choice(month_data.index, drop_count, replace=False)
        data.loc[drop_indices, input_features + columns_to_predict] = np.nan
    return data

# Пример использования функции для удаления 30% данных за случайные дни
data_with_drops = drop_random_days(data.copy(), 0.3)

# Заполнение пропущенных значений (импутация)
imputer = SimpleImputer(strategy='mean')  # Использование среднего значения для заполнения
data_with_drops[input_features + columns_to_predict] = imputer.fit_transform(data_with_drops[input_features + columns_to_predict])

# Формирование временных шагов
sequence_length = 30  # Длина временной последовательности для модели

def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

X, y = create_sequences(data_with_drops, input_features, columns_to_predict, sequence_length)

# Получаем параметры с консоли
test_size = float(input("Введите размер тестовой выборки (например, 0.1 для 10%): "))
num_encoder_layers = int(input("Введите количество слоев трансформера (например, 2): "))

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Преобразуем данные в 2D для подачи в модель
X_train_2D = X_train.reshape(X_train.shape[0], -1)
X_test_2D = X_test.reshape(X_test.shape[0], -1)

# Преобразуем в тензоры для PyTorch
X_train_tensor = torch.tensor(X_train_2D, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_2D, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Байесовская модель с трансформером
class BayesianTransformerModel(nn.Module):
    def __init__(self, input_size, seq_len, nhead=8, num_encoder_layers=1):
        super(BayesianTransformerModel, self).__init__()
        
        # Входной слой
        self.fc = nn.Linear(input_size, 128)  # Входной слой

        # Трансформер
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_encoder_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(128, 2)  # 2: среднее и логарифм дисперсии

    def forward(self, x):
        x = self.fc(x)  # Применяем полносвязный слой
        x = x.permute(1, 0, 2)  # Меняем размерность на [seq_len, batch_size, features]
        
        x = self.transformer(x)  # Применяем трансформер
        
        # Убираем размерность для seq_len
        x = x.squeeze(0)
        
        # Получаем предсказания: среднее и дисперсия
        mean = x[:, 0]  # Среднее
        log_variance = x[:, 1]  # Логарифм дисперсии
        variance = torch.exp(log_variance)  # Дисперсия
        
        return mean, variance

# Инициализация модели
model = BayesianTransformerModel(input_size=X_train_tensor.shape[1], seq_len=sequence_length, num_encoder_layers=num_encoder_layers)

# Настройка оптимизатора
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# Обучение байесовской модели с трансформером
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    mean, variance = model(X_train_tensor)  # Прогнозируем среднее и дисперсию
    nll_loss = 0.5 * torch.mean(variance + (y_train_tensor - mean) ** 2 / variance)  # Байесовская ошибка с учетом дисперсии
    loss = nll_loss  # Потери для оптимизации
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Оценка модели на тестовой выборке
model.eval()
with torch.no_grad():
    test_mean, test_variance = model(X_test_tensor)
    test_rmse = mean_squared_error(y_test, test_mean.numpy(), squared=False)
    test_mae = mean_absolute_error(y_test, test_mean.numpy())
    test_dispersion = test_variance.mean().item()

print(f"\nTest RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Dispersion (Mean Variance): {test_dispersion:.4f}")
