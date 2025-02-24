import sys
# Добавляем путь к BayeFormers (убедитесь, что путь указан правильно)
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from bayeformers import to_bayesian

# --- Для воспроизводимости:
np.random.seed(42)
torch.manual_seed(42)

# ======== 1. Функция sMAPE ========
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

# ======== 2. Функция для создания пропусков в данных =========
def create_missing_data(data, columns, missing_percentage=0.1):
    missing_days = int(len(data) * missing_percentage)
    missing_indices = np.random.choice(data.index, size=missing_days, replace=False)
    data.loc[missing_indices, columns] = np.nan
    return data

# ======== 3. Функция формирования последовательностей =========
def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

# ======== 4. Фиксированное разбиение данных =========
def fixed_split_data(df, test_size_ratio, random_seed=42):
    np.random.seed(random_seed)
    total_size = len(df)
    test_size = int(total_size * test_size_ratio)
    all_indices = np.arange(total_size)
    test_indices = np.random.choice(all_indices, size=test_size, replace=False)
    train_indices = np.setdiff1d(all_indices, test_indices)
    return df.iloc[train_indices], df.iloc[test_indices]

# ======== 5. Байесовская модель регрессии с оценкой дисперсии =========
class BayesianRegressionModelWithVariance(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_hidden_layers):
        super(BayesianRegressionModelWithVariance, self).__init__()
        layers = [nn.Linear(input_size, hidden_layer_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(nn.ReLU())
        # Выходной слой: два значения – предсказание и логарифм дисперсии
        layers.append(nn.Linear(hidden_layer_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        output = self.network(x)
        mean = output[:, 0]
        log_variance = output[:, 1]
        variance = torch.exp(log_variance)
        return mean, variance

# ======== 6. Обучение модели =========
def train_model(model, optimizer, X_train, y_train, epochs, device):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        mean, variance = model(X_train)
        loss = 0.5 * torch.mean(variance + (y_train - mean) ** 2 / variance)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ======== 7. Оценка модели =========
def evaluate_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        test_mean, test_variance = model(X_test)
        rmse = mean_squared_error(y_test.cpu().numpy(), test_mean.cpu().numpy(), squared=False)
        mae = mean_absolute_error(y_test.cpu().numpy(), test_mean.cpu().numpy())
        dispersion = test_variance.mean().item()
    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test Dispersion (Mean Variance): {dispersion:.4f}")
    return rmse, mae, dispersion

# ======== 8. Основной блок =========
def main():
    # Ввод параметров с консоли
    num_hidden_layers = int(input("Введите количество скрытых слоев: "))
    test_size_ratio = float(input("Введите долю данных для теста (0.1 - 0.5): "))

    # Загрузка данных
    data = pd.read_csv(
        r"D:\main for my it\my tasks\source\ResearchHub\Exp4_TimeSeriesPrediction\data\Microsoft_Stock.csv", 
        sep=','
    )
    
    # Определяем признаки
    input_features = ['Open', 'High', 'Low', 'Volume']
    target_columns = ['Close']
    
    # Нормализация данных
    scaler = MinMaxScaler()
    data[input_features + target_columns] = scaler.fit_transform(data[input_features + target_columns])
    
    # Создание пропусков (искусственно удаляем данные)
    data_with_missing = create_missing_data(data.copy(), input_features + target_columns, missing_percentage=0.2)
    # Заполняем пропуски методом "forward fill"
    data_with_missing.fillna(method='ffill', inplace=True)
    
    # Формирование последовательностей
    sequence_length = 30
    X, y = create_sequences(data_with_missing, input_features, target_columns, sequence_length)
    
    # Фиксированное разделение данных
    train_data, test_data = fixed_split_data(data_with_missing, test_size_ratio)
    X_train, y_train = create_sequences(train_data, input_features, target_columns, sequence_length)
    X_test, y_test = create_sequences(test_data, input_features, target_columns, sequence_length)
    
    # Преобразование последовательностей в тензоры
    X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1)
    
    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация модели
    input_size = X_train_tensor.shape[1]
    hidden_layer_size = 64
    model = BayesianRegressionModelWithVariance(input_size, hidden_layer_size, num_hidden_layers).to(device)
    
    # Преобразование модели в байесовскую
    bayesian_model = to_bayesian(model, delta=0.05, freeze=True)
    
    # Настройка оптимизатора и обучение
    optimizer = optim.Adam(bayesian_model.parameters(), lr=0.001, weight_decay=0.0001)
    epochs = 100
    print("\nНачало обучения...")
    train_model(bayesian_model, optimizer, X_train_tensor.to(device), y_train_tensor.to(device), epochs, device)
    
    # Оценка модели
    print("\nОценка модели...")
    evaluate_model(bayesian_model, X_test_tensor.to(device), y_test_tensor.to(device), device)

if __name__ == '__main__':
    main()
import sys
# Добавляем путь к BayeFormers (убедитесь, что путь указан правильно)
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from bayeformers import to_bayesian

# --- Для воспроизводимости:
np.random.seed(42)
torch.manual_seed(42)

# ======== 1. Функция sMAPE ========
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

# ======== 2. Функция для создания пропусков в данных =========
def create_missing_data(data, columns, missing_percentage=0.1):
    missing_days = int(len(data) * missing_percentage)
    missing_indices = np.random.choice(data.index, size=missing_days, replace=False)
    data.loc[missing_indices, columns] = np.nan
    return data

# ======== 3. Функция формирования последовательностей =========
def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

# ======== 4. Фиксированное разбиение данных =========
def fixed_split_data(df, test_size_ratio, random_seed=42):
    np.random.seed(random_seed)
    total_size = len(df)
    test_size = int(total_size * test_size_ratio)
    all_indices = np.arange(total_size)
    test_indices = np.random.choice(all_indices, size=test_size, replace=False)
    train_indices = np.setdiff1d(all_indices, test_indices)
    return df.iloc[train_indices], df.iloc[test_indices]

# ======== 5. Байесовская модель регрессии с оценкой дисперсии =========
class BayesianRegressionModelWithVariance(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_hidden_layers):
        super(BayesianRegressionModelWithVariance, self).__init__()
        layers = [nn.Linear(input_size, hidden_layer_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(nn.ReLU())
        # Выходной слой: два значения – предсказание и логарифм дисперсии
        layers.append(nn.Linear(hidden_layer_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        output = self.network(x)
        mean = output[:, 0]
        log_variance = output[:, 1]
        variance = torch.exp(log_variance)
        return mean, variance

# ======== 6. Обучение модели =========
def train_model(model, optimizer, X_train, y_train, epochs, device):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        mean, variance = model(X_train)
        loss = 0.5 * torch.mean(variance + (y_train - mean) ** 2 / variance)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ======== 7. Оценка модели =========
def evaluate_model(model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        test_mean, test_variance = model(X_test)
        rmse = mean_squared_error(y_test.cpu().numpy(), test_mean.cpu().numpy(), squared=False)
        mae = mean_absolute_error(y_test.cpu().numpy(), test_mean.cpu().numpy())
        dispersion = test_variance.mean().item()
    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test Dispersion (Mean Variance): {dispersion:.4f}")
    return rmse, mae, dispersion

# ======== 8. Основной блок =========
def main():
    # Ввод параметров с консоли
    num_hidden_layers = int(input("Введите количество скрытых слоев: "))
    test_size_ratio = float(input("Введите долю данных для теста (0.1 - 0.5): "))

    # Загрузка данных
    data = pd.read_csv(
        r"D:\main for my it\my tasks\source\ResearchHub\Exp4_TimeSeriesPrediction\data\Microsoft_Stock.csv", 
        sep=','
    )
    
    # Определяем признаки
    input_features = ['Open', 'High', 'Low', 'Volume']
    target_columns = ['Close']
    
    # Нормализация данных
    scaler = MinMaxScaler()
    data[input_features + target_columns] = scaler.fit_transform(data[input_features + target_columns])
    
    # Создание пропусков (искусственно удаляем данные)
    data_with_missing = create_missing_data(data.copy(), input_features + target_columns, missing_percentage=0.2)
    # Заполняем пропуски методом "forward fill"
    data_with_missing.fillna(method='ffill', inplace=True)
    
    # Формирование последовательностей
    sequence_length = 30
    X, y = create_sequences(data_with_missing, input_features, target_columns, sequence_length)
    
    # Фиксированное разделение данных
    train_data, test_data = fixed_split_data(data_with_missing, test_size_ratio)
    X_train, y_train = create_sequences(train_data, input_features, target_columns, sequence_length)
    X_test, y_test = create_sequences(test_data, input_features, target_columns, sequence_length)
    
    # Преобразование последовательностей в тензоры
    X_train_tensor = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1)
    
    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация модели
    input_size = X_train_tensor.shape[1]
    hidden_layer_size = 64
    model = BayesianRegressionModelWithVariance(input_size, hidden_layer_size, num_hidden_layers).to(device)
    
    # Преобразование модели в байесовскую
    bayesian_model = to_bayesian(model, delta=0.05, freeze=True)
    
    # Настройка оптимизатора и обучение
    optimizer = optim.Adam(bayesian_model.parameters(), lr=0.001, weight_decay=0.0001)
    epochs = 100
    print("\nНачало обучения...")
    train_model(bayesian_model, optimizer, X_train_tensor.to(device), y_train_tensor.to(device), epochs, device)
    
    # Оценка модели
    print("\nОценка модели...")
    evaluate_model(bayesian_model, X_test_tensor.to(device), y_test_tensor.to(device), device)

if __name__ == '__main__':
    main()
