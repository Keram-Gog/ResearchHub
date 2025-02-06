import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Кастомный Dataset для временных рядов
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X может быть многомерным (например, [num_samples, seq_len, features])
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Функция для тренировки модели
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.view(inputs.size(0), -1))  # выпрямляем вход
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Функция для тестирования модели
def test_model(model, test_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs.view(inputs.size(0), -1))  # выпрямляем вход
            predictions.extend(outputs.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    mean_pred = np.mean(predictions)
    var_pred = np.var(predictions)
    
    print(f"Среднее предсказание: {mean_pred:.4f}")
    print(f"Дисперсия предсказания: {var_pred:.4f}")

    return mean_pred, var_pred

# Класс для полносвязной модели
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=6, hidden_units=128):
        super(FullyConnectedModel, self).__init__()
        layers = []
        current_size = input_size
        curr_hidden = hidden_units

        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, curr_hidden))
            layers.append(nn.ReLU())
            current_size = curr_hidden
            curr_hidden //= 2  # Сокращаем число нейронов на каждом слое

        layers.append(nn.Linear(current_size, output_size))  # Выходной слой
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

# Функция для заполнения пропусков в данных
def fill_missing_data(data):
    return data.interpolate(method="linear", limit_direction="forward", axis=0)

# Функция для разбиения данных (работает с numpy-массивами произвольной размерности вдоль первой оси)
def split_data(data, test_size, random_seed=42):
    np.random.seed(random_seed)
    total_size = data.shape[0]
    test_count = int(total_size * test_size)
    indices = np.arange(total_size)
    test_indices = np.random.choice(indices, size=test_count, replace=False)
    train_indices = np.setdiff1d(indices, test_indices)
    train_data = data[train_indices]
    test_data = data[test_indices]
    return train_data, test_data

# Загрузка и подготовка данных
file_path = r"D:\main for my it\my tasks\source\ResearchHub\Exp4_TimeSeriesPrediction\data\Microsoft_Stock.csv"
data = pd.read_csv(file_path)

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").set_index("Date")

numerical_columns = ["Open", "High", "Low", "Close", "Volume"]
data = data[numerical_columns]

data = fill_missing_data(data)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Формирование временных последовательностей
seq_len = 30
X_seq, y_seq = [], []
for i in range(len(data_scaled) - seq_len):
    X_seq.append(data_scaled[i:i + seq_len])
    y_seq.append(data_scaled[i + seq_len, 3])  # прогнозируем "Close"

# Преобразуем списки в numpy-массивы, чтобы избежать предупреждения (и потом в тензоры)
X_seq = np.array(X_seq)  # форма: (num_samples, seq_len, features)
y_seq = np.array(y_seq).reshape(-1, 1)

# Преобразуем в тензоры
X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32)

# Определение параметров эксперимента
layer_options = [1, 2, 4, 6, 8, 10]
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]
epochs = 50
results = []

# Основной цикл эксперимента
for num_layers in layer_options:
    for test_size in test_size_options:
        # Разделение данных по индексам (используем numpy-массив)
        X_train_np, X_test_np = split_data(X_seq.numpy(), test_size)
        y_train_np, y_test_np = split_data(y_seq.numpy(), test_size)

        # Преобразование обратно в тензоры
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_np, dtype=torch.float32)

        # Подготовка DataLoader
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Параметры модели
        # Входной размер = seq_len * (количество признаков)
        input_size = X_train.size(1) * X_train.size(2)
        output_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Создание модели
        model = FullyConnectedModel(input_size=input_size, output_size=output_size, num_layers=num_layers).to(device)

        # Оптимизатор и функция потерь
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"\nНачало обучения модели с {num_layers} слоями и test_size={test_size}...")
        train_model(model, train_loader, criterion, optimizer, epochs, device)

        print(f"\nТестирование модели с {num_layers} слоями и test_size={test_size}...")
        mean_pred, var_pred = test_model(model, test_loader, device)

        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'mean_prediction': mean_pred,
            'variance_prediction': var_pred
        })

# Сохранение результатов в CSV
results_df = pd.DataFrame(results)
results_df.to_csv(r'Exp4_TimeSeriesPrediction\1_FullyConnectedNN\experiment_results.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results.csv'.")
