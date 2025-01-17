import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Датасет для временных рядов
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Полносвязная модель для временных рядов с 6 слоями
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# Функция для удаления дней
def drop_days(data, drop_percentage):
    """
    Убирает данные за случайные дни с заданным процентом пропусков.
    """
    unique_days = data.index.unique()
    num_days_to_drop = int(len(unique_days) * drop_percentage)
    days_to_drop = np.random.choice(unique_days, num_days_to_drop, replace=False)
    return data.drop(index=days_to_drop)

# Функция для заполнения пропусков
def fill_missing_data(data):
    """
    Заполняет пропущенные значения в данных методом линейной интерполяции.
    """
    return data.interpolate(method='linear', limit_direction='forward', axis=0)

# Функция для разделения данных
def split_data(data, test_size=0.1):
    """
    Делит данные на тренировочные и тестовые выборки.
    """
    train_data = data.iloc[:-int(len(data) * test_size)]
    test_data = data.iloc[-int(len(data) * test_size):]
    return train_data, test_data

# Функция для обучения модели
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Функция для тестирования модели
def test_model(model, test_loader):
    model.eval()
    total_loss = 0
    mae = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            mae += torch.sum(torch.abs(outputs - targets)).item()
            all_targets.append(targets)
            all_outputs.append(outputs)
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    variance = torch.mean((all_outputs - all_targets.mean())**2).item()
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    print(f"Mean Absolute Error (MAE): {mae / len(test_loader.dataset):.4f}")
    print(f"Variance: {variance:.4f}")

# Загрузка данных
file_path = r"D:\source\нир вр ряды\data\Microsoft_Stock.csv"  # Укажите путь к CSV-файлу
data = pd.read_csv(file_path)

# Преобразование столбца Date в datetime и сортировка
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").set_index("Date")

# Используем только числовые столбцы
numerical_columns = ["Open", "High", "Low", "Close", "Volume"]
data = data[numerical_columns]

# Прореживание данных
drop_percentage = 0.2  # Пример: исключить 20% дней
data = drop_days(data, drop_percentage)

# Заполнение пропусков
data = fill_missing_data(data)

# Стандартизация данных
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Формирование временных последовательностей
seq_len = 30
X_seq, y_seq = [], []
for i in range(len(data_scaled) - seq_len):
    X_seq.append(data_scaled[i:i + seq_len])
    y_seq.append(data_scaled[i + seq_len, 3])  # Прогнозируем "Close"

X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

# Разделение данных
train_data, test_data = split_data(pd.DataFrame(data_scaled), test_size=0.1)
X_train, X_test = torch.tensor(train_data.values[:-seq_len], dtype=torch.float32), torch.tensor(test_data.values, dtype=torch.float32)
y_train, y_test = y_seq[:len(X_train)], y_seq[-len(X_test):]

# Подготовка DataLoader
train_dataset = TimeSeriesDataset(X_train.view(X_train.size(0), -1), y_train)
test_dataset = TimeSeriesDataset(X_test.view(X_test.size(0), -1), y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Параметры модели
input_size = X_train.size(1) * seq_len
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullyConnectedModel(input_size, output_size).to(device)

# Оптимизатор и функция потерь
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
train_model(model, train_loader, criterion, optimizer, epochs=50)

# Тестирование
test_model(model, test_loader)
