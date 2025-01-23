import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
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


# Полносвязная модель с динамическим количеством слоёв
class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=6, hidden_units=128):
        super(FullyConnectedModel, self).__init__()
        layers = []
        current_size = input_size

        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_units))
            layers.append(nn.ReLU())
            current_size = hidden_units
            hidden_units //= 2  # Сокращаем количество нейронов вдвое на каждом слое

        layers.append(nn.Linear(current_size, output_size))  # Выходной слой
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


# Функция для заполнения пропусков
def fill_missing_data(data):
    return data.interpolate(method="linear", limit_direction="forward", axis=0)


# Функция для разделения данных (с фиксацией теста)
def split_data(data, test_size, random_seed=42):
    # Устанавливаем seed для воспроизводимости
    np.random.seed(random_seed)
    # Фиксируем случайную выборку теста
    total_size = len(data)
    test_size = int(total_size * test_size)
    all_indices = np.arange(total_size)
    test_indices = np.random.choice(all_indices, size=test_size, replace=False)
    train_indices = np.setdiff1d(all_indices, test_indices)
    
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    
    return train_data, test_data


# Загрузка данных
file_path = r"D:\source\нир вр ряды\data\Microsoft_Stock.csv"  # Укажите путь к CSV-файлу
data = pd.read_csv(file_path)

# Преобразование столбца Date в datetime и сортировка
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").set_index("Date")

# Используем только числовые столбцы
numerical_columns = ["Open", "High", "Low", "Close", "Volume"]
data = data[numerical_columns]

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

# Получение параметров от пользователя
test_size = float(input("Введите размер тестовой выборки (например, 0.2 для 20%): "))  # Фиксируем тестовый размер
num_layers = int(input("Введите количество скрытых слоёв модели: "))

# Разделение данных с фиксированными тестовыми данными
train_data, test_data = split_data(pd.DataFrame(X_seq.numpy()), test_size=test_size)
train_labels, test_labels = split_data(pd.DataFrame(y_seq.numpy()), test_size=test_size)

# Преобразование обратно в тензоры
X_train = torch.tensor(train_data.values, dtype=torch.float32)
y_train = torch.tensor(train_labels.values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(test_data.values, dtype=torch.float32)
y_test = torch.tensor(test_labels.values, dtype=torch.float32).unsqueeze(1)

# Подготовка DataLoader
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Параметры модели
input_size = X_train.size(1) * X_train.size(2)  # seq_len * feature_size
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создание модели
model = FullyConnectedModel(input_size=input_size, output_size=output_size, num_layers=num_layers).to(device)

# Оптимизатор и функция потерь
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
epochs = 50
print("\nНачало обучения...")
train_model(model, train_loader, criterion, optimizer, epochs)

# Тестирование модели
print("\nТестирование модели...")
test_model(model, test_loader)
