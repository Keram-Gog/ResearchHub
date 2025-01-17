import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Датасет для временных рядов
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Полносвязная модель с двумя слоями и трансформером
class FullyConnectedModelWithTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nhead=8, num_encoder_layers=2):
        super(FullyConnectedModelWithTransformer, self).__init__()
        
        # Один полносвязный слой
        self.fc = nn.Linear(input_size, hidden_size)
        
        # Трансформер (с двумя слоями)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_encoder_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Применяем один полносвязный слой
        x = self.fc(x)
        
        # Трансформер ожидает вход в формате (seq_len, batch_size, features)
        # Для этого добавляем размерность для последовательности
        x = x.unsqueeze(0)  # [1, batch_size, hidden_size] - добавление размерности для seq_len
        
        # Применяем трансформер
        x = self.transformer(x)
        
        # Убираем размерность для последовательности
        x = x.squeeze(0)  # Убираем размерность для seq_len
        
        # Прогнозируем выходное значение
        x = self.output_layer(x)
        
        return x

# Заполнение пропущенных данных с использованием линейной интерполяции
def fill_missing_data(data):
    # Линейная интерполяция для пропусков
    return pd.DataFrame(data).interpolate(method='linear', axis=0).values

# Генерация последовательностей с пропущенными днями
def create_sequences_with_missing_days(data, seq_len, missing_percentage):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_len):
        sequence = data[i:i+seq_len]
        num_missing_days = int(seq_len * missing_percentage)
        missing_indices = np.random.choice(seq_len, num_missing_days, replace=False)
        sequence[missing_indices] = np.nan  # Пропускаем данные
        X_seq.append(sequence)
        y_seq.append(data[i+seq_len, 3])  # Прогнозируем "Close"
    return X_seq, y_seq

# Функция для обучения модели
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            # Прямой проход
            outputs = model(data)

            # Расчёт потерь
            loss = criterion(outputs, targets)

            # Обратный проход и обновление весов
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

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

            # Вычисление MAE
            mae += torch.sum(torch.abs(outputs - targets)).item()

            # Сохранение для вычисления дисперсии
            all_targets.append(targets)
            all_outputs.append(outputs)

    # Объединение всех предсказаний и целей
    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)

    # Вычисление дисперсии
    variance = torch.mean((all_outputs - all_targets.mean())**2).item()

    print(f"Test Loss: {total_loss/len(test_loader):.4f}")
    print(f"Mean Absolute Error (MAE): {mae/len(test_loader.dataset):.4f}")
    print(f"Variance: {variance:.4f}")

# Загрузка данных
file_path = r"D:\source\нир вр ряды\data\Microsoft_Stock.csv"  # Укажите путь к CSV-файлу
data = pd.read_csv(file_path)

# Преобразование столбца Date в datetime и сортировка
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

# Используем только числовые столбцы
numerical_columns = ["Open", "High", "Low", "Close", "Volume"]
data = data[numerical_columns]

# Стандартизация данных
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Формирование временных последовательностей с пропущенными днями
seq_len = 30  # Длина окна (настраиваемая переменная)
missing_percentage = 0.3  # Примерный процент пропущенных дней
X_seq, y_seq = create_sequences_with_missing_days(data_scaled, seq_len, missing_percentage)

# Заполнение пропущенных данных
X_seq = fill_missing_data(X_seq)

X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42, shuffle=False)

# Создание DataLoader
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Параметры модели
input_size = X_seq.shape[2] * seq_len
hidden_size = 128
output_size = 1

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullyConnectedModelWithTransformer(input_size, hidden_size, output_size).to(device)

# Оптимизатор и функция потерь
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Преобразование входных данных для полносвязной сети
X_train = X_train.view(X_train.size(0), -1)  # [batch_size, seq_len * input_size]
X_test = X_test.view(X_test.size(0), -1)

# Обновляем DataLoader с новыми размерами
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Обучение модели
train_model(model, train_loader, criterion, optimizer, epochs=50)

# Тестирование модели
test_model(model, test_loader)
