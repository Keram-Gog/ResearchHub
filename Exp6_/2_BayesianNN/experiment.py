import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import sys

# Подключаем путь к BayeFormers
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')

from bayeformers import to_bayesian
from tqdm import tqdm
import os

# Устанавливаем устройство (GPU, если доступно)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1. Загрузка и предобработка данных
# -------------------------------
data_path = r"Exp6_\data\data.csv"
df = pd.read_csv(data_path)

# Приводим столбец с датой к формату datetime и сортируем по дате
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')

# Очищаем столбец цены закрытия: убираем знак "$" и преобразуем в float
df['Close'] = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

# Создаем лаговый признак: значение цены закрытия предыдущего дня
df['Lag1'] = df['Close'].shift(1)
df = df.dropna().reset_index(drop=True)

# Формируем признаки и целевую переменную
X_np = df['Lag1'].values.reshape(-1, 1)  # входной признак
y_np = df['Close'].values                # целевая переменная

# Приводим данные к тензорам
X_all = torch.tensor(X_np, dtype=torch.float32)
y_all = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)  # размерность (N,1)

total_samples = len(X_all)

# -------------------------------
# 2. Определение модели (частотная реализация)
# -------------------------------
class FCNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int):
        super(FCNet, self).__init__()
        layers = []
        # Первый слой с input_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Добавляем (num_hidden_layers - 1) скрытых слоев
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Выходной слой
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# -------------------------------
# 3. Параметры экспериментов
# -------------------------------
train_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]     # доля обучающих данных
num_layers_list = [1, 2, 4, 6, 8, 10]         # число скрытых слоев
hidden_dim = 32                             # число нейронов в каждом скрытом слое

results = []

# Параметры обучения
epochs = 50
batch_size = 32
learning_rate = 1e-3
SAMPLES = 10  # число сэмплов для оценки в байесовском подходе

# Функция для обучения байесовской модели на одном эксперименте
def train_bayesian_model(model, train_loader):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.train()
    n_batches = len(train_loader)
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            # Сохраняем сэмплированные предсказания и регуляризаторы
            predictions = []
            log_priors = []
            log_variational_posteriors = []
            
            for _ in range(SAMPLES):
                pred = model(x_batch)
                predictions.append(pred)
                log_priors.append(model.log_prior())
                log_variational_posteriors.append(model.log_variational_posterior())
            
            # Преобразуем списки в тензоры и усредняем по сэмплам
            preds = torch.stack(predictions, dim=0).mean(dim=0)
            avg_log_prior = torch.stack(log_priors).mean()
            avg_log_var_post = torch.stack(log_variational_posteriors).mean()
            
            # Вычисляем MSE как ошибку (можно использовать reduction='sum' или 'mean')
            mse_loss = F.mse_loss(preds, y_batch, reduction='mean')
            
            # Итоговая функция потерь с учетом байесовской регуляризации
            loss = (avg_log_var_post - avg_log_prior) / n_batches + mse_loss
            loss.backward()
            optimizer.step()

# Функция для оценки модели на тестовой выборке
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            predictions = []
            for _ in range(SAMPLES):
                pred = model(x_batch)
                predictions.append(pred)
            preds = torch.stack(predictions, dim=0).mean(dim=0)
            
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
            
    all_preds = torch.cat(all_preds, dim=0).squeeze().numpy()
    all_targets = torch.cat(all_targets, dim=0).squeeze().numpy()
    
    errors = all_preds - all_targets
    variance_error = np.var(errors)
    mae_error = np.mean(np.abs(errors))
    return variance_error, mae_error

# -------------------------------
# 4. Проведение экспериментов
# -------------------------------
for train_size in train_sizes:
    n_train = int(total_samples * train_size)
    # Разбиваем данные: первые n_train для обучения, оставшиеся для теста
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]
    
    # Создаем DataLoader-ы
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for num_layers in num_layers_list:
        try:
            # Создаем частотную (frequentist) модель
            model_frequentist = FCNet(input_dim=1, hidden_dim=hidden_dim, num_hidden_layers=num_layers, output_dim=1)
            model_frequentist.to(device)
            
            # Преобразуем модель в байесовскую с помощью bayeformers (MOPED инициализация)
            model_bayesian = to_bayesian(model_frequentist, delta=0.05, freeze=True)
            model_bayesian.to(device)
            
            # Обучаем модель
            train_bayesian_model(model_bayesian, train_loader)
            
            # Оцениваем модель на тестовой выборке
            variance_error, mae_error = evaluate_model(model_bayesian, test_loader)
            
            results.append({
                'train_size': train_size,
                'num_layers': num_layers,
                'variance': variance_error,
                'MAE': mae_error
            })
            print(f"train_size={train_size}, num_layers={num_layers} -> Var: {variance_error:.4f}, MAE: {mae_error:.4f}")
        except Exception as e:
            print(f"Ошибка для train_size={train_size} и num_layers={num_layers}: {e}")
            results.append({
                'train_size': train_size,
                'num_layers': num_layers,
                'variance': np.nan,
                'MAE': np.nan
            })

# -------------------------------
# 5. Сохранение результатов
# -------------------------------
results_df = pd.DataFrame(results)
output_dir = r"Exp6_\data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "experiment_results_bayeformers.csv")
results_df.to_csv(output_path, index=False)

print("Результаты экспериментов сохранены в", output_path)
