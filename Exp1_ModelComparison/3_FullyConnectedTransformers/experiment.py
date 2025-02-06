import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

# 1. Загрузка и подготовка данных
try:
    data = pd.read_csv('Exp1_ModelComparison\\data\\student-mat.csv', sep=';')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['G3'])  # Используем G3 как целевую переменную
y = data['G3']

# Преобразование категориальных переменных в числовые (one-hot encoding)
X = pd.get_dummies(X)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Определение параметров эксперимента
layer_options = [1, 2, 4, 6, 8, 10]  # Количество слоев
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Размер тестовой выборки
train_size = 100  # Фиксированный размер обучающей выборки
epochs = 100  # Число эпох обучения

# Результаты эксперимента
results = []

# 3. Класс модели
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Устанавливаем batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        positional_encoding = self.positional_encoding.expand(batch_size, -1, -1)
        
        x = self.input_projection(x) + positional_encoding
        x = self.transformer_encoder(x)  # Теперь вход будет в формате (batch_size, seq_len, d_model)
        x = x.mean(dim=0)  # Усреднение по временной оси
        x = self.output_layer(x)
        return x


# 4. Запуск эксперимента для разных параметров
for num_layers in layer_options:
    for test_size in test_size_options:
        # Динамически вычисляем размер обучающей выборки
        train_size_dynamic = min(train_size, len(X_scaled))  # Адаптируем размер обучающей выборки

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        # Теперь нет пропуска эксперимента, если размер данных меньше train_size
        # Просто адаптируем обучающую выборку
        if len(X_train) < train_size_dynamic:
            print(f"Размер обучающей выборки ({len(X_train)}) меньше заданного значения ({train_size_dynamic}). Используем доступный размер.")
        
        # Подготовка данных
        X_train_sub = torch.tensor(X_train[:train_size_dynamic], dtype=torch.float32)
        y_train_sub = torch.tensor(y_train.values[:train_size_dynamic], dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Инициализация модели
        input_size = X_train.shape[1]
        model = TransformerModel(input_dim=input_size, num_layers=num_layers)

        # Настройка параметров обучения
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Обучение модели
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train_sub)
            loss = criterion(predictions, y_train_sub)
            loss.backward()
            optimizer.step()

        # Оценка модели
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            test_mae = mean_absolute_error(y_test, test_predictions.numpy())
            test_variance = np.var(test_predictions.numpy())  # Вычисление дисперсии

        # Сохранение результата
        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'train_size': train_size_dynamic,
            'mae': test_mae,
            'variance': test_variance
        })

        print(f"Слои: {num_layers}, Тестовая выборка: {test_size}, Train size: {train_size_dynamic}, MAE: {test_mae:.4f}, Variance: {test_variance:.4f}")

# 5. Создание таблицы результатов
results_df = pd.DataFrame(results)

# 6. Сохранение таблицы в CSV
results_df.to_csv('Exp1_ModelComparison\\3_FullyConnectedTransformers\\experiment_results.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results_mae_variance.csv'.")
