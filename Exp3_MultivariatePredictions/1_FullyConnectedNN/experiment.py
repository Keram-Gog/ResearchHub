import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

# 1. Загрузка и подготовка данных
try:
    data = pd.read_csv('D:\\питон\\MO\\1\\этот\\global_mean_sea_level_1993-2024.csv', sep=',')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# Выбор параметров для прогнозирования и входных данных
columns_to_predict = ['GMSLNoGIA', 'SmoothedGMSLWithGIA', 'SmoothedGMSLWithGIASigremoved']
input_features = [
    'YearPlusFraction', 'NumberOfObservations', 'NumberOfWeightedObservations', 
    'StdDevGMSLNoGIA', 'StdDevGMSLWithGIA', 
    'AltimeterType', 'MergedFileCycle', 'SmoothedGMSLNoGia', 'SmoothedGMSLNoGIASigremoved'
]

# Нормализация данных
scaler = MinMaxScaler()
data[input_features + columns_to_predict] = scaler.fit_transform(data[input_features + columns_to_predict])

# Формирование временных шагов
def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

sequence_length = 30  # Примерная длина последовательности
X, y = create_sequences(data, input_features, columns_to_predict, sequence_length)

# 2. Определение параметров эксперимента
layer_options = [1, 2, 4, 6, 8, 10]  # Количество слоев
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Размер тестовой выборки
epochs = 100  # Число эпох обучения

# Результаты эксперимента
results = []

# 3. Класс модели
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_size, num_layers):
        super(FullyConnectedNN, self).__init__()
        layers = []
        current_size = input_dim

        # Добавляем скрытые слои
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_layer_size))
            layers.append(nn.ReLU())
            current_size = hidden_layer_size

        # Выходной слой
        layers.append(nn.Linear(current_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 4. Запуск эксперимента
for num_layers in layer_options:
    for test_size in test_size_options:
        # Динамически вычисляем размер обучающей выборки
        train_size = int(len(X) * (1 - test_size))

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if len(X_train) < train_size:
            print(f"Пропуск эксперимента: train_size ({train_size}) превышает доступное число данных ({len(X_train)})")
            continue

        # Преобразование данных в тензоры
        X_train_tensor = torch.tensor(X_train[:train_size], dtype=torch.float32).reshape(X_train[:train_size].shape[0], -1)
        y_train_tensor = torch.tensor(y_train[:train_size], dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(X_test.shape[0], -1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Инициализация модели
        input_dim = X_train_tensor.shape[1]
        output_dim = y_train_tensor.shape[1]
        hidden_layer_size = 256  # Фиксированный размер скрытых слоёв
        model = FullyConnectedNN(input_dim, output_dim, hidden_layer_size, num_layers)

        # Настройка параметров обучения
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Обучение модели
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train_tensor)
            loss = criterion(predictions, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Оценка модели
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            test_mae = mean_absolute_error(y_test, test_predictions.numpy())
            test_variance = np.var(test_predictions.numpy())

        # Сохранение результата
        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'mae': test_mae,
            'variance': test_variance
        })

        print(f"Слои: {num_layers}, Тестовая выборка: {test_size}, MAE: {test_mae:.4f}, Variance: {test_variance:.4f}")

# 5. Создание таблицы результатов
results_df = pd.DataFrame(results)

# Сохранение таблицы в CSV
results_df.to_csv('Exp3_MultivariatePredictions\\1_FullyConnectedNN\\experiment_results.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results.csv'.")
