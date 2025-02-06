import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

# Функция для заполнения пропусков (линейная интерполяция)
def fill_missing_data(data):
    return data.interpolate(method="linear", limit_direction="forward", axis=0)

# Функция для получения индексов для обучения и тестирования
def split_indices(total_size, test_size, random_seed=42):
    np.random.seed(random_seed)
    indices = np.arange(total_size)
    test_count = int(total_size * test_size)
    test_indices = np.random.choice(indices, size=test_count, replace=False)
    train_indices = np.setdiff1d(indices, test_indices)
    return train_indices, test_indices

# Загрузка и подготовка данных
file_path = r"D:\main for my it\my tasks\source\ResearchHub\Exp4_TimeSeriesPrediction\data\Microsoft_Stock.csv"
data = pd.read_csv(file_path)

# Преобразование столбца с датами, сортировка и выбор нужных колонок
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").set_index("Date")
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
# Для каждого окна из 30 значений целевой переменной будет значение "Close" следующего временного шага (индекс 3)
for i in range(len(data_scaled) - seq_len):
    X_seq.append(data_scaled[i:i + seq_len])
    y_seq.append(data_scaled[i + seq_len, 3])  # прогнозируем "Close"

# Преобразуем списки в numpy-массивы
X_seq = np.array(X_seq)  # форма: (num_samples, seq_len, features)
y_seq = np.array(y_seq).reshape(-1, 1)

# Для градиентного бустинга будем использовать выпрямлённые (flattened) последовательности:
num_samples, seq_length, num_features = X_seq.shape
X_seq_flat = X_seq.reshape(num_samples, seq_length * num_features)

# Определение параметров эксперимента:
n_estimators_options = [10, 50, 100, 200, 300]
max_depth_options = [1, 2, 4, 6, 8, 10]
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]
results = []

# Основной цикл эксперимента (тройной цикл по n_estimators, max_depth и test_size)
for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        for test_size in test_size_options:
            # Разбиение данных по индексам
            train_idx, test_idx = split_indices(num_samples, test_size, random_seed=42)
            X_train_np = X_seq_flat[train_idx]
            X_test_np = X_seq_flat[test_idx]
            y_train_np = y_seq[train_idx]
            y_test_np = y_seq[test_idx]
            
            # Создание и обучение модели градиентного бустинга с заданными параметрами
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train_np, y_train_np.ravel())
            
            # Предсказание на тестовой выборке
            predictions = model.predict(X_test_np)
            
            # Вычисление статистик предсказаний: дисперсия и MAE
            var_pred = np.var(predictions)
            mae = mean_absolute_error(y_test_np, predictions)
            
            print(f"n_estimators={n_estimators}, max_depth={max_depth}, test_size={test_size}")
            print(f"  Дисперсия предсказания: {var_pred:.4f}")
            print(f"  MAE: {mae:.4f}\n")
            
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'test_size': test_size,
                'variance_prediction': var_pred,
                'MAE': mae
            })

# Сохранение результатов в CSV-файл
results_df = pd.DataFrame(results)
output_dir = r'Exp4_TimeSeriesPrediction\5_GradientBoosting'
os.makedirs(output_dir, exist_ok=True)
results_csv_path = os.path.join(output_dir, 'experiment_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"\nЭксперименты завершены. Результаты сохранены в '{results_csv_path}'.")
