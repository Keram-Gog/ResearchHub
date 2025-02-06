import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
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
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Размер тестовой выборки
max_depth_options = [1, 2, 4, 6, 8, 10]  # Разные значения max_depth (количество слоёв)

# Результаты эксперимента
results = []

# 3. Запуск эксперимента с градиентным бустингом (многовыходная регрессия)
for max_depth in max_depth_options:
    for test_size in test_size_options:
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Преобразование данных в нужный формат (выпрямляем последовательности)
        X_train_flattened = X_train.reshape(X_train.shape[0], -1)
        X_test_flattened = X_test.reshape(X_test.shape[0], -1)

        # Инициализация модели градиентного бустинга с обёрткой для многовыходной регрессии
        base_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=max_depth, random_state=42)
        model = MultiOutputRegressor(base_model)

        # Обучение модели
        model.fit(X_train_flattened, y_train)

        # Оценка модели
        y_pred = model.predict(X_test_flattened)
        
        # Рассчитываем MAE и дисперсию для вектора предсказаний
        test_mae = mean_absolute_error(y_test, y_pred)
        test_variance = np.var(y_pred, axis=0)

        # Сохранение результата (средняя дисперсия по всем переменным)
        results.append({
            'max_depth': max_depth,
            'test_size': test_size,
            'mae': test_mae,
            'variance': test_variance.mean()
        })

        print(f"max_depth: {max_depth}, Test size: {test_size}, MAE: {test_mae:.4f}, Variance: {test_variance.mean():.4f}")

# 4. Создание таблицы результатов
results_df = pd.DataFrame(results)

# Сохранение таблицы в CSV
results_df.to_csv('Exp3_MultivariatePredictions\\5_GradientBoosting\\experiment_results_with_depth.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results_with_depth.csv'.")
