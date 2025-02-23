import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# 1. Загрузка датасета (замените путь на ваш)
file_path = r'Exp5_\data\cleaned_weather.csv'  # <-- Укажите актуальный путь к файлу
data = pd.read_csv(file_path)

# 2. Преобразуем столбец 'date' в datetime и устанавливаем его в качестве индекса
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 3. Выбираем 20 метеорологических показателей
features = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 
            'sh', 'H2OC', 'rho', 'wv', 'max. wv', 'wd', 'rain', 
            'raining', 'SWDR', 'PAR', 'max. PAR', 'Tlog']
data = data[features]

# 4. Нормализация данных
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 5. Функция для создания обучающего набора с использованием скользящего окна
def create_dataset(data_array, look_back=24):
    X, y = [], []
    # Предсказываем следующий шаг для столбца 'T' (индекс 1)
    for i in range(len(data_array) - look_back):
        X.append(data_array[i : i + look_back])
        y.append(data_array[i + look_back, 1])
    return np.array(X), np.array(y)

look_back = 24  # Количество предыдущих временных шагов
X, y = create_dataset(data_scaled, look_back)

# Преобразуем входные данные в 2D для градиентного бустинга
num_features = X.shape[2]
X_flat = X.reshape(X.shape[0], look_back * num_features)

# 6. Параметры эксперимента: доли обучающей выборки и число базовых моделей
training_fractions = [0.1, 0.2, 0.5, 0.8, 0.9]
n_estimators_list = [1, 2, 4, 6, 8, 10]

results = []

# 7. Эксперимент: перебор долей обучающей выборки и числа базовых моделей (n_estimators)
for frac in training_fractions:
    train_size = int(len(X_flat) * frac)
    X_train = X_flat[:train_size]
    y_train = y[:train_size]
    X_test = X_flat[train_size:]
    y_test = y[train_size:]
    
    for n_est in n_estimators_list:
        # Создаем и обучаем модель XGBoost для регрессии
        model = xgb.XGBRegressor(objective='reg:squarederror',
                                 n_estimators=n_est,
                                 verbosity=0,
                                 seed=42)
        model.fit(X_train, y_train)
        
        # Предсказание на тестовой выборке
        y_pred = model.predict(X_test)
        
        # Вычисляем MAE и дисперсию предсказанных значений
        mae_value = mean_absolute_error(y_test, y_pred)
        variance_value = np.var(y_pred)
        
        results.append({
            'training_fraction': frac,
            'n_estimators': n_est,
            'mae': mae_value,
            'variance': variance_value
        })
        
        print(f"Train Fraction: {frac}, n_estimators: {n_est}, MAE: {mae_value:.4f}, Variance: {variance_value:.4f}")

# 8. Запись результатов эксперимента в CSV файл
results_df = pd.DataFrame(results)
results_df.to_csv("gb_experiment_results.csv", index=False)
print("Результаты эксперимента сохранены в 'gb_experiment_results.csv'")
