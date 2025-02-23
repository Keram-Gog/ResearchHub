import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# 1. Загрузка датасета (замените путь на актуальный)
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

# Для ARIMA используем целевой временной ряд – нормализованные значения температуры "T" (второй столбец, индекс 1)
# Восстанавливаем индекс из исходного DataFrame
ts = pd.Series(data_scaled[:, 1], index=data.index)

# 5. Параметры эксперимента:
# Доли обучающей выборки и список ARIMA порядков (p,d,q)
training_fractions = [0.1, 0.2, 0.5, 0.8, 0.9]
orders_list = [(1, 1, 1), (2, 1, 2), (0, 1, 1), (1, 1, 0), (2, 1, 1), (1, 1, 2)]

results = []

# 6. Эксперимент: перебор долей обучающей выборки и ARIMA порядков
for frac in training_fractions:
    train_size = int(len(ts) * frac)
    train_ts = ts.iloc[:train_size]
    test_ts = ts.iloc[train_size:]
    
    for order in orders_list:
        try:
            # Обучаем модель ARIMA с заданным порядком на обучающем наборе
            model = ARIMA(train_ts, order=order)
            model_fit = model.fit()
            
            # Прогнозируем следующие len(test_ts) шагов (одно шаговое предсказание за раз)
            forecast = model_fit.forecast(steps=len(test_ts))
            
            # Вычисляем метрики: MAE и дисперсию предсказанных значений
            mae_value = mean_absolute_error(test_ts, forecast)
            variance_value = np.var(forecast)
            
            results.append({
                'training_fraction': frac,
                'order': str(order),
                'mae': mae_value,
                'variance': variance_value
            })
            
            print(f"Train Fraction: {frac}, Order: {order}, MAE: {mae_value:.4f}, Variance: {variance_value:.4f}")
        except Exception as e:
            print(f"Train Fraction: {frac}, Order: {order}, Exception: {e}")
            results.append({
                'training_fraction': frac,
                'order': str(order),
                'mae': None,
                'variance': None
            })

# 7. Сохранение результатов эксперимента в CSV файл
results_df = pd.DataFrame(results)
results_df.to_csv("arima_experiment_results.csv", index=False)
print("Результаты эксперимента сохранены в 'arima_experiment_results.csv'")
