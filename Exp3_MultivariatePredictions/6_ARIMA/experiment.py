import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Подавление предупреждений (опционально)
warnings.filterwarnings("ignore")

# 1. Загрузка данных
try:
    data = pd.read_csv('D:\\питон\\MO\\1\\этот\\global_mean_sea_level_1993-2024.csv', sep=',')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# 2. Нормализация целевых столбцов (как в других экспериментах)
columns_to_predict = ['GMSLNoGIA', 'SmoothedGMSLWithGIA', 'SmoothedGMSLWithGIASigremoved']
scaler = MinMaxScaler()
data[columns_to_predict] = scaler.fit_transform(data[columns_to_predict])

# Выбираем для ARIMA только один временной ряд – например, 'GMSLNoGIA'
series = data['GMSLNoGIA']

# 3. Определение параметров эксперимента
layer_options = [1, 2, 4, 6, 8, 10]       # Значения для параметра p (аналог количества слоёв)
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Доли тестовой выборки
results = []

# Фиксированные параметры ARIMA
d = 1
q = 1

# 4. Запуск эксперимента
for p in layer_options:
    for test_size in test_size_options:
        n_test = int(len(series) * test_size)
        if n_test < 1:
            print(f"Test size слишком мал для test_size = {test_size}. Пропуск эксперимента.")
            continue

        # Для временных рядов используем последние n_test наблюдений для теста
        train = series[:-n_test]
        test = series[-n_test:]
        
        order = (p, d, q)
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
        except Exception as e:
            print(f"Ошибка при обучении ARIMA с order={order} и test_size={test_size}: {e}")
            continue

        # Прогнозирование на n_test шагов вперёд
        forecast = model_fit.forecast(steps=n_test)

        # Вычисление метрики MAE и дисперсии прогнозов
        mae = mean_absolute_error(test, forecast)
        forecast_variance = np.var(forecast)

        results.append({
            'num_layers': p,
            'test_size': test_size,
            'order': str(order),
            'mae': mae,
            'forecast_variance': forecast_variance
        })

        print(f"Num_layers: {p}, Test_size: {test_size}, Order: {order}, MAE: {mae:.4f}, Variance: {forecast_variance:.4f}")

# 5. Сохранение результатов в CSV
output_path = 'Exp3_MultivariatePredictions/6_ARIMA/experiment_results_arima.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"\nЭксперименты с ARIMA завершены. Результаты сохранены в '{output_path}'.")
