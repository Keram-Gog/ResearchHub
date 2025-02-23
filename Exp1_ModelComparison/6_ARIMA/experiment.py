import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Подавление предупреждений (опционально)
warnings.filterwarnings("ignore")

# 1. Загрузка данных
try:
    data = pd.read_csv('Exp1_ModelComparison/data/student-mat.csv', sep=';')
    print("Данные успешно загружены!")
except Exception as e:
    print("Ошибка при загрузке данных:", e)
    exit()

# 2. Извлечение временного ряда целевой переменной
# Предполагается, что записи упорядочены по времени (если нет, отсортируйте их соответствующим образом)
y = data['G3']

# 3. Определение параметров эксперимента
# Здесь параметр p (аналог количества слоёв) будем перебирать по списку
layer_options = [1, 2, 4, 6, 8, 10]  # p в ARIMA, аналог количества слоёв
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Размер тестовой выборки
results = []

# Фиксированные параметры для ARIMA: d и q
d = 1
q = 1

# 4. Запуск эксперимента: для каждой комбинации параметра p и test_size
for num_layers in layer_options:
    for test_size in test_size_options:
        n_test = int(len(y) * test_size)
        if n_test < 1:
            print(f"Тестовая выборка слишком мала для test_size = {test_size}. Пропуск эксперимента.")
            continue

        # Для временного ряда используем последние n_test наблюдений для теста
        train = y[:-n_test]
        test = y[-n_test:]

        # Определяем порядок ARIMA: p = num_layers, d = 1, q = 1
        order = (num_layers, d, q)

        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
        except Exception as e:
            print(f"Ошибка при обучении ARIMA с order={order} для test_size = {test_size}: {e}")
            continue

        # Прогнозирование на n_test шагов вперед
        forecast = model_fit.forecast(steps=n_test)

        # Вычисляем MAE и дисперсию прогнозов
        mae = mean_absolute_error(test, forecast)
        forecast_variance = np.var(forecast)

        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'order': str(order),
            'mae': mae,
            'forecast_variance': forecast_variance
        })

        print(f"Num_layers: {num_layers}, Test size: {test_size}, Order: {order}, MAE: {mae:.4f}, Variance: {forecast_variance:.4f}")

# 5. Сохранение результатов в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Exp1_ModelComparison/6_ARIMA/experiment_results_arima.csv', index=False)
print("\nЭксперименты с ARIMA завершены. Результаты сохранены в 'experiment_results_arima.csv'.")
