import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Подавление предупреждений (опционально)
warnings.filterwarnings("ignore")

# 1. Загрузка данных
try:
    data = pd.read_csv(r"D:\main for my it\my tasks\source\ResearchHub\Exp4_TimeSeriesPrediction\data\Microsoft_Stock.csv", sep=',')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# 2. Преобразование даты, сортировка и установка в качестве индекса
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")
data.set_index("Date", inplace=True)

# 3. Выбор временного ряда для прогнозирования: столбец "Close"
series = data["Close"]

# 4. Разбиение на обучающую и тестовую выборки:
# Фиксируем тестовую выборку как последние 20% наблюдений (это остаётся неизменным во всех экспериментах)
fixed_test_fraction = 0.2
n_test = int(len(series) * fixed_test_fraction)
if n_test < 1:
    print("Тестовая выборка слишком мала.")
    exit()
test = series[-n_test:]
train_full = series[:-n_test]

# 5. Функция прореживания обучающей выборки по месяцам
def thin_training_series(series, thinning_fraction, random_state=42):
    """
    Для каждого месяца из series (с индексом datetime) случайным образом устанавливаем в NaN заданную долю наблюдений,
    а затем заполняем пропуски линейной интерполяцией.
    """
    df = series.copy().to_frame(name='Close')
    # Группируем по месяцам
    groups = df.groupby(pd.Grouper(freq="M"))
    np.random.seed(random_state)
    for group_name, group in groups:
        if len(group) > 0:
            n_to_remove = int(np.floor(len(group) * thinning_fraction))
            if n_to_remove > 0:
                indices_to_remove = np.random.choice(group.index, size=n_to_remove, replace=False)
                df.loc[indices_to_remove, 'Close'] = np.nan
    # Заполнение пропусков методом линейной интерполяции
    df['Close'] = df['Close'].interpolate(method='linear', limit_direction='forward')
    return df['Close']

# 6. Определение параметров эксперимента
layer_options = [1, 2, 4, 6, 8, 10]           # Значения p для ARIMA (аналог количества слоёв)
thinning_options = [0.1, 0.2, 0.5, 0.8, 0.9]    # Доля пропущенных (удалённых) дней в каждом месяце обучающей выборки
results = []

# Фиксированные параметры ARIMA
d = 1
q = 1

# 7. Запуск эксперимента:
for p in layer_options:
    for thinning_fraction in thinning_options:
        # Прореживаем обучающую выборку заданной долей по месяцам и заполняем пропуски
        train_thinned = thin_training_series(train_full, thinning_fraction, random_state=42)
        
        order = (p, d, q)
        try:
            model = ARIMA(train_thinned, order=order)
            model_fit = model.fit()
        except Exception as e:
            print(f"Ошибка при обучении ARIMA с order={order} и thinning_fraction={thinning_fraction}: {e}")
            continue

        # Прогнозирование на период тестовой выборки (фиксированного объёма)
        forecast = model_fit.forecast(steps=n_test)

        # Вычисление MAE и дисперсии прогнозов
        mae = mean_absolute_error(test, forecast)
        forecast_variance = np.var(forecast)

        results.append({
            'num_layers': p,
            'thinning_fraction': thinning_fraction,
            'order': str(order),
            'mae': mae,
            'forecast_variance': forecast_variance
        })

        print(f"Num_layers: {p}, Thinning: {thinning_fraction}, Order: {order}, MAE: {mae:.4f}, Variance: {forecast_variance:.4f}")

# 8. Сохранение результатов в CSV
output_path = r"Exp4_TimeSeriesPrediction\6_ARIMA\experiment_results_arima_updated.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"\nЭксперименты с ARIMA (обновлённые) завершены. Результаты сохранены в '{output_path}'.")
