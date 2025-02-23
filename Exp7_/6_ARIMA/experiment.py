import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# 1. Загрузка датасета (замените путь на актуальный)
file_path = r'Exp7_\data\MSFT_Monthly_stock_prizes.csv'
data = pd.read_csv(file_path)

# 2. Преобразуем столбец 'Date' в datetime и устанавливаем его в качестве индекса
data.rename(columns={
    'Unnamed: 0': 'Date',
    'Openning Values': 'Open',
    'Highest Values': 'High',
    'Lowest Values': 'Low',
    'Closing Values': 'Close',
    'Volumes of Stocks': 'Volume'
}, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

ts = data['Close']  # Используем закрытые цены акций для ARIMA

# 3. Параметры эксперимента:
test_fractions = [0.1, 0.2, 0.5, 0.8, 0.9]  # Доли тестовой выборки
layers_list = [1, 2, 4, 6, 8, 10]  # Количество слоев (p-параметр в ARIMA)

d = 1  # Фиксируем порядок дифференцирования
q = 1  # Фиксируем порядок MA

results = []

# 4. Эксперимент: перебор долей тестовой выборки и количества слоев
for frac in test_fractions:
    test_size = int(len(ts) * frac)
    train_ts = ts.iloc[:-test_size]
    test_ts = ts.iloc[-test_size:]
    
    for layers in layers_list:
        try:
            order = (layers, d, q)
            
            # Обучаем модель ARIMA
            model = ARIMA(train_ts, order=order)
            model_fit = model.fit()
            
            # Прогнозируем
            forecast = model_fit.forecast(steps=len(test_ts))
            
            # Вычисляем метрики: MAE и дисперсию предсказаний
            mae_value = mean_absolute_error(test_ts, forecast)
            variance_value = np.var(forecast)
            
            results.append({
                'test_fraction': frac,
                'layers': layers,
                'mae': mae_value,
                'variance': variance_value
            })
            
            print(f"Test Fraction: {frac}, Layers: {layers}, MAE: {mae_value:.4f}, Variance: {variance_value:.4f}")
        except Exception as e:
            print(f"Test Fraction: {frac}, Layers: {layers}, Exception: {e}")
            results.append({
                'test_fraction': frac,
                'layers': layers,
                'mae': None,
                'variance': None
            })

# 5. Сохранение результатов в CSV
results_df = pd.DataFrame(results)
results_df.to_csv("Exp7_\\6_ARIMA\\arima_msft_results.csv", index=False)
print("Результаты эксперимента сохранены в 'arima_msft_results.csv'")
