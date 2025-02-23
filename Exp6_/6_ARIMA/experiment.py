import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# Загрузка данных
data_path = r"Exp6_\data\data.csv"
df = pd.read_csv(data_path)

# Преобразование столбца с датой в datetime и сортировка по дате
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')

# Очистка столбца цены: удаляем знак "$" и преобразуем в float
df['Close'] = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

# Извлекаем временной ряд (например, цены закрытия)
ts = df['Close'].values

# Задаем параметры экспериментов
train_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]          # доля данных для обучения
num_layers_list = [1, 2, 4, 6, 8, 10]              # интерпретируем как порядок AR (p)

results = []

# Запускаем эксперименты
for train_size in train_sizes:
    n_train = int(len(ts) * train_size)
    train, test = ts[:n_train], ts[n_train:]
    
    for p in num_layers_list:
        try:
            # Создаем и обучаем модель ARIMA с параметрами (p, d=1, q=0)
            model = ARIMA(train, order=(p, 1, 0))
            model_fit = model.fit()
            
            # Делаем прогноз на период тестовой выборки
            forecast = model_fit.forecast(steps=len(test))
            
            # Вычисляем ошибки: разница между прогнозом и фактическими значениями
            errors = forecast - test
            
            # Дисперсия ошибок
            variance_error = np.var(errors)
            # MAE (средняя абсолютная ошибка)
            mae_error = np.mean(np.abs(errors))
            
            results.append({
                'train_size': train_size,
                'num_layers': p,
                'variance': variance_error,
                'MAE': mae_error
            })
        except Exception as e:
            print(f"Ошибка для train_size={train_size} и num_layers={p}: {e}")
            results.append({
                'train_size': train_size,
                'num_layers': p,
                'variance': np.nan,
                'MAE': np.nan
            })

# Сохраняем результаты экспериментов в CSV-файл в той же папке, что и исходный датасет
results_df = pd.DataFrame(results)
output_path = r"Exp6_\data\experiment_results.csv"
results_df.to_csv(output_path, index=False)

print("Результаты экспериментов сохранены в", output_path)
