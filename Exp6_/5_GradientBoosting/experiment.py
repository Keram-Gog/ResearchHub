import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# Загрузка и предобработка данных
data_path = r"Exp6_\data\data.csv"
df = pd.read_csv(data_path)

# Приведение столбца даты к формату datetime и сортировка
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')

# Очистка цены закрытия: удаляем знак "$" и преобразуем в float
df['Close'] = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

# Создаем лаговый признак: для каждой строки берем значение цены закрытия предыдущего дня
df['Lag1'] = df['Close'].shift(1)
df = df.dropna().reset_index(drop=True)

# Определяем признаки и целевую переменную
X = df[['Lag1']].values   # признак – значение предыдущего дня
y = df['Close'].values      # целевая переменная – текущая цена закрытия

# Параметры экспериментов
train_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]         # доля данных для обучения
max_depths = [1, 2, 4, 6, 8, 10]                   # интерпретируем как число "слоёв"

results = []
n = len(df)

# Проведение экспериментов
for train_size in train_sizes:
    n_train = int(n * train_size)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    for depth in max_depths:
        try:
            # Инициализация и обучение модели градиентного бустинга
            model = GradientBoostingRegressor(max_depth=depth, n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Прогнозирование для тестовой выборки
            y_pred = model.predict(X_test)
            
            # Вычисление ошибок
            errors = y_pred - y_test
            variance_error = np.var(errors)
            mae_error = mean_absolute_error(y_test, y_pred)
            
            results.append({
                'train_size': train_size,
                'max_depth': depth,
                'variance': variance_error,
                'MAE': mae_error
            })
        except Exception as e:
            print(f"Ошибка для train_size={train_size} и max_depth={depth}: {e}")
            results.append({
                'train_size': train_size,
                'max_depth': depth,
                'variance': np.nan,
                'MAE': np.nan
            })

# Сохранение результатов экспериментов в CSV-файл в той же папке
results_df = pd.DataFrame(results)
output_path = r"Exp6_\5_GradientBoosting\experiment_results_gradient_boosting.csv"
results_df.to_csv(output_path, index=False)

print("Результаты экспериментов сохранены в", output_path)
