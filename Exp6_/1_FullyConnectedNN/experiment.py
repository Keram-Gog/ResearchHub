import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

# Для воспроизводимости результатов
np.random.seed(42)
tf.random.set_seed(42)

# Загрузка и предобработка данных
data_path = r"Exp6_\data\data.csv"
df = pd.read_csv(data_path)

# Приводим дату к формату datetime и сортируем по дате
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')

# Очистка цены закрытия: убираем знак "$" и преобразуем в float
df['Close'] = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

# Создаем лаговый признак: значение цены закрытия предыдущего дня
df['Lag1'] = df['Close'].shift(1)
df = df.dropna().reset_index(drop=True)

# Определяем признаки и целевую переменную
X = df[['Lag1']].values    # Признак – значение предыдущего дня
y = df['Close'].values       # Целевая переменная – текущая цена закрытия

# Параметры экспериментов
train_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]         # Доля данных для обучения
num_layers_list = [1, 2, 4, 6, 8, 10]             # Количество скрытых слоев

results = []
n = len(df)

# Параметры обучения модели
epochs = 50
batch_size = 32

# Проведение экспериментов
for train_size in train_sizes:
    n_train = int(n * train_size)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    for num_layers in num_layers_list:
        try:
            # Создание модели: входной размер 1, указанное число скрытых слоев,
            # каждый слой имеет 32 нейрона с активацией ReLU, и выходной слой - 1 нейрон.
            model = Sequential()
            # Первый скрытый слой с указанием input_dim
            model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
            # Добавляем оставшиеся скрытые слои
            for _ in range(num_layers - 1):
                model.add(Dense(32, activation='relu'))
            # Выходной слой для регрессии
            model.add(Dense(1, activation='linear'))
            
            # Компиляция модели
            model.compile(optimizer=Adam(), loss='mse')
            
            # Обучение модели
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Прогнозирование для тестовой выборки
            y_pred = model.predict(X_test).flatten()
            
            # Вычисление ошибок
            errors = y_pred - y_test
            variance_error = np.var(errors)
            mae_error = mean_absolute_error(y_test, y_pred)
            
            results.append({
                'train_size': train_size,
                'num_layers': num_layers,
                'variance': variance_error,
                'MAE': mae_error
            })
        except Exception as e:
            print(f"Ошибка для train_size={train_size} и num_layers={num_layers}: {e}")
            results.append({
                'train_size': train_size,
                'num_layers': num_layers,
                'variance': np.nan,
                'MAE': np.nan
            })

# Сохранение результатов экспериментов в CSV-файл в той же папке
results_df = pd.DataFrame(results)
output_path = r"Exp6_\1_FullyConnectedNN\experiment_results_fully_connected.csv"
results_df.to_csv(output_path, index=False)

print("Результаты экспериментов сохранены в", output_path)
