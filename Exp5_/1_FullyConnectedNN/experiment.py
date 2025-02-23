import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 1. Загрузка датасета (замените путь на ваш)
file_path = r'Exp5_\data\cleaned_weather.csv'  # <-- Укажите актуальный путь к файлу
data = pd.read_csv(file_path)

# 2. Преобразуем столбец 'date' в datetime и устанавливаем его в качестве индекса
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 3. Выбираем 20 метеорологических показателей (без столбцов 'date' и 'OT')
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

# Преобразуем входные данные в 2D для полносвязной сети
num_features = X.shape[2]

# Списки для экспериментов
training_fractions = [0.1, 0.2, 0.5, 0.8, 0.9]
layer_counts = [1, 2, 4, 6, 8, 10]

results = []

# Цикл по различным размерам обучающей выборки
for frac in training_fractions:
    train_size = int(len(X) * frac)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Преобразуем в 2D для полносвязной сети
    X_train_flat = X_train.reshape(X_train.shape[0], look_back * num_features)
    X_test_flat = X_test.reshape(X_test.shape[0], look_back * num_features)
    
    # Цикл по количеству скрытых слоев
    for n_layers in layer_counts:
        # Строим модель с n_layers скрытых слоев, каждый с 64 нейронами и ReLU-активацией.
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(look_back * num_features,)))
        for i in range(n_layers - 1):
            model.add(Dense(64, activation='relu'))
        model.add(Dense(1))  # Выходной слой: предсказываем температуру 'T'
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Обучение модели (50 эпох, батч=32, вывод обучения выключен для краткости)
        history = model.fit(X_train_flat, y_train, epochs=50, batch_size=32, verbose=0,
                            validation_data=(X_test_flat, y_test))
        
        # Предсказание на тестовой выборке
        y_pred = model.predict(X_test_flat)
        
        # Вычисляем MAE и дисперсию предсказаний
        mae_value = mean_absolute_error(y_test, y_pred)
        variance_value = np.var(y_pred)
        
        results.append({
            'training_fraction': frac,
            'n_layers': n_layers,
            'mae': mae_value,
            'variance': variance_value
        })
        
        print(f"Train Fraction: {frac}, Layers: {n_layers}, MAE: {mae_value:.4f}, Variance: {variance_value:.4f}")

# Записываем результаты в CSV файл
results_df = pd.DataFrame(results)
results_df.to_csv("experiment_results.csv", index=False)
print("Результаты эксперимента сохранены в 'experiment_results.csv'")
