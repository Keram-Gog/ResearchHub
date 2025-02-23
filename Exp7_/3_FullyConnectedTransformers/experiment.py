import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Загружаем датасет
file_path = r'Exp7_\data\MSFT_Monthly_stock_prizes.csv'
data = pd.read_csv(file_path)

# Переименовываем столбцы
data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Преобразуем "Date" в datetime и сортируем
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

# Выбираем признаки и целевую переменную
features = ["Open", "High", "Low", "Volume"]
target = "Close"

# Экспериментальные параметры
test_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]
num_layers_list = [1, 2, 4, 6, 8, 10]  # Количество слоев трансформера
sequence_length = 10  # Длина временного окна

# Нормализация данных
scaler = StandardScaler()

# Функция для создания временных последовательностей
def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y.iloc[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

# Функция создания трансформера с разным количеством слоев
def build_transformer_model(input_shape, num_layers, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Добавляем num_layers слоев трансформера
    for _ in range(num_layers):
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs  # Skip connection

        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=input_shape[-1], kernel_size=1)(x)
        x = x + res  # Skip connection

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model

results = []

# Запускаем эксперимент
for test_size in test_sizes:
    print(f"\n🔹 Запуск эксперимента с test_size={test_size}...")

    # Разбиение на train/test без перемешивания
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]

    # Проверяем размер данных перед масштабированием
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}, Sequence length: {sequence_length}")

    if X_train.shape[0] <= sequence_length or X_test.shape[0] <= sequence_length:
        print(f"⚠️ Пропускаем test_size={test_size}, так как данных меньше sequence_length.")
        continue

    # Масштабируем данные
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Преобразуем данные в последовательности
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)

    # Перебираем количество слоев в трансформере
    for num_layers in num_layers_list:
        print(f"\n🟢 Обучение модели с {num_layers} слоями...")

        model = build_transformer_model(input_shape=(sequence_length, len(features)), num_layers=num_layers)
        
        # Обучаем модель
        model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=1)

        # Прогнозируем
        print(f"🔵 Прогнозирование для {num_layers} слоев...")
        y_pred = model.predict(X_test_seq)

        # Вычисляем метрики: MAE и дисперсию предсказаний
        mae = mean_absolute_error(y_test_seq, y_pred)
        variance = np.var(y_pred)

        results.append({
            "test_size": test_size,
            "num_layers": num_layers,
            "mae": mae,
            "variance": variance
        })

        print(f"✅ Test size: {test_size}, Layers: {num_layers}, MAE: {mae:.4f}, Variance: {variance:.4f}")

# Сохраняем результаты
results_df = pd.DataFrame(results)
results_df.to_csv("transformer_results.csv", index=False)
print("📁 Результаты сохранены в 'transformer_results.csv'")
