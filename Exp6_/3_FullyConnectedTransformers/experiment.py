import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

# Для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

# Функция для создания обучающего набора на основе скользящего окна
def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Определяем блок трансформера (энкодер)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Многоголовое внимание
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([inputs, x])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward
    x_ff = layers.Dense(ff_dim, activation="relu")(x)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)
    x_ff = layers.Dropout(dropout)(x_ff)
    x = layers.Add()([x, x_ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# Функция для построения модели трансформера
def build_transformer_model(window_size, num_layers, head_size=32, num_heads=2, ff_dim=32, dropout=0.1):
    inputs = Input(shape=(window_size, 1))
    # Проецируем вход в пространство размерности head_size (embed_dim)
    x = layers.Dense(head_size)(inputs)
    
    # Применяем заданное число трансформерных энкодерных блоков
    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Усреднение по временной оси
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="linear")(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss="mse")
    return model

# Загрузка и предобработка данных
data_path = r"Exp6_\data\data.csv"
df = pd.read_csv(data_path)

# Приводим столбец даты к формату datetime и сортируем
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')

# Очистка столбца цены закрытия: убираем знак "$" и преобразуем в float
df['Close'] = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)

# Извлекаем временной ряд цен закрытия
ts = df['Close'].values

# Параметры для формирования последовательностей
window_size = 10  # длина окна (можно изменить)
X_all, y_all = create_dataset(ts, window_size)

# Изменяем форму признаков для соответствия входу модели (samples, window_size, 1)
X_all = X_all.reshape(-1, window_size, 1)

# Параметры экспериментов
train_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]
num_layers_list = [1, 2, 4, 6, 8, 10]

results = []
n_samples = len(X_all)

# Параметры обучения
epochs = 50
batch_size = 32

# Проведение экспериментов
for train_size in train_sizes:
    n_train = int(n_samples * train_size)
    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]
    
    for num_layers in num_layers_list:
        try:
            model = build_transformer_model(window_size, num_layers)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            y_pred = model.predict(X_test).flatten()
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

# Сохранение результатов в CSV-файл
results_df = pd.DataFrame(results)
output_path = r"Exp6_\data\experiment_results_transformer.csv"
results_df.to_csv(output_path, index=False)

print("Результаты экспериментов сохранены в", output_path)
