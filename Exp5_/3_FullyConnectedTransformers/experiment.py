import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Add, Embedding

# --------------------------
# 1. Загрузка и предобработка данных
# --------------------------

# Замените путь на актуальный путь к вашему файлу
file_path = r'Exp5_\data\cleaned_weather.csv'
data = pd.read_csv(file_path)

# Преобразуем столбец 'date' в datetime и устанавливаем его в качестве индекса
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Выбираем 20 метеорологических показателей
features = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 
            'sh', 'H2OC', 'rho', 'wv', 'max. wv', 'wd', 'rain', 
            'raining', 'SWDR', 'PAR', 'max. PAR', 'Tlog']
data = data[features]

# Нормализуем данные
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --------------------------
# 2. Формирование обучающих примеров с использованием скользящего окна
# --------------------------

def create_dataset(data_array, look_back=24):
    X, y = [], []
    # Будем предсказывать следующий шаг для столбца 'T' (индекс 1)
    for i in range(len(data_array) - look_back):
        X.append(data_array[i : i + look_back])
        y.append(data_array[i + look_back, 1])
    return np.array(X), np.array(y)

look_back = 24  # Количество временных шагов в окне
X, y = create_dataset(data_scaled, look_back)
# Исходная форма X: (num_samples, look_back, num_features)
input_shape = X.shape[1:]  # (look_back, num_features)

# --------------------------
# 3. Определение функции построения модели-трансформера
# --------------------------

def build_transformer_model(n_layers, input_shape, head_size=64, num_heads=4, ff_dim=64, dropout=0.1):
    """
    Строит модель трансформера для временных рядов.
    :param n_layers: число трансформерных блоков
    :param input_shape: форма входного тензора (look_back, num_features)
    :param head_size: размерность ключей в MultiHeadAttention
    :param num_heads: число голов
    :param ff_dim: размерность внутреннего слоя feed-forward
    :param dropout: dropout для блоков
    :return: скомпилированная модель
    """
    inputs = Input(shape=input_shape)  # (look_back, num_features)
    
    # Проецируем вход в пространство размерности head_size
    x = Dense(head_size)(inputs)
    
    # Добавляем обучаемое позиционное кодирование
    # Создаем позиционные индексы и передаем через Embedding (получаем форму: (look_back, head_size))
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embedding = Embedding(input_dim=input_shape[0], output_dim=head_size)(positions)
    x = x + pos_embedding
    
    # Применяем n_layers трансформерных блоков
    for _ in range(n_layers):
        # --- MultiHeadAttention ---
        attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        attn_output = Dropout(dropout)(attn_output)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # --- Feed-Forward Network ---
        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dropout(dropout)(ff_output)
        ff_output = Dense(head_size)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Глобальное усреднение по временной оси
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)  # Выход: предсказание температуры
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --------------------------
# 4. Эксперимент: перебор долей обучающей выборки и числа трансформерных блоков
# --------------------------

training_fractions = [0.1, 0.2, 0.5, 0.8, 0.9]
layer_counts = [1, 2, 4, 6, 8, 10]

results = []

for frac in training_fractions:
    train_size = int(len(X) * frac)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Для трансформера оставляем исходную форму: (num_samples, look_back, num_features)
    for n_layers in layer_counts:
        model = build_transformer_model(n_layers, input_shape, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)
        
        # Обучаем модель (50 эпох, батч=32, вывод обучения выключен для краткости)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0,
                            validation_data=(X_test, y_test))
        
        # Предсказание
        y_pred = model.predict(X_test)
        
        mae_value = mean_absolute_error(y_test, y_pred)
        variance_value = np.var(y_pred)
        
        results.append({
            'training_fraction': frac,
            'n_layers': n_layers,
            'mae': mae_value,
            'variance': variance_value
        })
        
        print(f"Train Fraction: {frac}, Layers: {n_layers}, MAE: {mae_value:.4f}, Variance: {variance_value:.4f}")

# --------------------------
# 5. Сохранение результатов эксперимента в CSV
# --------------------------

results_df = pd.DataFrame(results)
results_df.to_csv("Exp5_\\3_FullyConnectedTransformers\\transformer_experiment_results.csv", index=False)
print("Результаты эксперимента сохранены в 'transformer_experiment_results.csv'")
