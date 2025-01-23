import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, MultiHeadAttention, GlobalAveragePooling1D

# Установка фиксированного random seed
random_seed = 42
np.random.seed(random_seed)

# Получение параметров от пользователя
train_size = int(input("Введите размер обучающей выборки (например, 100): "))  # Размер обучающей выборки
test_size = 100  # Размер тестовой выборки фиксирован
sequence_length = int(input("Введите длину временной последовательности (например, 30): "))
epochs = int(input("Введите количество эпох для обучения (например, 50): "))
num_layers = int(input("Введите количество слоев внимания в модели (например, 8): "))

# Загрузка данных
data = pd.read_csv("D:\\питон\\MO\\1\\этот\\global_mean_sea_level_1993-2024.csv", sep=',')

# Выбор параметров для прогнозирования и входных данных
columns_to_predict = ['GMSLNoGIA', 'SmoothedGMSLWithGIA', 'SmoothedGMSLWithGIASigremoved']
input_features = [
    'YearPlusFraction', 'NumberOfObservations', 'NumberOfWeightedObservations', 
    'StdDevGMSLNoGIA', 'StdDevGMSLWithGIA', 
    'AltimeterType', 'MergedFileCycle', 'SmoothedGMSLNoGia', 'SmoothedGMSLNoGIASigremoved'
]

# Нормализация данных
scaler = MinMaxScaler()
data[input_features + columns_to_predict] = scaler.fit_transform(data[input_features + columns_to_predict])

# Формирование временных шагов
def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

X, y = create_sequences(data, input_features, columns_to_predict, sequence_length)

# Фиксируем тестовую выборку
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_test, _, y_test, _ = train_test_split(X_temp, y_temp, test_size=test_size / (1 - 0.2), random_state=random_seed)

# Преобразование в 2D для полносвязной сети
X_train_2D = X_train.reshape(X_train.shape[0], -1)
X_test_2D = X_test.reshape(X_test.shape[0], -1)

# Функция для создания модели с трансформером
def create_transformer_model(input_shape, num_layers):
    inputs = Input(shape=input_shape)
    
    # Добавление заданного числа слоев MultiHeadAttention
    x = inputs
    for _ in range(num_layers):
        x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    
    # Преобразование: глобальное усреднение по временным шагам
    x = GlobalAveragePooling1D()(x)
    
    # Выходной слой (полносвязный слой)
    outputs = Dense(len(columns_to_predict))(x)
    
    # Компиляция модели
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

# Создание модели
input_shape = (X_train.shape[1], X_train.shape[2])  # Размерность входных данных
model = create_transformer_model(input_shape, num_layers)

# Обучение модели
history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, validation_split=0.2)

# Оценка модели
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}, Average MAE: {test_mae:.4f}")

# Предсказания
y_pred = model.predict(X_test)

# Расчет ошибки (остатков)
errors = y_test - y_pred

# Вычисление общей MAE (средней по всем параметрам)
overall_mae = np.mean(np.abs(errors))

# Вычисление общей дисперсии ошибки
overall_variance = np.mean(np.var(errors, axis=0))

# Вывод результатов
print(f"Overall MAE (mean across all parameters): {overall_mae:.4f}")
print(f"Overall Variance (mean across all parameters): {overall_variance:.4f}")
