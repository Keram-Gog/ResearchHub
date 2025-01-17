import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, MultiHeadAttention, GlobalAveragePooling1D

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
sequence_length = 30

def create_sequences(data, input_features, target_columns, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[input_features].iloc[i:i+seq_length].values)
        y.append(data[target_columns].iloc[i+seq_length].values)
    return np.array(X), np.array(y)

X, y = create_sequences(data, input_features, columns_to_predict, sequence_length)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# Функция для создания модели с трансформером (с 8 слоями)
def create_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Добавление нескольких слоев MultiHeadAttention
    x = inputs
    for _ in range(8):  # 8 слоев внимания
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
model = create_transformer_model(input_shape)

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)

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
