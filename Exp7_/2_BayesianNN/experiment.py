import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import sys

sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')
from bayeformers import to_bayesian

# Загружаем датасет
file_path = r'Exp7_\data\MSFT_Monthly_stock_prizes.csv'
data = pd.read_csv(file_path)

# Переименовываем столбцы для удобства
data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Преобразуем "Date" в формат datetime и сортируем
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

# Выбираем признаки (используем предыдущие значения цен и объемов)
features = ["Open", "High", "Low", "Volume"]
target = "Close"

# Экспериментальные параметры
test_sizes = [0.1, 0.2, 0.5, 0.8, 0.9]
hidden_layer_sizes_list = [(10,), (50,), (100,), (200,), (300,)]  # Списки размеров скрытых слоев

results = []

# Нормализация данных
scaler = StandardScaler()

# Запускаем эксперимент
for test_size in test_sizes:
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]

    # Масштабируем данные
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for hidden_layer_sizes in hidden_layer_sizes_list:
        # Создаем и обучаем модель
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=42, max_iter=1000)
        model = to_bayesian(model, delta=0.05, freeze=True)

        model.fit(X_train_scaled, y_train)

        # Прогнозируем
        y_pred = model.predict(X_test_scaled)

        # Вычисляем метрики: MAE и дисперсию предсказаний
        mae = mean_absolute_error(y_test, y_pred)
        variance = np.var(y_pred)

        results.append({
            "test_size": test_size,
            "hidden_layer_sizes": hidden_layer_sizes,
            "mae": mae,
            "variance": variance
        })

        print(f"Test size: {test_size}, Hidden Layers: {hidden_layer_sizes}, MAE: {mae:.4f}, Variance: {variance:.4f}")

# Сохраняем результаты
results_df = pd.DataFrame(results)
results_df.to_csv(r"D:\main for my it\my tasks\source\ResearchHub\Exp7_\1_FullyConnectedNN\results.csv", index=False)
print("Результаты сохранены в 'mlp_results.csv'")
