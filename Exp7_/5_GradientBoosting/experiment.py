import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

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
n_estimators_list = [10, 50, 100, 200, 500, 1000]

results = []

# Запускаем эксперимент
for test_size in test_sizes:
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)

    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]

    for n_estimators in n_estimators_list:
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        variance = np.var(y_pred)

        results.append({
            "test_size": test_size,
            "n_estimators": n_estimators,
            "mae": mae,
            "variance": variance
        })

        print(f"Test size: {test_size}, Trees: {n_estimators}, MAE: {mae:.4f}, Variance: {variance:.4f}")

# Сохраняем результаты
results_df = pd.DataFrame(results)
results_df.to_csv("gradient_boosting_results.csv", index=False)
print("Результаты сохранены в 'gradient_boosting_results.csv'")
