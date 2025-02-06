import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# 1. Загрузка данных
try:
    data = pd.read_csv('Exp1_ModelComparison\\data\\student-mat.csv', sep=';')
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# 2. Подготовка данных
X = data.drop(columns=['G3'])  # Признаки
y = data['G3']  # Целевая переменная

# One-hot encoding для категориальных переменных
X = pd.get_dummies(X)

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Определение параметров эксперимента
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]  # Доли тестовой выборки
depth_options = [1, 2, 4, 6, 8, 10]  # Количество слоев (глубина деревьев)
results = []

# 4. Запуск эксперимента
for max_depth in depth_options:
    for test_size in test_size_options:
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        # 5. Обучение модели
        model = lgb.LGBMRegressor(
            n_estimators=500,  # Количество деревьев
            learning_rate=0.05,  # Скорость обучения
            max_depth=max_depth,  # Глубина деревьев
            num_leaves=31,  # Число листьев в дереве
            random_state=42
        )

        model.fit(X_train, y_train)

        # 6. Оценка модели
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        variance = np.var(y_pred - y_test)

        # 7. Сохранение результатов
        results.append({'max_depth': max_depth, 'test_size': test_size, 'MAE': mae, 'Variance': variance})
        print(f"Max Depth: {max_depth}, Test Size: {test_size}, MAE: {mae:.4f}, Variance: {variance:.4f}")

# 8. Сохранение таблицы результатов
results_df = pd.DataFrame(results)
results_df.to_csv('Exp1_ModelComparison\\5_GradientBoosting\\experiment_results_gb.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results_gb.csv'.")
