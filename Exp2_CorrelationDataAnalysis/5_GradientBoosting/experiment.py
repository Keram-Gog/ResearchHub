import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# 1. Загрузка и подготовка данных
try:
    data = pd.read_csv(
        'D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp2_CorrelationDataAnalysis\\data\\nifty_500.csv', 
        sep=','
    )
    print("Данные успешно загружены!")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])
y = data['Last Traded Price']

# Преобразование категориальных переменных в числовые (one-hot encoding)
X = pd.get_dummies(X)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Определение параметров эксперимента
num_estimators_options = [50, 100, 200, 300, 500]
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]
depth_options = [1, 2, 4, 6, 8, 10]  # Значения глубины деревьев (слоев)
results = []

# 3. Проведение эксперимента
for n_estimators in num_estimators_options:
    for max_depth in depth_options:
        for test_size in test_size_options:
            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            
            # Инициализация и обучение модели градиентного бустинга
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, 
                learning_rate=0.1, 
                max_depth=max_depth, 
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Предсказания на тестовой выборке
            y_pred = model.predict(X_test)
            
            # Вычисление метрик качества
            mae = mean_absolute_error(y_test, y_pred)
            variance = np.var(y_pred - y_test)
            
            # Запись результатов эксперимента
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'test_size': test_size,
                'MAE': mae,
                'Variance': variance
            })
            
            print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, test_size: {test_size}, "
                  f"MAE: {mae:.4f}, Variance: {variance:.4f}")

# 4. Сохранение результатов в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Exp2_CorrelationDataAnalysis\\5_GradientBoosting\\experiment_results.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results.csv'.")
