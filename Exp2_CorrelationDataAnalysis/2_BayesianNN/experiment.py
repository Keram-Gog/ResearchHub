import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import sys

# Подключаем путь к BayeFormers
sys.path.append(r'D:\main for my it\my tasks\source\ResearchHub\BayeFormers-master')
from bayeformers import to_bayesian
import bayeformers.nn as bnn

# Загрузка и подготовка данных
data = pd.read_csv('D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp2_CorrelationDataAnalysis\\data\\nifty_500.csv', sep=',')
print("Данные успешно загружены!")

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])
y = data['Last Traded Price']

# Преобразование категориальных переменных в числовые (one-hot encoding)
X = pd.get_dummies(X)

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Фиксируем random_state для воспроизводимости
random_state = 42

# Определение параметров эксперимента
num_layers_options = [1, 2, 4, 6, 8, 10]        # Возможные значения для количества слоёв
test_size_options = [0.1, 0.2, 0.5, 0.8, 0.9]     # Возможные значения для тестовой выборки
epochs = 100                                      # Число эпох обучения

# Список для хранения результатов эксперимента
results = []

# Функция для подгонки размера обучающей выборки, если он больше доступных данных
def adjust_train_size(X, y, test_size):
    num_train_samples = int(len(X) * (1 - test_size))
    return X[:num_train_samples], y[:num_train_samples]

# Класс исходной регрессионной модели с динамическим числом скрытых слоёв
class RegressionModel(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(RegressionModel, self).__init__()
        layers = []
        current_dim = input_dim
        # Формируем скрытые слои
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, current_dim // 2))
            layers.append(nn.ReLU())
            current_dim //= 2  # уменьшаем размерность в два раза
        # Выходной слой
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Перебор комбинаций параметров эксперимента
for num_layers in num_layers_options:
    for test_size in test_size_options:
        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        # Подгонка размера обучающей выборки (если требуется)
        X_train, y_train = adjust_train_size(X_train, y_train, test_size)

        # Преобразование данных в тензоры
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Инициализация исходной модели
        input_dim = X_train_tensor.shape[1]
        model = RegressionModel(input_dim, num_layers)

        # Преобразование модели в байесовскую с помощью BayeFormers
        bayesian_model = to_bayesian(model, delta=0.05, freeze=True)

        # Настройка оптимизатора и функции потерь
        optimizer = torch.optim.Adam(bayesian_model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        # Обучение модели
        for epoch in range(epochs):
            bayesian_model.train()
            optimizer.zero_grad()
            predictions = bayesian_model(X_train_tensor)
            loss = loss_fn(predictions, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Оценка модели на тестовой выборке
        bayesian_model.eval()
        with torch.no_grad():
            test_predictions = bayesian_model(X_test_tensor)
            # Вычисляем дисперсию предсказаний
            test_variance = test_predictions.var().item()
            # Вычисляем MAE
            test_mae = mean_absolute_error(y_test, test_predictions.numpy())

        # Сохранение результатов эксперимента
        results.append({
            'num_layers': num_layers,
            'test_size': test_size,
            'test_variance': test_variance,
            'test_mae': test_mae
        })

        print(f"Слои: {num_layers}, Тестовая выборка: {test_size}, MAE: {test_mae:.4f}, Variance: {test_variance:.4f}")

# Создание таблицы результатов и сохранение в CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Exp2_CorrelationDataAnalysis\\2_BayesianNN\\experiment_results.csv', index=False)
print("\nЭксперименты завершены. Результаты сохранены в 'experiment_results.csv'.")
