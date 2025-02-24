import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

class DynamicRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(DynamicRegressionModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(torch.nn.Linear(input_dim, 1))
        self.final_layer = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = F.relu(layer(out))
        return self.final_layer(out)

def to_bayesian(model, delta=0.05, freeze=True):
    # Добавление байесовских параметров, для простоты, вместо реального превращения в байесовский слой
    for param in model.parameters():
        param.requires_grad = not freeze
    return model

def smape(y_true, y_pred):
    """
    Вычисляет Symmetric Mean Absolute Percentage Error (sMAPE), в %.
    Формула:
      sMAPE = 100 * (2 * |y - y_hat|) / (|y| + |y_hat| + 1e-8)
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    # Обработка нулевых значений в true и pred
    y_true = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)  # Минимизируем значения
    y_pred = np.where(np.abs(y_pred) < 1e-8, 1e-8, y_pred)  # Минимизируем значения

    # Ограничиваем величину предсказаний, чтобы не выходили за пределы
    y_true = np.clip(y_true, 1e-8, np.inf)
    y_pred = np.clip(y_pred, 1e-8, np.inf)

    # Вычисление sMAPE
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def introduce_sparsity(X, fraction):
    """
    fraction - доля элементов, которые заменяются на NaN.
    Пример: fraction=0.25 означает, что 25% значений X станут NaN.
    """
    X_sp = X.copy()
    mask = np.random.rand(*X_sp.shape) < fraction
    X_sp[mask] = np.nan
    return X_sp

def main():
    # ======== 4. Загрузка данных ========
    try:
        data = pd.read_csv(
            'D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp2_CorrelationDataAnalysis\\data\\nifty_500.csv',
            sep=','
        )
        print("Данные успешно загружены!")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    # ======== 5. Подготовка X и y ========
    X = data.drop(columns=['Last Traded Price', 'Company Name', 'Symbol'])  # Признаки
    y = data['Last Traded Price']                                          # Целевой столбец
    
    # Преобразуем категориальные в One-Hot
    X = pd.get_dummies(X)
    
    # Масштабируем
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ======== 6. Параметры перебора ========
    size_options = [1.0, 0.5, 0.25, 0.1]       # 100%, 50%, 25%, 10%
    sparsity_options = [0.0, 0.25, 0.5, 0.9]   # 0%, 25%, 50%, 90%
    layer_options = [1, 2, 4, 6, 8, 10, 12]    # Сколько слоёв

    epochs = 100
    learning_rate = 0.0001
    weight_decay = 0.0001

    results = []

    # ======== 7. Цикл по Size / Sparsity / Complexity ========
    for size in size_options:
        for sparsity in sparsity_options:
            # -- Имитация sparsity
            X_sp = introduce_sparsity(X_scaled, fraction=sparsity)
            # -- Заполняем NaN нулями (можно заменить на любой другой способ)
            X_sp = np.nan_to_num(X_sp, nan=0.0)

            # -- train_size
            train_size = int(len(X_sp) * size)
            if train_size < 1 or train_size >= len(X_sp):
                print(f"[Пропуск] size={size} -> train_size={train_size} недопустимо.")
                continue

            # -- Разделяем (без shuffle)
            X_train = X_sp[:train_size]
            y_train = y[:train_size]
            X_test  = X_sp[train_size:]
            y_test  = y[train_size:]

            # -- Тензоры
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
            y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).view(-1, 1)

            input_dim = X_train_t.shape[1]

            for num_layers in layer_options:
                # 1) Создаём классическую сеть
                base_model = DynamicRegressionModel(input_dim, num_layers)
                # 2) Преобразуем её в байесовскую
                bayesian_model = to_bayesian(base_model, delta=0.05, freeze=True)

                # -- Оптимизатор Adam
                optimizer = torch.optim.Adam(bayesian_model.parameters(), 
                                             lr=learning_rate, 
                                             weight_decay=weight_decay)

                # ======== 8. Обучение ========
                for epoch in range(epochs):
                    bayesian_model.train()
                    optimizer.zero_grad()
                    pred = bayesian_model(X_train_t)
                    loss = F.mse_loss(pred, y_train_t)
                    loss.backward()
                    optimizer.step()

                # ======== 9. Оценка ========
                bayesian_model.eval()
                with torch.no_grad():
                    test_pred = bayesian_model(X_test_t).numpy().ravel()

                # Метрики
                test_mae = mean_absolute_error(y_test, test_pred)
                test_smape = smape(y_test, test_pred)
                test_variance = np.var(test_pred)

                results.append({
                    'size': size,
                    'sparsity': sparsity,
                    'num_layers': num_layers,
                    'mae': test_mae,
                    'smape': test_smape,
                    'variance': test_variance
                })

                print(f"[BayesFCNN] Size={size}, Spars={sparsity}, Layers={num_layers} "
                      f"-> MAE={test_mae:.4f}, sMAPE={test_smape:.2f}%, Var={test_variance:.4f}")

    # ======== 10. Сохраняем результаты ========
    results_df = pd.DataFrame(results)
    results_df.to_csv('FCNN_bayes_results.csv', index=False)
    print("\nЭксперименты завершены. Итоги в 'FCNN_bayes_results.csv'.")

if __name__ == '__main__':
    main()
