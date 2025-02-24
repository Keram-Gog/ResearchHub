import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# -- Для воспроизводимости:
np.random.seed(42)
torch.manual_seed(42)

# ======== 1. Функция sMAPE ========
def smape(y_true, y_pred):
    """
    Вычисляет Symmetric Mean Absolute Percentage Error (sMAPE), в %.
    Формула:
      sMAPE = 100 * (2 * |y - y_hat|) / (|y| + |y_hat| + 1e-8)
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    return 100.0 * np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

# ======== 2. Функция, вносящая "разреженность" (sparsity) ========
def introduce_sparsity(X, fraction):
    """
    fraction - доля элементов, которые заменяются на NaN.
    Пример: fraction=0.25 означает, что 25% значений X станут NaN.
    """
    X_sp = X.copy()
    mask = np.random.rand(*X_sp.shape) < fraction
    X_sp[mask] = np.nan
    return X_sp

# ======== 3. Класс динамической сети ========
class DynamicRegressionModel(nn.Module):
    def __init__(self, input_size, num_layers):
        super(DynamicRegressionModel, self).__init__()
        layers = []
        current_size = input_size
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, max(current_size // 2, 1)))
            layers.append(nn.ReLU())
            current_size = max(current_size // 2, 1)
        
        # Выходной слой
        layers.append(nn.Linear(current_size, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ======== 4. Основной скрипт ========
def main():
    # 4.1. Загрузка данных
    try:
        data = pd.read_csv(
            'D:\\main for my it\\my tasks\\source\\ResearchHub\\Exp1_ModelComparison\\data\\student-mat.csv',
            sep=';'
        )
        print("Данные успешно загружены!")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return
    
    # 4.2. Подготовка X и y
    X = data.drop(columns=['G3'])  # Признаки
    y = data['G3']                # Целевой столбец
    
    # Преобразуем категориальные в One-Hot
    X = pd.get_dummies(X)
    
    # Масштабируем
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4.3. Параметры перебора
    size_options = [1.0, 0.5, 0.25, 0.1]       # 100%, 50%, 25%, 10%
    sparsity_options = [0.0, 0.25, 0.5, 0.9]   # 0%, 25%, 50%, 90%
    layer_options = [1, 2, 4, 6, 8, 10, 12]    # Сколько слоёв
    
    epochs = 100
    learning_rate = 0.0001
    weight_decay = 0.0001  # <-- важное изменение
    
    results = []  # Сюда будем складывать результаты (MAE, sMAPE, Variance)
    
    # 4.4. Перебор параметров
    for size in size_options:
        for sparsity in sparsity_options:
            # -- Вносим sparsity в данные:
            X_sp = introduce_sparsity(X_scaled, fraction=sparsity)
            # -- Примитивно заполняем NaN нулями (можно заменить на любой другой способ):
            X_sp = np.nan_to_num(X_sp, nan=0.0)
            
            # -- Определяем количество обучающих примеров
            train_size = int(len(X_sp) * size)
            if train_size < 1 or train_size >= len(X_sp):
                # Если слишком мало (или слишком много) данных остаётся на обучение, пропустим
                print(f"[Пропуск] size={size} -> train_size={train_size} недопустимо.")
                continue
            
            # -- Разделяем на train/test без перемешивания (можно заменить на train_test_split)
            X_train = X_sp[:train_size]
            y_train = y[:train_size]
            
            X_test = X_sp[train_size:]
            y_test = y[train_size:]
            
            # -- Преобразуем в тензоры
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
            
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
            
            input_dim = X_train_t.shape[1]
            
            # 4.5. Перебираем число слоёв
            for num_layers in layer_options:
                model = DynamicRegressionModel(input_dim, num_layers)
                
                # -- Оптимизатор Adam с lr=0.0001 и weight_decay=0.0001
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                criterion = nn.MSELoss()
                
                # -- Обучение
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    pred = model(X_train_t)
                    loss = criterion(pred, y_train_t)
                    loss.backward()
                    optimizer.step()
                
                # -- Оценка
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test_t).numpy().ravel()
                
                test_mae = mean_absolute_error(y_test, test_pred)
                test_var = np.var(test_pred)
                test_smape = smape(y_test, test_pred)
                
                results.append({
                    'size': size,
                    'sparsity': sparsity,
                    'num_layers': num_layers,
                    'mae': test_mae,
                    'smape': test_smape,
                    'variance': test_var
                })
                
                print(f"Size={size}, Sparsity={sparsity}, Layers={num_layers} "
                      f"-> MAE={test_mae:.4f}, sMAPE={test_smape:.2f}%, Var={test_var:.4f}")
    
    # 4.6. Сохраняем результаты
    results_df = pd.DataFrame(results)
    results_df.to_csv('FCNN_classic_results.csv', index=False)
    print("\nЭксперименты завершены. Итоги в 'FCNN_classic_results.csv'.")

# Запуск
if __name__ == '__main__':
    main()
