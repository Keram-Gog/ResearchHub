import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# загрузим датасет
file_path = 'Exp5_\data\cleaned_weather.csv'
data = pd.read_csv(file_path)

# первые строки 
print(data.head())

# Информация о датасете (количество строк, типы данных)
print(data.info())

# Проверка на пропущенные значения
print(data.isnull().sum())

# Статистическое описание данных (среднее, стандартное отклонение, минимум, максимум)
print(data.describe())