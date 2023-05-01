import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Part 1 - 資料前處理

# 取資料
dataset = pd.read_csv('ann/Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# 編碼
label_encoder_x_1 = LabelEncoder()
x[:, 1] = label_encoder_x_1.fit_transform(x[:, 1])
label_encoder_x_2 = LabelEncoder()
x[:, 2] = label_encoder_x_2.fit_transform(x[:, 2])
column_transformer = ColumnTransformer([('Geography', OneHotEncoder(), [1])], remainder='passthrough')
x = column_transformer.fit_transform(x)
x = x[:, 1:]

# 分割測試集及訓練集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 特徵歸一化
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print(x)