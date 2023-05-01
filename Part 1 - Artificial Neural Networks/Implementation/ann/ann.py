import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

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

# 訓練ANN
classifier = Sequential()

# 選擇激活函數建立隱藏層
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# 選擇激活函數建立輸出層
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# 定義計算
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 加入訓練集進行訓練
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

# 進行測試集預測
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# 單一預測
new_pred = classifier.predict(sc.fit_transform(np.array([[0, 0, 600, 1, 40, 3, 6000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)

# 分析混淆矩陣
cm = confusion_matrix(y_test, y_pred)