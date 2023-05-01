import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

def build_classifier(optimizer):
    # 建立分類器
    classifier = Sequential()
    # 選擇激活函數建立隱藏層
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    # 防止過擬合
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    # 選擇激活函數建立輸出層
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # 定義計算
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

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

# 建立分類器 ===============================================================================================
'''
classifier = build_classifier()

# 加入訓練集進行訓練
classifier.fit(x_train, y_train, batch_size=10, epochs=100)

# 進行測試集預測
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# 單一預測
new_pred = classifier.predict(sc.fit_transform(np.array([[0, 0, 600, 1, 40, 3, 6000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)

# 建立混淆矩陣
cm = confusion_matrix(y_test, y_pred)
'''
# ==========================================================================================================

# 建立分類器 - 交叉驗證法 ====================================================================================
'''
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()
'''
# ==========================================================================================================

# 建立分類器 - 網格搜索法 ====================================================================================
classifier = KerasClassifier(build_fn=build_classifier)
parameters = { 'batch_size': [25, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop'] }
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
