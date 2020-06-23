##############################
# libraries
##############################
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
import matplotlib.pyplot as plt
# %matplotlib inline

##############################
# 定数
##############################
COLAB_FLG = 1
TRAIN_PATH = '/kaggle/input/titanic/train.csv'
TEST_PATH = '/kaggle/input/titanic/test.csv'
COLAB_TRAIN_PATH = '/content/drive/My Drive/MachineLeaning/kaggle_titanic/train.csv'
COLAB_TEST_PATH = '/content/drive/My Drive/MachineLeaning/kaggle_titanic/test.csv'
COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Ticket_Left']

##############################
# データ読み込み
##############################
def read_data():
  ## 読み込むデータのパス切り替え
  if COLAB_FLG:
    TRAIN_PATH = COLAB_TRAIN_PATH
    TEST_PATH = COLAB_TEST_PATH

  print('Train data reading...')
  train = pd.read_csv(TRAIN_PATH)
  print("Train data is column {}, rows: {}".format(train.shape[0], train.shape[1]))

  print('Test data reading...')
  test = pd.read_csv(TEST_PATH)
  print("Test data is column {}, rows: {}".format(test.shape[0], test.shape[1]))

  return train, test

##############################
# 前処理
##############################
def pre_processing(train, test):
  # train と test を合わせて前処理
  print('Combine train and test')
  train['is_train'] = 1
  test['is_train'] = 0
  train_test = pd.concat([train.drop('Survived', axis=1),test], axis=0)

  # Sex
  print('Modifying Sex column')
  train_test = train_test.replace('female', 0).replace('male', 1)

  # Ticket
  print('Modifying Ticket column')
  ticket_to_num = {'A':0, 'P':1, 'S':2, '1':3, '3':4, '2':5, 'C':6, '7':7, 'W':8, '4':9, 'F':9, 'L':10, '9':11,'6':12, '5':13, '8':14}
  train_test['Ticket_Left'] = train_test['Ticket'].apply(lambda x: str(x)[0])
  train_test['Ticket_Left'] = train_test['Ticket_Left'].replace(ticket_to_num)
  train_test = train_test.drop('Ticket', axis=1)

  # Embarked
  print('Modifying Embarked column')
  embarked_to_num = {'S':0, 'C':1, 'Q':2}
  train_test['Embarked'] = train_test['Embarked'].replace(embarked_to_num)
  train_test['Embarked'] = train_test['Embarked'].fillna(train_test['Embarked'].mean())
  train_test = train_test.drop('Embarked', axis=1)

  # Cabin
  # 204/891 したデータが入っていないため削除
  print('Modifying Cabin column')
  train_test = train_test.drop('Cabin', axis=1)

  # Name
  # 関係なさそうだから削除
  print('Modifying Name column')
  train_test = train_test.drop('Name', axis=1)

  # train test
  print('Split train and test')
  train_result = train_test.loc[train_test['is_train'] == 1]
  test_result = train_test.loc[train_test['is_train'] == 0]

  train_result = train_result.drop('is_train', axis=1)
  test_result = test_result.drop('is_train', axis=1)

  # add Survived
  train_result['Survived'] = train['Survived']

  # Delete Age and Fare
  train_result = train_result.dropna(subset=['Age', 'Fare'])
  test_result = test_result.fillna(test_result.mean())

  return train_result, test_result


##############################
# モデル構築 5層パーセプトロン
##############################
def create_model_5dim_layer(activation="relu", optimizer="adam", out_dim=100, dropout=0.5):
    columns = COLUMNS

    model = Sequential()

    # 入力層 - 隠れ層1
    model.add(Dense(input_dim=len(columns), units=out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 隠れ層1 - 隠れ層2
    model.add(Dense(units=out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 隠れ層2 - 隠れ層3
    model.add(Dense(units=out_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # 隠れ層3 - 出力層
    model.add(Dense(units=1))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


##############################
# メイン処理
##############################
# データ読み込み
train, test = read_data()

# 前処理
train, test = pre_processing(train, test)

# 使用するカラム
columns = COLUMNS
train_data = train[columns].values
train_lavels = train['Survived'].values

# 型を変換
x_train = np.asarray(train_data).astype('float32')
y_train = np.asarray(train_lavels).astype('float32')
test_data = test[columns].values.astype('float32')

#正規化
for i in range(len(columns)-1):
    mean = x_train.mean(axis=0)[i]
    std = x_train.std(axis=0)[i]

    x_train[:, i] = (x_train[:, i] - mean) / std
    test_data[:, i] = (test_data[:, i] - mean) / std

# モデル作成
# model = create_model_5dim_layer(columns)
# {'activation': 'tanh', 'batch_size': 16, 'dropout': 0.5, 'nb_epoch': 25, 'optimizer': 'adagrad', 'out_dim': 234}
# create_model_5dim_layer(activation="relu", optimizer="adam", out_dim=100, dropout=0.5)
activation = 'tanh'
optimizer = 'adagrad'
out_dim = 234
dropout = 0.5
batch_size = 16
nb_epoch = 25

model = create_model_5dim_layer(activation=activation, optimizer=optimizer, out_dim=out_dim)

# fitting
model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size)

# 点数算出
train_loss, train_acc = model.evaluate(x_train, y_train)
print('train_acc : ', train_acc)

# テストデータを入力
Y_pred = model.predict(test_data)

import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test['PassengerId'].astype(int), Y_pred[:, 0].astype(int)):
        writer.writerow([pid, survived])


##############################
# ハイパーパラメータ探求
##############################
# データ読み込み
train, test = read_data()

# 前処理
train, test = pre_processing(train, test)

# 使用するカラム
columns = COLUMNS
train_data = train[columns].values
train_lavels = train['Survived'].values

# 型を変換
x_train = np.asarray(train_data).astype('float32')
y_train = np.asarray(train_lavels).astype('float32')
test_data = test[columns].values.astype('float32')

#正規化
for i in range(len(columns)-1):
    mean = x_train.mean(axis=0)[i]
    std = x_train.std(axis=0)[i]

    x_train[:, i] = (x_train[:, i] - mean) / std
    test_data[:, i] = (test_data[:, i] - mean) / std

##############################
# GridSearch
##############################
from keras.wrappers.scikit_learn import KerasClassifier
# model = KerasClassifier(build_fn=create_model_5dim_layer(columns), verbose=0)
model = KerasClassifier(build_fn=create_model_5dim_layer, verbose=0)
from sklearn.model_selection import GridSearchCV
# Define options for parameters
activation = ["tanh", "relu"]
optimizer = ["adam", "adagrad"]
out_dim = [234, 468, 702]
nb_epoch = [25, 50]
batch_size = [8, 16]
dropout = [0.2, 0.4, 0.5]

param_grid = dict(activation=activation,
                  optimizer=optimizer,
                  out_dim=out_dim,
                  nb_epoch=nb_epoch,
                  batch_size=batch_size,
                  dropout=dropout)
grid = GridSearchCV(estimator=model, param_grid=param_grid)

# Run grid search
grid_result = grid.fit(x_train, y_train)

print(grid_result.best_score_)
print(grid_result.best_params_)