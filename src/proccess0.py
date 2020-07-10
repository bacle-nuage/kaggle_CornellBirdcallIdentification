# 参照：https://www.infiniteloop.co.jp/blog/2018/01/learning-keras-04/

# ライブラリ
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os

# path
# audio_dir = '/path'
# file_name = 'file_name'
# audio_path = os.path.join(audio_dir, file_name)

# load
# audio, sr_ = librosa.load(audio_path)

# get training dataset and target dataset
x = list(meta_data.loc[:,"filename"])
y = list(meta_data.loc[:, "target"])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, stratify=y)
print("x train:{0}\ny train:{1}\nx test:{2}\ny test:{3}".format(len(x_train),
                                                                len(y_train),
                                                                len(x_test),
                                                                len(y_test)))

# showing the classes are equally splitted
a = np.zeros(50)
for c in y_test:
    a[c] += 1
print(a)




##############################
# libraries をインポート
##############################
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn import datasets
import numpy as np

##############################
# 関数
##############################
# データ準備
def read_data():
    X =
    T =

    T = np_utils.to_categorical(T) # one hot encoding

    train_x, test_x, train_y, test_y = train_test_split(X, T, train_size=0.8, test_size=0.2) # トレーニングとテストデータに分割

    return train_x, test_x, train_y, test_y


##############################
# モデル構築 5層パーセプトロン
##############################
def create_model(out_dim=100, activation="relu", optimizer=SGD(lr=0.1), loss="categorical_crossentropy"):
    input_dim =

    model = Sequential()

    model.add(Dense(input_dim=input_dim, units=out_dim))
    model.add(Activation(Activation))

    model.compile(loss=loss, optimizer=optimizer)

    return model


##############################
# メイン処理
##############################
# データ準備
train_x, test_x, train_y, test_y = read_data()

# モデル作成
activation="relu"
optimizer=SGD(lr=0.1)
out_dim=100
loss="categorical_crossentropy"
model = create_model(out_dim=out_dim, activation=activation, optimizer=optimizer, loss=loss)

# トレーニング
batch_size = 16
nb_epoch = 25
model.fit(train_x, train_y, epochs=nb_epoch, batch_size=batch_size)

# 学習済モデルでテストデータを分類
Y = model.predict_classes(test_x, batch_size=batch_size)

# 結果検証
_, T_index = np.where(test_t > 0) # to_categorical の逆変換
print()
print('RESULT')
print(Y == T_index)
