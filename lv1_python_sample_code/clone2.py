# coding: UTF-8

import sys
import numpy as np
from PIL import Image
from sklearn import neighbors
from labels import COLOR2ID
from evaluation import IMAGE_SIZE
from evaluation import LV1_Evaluator

from keras.models import Sequential, Model
from keras.layers import Activation, Dense
from keras.callbacks import EarlyStopping
from keras import optimizers
import matplotlib.pyplot as plt


# ターゲット認識器を表現するクラス
# ターゲット認識器は2次元パターン（512x512の画像）で与えられるものとする
class LV1_TargetClassifier:

    # ターゲット認識器をロード
    #   filename: ターゲット認識器を表現する画像のファイルパス
    def load(self, filename):
        self.img = Image.open(filename)

    # 入力された二次元特徴量に対し，その認識結果（クラスラベルID）を返す
    def predict_once(self, x1, x2):
        h = IMAGE_SIZE // 2
        x = max(0, min(IMAGE_SIZE - 1, np.round(h * x1 + h)))
        y = max(0, min(IMAGE_SIZE - 1, np.round(h - h * x2)))
        return COLOR2ID(self.img.getpixel((x, y)))

    # 入力された二次元特徴量の集合に対し，各々の認識結果を返す
    def predict(self, features):
        labels = np.zeros(features.shape[0])
        for i in range(0, features.shape[0]):
            labels[i] = self.predict_once(features[i][0], features[i][1])
        return np.int32(labels)

# クローン認識器を表現するクラス
# このサンプルコードでは単純な 1-nearest neighbor 認識器とする（sklearnを使用）
# 下記と同型の fit メソッドと predict メソッドが必要
class LV1_UserDefinedClassifier:

    # クローン認識器の設定
    def __init__(self):
        self.clf = Sequential()
        self.clf.add(Dense(5, input_dim=2, activation='tanh'))
        self.clf.add(Dense(10, activation='softmax'))
        optimizer1 = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        self.clf.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
        #early_stopping = EarlyStopping(patience=5, verbose=1)

    # クローン認識器の学習
    #   (features, labels): 訓練データ（特徴量とラベルのペアの集合）
    def fit(self, features, labels):
        Y = np.eye(10)[labels.astype(int)] 
        epochs = 1000
        batch_size = 5
        self.history = self.clf.fit(features, Y, batch_size=batch_size, epochs = epochs, shuffle=True, validation_split=0.1)

    # 未知の二次元特徴量を認識
    #   features: 認識対象の二次元特徴量の集合
    def predict(self, features):
        Y = self.clf.predict(features)
        labels=np.argmax(Y,1)
        return np.int32(labels)

# ターゲット認識器に入力する二次元特徴量をサンプリングする関数
#   n_samples: サンプリングする特徴量の数
def LV1_user_function_sampling(n_samples=1):
    features = np.zeros((n_samples, 2))
    for i in range(0, n_samples):
        # このサンプルコードでは[-1, 1]の区間をランダムサンプリングするものとする
        features[i][0] = 2 * np.random.rand() - 1
        features[i][1] = 2 * np.random.rand() - 1
    return np.float32(features)

def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.pause(0.1)

    # 損失の履歴をプロット
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.show()
    
# クローン処理の実行
# 第一引数でターゲット認識器を表す画像ファイルのパスを，
# 第二引数でクローン認識器の可視化結果を保存する画像ファイルのパスを，
# それぞれ指定するものとする
if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("usage: clone.py /target/classifier/image/path /output/image/path")
        exit(0)

    # ターゲット認識器を用意
    target = LV1_TargetClassifier()
    target.load(sys.argv[1]) # 第一引数で指定された画像をターゲット認識器としてロード
    print("\nA target recognizer was loaded from {0} .".format(sys.argv[1]))

    # ターゲット認識器への入力として用いる二次元特徴量を用意
    # このサンプルコードではひとまず1000サンプルを用意することにする
    n = 200
    features = LV1_user_function_sampling(n_samples=n)
    print("\n{0} features were sampled.".format(n))

    # ターゲット認識器に用意した入力特徴量を入力し，各々に対応するクラスラベルIDを取得
    labels = target.predict(features)
    print("\nThe sampled features were recognized by the target recognizer.")

    # クローン認識器を学習
    model = LV1_UserDefinedClassifier()
    model.fit(features, labels)
    print("\nA clone recognizer was trained.")
    print(model.history)

    # 学習したクローン認識器を可視化し，精度を評価
    evaluator = LV1_Evaluator()
    evaluator.visualize(model, sys.argv[2])
    print("\nThe clone recognizer was visualized and saved to {0} .".format(sys.argv[2]))
    print("\naccuracy: {0}".format(evaluator.calc_accuracy(target, model)))

    plot_history(model.history)

