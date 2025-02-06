# 以下のプログラムは、TeachableMachineの「モデルをエクスポートしてプロジェクトで使用する。」のTensorflowタブのKerasの
# サンプルプログラムを編集して作成したもの

# coding: utf-8

from keras.models import load_model
import numpy as np
import glob
from PIL import Image, ImageOps
import math
import statistics

data_path = 'C:/GrainBoundaries/'
h5_file = 'C:/Python/keras_Model.h5'
KMP_WARNINGS = 0

IMAGE_SIZE = 224
label=["2.0", "2.5", "3.0", "3.5","4.0", "4.5", "5.0","5.5","6.0","6.5","7.0"]
folders=["304-200","304-500","435Q-500","435Q-1000","435QT-500","435QT-1000","SUJ2QT-500","SUJ2QT-1000"]
nb_classes = len(label)

model = load_model(h5_file, compile=False)

print("Model:{}", h5_file)
print(folders)
print("Results: folder, ave., std.")
for index, name in enumerate(folders):

    MobileNet_grain_number = []
    MobileNet_grain_number_score = []

    data_dir = data_path + '/' + name

    # 画像ファイルのフォルダにあるcond.iniの値の読み込み
    f = open(data_dir + "\cond.ini", 'r')
    magnification = eval(f.readline().rstrip())
    picturewidth = eval(f.readline().rstrip())  #画像処理および出力時の画像の幅（高さは元画像の縦横比から計算）

    f.close()
    #print("Magnification : {:.1f}, PictureWidth : {:.1f}".format(magnification, picturewidth))
    data_files = glob.glob(data_dir + "/*.jpg")
    #print(data_files)

    grain_number = []
    confidence_score = []

    for i, data_file in enumerate(data_files):

        data = np.ndarray(shape=(1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(data_file).convert("RGB")

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        grain_pred = label[index]
        T = magnification

        # Calcurate Grain Size Index
        G = float(grain_pred) + 6.64 * math.log10(T/100)
        grain_number.append(G)
        #confidence_score.append(prediction[0][index])

    print("{}, {:.2f}, {:.2f}"
      .format(data_dir, statistics.mean(grain_number), statistics.stdev(grain_number)))







