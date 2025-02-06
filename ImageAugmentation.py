import os
import cv2
import numpy as np
import random

InputData = 'c:/data/1-200'
OutputData = 'c:/data/1-200'

# https://github.com/FaisalAhmedBijoy/Image_Cryptography_with_Autoencoders/blob/main/salt_and_pepper_noise.py からsaltの部分を削除
def add_pepper_noise(image, amount):
    # 入力画像を変更せずにノイズを加えるため、画像のコピーを作成します。
    noisy_image = np.copy(image)

    # 画像の高さ（行数）と幅（列数）を取得します。
    height, width = noisy_image.shape[:2]

    # amountで指定された割合に基づき、ノイズを加えるべきピクセル数を計算します。
    # amountが0.1なら、画像の10%のピクセルにノイズを加えることになります。
    num_pixels = int(amount * height * width)

    # ランダムなピクセルの座標を生成
    # np.random.randintを使って、画像内のランダムなピクセル座標（高さ、幅）を
    # num_pixels個生成します。この座標にノイズを加えます。
    coords = [np.random.randint(0, height, num_pixels),
            np.random.randint(0, width, num_pixels)]

    # 「胡椒」ノイズ（黒いピクセル）を加える
    # 同じ座標に黒いピクセル（値0）を設定します。これが「胡椒ノイズ」です。
    noisy_image[coords[0], coords[1]] = 0  # Pepper (black)

    # ノイズが加わった画像を返す
    # 最後に、ノイズが加えられた画像を返します。
    return noisy_image

for pathname, dirnames, filenames in os.walk(InputData):
    for filename in filenames:
        img_path = (os.path.join(pathname,filename))

        # 画像を読み込む
        image = cv2.imread(img_path)

        for noise in range(1, 8):
            noisy_image = add_pepper_noise(image, amount = noise/10)

            filen = filename[:-4] #.jpgを削除

            # 結果を保存
            cv2.imwrite(OutputData + '/' + filen + '_' + str(noise/10) + '.jpg', noisy_image)
