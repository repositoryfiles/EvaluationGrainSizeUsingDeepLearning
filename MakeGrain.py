# 種々の粒度番号の画像を自動生成するプログラム

# coding: utf-8
import sys
import numpy as np
import os
import math
from PIL import Image, ImageFilter

#試料の結晶粒度（あらかじめ求めておく値）
ApparentGrainSize = 9.40
#元画像の撮影倍率
Magnification = 200

width = []
height = []
GrainSize = []

#画像のサイズ
width.append(1920)
height.append(1440)

VGG16_size = 224

#元画像の格納場所
InputData = 'c:/data/1-200/'
#トリミング後の種々の結晶粒度の画像の格納場所
OutputData = 'c:/data1/1-200/'

#見かけの粒度番号の計算
GrainSize.append(round(ApparentGrainSize - 6.64 * math.log10(Magnification/100), 1))
origin_m = 8 * 2 ** GrainSize[0]
intGrainSize = math.floor(GrainSize[0]) #小数点以下を切り捨て

#画像の中心を任意のサイズでトリミング
def crop_center(pil_img, crop_width, crop_height):
    img_width = width[0]
    img_height = height[0]
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

#粒度番号とトリミングサイズの計算
for num in range(20):
    Grain = intGrainSize-num*0.5
    GrainSize.append(Grain)
    m = 8 * 2 ** (Grain)
    wid = width[0] * math.sqrt(m) / math.sqrt(origin_m)
    hei = height[0] * math.sqrt(m) / math.sqrt(origin_m)
    width.append(wid)
    height.append(hei)
    if hei < VGG16_size: #width.appendの前に置く手もあるが、224をわずかに下回った値も含めるためここに置いた
        break

#粒度番号名のフォルダの作成
for num in GrainSize:
    if num == GrainSize[0]:
        continue
    Fol = OutputData + str(num)
    os.makedirs(Fol, exist_ok=True)

for pathname, dirnames, filenames in os.walk(InputData):
    for filename in filenames:
        img_path = (os.path.join(pathname,filename))
        im = Image.open(img_path)
        count = 0
        #print(img_path)
        for num in GrainSize:
            if num == GrainSize[0]:
                continue
            count = count + 1
            #幅と高さが異なる場合
            im_new=crop_center(im, width[count], height[count])
            #print(width[count], height[count])

            w = im_new.width
            h = im_new.height

            filen = filename[:-4] #.jpgを削除

            #リサイズ
            im_resize = im_new.resize((int(224*w/h), 224))

            im_resize.save(OutputData + str(num) + '/' + filen + '_' + str(num)  + '.jpg', quality=95)

