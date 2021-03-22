import glob
import os
import numpy as np
from PIL import Image

# 코드 실행시 모든 파일에 000을 붙여준다!
for i in range(1000):
    os.mkdir('C:/workspace/lotte/train_new/{0:04}'.format(i))

    for img in range(48):
        image = Image.open(f'C:/workspace/lotte/train/{i}/{img}.jpg')
        image.save('C:/workspace/lotte/train_new/{0:04}/{1:02}.jpg'.format(i, img))

for i in range(72000):
    image = Image.open(f'C:/workspace/lotte/test/test/{i}.jpg')
    image.save('C:/workspace/lotte/test_new/test_new/{0:05}.jpg'.format(i))