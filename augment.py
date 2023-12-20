#Editor: 신성호
from torchvision import datasets, transforms
import torch
# to split test_data always same
torch.manual_seed(1004)

import torch.nn as nn
from torch.utils.data import DataLoader

import os
from PIL import Image, UnidentifiedImageError, ImageFile,ImageEnhance
import PIL.ImageOps
import random
import plotly.express as px

custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def augment():
    defaultPath = 'step_3/'
    #defaultPath = 'step_1/'
    #defaultPath = 'Class2/'
    file_names = os.listdir(defaultPath)
    for i in range(len(file_names)):
        for j in range(10):
            filePath = defaultPath + file_names[i]
            image = Image.open(filePath)
            random_augment = random.randrange(1, 4)
            if (random_augment == 1):
            # 이미지 좌우 반전
            # print("invert"
                inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # inverted_image.save(filePath + 'inverted_' + str(k) + '.png')
            elif (random_augment == 2):
            # print("invert")
                inverted_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            # inverted_image.save(filePath + 'inverted_' + str(k) + '.png')
            else:
                inverted_image = image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
                random_augment = random.randrange(1, 4)
            if (random_augment != 1):
            # 이미지 회전
            # print("invert")
                inverted_image = inverted_image.rotate(random.randrange(-30, 30))
            random_augment = random.randrange(1, 7)

            if (random_augment == 1):
                # 이미지 흑백변환
                # print("invert")
                inverted_image = inverted_image.convert('L')
            elif (random_augment == 2 or random_augment == 3 or random_augment == 4):
                inverted_image = ImageEnhance.Brightness(inverted_image).enhance(random.randrange(30, 115) / 100)

            inverted_image = inverted_image.resize((224, 224))  # 수정됨
            if filePath.split('.')[1] == 'png':
                inverted_image.save(filePath.split('.')[0] + 'augmented' + str(j) + '.png')
            elif filePath.split('.')[1] == 'jpg':
                inverted_image.save(filePath.split('.')[0] + 'augmented' + str(j) + '.jpg')
            else:
                inverted_image.save(filePath.split('.')[0] + 'augmented' + str(j) + '.jpeg')

if __name__ == '__main__':
    augment()