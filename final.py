import pandas as pd
import os
import numpy as np
import codecs
import keras
import random
import cv2

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from imutils import paths
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
#from keras.utils import to_categorical

#如果没有这两行代码，可能会报错OSError: image file is truncated
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


config_path = '/openbayes/input/input1/BERT/cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/openbayes/input/input1/BERT/cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/openbayes/input/input1/BERT/cased_L-12_H-768_A-12/vocab.txt'

maxlen = 200
CLASS_NUM = 3
CLASS_NUM = 3
EPOCHS = 20
#后期可以用学习率衰减来训练
INIT_LR = 1e-5

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

train_text_path = r'/openbayes/input/input0/taskA/Task_A_train.csv'
trial_text_path = r'/openbayes/input/input0/taskA/Task_A_trial.csv'

train_text_data = pd.read_csv(train_text_path)
train_text_data = np.array(train_text_data)
train_text_data = train_text_data.tolist()

trial_text_data = pd.read_csv(trial_text_path)
trial_text_data = np.array(trial_text_data)
trial_text_data = trial_text_data.tolist()

image_path = r'/openbayes/home/img'

class data_generator:
    def __init__(self, data, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, X3, Y = [], [], [], []
            for i in idxs:
                d = self.data[i]
                text = str(d[1])[:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                imagePath = os.path.join(image_path,d[2])
                img = image.load_img(imagePath, target_size=(224, 224))
                x3 = image.img_to_array(img)
                y = d[0]
                X1.append(x1)
                X2.append(x2)
                X3.append(x3)
                Y.append([y])               
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    X3 = np.array(X3, dtype="float") / 255.0
                    Y = seq_padding(Y)
#                     Y = np.array(Y)
#                     Y = to_categorical(Y, num_classes=CLASS_NUM)
                    yield [X1, X2, X3], Y
                    [X1, X2, X3, Y] = [], [], [], []
        
train_D = data_generator(train_text_data)
valid_D = data_generator(trial_text_data)

#文字处理模型部分
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

bert_x1_in = Input(shape=(None,))
bert_x2_in = Input(shape=(None,))

x = bert_model([bert_x1_in, bert_x2_in])
text_output = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类

#图像模型部分
model = ResNet50()
#加一个dense层吧
img_in = Input(shape=(None,None,None,))
x = model(inputs = [img_in])
img_output = Dense(10)(x)

#将文字处理部分与图像处理部分连接起来
x = keras.layers.concatenate([img_output, text_output])

# 堆叠多个全连接网络层
# x = Dense(64, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(64, activation='relu')(x)


#顶层套一个全连接层
p = Dense(CLASS_NUM, activation='softmax')(x)

#重新定义一个模型，BERT本身有两个输入
model = Model([bert_x1_in, bert_x2_in, img_in], p)

model.compile(
    optimizer=Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
             )
model.summary()

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=EPOCHS,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)