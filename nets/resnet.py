
import tensorflow as tf
from tensorflow import keras
from pixel_based_module import Pixel_Anchor

class BasicBlock(keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.ReLU()
        
        self.conv2 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

        self.identity = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='same')

        # if strides > 1:
        #     self.identity = keras.Sequential()
        #     self.identity.add(keras.layers.Conv2D(filters, (1, 1), strides=strides))
        # else:
        #     self.identity = lambda x: x

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.identity(inputs)

        x = keras.layers.add([x, identity])
        x = self.relu(x)

        return x

class ResNET50(keras.layers.Layer):
    def __init__(self):
        super(ResNET50, self).__init__()

        self.preprocess = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])

        self.resblock_1 = self.__build_resblock(64, 2)
        self.resblock_2 = self.__build_resblock(64, 2, strides=2)
        self.resblock_3 = self.__build_resblock(128, 2, strides=1)
        self.resblock_4 = self.__build_resblock(128, 2, strides=2)
        self.resblock_5 = self.__build_resblock(256, 2, strides=1)
        self.resblock_6 = self.__build_resblock(256, 2, strides=2)
        self.resblock_7 = self.__build_resblock(512, 2, strides=1)
        self.resblock_8 = self.__build_resblock(512, 2, strides=1)
        self.resblock_9 = self.__build_resblock(512, 2, strides=2)
        self.resblock_10 = self.__build_resblock(1024, 2, strides=1)
        self.resblock_11 = self.__build_resblock(1024, 2, strides=1)
        self.resblock_12 = self.__build_resblock(1024, 2, strides=2)


    def call(self, inputs, training=None):
        x = self.preprocess(inputs)
        x = self.resblock_1(x)
        x = self.resblock_2(x)
        x = self.resblock_3(x)
        x = self.resblock_4(x)
        x = self.resblock_5(x)
        x = self.resblock_6(x)
        x = self.resblock_7(x)
        x = self.resblock_8(x)
        x = self.resblock_9(x)
        x = self.resblock_10(x)
        x = self.resblock_11(x)
        x = self.resblock_12(x) 

        return x

    def __build_resblock(self, filters, basic_block_num, strides=1):
        resblocks = keras.Sequential()

        resblocks.add(BasicBlock(filters, strides))

        for _ in range(basic_block_num):
            resblocks.add(BasicBlock(filters, strides=1))

        return resblocks


class ResNET50_Pixel_Anchor(keras.layers.Layer):
    def __init__(self):
        super(ResNET50_Pixel_Anchor, self).__init__()

        self.preprocess = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])

        self.resblock_1 = self.__build_resblock(64, 2)
        self.resblock_2 = self.__build_resblock(64, 2, strides=2)
        self.resblock_3 = self.__build_resblock(128, 2, strides=2)
        self.resblock_4 = self.__build_resblock(128, 2, strides=1)
        self.resblock_5 = self.__build_resblock(256, 2, strides=2)
        self.resblock_6 = self.__build_resblock(256, 2, strides=1)
        self.resblock_7 = self.__build_resblock(512, 2, strides=2)
        self.resblock_8 = self.__build_resblock(512, 2, strides=1)
        self.resblock_9 = self.__build_resblock(512, 2, strides=1)
        self.resblock_10 = self.__build_resblock(1024, 2, strides=2)
        self.resblock_11 = self.__build_resblock(1024, 2, strides=1)
        self.resblock_12 = self.__build_resblock(1024, 2, strides=1)


    def call(self, inputs, training=None):
        x = self.preprocess(inputs) 
        x = self.resblock_1(x)      # [None, 1 times, 1 times, 64]
        x = self.resblock_2(x)      # [None, 1/2 times, 1/2 times, 64]
        x = self.resblock_3(x)      # [None, 1/4 times, 1/4 times, 128]
        x = self.resblock_4(x)      # [None, 1/4 times, 1/4 times, 128]
        out_1_4 = x
        x = self.resblock_5(x)      # [None, 1/8 times, 1/8 times, 256]
        x = self.resblock_6(x)      # [None, 1/8 times, 1/8 times, 256]
        out_1_8 = x
        x = self.resblock_7(x)      # [None, 1/16 times, 1/16 times, 512]
        x = self.resblock_8(x)      # [None, 1/16 times, 1/16 times, 512]
        x = self.resblock_9(x)      # [None, 1/16 times, 1/16 times, 512]
        out_1_16 = x
        # x = self.resblock_10(x)     # [None, 1/32 times, 1/32 times, 1024]
        # x = self.resblock_11(x)     # [None, 1/32 times, 1/32 times, 1024]
        # x = self.resblock_12(x)     # [None, 1/32 times, 1/32 times, 1024]

        return out_1_4, out_1_8, out_1_16

    def __build_resblock(self, filters, basic_block_num, strides=1):
        resblocks = keras.Sequential()

        resblocks.add(BasicBlock(filters, strides))

        for _ in range(basic_block_num):
            resblocks.add(BasicBlock(filters, strides=1))

        return resblocks


if __name__ == "__main__":
    import cv2
    import numpy as np
    r = ResNET50_Pixel_Anchor()
    im_data =  cv2.imread('D:\\github_projects\\pixel_text\\images\\test1.jpg')
    im_data = cv2.resize(im_data, (224, 224))
    im_data = np.expand_dims(im_data, 0)
    im_data = tf.dtypes.cast(im_data, dtype=tf.float32)

    x1, x2, x3 = r(im_data)
    

    p = Pixel_Anchor()
    x = p([x1, x2, x3])
    
    cv2.imshow('123', im_data)
    cv2.waitKey(0)
    a = 10

