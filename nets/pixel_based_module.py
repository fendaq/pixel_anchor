import tensorflow as tf
from tensorflow import keras

class Pixel_Anchor(keras.layers.Layer):
    def __init__(self, filters=[128, 256, 512]):
        '''
        filters= [1/4 times's channels, 1/8 times's channels, 1/16 times's channels]
        '''
        super(Pixel_Anchor, self).__init__()
        self.aspp_1 = keras.layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=6, padding='same')
        self.aspp_2 = keras.layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=12, padding='same')
        self.aspp_3 = keras.layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=18, padding='same')
        self.aspp_4 = keras.layers.Conv2D(filters=filters[2], kernel_size=(3, 3), dilation_rate=24, padding='same')
        self.conv1 = keras.layers.Conv2D(filters=filters[1], kernel_size=(3, 3), padding='same')
        self.upsample1 = keras.layers.UpSampling2D(2)
        self.conv2 = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3), padding='same')
        self.upsample2 = keras.layers.UpSampling2D(2)
        self.conv3 = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3), padding='same')

    def call(self, inputs, training=None):
        # x1: 1/4, x2: 1/8, x3: 1/16
        x1, x2, x3 = inputs[0], inputs[1], inputs[2]
        aspp1 = self.aspp_1(x3)
        aspp2 = self.aspp_2(x3)
        aspp3 = self.aspp_3(x3)
        x = tf.concat([aspp1, aspp2, aspp3], axis=-1)
        x = self.conv1(x)
        x = self.upsample1(x)
        x = tf.concat([x2, x], axis=-1)
        x = self.conv2(x)
        x = self.upsample2(x)
        x = tf.concat([x1, x], axis=-1)
        x = self.conv3(x)
        return x
