from icecream import ic
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D
from tensorflow.keras.layers import GlobalMaxPooling2D, Add, MaxPooling2D, Activation
from tensorflow.keras.layers import Concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.activations import sigmoid
import cfg

MODEL_W = cfg.MODEL_INPUT_SIZE[0]
MODEL_H = cfg.MODEL_INPUT_SIZE[1]

def get_primary_model():
    inp_2047 = tf.keras.Input(shape=(1,1,2048))
    inp_224 = tf.keras.Input(shape=(3,3,1024))

    backbone = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(int(MODEL_H), int(MODEL_W), 3),
            pooling=None,
        )

    yolo = MaxPooling2D((2,2))(backbone.layers[-1].output)

    yolo = Conv2D(1024, 3, activation=None, padding="same")(yolo)
    yolo = LeakyReLU()(yolo)

    yolo = Conv2D(1024, 1, activation=None, padding="valid")(yolo)
    yolo = LeakyReLU()(yolo)

    tap_224 = yolo

    yolo = Concatenate()([yolo, inp_224])

    yolo = Conv2D(1024, 2, activation=None, padding="valid")(yolo)
    yolo = LeakyReLU()(yolo)
    tap_2047 = yolo

    yolo = Concatenate()([yolo, inp_2047])

    yolo = tf.keras.layers.Flatten()(yolo)
    yolo = tf.keras.layers.Dense(units=2048)(yolo)
    yolo = LeakyReLU()(yolo)
    yolo = tf.keras.layers.Dense(units=2048)(yolo)
    yolo = LeakyReLU()(yolo)
    yolo = tf.expand_dims(yolo, axis=1)
    yolo = tf.expand_dims(yolo, axis=1)

    yolo = Conv2D(5, 1, activation=None, padding="same", name='YOLO')(yolo)
    yolo = Activation(tf.keras.activations.linear)(yolo)
    model = tf.keras.Model( inputs=[backbone.inputs, inp_2047, inp_224], outputs=[yolo, tap_2047, tap_224] )
    return model