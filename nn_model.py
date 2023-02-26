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
    main_inp = tf.keras.Input(shape=(192,192,3))
    backbone = tf.keras.applications.resnet.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(int(MODEL_H), int(MODEL_W), 3),
            pooling=None,
        )(main_inp)
    yolo = Conv2D(64, 3, activation=None, padding="same")(backbone)
    yolo = LeakyReLU()(yolo)
    yolo = Conv2D(128, 3, activation=None, padding="same")(yolo)
    yolo = LeakyReLU()(yolo)
    yolo = Conv2D(256, 6, activation=None, padding="valid")(yolo)
    yolo = LeakyReLU()(yolo)
    tap_7 = yolo
    yolo = Conv2D(5, 1, padding="same", name='TANH')(yolo)                     # activation='tanh'
    model = tf.keras.Model( inputs=main_inp, outputs=[yolo, backbone] )
    return model


def get_unet_pp():
    main_inp = tf.keras.Input(shape=(192,192,3))
    yolo = Conv2D(32, 3, activation=None, padding="same")(main_inp)
    yolo = LeakyReLU()(yolo)
    tap_1 = yolo
    yolo = MaxPooling2D((2,2))(yolo)
    yolo = Conv2D(64, 3, activation=None, padding="same")(yolo)
    yolo = LeakyReLU()(yolo)
    yolo = UpSampling2D((2,2))(yolo)
    yolo = Concatenate()([yolo, tap_1])
    yolo = Conv2D(32, 3, activation=None, padding="same")(yolo)
    yolo = LeakyReLU()(yolo)

    
def gen_shape_model():
    yolo = Conv2D(32, 3, activation=None, padding="same")(yolo)    
    yolo = tf.keras.layers.Multiply()([main_inp, att])
    ran_unif = tf.keras.initializers.RandomUniform(minval=-100.0, maxval=100., seed=None)
    regu = tf.keras.regularizers.L2(l2=0.00000001)
    # trainnable pixel-wise multiplication layer
    yolo = tf.keras.layers.LocallyConnected2D(1, kernel_size=(1, 1), 
        activation=None, 
        use_bias=None, 
        kernel_initializer=ran_unif,
        kernel_regularizer=regu,
        implementation=3,
        )(inp_2047)
    yolo = tf.keras.layers.Add()([main_inp, yolo])  
    yolo = tf.image.grayscale_to_rgb(yolo)
    b = tf.keras.layers.Lambda(lambda x:x+tf.constant(-128.))(yolo)
    yolo = tf.keras.layers.Add()([yolo, b])



if __name__=='__main__':

    model = get_primary_model()

    model.summary()


    #att = Conv2DTranspose(16, 3, strides=(2,2), padding='same')(att)
