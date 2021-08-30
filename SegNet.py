from tensorflow import keras
from tensorflow.keras.layers import *

def SegNet(input_height=512, input_width=512):
    input_shape = (input_height, input_width, 3)
    img_input = keras.Input(shape=input_shape)
    image_ordering = "channels_last"
    # ##################################### Encoder ########################################
    # Block 1
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1', data_format=image_ordering)(img_input)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2', data_format=image_ordering)(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool',data_format=image_ordering)(x)
    # Block 2
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1', data_format=image_ordering)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2', data_format=image_ordering)(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool',data_format=image_ordering)(x)
    # Block 3
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1', data_format=image_ordering)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2', data_format=image_ordering)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3', data_format=image_ordering)(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool',data_format=image_ordering)(x)
    # Block 4
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1', data_format=image_ordering)(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2', data_format=image_ordering)(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3', data_format=image_ordering)(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool',data_format=image_ordering)(x)
    # Block 5
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1', data_format=image_ordering)(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2', data_format=image_ordering)(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3', data_format=image_ordering)(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block5_pool',data_format=image_ordering)(x)

    vgg = keras.Model(img_input, x)
    vgg.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    for layer in vgg.layers:
        layer.trainable = False

    # ###################################### Decoder #######################################
    # Block 6
    o = UpSampling2D(name='block6_upsamp', size=(2,2))(x)
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='block6_upconv1', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='block6_upconv2', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Conv2D(512, (3,3), activation='relu', padding='same', name='block6_upconv3', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    # Block 7
    o = UpSampling2D(name='block7_upsamp', size=(2,2))(o)
    o = Conv2D(256, (3,3), activation='relu', padding='same', name='block7_conv1', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Conv2D(256, (3,3), activation='relu', padding='same', name='block7_conv2', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Conv2D(256, (3,3), activation='relu', padding='same', name='block7_conv3', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    # Block 8
    o = UpSampling2D(name='block8_upsamp', size=(2,2))(o)
    o = Conv2D(128, (3,3), activation='relu', padding='same', name='block8_conv1', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Conv2D(128, (3,3), activation='relu', padding='same', name='block8_conv2', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Conv2D(128, (3,3), activation='relu', padding='same', name='block8_conv3', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    # Block 9
    o = UpSampling2D(name='block9_upsamp', size=(2,2))(o)
    o = Conv2D(64, (3,3), activation='relu', padding='same', name='block9_conv1', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Conv2D(64, (3,3), activation='relu', padding='same', name='block9_conv2', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    # Block 10
    o = UpSampling2D(name='block10_upsamp', size=(2,2))(o)
    o = Conv2D(64, (3,3), activation='relu', padding='same', name='block10_conv1', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Conv2D(64, (3,3), activation='relu', padding='same', name='block10_conv2', data_format=image_ordering,
               kernel_initializer='he_normal')(o)
    o = Dense(2, activation='softmax')(o)
    model = keras.Model(inputs=img_input, outputs=o)
    return model