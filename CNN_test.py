from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D

def CNN_test():
    model = Sequential()
    model.add(Conv2D(6,(3,3), padding='same', activation='relu', input_shape=(512,512,3)))
    model.add(Conv2D(6, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Conv2D(9,(3,3), padding='same', activation='relu'))
    model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(9, (3, 3), padding='same', activation='relu'))

    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(6, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(6, (3, 3), padding='same', activation='relu'))
    
    model.add(Dense(1,activation='softmax'))
    model.summary()
    return model
