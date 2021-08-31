from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D

def CNN_test():
    model = Sequential()
    model.add(Conv2D(32,(5,5),activation='relu', input_shape=(512,512,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64,(5,5), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(Dense(1, activation='softmax'))
    return model