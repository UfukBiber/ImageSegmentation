import tensorflow as tf
import random, os
import matplotlib.pyplot as plt
import numpy as np


inputs = tf.keras.Input(shape=(200, 200, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = tf.keras.layers.Conv2D(32, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
x = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x)

x = tf.keras.layers.Conv2DTranspose(256, 3,  activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)

outputs = tf.keras.layers.Conv2D(3, 3, activation="softmax",padding="same")(x)

model = tf.keras.Model(inputs, outputs)
model.load_weights("Model_1")




def Inppath2Image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = (200, 200))
    img = tf.cast(img, tf.uint8)
    return img

def tarPath2Image(path):
    tar = tf.io.read_file(path)
    tar = tf.io.decode_png(tar, channels = 1)
    tar = tf.image.resize(tar, size = (200, 200))
    tar = (tf.cast(tar, tf.uint8) -1) * 127
    return tar

Inppaths = sorted([path for path in os.listdir("images") if path.endswith(".jpg")])
imgPath = os.path.join("images", Inppaths[-15])

tarPaths = sorted([path for path in os.listdir("annotations/trimaps") if (not path.startswith(".") and path.endswith(".png"))])
tarPath = os.path.join("annotations/trimaps", tarPaths[-15])

img = Inppath2Image(imgPath)
tar = tarPath2Image(tarPath)

output = model.predict(img[tf.newaxis,:])[0]
output = np.argmax(output, axis = -1)
output = output * 127


fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.imshow(img)
ax2.imshow(tar)
ax3.imshow(output)
plt.show()