import tensorflow as tf
import random, os
import matplotlib.pyplot as plt
import numpy as np

Validation_Split = 1000
img_size = (200, 200)
input_dir = "images"
target_dir = "annotations/trimaps"

input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
target_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith(".png") and not fname.startswith(".")])

input_img_paths = input_img_paths[:4000]
target_paths = target_paths[:4000]

num_imgs = len(input_img_paths)

random.Random(1231).shuffle(input_img_paths)
random.Random(1231).shuffle(target_paths)

trainInpPaths, trainTarPaths = input_img_paths[:-1000], target_paths[:-1000]
valInpPaths, valTarPaths = input_img_paths[-1000:], target_paths[-1000:]


train_ds = tf.data.Dataset.from_tensor_slices((trainInpPaths, trainTarPaths))
val_ds = tf.data.Dataset.from_tensor_slices((valInpPaths, valTarPaths))


def Inppath2Image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, size = (200, 200))
    img = tf.cast(img, tf.float32)
    return img

def tarPath2Image(path):
    tar = tf.io.read_file(path)
    tar = tf.io.decode_png(tar, channels = 1)
    tar = tf.image.resize(tar, size = (200, 200))
    tar = tf.cast(tar, tf.uint8) - 1
    return tar

train_ds = train_ds.map(lambda x, y:(Inppath2Image(x), tarPath2Image(y)), num_parallel_calls=4)
val_ds = val_ds.map(lambda x, y:(Inppath2Image(x), tarPath2Image(y)), num_parallel_calls=4)

train_ds = train_ds.batch(16).shuffle(512).prefetch(32)
val_ds = val_ds.batch(16).shuffle(512).prefetch(32)

# for x, y in train_ds.take(1):
#     fig, (ax1, ax2) = plt.subplots(2)
#     ax1.imshow(x[0])
#     ax2.imshow(y[0])
#     plt.show()

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
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])
callbacks = [ tf.keras.callbacks.ModelCheckpoint("oxford_segmentation.keras", save_best_only=True)]

model.fit(train_ds, validation_data = val_ds, callbacks = [tf.keras.callbacks.ModelCheckpoint("Model_1", save_best_only = True, save_weights_only = True)],  epochs = 30)