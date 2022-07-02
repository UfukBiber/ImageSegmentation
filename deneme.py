import tensorflow as tf 
import os 
import matplotlib.pyplot as plt 


Inppaths = sorted([path for path in os.listdir("images") if path.endswith(".jpg")])
print(Inppaths[0:5])
imgPath = os.path.join("images", Inppaths[5])

tarPaths = sorted([path for path in os.listdir("annotations/trimaps") if (not path.startswith(".") and path.endswith(".png"))])
print(tarPaths[0:5])
tarPath = os.path.join("annotations/trimaps", tarPaths[5])

print(imgPath)
print(tarPath)

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

img = Inppath2Image(imgPath)
tar = tarPath2Image(tarPath)

fig, (ax1, ax2) = plt.subplots(2)
ax1.imshow(img)
ax2.imshow(tar)
plt.show()