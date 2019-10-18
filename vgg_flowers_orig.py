import pathlib
import random
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

AUTOTUNE = tf.data.experimental.AUTOTUNE

# data_dir = tf.keras.utils.get_file(
    # 'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
data_dir = 'data/flowers/raw'
data_dir = pathlib.Path(data_dir)

label_names = {'daisy': 0, 'dandelion': 1,
               'roses': 2, 'sunflowers': 3, 'tulips': 4}
label_key = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

all_images = list(data_dir.glob('*/*'))
all_images = [str(path) for path in all_images]
random.shuffle(all_images)

all_labels = [label_names[pathlib.Path(path).parent.name]
              for path in all_images]

data_size = len(all_images)

train_test_split = (int)(data_size*0.2)

x_train = all_images[train_test_split:]
x_test = all_images[:train_test_split]

y_train = all_labels[train_test_split:]
y_test = all_labels[:train_test_split]

IMG_SIZE = 160

BATCH_SIZE = 32


def _parse_data(x, y):
    image = tf.compat.v1.read_file(x)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, y


def _input_fn(x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(_parse_data)
    ds = ds.shuffle(buffer_size=data_size)

    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_ds = _input_fn(x_train, y_train)
validation_ds = _input_fn(x_test, y_test)

breakpoint()

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

model_file = 'models/vgg_flowers_orig.h5'

VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                          include_top=False,
                                          weights='imagenet')
VGG16_MODEL.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(
    len(label_names), activation='softmax')
model = tf.keras.Sequential([
    VGG16_MODEL,
    global_average_layer,
    prediction_layer
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
history = model.fit(train_ds,
                    epochs=100,
                    steps_per_epoch=2,
                    validation_steps=2,
                    validation_data=validation_ds)
model.save(model_file)


validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_ds, steps=validation_steps)

print("loss: {:.2f}".format(loss0))
print("accuracy: {:.2f}".format(accuracy0))
