import tensorflow as tf


def preprocess(x, y, img_size):
    image = tf.io.read_file(x)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (img_size, img_size))

    return image, y


def preprocess_flowers(x, y):
    return preprocess(x, y, 224)


def preprocess_lfw(x, y):
    return preprocess(x, y, 250)


def preprocess_faces(x, y):
    return preprocess(x, y, 224)


def preprocess_224(x, y):
    return preprocess(x, y, 224)


def preprocess_250(x, y):
    return preprocess(x, y, 250)


def generate_dataset(x, y, preprocess_function=preprocess_224,
                     batch_size=32, repeat=True, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess_function)

    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size)
    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds
