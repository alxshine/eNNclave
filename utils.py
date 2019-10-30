import tensorflow as tf


def preprocess_flowers(x, y, img_size=224):
    image = tf.compat.v1.read_file(x)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (img_size, img_size))

    return image, y


def preprocess_lfw(x, y, img_size=250):
    image = tf.io.read_file(x)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (img_size, img_size))

    return image, y


def generate_dataset(x, y, preprocess_function=preprocess_flowers,
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
