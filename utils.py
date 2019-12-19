import tensorflow as tf

def get_all_layers(model):
    """ Get all layers of model, including ones inside a nested model """
    layers = []
    for l in model.layers:
        if hasattr(l, 'layers'):
            layers += get_all_layers(l)
        else:
            layers.append(l)
    return layers

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

def preprocess_mnist(x, y):
    x = tf.cast(x, tf.float32)
    x = (x/127.5) - 1
    return x, y

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

def get_num_classes(labels):
    """Gets the total number of classes.

    # Arguments
        labels: list, label values.
            There should be at lease one sample for values in the
            range (0, num_classes -1)

    # Returns
        int, total number of classes.

    # Raises
        ValueError: if any label value in the range(0, num_classes - 1)
            is missing or if number of classes is <= 1.
    """
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}. Please make sure you have '
                         'at least one sample for every label value '
                         'in the range(0, {max_class})'.format(
                            missing_classes=missing_classes,
                            max_class=num_classes - 1))

    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes


def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)
