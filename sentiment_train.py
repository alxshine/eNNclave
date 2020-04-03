"""Model for sentiment analysis.

The model makes use of concatenation of two CNN layers with
different kernel sizes.
"""
import argparse

import tensorflow as tf
import numpy as np

_DROPOUT_RATE = 0.95

DATASET_IMDB = "imdb"


def load(dataset, vocabulary_size, sentence_length):
  """Returns training and evaluation input.
  Args:
    dataset: Dataset to be trained and evaluated.
      Currently only imdb is supported.
    vocabulary_size: The number of the most frequent tokens
      to be used from the corpus.
    sentence_length: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    A tuple of length 4, for training sentences, labels,
    evaluation sentences, and evaluation labels,
    each being an numpy array.
  """
  if dataset == DATASET_IMDB:
    return _load(vocabulary_size, sentence_length)
  else:
    raise ValueError("unsupported dataset: " + dataset)


def get_num_class(dataset):
  """Returns an integer for the number of label classes.
  Args:
    dataset: Dataset to be trained and evaluated.
      Currently only imdb is supported.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    int: The number of label classes.
  """
  if dataset == DATASET_IMDB:
    return NUM_CLASS
  else:
    raise ValueError("unsupported dataset: " + dataset)

NUM_CLASS = 2


def _load(vocabulary_size, sentence_length):
  """Returns training and evaluation input for imdb dataset.
  Args:
    vocabulary_size: The number of the most frequent tokens
      to be used from the corpus.
    sentence_length: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
  Raises:
    ValueError: if the dataset value is not valid.
  Returns:
    A tuple of length 4, for training and evaluation data,
    each being an numpy array.
  """
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
      path="imdb.npz",
      num_words=vocabulary_size,
      skip_top=0,
      maxlen=None,
      seed=113,
      start_char=START_CHAR,
      oov_char=OOV_CHAR,
      index_from=OOV_CHAR+1)

  x_train_processed = []
  for sen in x_train:
    sen = pad_sentence(sen, sentence_length)
    x_train_processed.append(np.array(sen))
  x_train_processed = np.array(x_train_processed)

  x_test_processed = []
  for sen in x_test:
    sen = pad_sentence(sen, sentence_length)
    x_test_processed.append(np.array(sen))
  x_test_processed = np.array(x_test_processed)

  return x_train_processed, np.eye(NUM_CLASS)[y_train], \
         x_test_processed, np.eye(NUM_CLASS)[y_test]

START_CHAR = 1
END_CHAR = 2
OOV_CHAR = 3


def pad_sentence(sentence, sentence_length):
  """Pad the given sentense at the end.
  If the input is longer than sentence_length,
  the remaining portion is dropped.
  END_CHAR is used for the padding.
  Args:
    sentence: A numpy array of integers.
    sentence_length: The length of the input after the padding.
  Returns:
    A numpy array of integers of the given length.
  """
  sentence = sentence[:sentence_length]
  if len(sentence) < sentence_length:
    sentence = np.pad(sentence, (0, sentence_length - len(sentence)),
                      "constant", constant_values=(START_CHAR, END_CHAR))

  return sentence

class CNN(tf.keras.models.Model):
  """CNN for sentimental analysis."""

  def __init__(self, emb_dim, num_words, sentence_length, hid_dim,
               class_dim, dropout_rate):
    """Initialize CNN model.

    Args:
      emb_dim: The dimension of the Embedding layer.
      num_words: The number of the most frequent tokens
        to be used from the corpus.
      sentence_length: The number of words in each sentence.
        Longer sentences get cut, shorter ones padded.
      hid_dim: The dimension of the Embedding layer.
      class_dim: The number of the CNN layer filters.
      dropout_rate: The portion of kept value in the Dropout layer.
    Returns:
      tf.keras.models.Model: A Keras model.
    """

    input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)

    layer = tf.keras.layers.Embedding(num_words, output_dim=emb_dim)(input_layer)

    layer_conv3 = tf.keras.layers.Conv1D(hid_dim, 3, activation="relu")(layer)
    layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)

    layer_conv4 = tf.keras.layers.Conv1D(hid_dim, 2, activation="relu")(layer)
    layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)

    layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3], axis=1)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    output = tf.keras.layers.Dense(class_dim, activation="softmax")(layer)

    super(CNN, self).__init__(inputs=[input_layer], outputs=output)

"""Main function for the sentiment analysis model.

The model makes use of concatenation of two CNN layers with
different kernel sizes. See `sentiment_model.py`
for more details about the models.
"""

def run_model(dataset_name, emb_dim, voc_size, sen_len,
              hid_dim, batch_size, epochs):
  """Run training loop and an evaluation at the end.

  Args:
    dataset_name: Dataset name to be trained and evaluated.
    emb_dim: The dimension of the Embedding layer.
    voc_size: The number of the most frequent tokens
      to be used from the corpus.
    sen_len: The number of words in each sentence.
      Longer sentences get cut, shorter ones padded.
    hid_dim: The dimension of the Embedding layer.
    batch_size: The size of each batch during training.
    epochs: The number of the iteration over the training set for training.
  """

  model = CNN(emb_dim, voc_size, sen_len,
                              hid_dim, get_num_class(dataset_name),
                              _DROPOUT_RATE)
  model.summary()

  model.compile(loss="categorical_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

  tf.compat.v1.logging.info("Loading the data")
  x_train, y_train, x_test, y_test = load(
      dataset_name, voc_size, sen_len)

  model.fit(x_train, y_train, batch_size=batch_size,
            validation_split=0.4, epochs=epochs)
  score = model.evaluate(x_test, y_test, batch_size=batch_size)
  tf.compat.v1.logging.info("Score: {}".format(score))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--dataset", help="Dataset to be trained "
                                              "and evaluated.",
                      type=str, choices=["imdb"], default="imdb")

  parser.add_argument("-e", "--embedding_dim",
                      help="The dimension of the Embedding layer.",
                      type=int, default=512)

  parser.add_argument("-v", "--vocabulary_size",
                      help="The number of the words to be considered "
                           "in the dataset corpus.",
                      type=int, default=6000)

  parser.add_argument("-s", "--sentence_length",
                      help="The number of words in a data point."
                           "Entries of smaller length are padded.",
                      type=int, default=600)

  parser.add_argument("-c", "--hidden_dim",
                      help="The number of the CNN layer filters.",
                      type=int, default=512)

  parser.add_argument("-b", "--batch_size",
                      help="The size of each batch for training.",
                      type=int, default=500)

  parser.add_argument("-p", "--epochs",
                      help="The number of epochs for training.",
                      type=int, default=55)

  args = parser.parse_args()

  run_model(args.dataset, args.embedding_dim, args.vocabulary_size,
            args.sentence_length, args.hidden_dim,
            args.batch_size, args.epochs)
