#!/bin/bash
#get data from here: https://www.kaggle.com/bittlingmayer/amazonreviews

train_file='data/amazon/train.ft.txt'
train_dest_file='data/amazon/train_cleaned.txt'
test_file='data/amazon/test.ft.txt'
test_dest_file='data/amazon/test_cleaned.txt'

# train file
sed -e '1s;^;Sentiment\tPhrase\r\n;' -e 's/__label__1 /0\t/' -e 's/__label__2 /1\t/' $train_file > $train_dest_file

# test file
sed -e '1s;^;Sentiment\tPhrase\r\n;' -e 's/__label__1 /0\t/' -e 's/__label__2 /1\t/' $test_file > $test_dest_file
