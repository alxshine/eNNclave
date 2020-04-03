#!/bin/bash
#get data from here: https://www.kaggle.com/bittlingmayer/amazonreviews

train_file='data/amazon/train.ft.txt'
test_file='data/amazon/test.ft.txt'

# train file
echo -e "Sentiment\tPhrase" | cat $train_file > /tmp/train.ft.txt && mv /tmp/train.ft.txt $train_file
sed -i 's/^__label__1 /1\t/' data/amazon/train.ft.txt
sed -i 's/^__label__2 /2\t/' data/amazon/train.ft.txt

# test file
echo -e "Sentiment\tPhrase" | cat $test_file > /tmp/test.ft.txt && mv /tmp/test.ft.txt $test_file
sed -i 's/^__label__1 /1\t/' data/amazon/test.ft.txt
sed -i 's/^__label__2 /2\t/' data/amazon/test.ft.txt
