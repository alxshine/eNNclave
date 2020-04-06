import json
import os

DATA_DIR = '/data/datasets/amazon'
JSON_FILE = 'Books.json'
CSV_FILE = 'books.csv'

with open(os.path.join(DATA_DIR, JSON_FILE), 'r') as input_file, open(os.path.join(DATA_DIR, CSV_FILE), 'w+') as output_file:
    output_file.write('rating,text\n')

    for line in input_file:
        json_dict = json.loads(line)
        rating = json_dict['overall']
        try:
            text = json_dict['reviewText'].replace('"','')

            output_file.write(f'{rating},"{text}"')
        except KeyError:
            continue
