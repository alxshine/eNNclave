import json

JSON_FILE = '/data/datasets/Books.json'
CSV_FILE = '/data/datasets/books.csv'

with open(JSON_FILE, 'r') as input_file, open(CSV_FILE, 'w+') as output_file:
    output_file.write('rating,text\n')

    for line in input_file:
        json_dict = json.loads(line)
        rating = json_dict['overall']
        try:
            text = json_dict['reviewText'].replace('"','')

            output_file.write(f'{rating},"{text}"')
        except KeyError:
            continue
