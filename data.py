import csv
# import numpy as np

with open ('./diagnosis.data', 'r', encoding='utf-16') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        print(row)
