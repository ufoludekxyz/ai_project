import pandas as pd

# Names for header
par_names = ['Temperature of patient',
             'Occurrence of nausea',
             'Lumbar pain',
             'Urine pushing',
             'Micturition pains',
             'Burning of urethra, itch, swelling of urethra outlet',
             'decision: Inflammation of urinary bladder',
             'decision: Nephritis of renal pelvis origin'
             ]


# Reading file with data (setting tab as values separator, encoding to utf-16 and float separator to comma)
data_list = pd.read_csv('./diagnosis.data', sep='\t', encoding='utf-16', header=None, thousands=',', names=par_names)

# Converting yes to 1 and no to 0
data_list = data_list.replace({'no': 0, 'yes': 1})

# Converting string to float
data_list['Temperature of patient'] = data_list['Temperature of patient'].astype(float)

# Min-max normalization function definition
def min_max_norm(column):
    return (column - column.min()) / (column.max() - column.min())

# Copy of initial DataFrame
norm_data = data_list.copy()

print(data_list.dtypes)
print(data_list)
