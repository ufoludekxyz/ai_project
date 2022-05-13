import pandas as pd

par_names = ['Temperature of patient',
             'Occurrence of nausea',
             'Lumbar pain',
             'Urine pushing',
             'Micturition pains',
             'Burning of urethra, itch, swelling of urethra outlet',
             'decision: Inflammation of urinary bladder',
             'decision: Nephritis of renal pelvis origin'
             ]

data_list = pd.read_csv('./diagnosis.data', sep='\t', encoding='utf-16', header=None, names=par_names)

data_list = data_list.replace({'no': 0, 'yes': 1})
