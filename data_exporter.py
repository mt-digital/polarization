from glob import glob

import io
import numpy as np
import os
import pandas as pd
import pickle
import sys

help_msg = '''
Usage:
    python data_exporter.py <data_dir> <write_path>
'''

try:
    data_dir = sys.argv[1]
    write_path = sys.argv[2]
except:
    print(help_msg)
    sys.exit(1)

g = glob(os.path.join(data_dir, 'outputs_*'))

COLUMNS = ['trial', 'box_width', 'noise_level', 'iteration', 'polarization']
# full_df = pd.DataFrame(
#     columns=COLUMNS
# )
open(write_path, 'w').write(','.join(COLUMNS) + '\n')

iostream = io.StringIO()

for d in g:
    print('writing directory ' + d)

    files = os.listdir(d)
    trial_index = int(d.split('_')[-1])

    for i, f in enumerate(files):
        print(i)
        bw_str, nl_str = f.split('_')[1:]
        box_width = float(bw_str.split('=')[-1])
        noise_level = float(nl_str.split('=')[-1])

        full_file_path = os.path.join(d, f)
        e = pickle.load(open(full_file_path, 'rb'))

        history = e.history

        df = pd.DataFrame(columns=COLUMNS)

        df['iteration'] = np.array(list(history.keys()))
        df['polarization'] = np.array(
            [v['polarization'] for v in history.values()]
        )

        df['trial'] = trial_index
        df['box_width'] = box_width
        df['noise_level'] = noise_level

        df.to_csv(iostream, mode='a', header=False,
                  columns=COLUMNS, index=False)
        # full_df = full_df.append(df)

    open(write_path, 'a').write(iostream.getvalue())

# full_df.to_csv(write_path)
