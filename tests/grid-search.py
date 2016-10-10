import numpy as np
import pandas as pd
import subprocess

res = []
lrs = [2 ** -e for e in range(1, 7)]

for lr in lrs:
    for lr_wv in lrs:
        print (lr, lr_wv)
        train_cmd = ("fasttext bilingual "
            "-input s-5000.txt "
            "-input-par1 s-100-u.txt "
            "-input-par2 t-100-u.txt "
            "-output ./models/dev "
            "-dim 10 -lr %f -lr_wv %f") % (lr, lr_wv)
        
        test_source_cmd = 'fasttext test ./models/dev-sup.bin ./s-5000-2.txt'
        test_target_cmd = 'fasttext test ./models/dev-sup.bin ./t-5000-2.txt'
        clean_cmd = 'rm ./models/dev-sup.bin'
        
        _ = subprocess.call(train_cmd.split())
        
        res.append((
            lr, 
            lr_wv,
            subprocess.check_output(test_source_cmd, shell=True),
            subprocess.check_output(test_target_cmd, shell=True)
        ))
        
        print
        print '!! Test'
        for r in res[-1]:
            print r
        
        print
        
        subprocess.call(clean_cmd.split())

pickle.dump(res, open('./res-100.pkl', 'w'))


def parse_(x):
    return float(x.split()[1])    

def parse(x):
    return [x[0], x[1], parse_(x[2]), parse_(x[3])]

df = pd.DataFrame(map(parse, res))
df.columns = ('lr', 'lr_wv', 'source', 'target')

from seaborn import plt
source_perf = pd.pivot_table(df[['lr', 'lr_wv', 'source']], columns=['lr'], index=['lr_wv'])
seaborn.heatmap(np.array(source_perf))
plt.show()

target_perf = pd.pivot_table(df[['lr', 'lr_wv', 'target']], columns=['lr'], index=['lr_wv'])
seaborn.heatmap(np.array(target_perf))
plt.show()
