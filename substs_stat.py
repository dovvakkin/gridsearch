from pathlib import Path
import pandas as pd
import numpy as np 
from time import time

def load_substs(substs_fname, limit=None):
    st = time()
    p = Path(substs_fname)
    print(time()-st, 'Loading substs from ', p)
    if substs_fname.endswith('.npz'):
        arr_dict = np.load(substs_fname, allow_pickle=True)
        print(arr_dict['substs'])
        substs_probs = pd.DataFrame({'substs':arr_dict['substs'], 'probs':arr_dict['probs']}).apply(lambda r: [(p,s) for s,p in zip(r.substs, r.probs)], axis=1)
        print(substs_probs.head(3))
    else:
        substs_probs = pd.read_csv(p, index_col=0, nrows=limit)['0']
        print(time()-st, 'Eval... ', p)
        substs_probs = substs_probs.apply(pd.eval)
        print(time()-st, 'Reindexing... ', p)
        substs_probs.reset_index(inplace = True, drop = True)

        szip = substs_probs.apply(lambda l: zip(*l)).apply(list)
        res_probs, res_substs = szip.str[0].apply(list), szip.str[1].apply(list)
        pd.DataFrame({'probs':res_probs, 'substs':res_substs}).to_csv(p.parent/(p.name.rstrip('.bz2')[:-1]+'.fast.bz2'),sep='\t')

    p_ex = p.parent / (p.name+'.input')
    print(time()-st,'Loading examples from ', p_ex)
    dfinp = pd.read_csv(p_ex, nrows=limit)
    dfinp['substs_probs'] = substs_probs
    print(dfinp.head())
    return dfinp

def sstat(substs_fname, limit=None):
    dfinp = load_substs(substs_fname, limit)
    print('Counting stat')
    rdf = dfinp.groupby('word').agg({'word_at':lambda x:pd.Series(x).value_counts().to_dict(), 'substs_probs':lambda x:pd.Series(s for l in x for p,s in l).value_counts().to_dict()})
    for col in ['word_at', 'substs_probs']:
        rdf[col+'_head'] = rdf[col].apply(lambda d: sorted(d.items(), key=lambda e: -e[1])[:15])
    stats_fname = substs_fname + '.sstat.tsv'
    rdf.to_csv(stats_fname, index=False, sep='\t')
    print('Stats saved to ', stats_fname)

from fire import Fire
if __name__=='__main__':
    Fire(sstat)
