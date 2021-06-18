import pandas as pd
import numpy as np
import scipy as sp
import pingouin


def reject_outliers(df, depvars=None, group_columns=None, thr=2.5, outfile=None):
    if group_columns is None:
        group_columns = ['subnum']

    remove_rows = np.zeros_like(df[group_columns[0]], dtype='bool')
    for groupcol in group_columns:
        df_aggr = df.groupby(groupcol).mean()

        remove_inds = []
        for depvar in depvars:
            z = np.abs(sp.stats.zscore(df_aggr[depvar], nan_policy='omit'))
            to_remove = df_aggr.index[z > thr]
            # give some feedback
            if outfile is not None:
                for ind in to_remove:
                    print('removing {} {} as outlier for {}'.format(groupcol, ind, depvar),
                        file=outfile)
            remove_inds.extend(to_remove)

        remove_rows |= np.in1d(df[groupcol], remove_inds)

    return df[~remove_rows]


def many_bfs(t, n, **kwargs):
    scales = np.linspace(0.1, 1, num=100)
    bfs = np.asarray([pingouin.bayesfactor_ttest(t, n, r=r, **kwargs) for r in scales])
    return bfs, scales

# t-based 95% confidence interval
def compute_ci(dat, alpha=0.05):
    return sp.stats.t.interval(1-alpha, len(dat) - 1,
        loc=np.mean(dat),
        scale=sp.stats.sem(dat))
