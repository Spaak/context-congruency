import itertools as it

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()

mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5

import seaborn as sns

import os
import time

# prevent pingouin warnings about outdated versions
os.environ['OUTDATED_IGNORE'] = "1"
import pingouin
pingouin.options['round.column.CI95%'] = 4

import beh_analysis
import plots

## set up figures etc.

start_time = time.time()

# note: single-column figure is 3.3 inch wide
# double column figure is 6.89 inch wide

save_figs = 'paper'

# this is optional if you want publication-quality fonts
# if save_figs == 'paper':
#     plots.set_font(font='Helvetica Neue LT Std', stretch='condensed', size=14)
# else: 
#     plots.set_font(font='Muli', size=16)

fig_counter = 0
def savefig(fig, title):
    global fig_counter
    fig_counter += 1
    sns.despine(fig)
    fig.tight_layout()
    if save_figs == 'paper':
        fig.savefig('../results/{:03d}-{}.pdf'.format(fig_counter, title))
    elif save_figs == 'png':
        fig.savefig('../results/{:03d}-{}.png'.format(fig_counter, title), dpi=300)


outfile = open('../results/statresults.txt', 'w', buffering=1)
pd.set_option('display.max_columns', 12)


## load data

# data on disk still has outliers, this is only used in the very first plot for showing
# all data points and marking outliers
df_cd_outliers = pd.read_pickle('../data/exp1-cd.pkl.gz')
df_2afc_outliers = pd.read_pickle('../data/exp2-2afc.pkl.gz')

# base analysis on data without outliers
df_all_cd = beh_analysis.reject_outliers(df_cd_outliers,
    depvars=['log_rt', 'log_loc_error', 'log_loc_rt'],
    group_columns=['subnum', 'item'], outfile=outfile)
df_all_2afc = beh_analysis.reject_outliers(df_2afc_outliers,
    depvars=['log_correct_rt', 'correct'],
    group_columns=['subnum', 'item'], outfile=outfile)

# also load inconsistency scores
df_scores = pd.read_pickle('../data/consistency-scores.pkl.gz')


## report demographics

def _report_demographics(label, df):
    print('{} task:\nage\n'.format(label), file=outfile)
    print(df.groupby('subnum').mean()['age'].aggregate(['mean', 'std']).to_string(), file=outfile)
    print(df.groupby('subnum').first().groupby('sex').count()['prolific_id'].to_string() + '\n', file=outfile)

# report demographics for entire sample, including outliers
_report_demographics('CD', df_cd_outliers)
_report_demographics('2AFC', df_2afc_outliers)


## cd: overall plots

df = df_cd_outliers
df = df.groupby('subnum').mean()

fig, ax = plt.subplots(1, 2, figsize=(7, 5))
plots.distplot(df.rt/1000, lab='Reaction time (s)', ax=ax[0], mark_outliers=2.5)
plots.distplot(df.loc_error, lab='Localization error (%)', ax=ax[1], mark_outliers=2.5)

savefig(fig, 'overview-cd')

print('CD overall', file=outfile)
print(df_all_cd.groupby('subnum').mean()
    .aggregate(['mean', 'std']).T.to_string() + '\n', file=outfile)


## cd: condition scatters

def _cd_scatters(groupby):
    df_rt = df_all_cd.pivot_table(index=groupby, columns=['sCongruency'],
        values='log_rt')
    df_loc_error = df_all_cd.pivot_table(index=groupby, columns=['sCongruency'],
        values='log_loc_error')
    
    fig_scatter, ax_scatter = plt.subplots(1, 2, figsize=(9, 5))
    fig_diffs, ax_diffs = plt.subplots(2, 1, sharey=False, figsize=(3, 5))
    
    if groupby == 'item':
        kwargs = dict(s=14, marker='x')
    else:
        kwargs = dict()
    
    allstat = [
        plots.condition_scatter(ax_scatter[0], df_rt,
            title='Reaction time (log$_{10}$(RT/s))', tail='greater', **kwargs),
        plots.condition_scatter(ax_scatter[1], df_loc_error,
            title='Localization error (log$_{10}$)', tail='greater', **kwargs)
    ]
    
    print('CD, {}'.format(groupby), file=outfile)
    allstat = pd.concat(allstat)
    print(allstat.to_string() + '\n', file=outfile)
    
    # also output raw (non-transformed) condition means
    print('CD, {} raw means RT'.format(groupby), file=outfile)
    df_rt = df_all_cd.pivot_table(index=groupby, columns=['sCongruency'],
        values='rt')
    print(df_rt.mean().to_string() + '\n', file=outfile)
    
    plots.ci95_plot(ax_diffs[0], allstat[:1], 'Reaction time difference (log$_{10}$(RT/s))')
    plots.ci95_plot(ax_diffs[1], allstat[1:], 'Localization error difference (log$_{10}$)')
    
    savefig(fig_scatter, 'scatters-cd-{}'.format(groupby))
    savefig(fig_diffs, 'diff95ciplots-cd-{}'.format(groupby))

_cd_scatters('subnum')
_cd_scatters('item')


## cd: predictors of item-level effect

# looking at absolute predictors, not condition differences

df = df_all_cd[df_all_cd.con_incon==1]
df = df.groupby(['item', 'sCongruency']).mean()
df = df.merge(df_scores, left_index=True, right_index=True)
df = df.reset_index()

fig, allax = plt.subplots(2, 1, sharex='col', figsize=(3,6))

# labels for different variables
labdict = {
    'consistency_rating': 'Inconsistency rating', # higher = less consistent
    'correct': 'Accuracy (%)',
    'log_loc_error': 'Localization error (log$_{10}$)', # for CD task
    'log_rt': 'Reaction time (log$_{10}$(RT/s))', # for CD task
    'log_correct_rt': 'Reaction time (log$_{10}$(RT/s))'
}

allstat = []
xvar = 'consistency_rating'
for ax, yvar in zip(allax, ('log_rt', 'log_loc_error')):
    plots.regplot(df[xvar], df[yvar], ax=ax, marker='x', scatter_kwargs=dict(s=14))
    
    xlab = labdict[xvar] if np.equal(ax, allax[-1]).all() else None
    ylab = labdict[yvar]
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    stat = pingouin.corr(df[xvar], df[yvar], tail='less')
    stat.insert(0, 'label', xvar + ' X ' + yvar)
    allstat.append(stat)
    
print('CD, ratings X items', file=outfile)
allstat = pd.concat(allstat)
print(allstat.to_string() + '\n', file=outfile)

savefig(fig, 'item-correlations-cd')


## 2afc: overall plots

df = df_2afc_outliers
df = df.groupby('subnum').mean()

fig, allax = plt.subplots(1, 2, figsize=(7, 5))
plots.distplot(df.correct*100, lab='Accuracy (%)', ax=allax[0], mark_outliers=2.5)
plots.distplot(df.correct_rt/1000, lab='Reaction time (s)', ax=allax[1], mark_outliers=2.5)

savefig(fig, 'overview-2afc')

print('2AFC overall', file=outfile)
print(df_all_2afc.groupby('subnum').mean()
    .aggregate(['mean', 'std']).T.to_string() + '\n', file=outfile)


## 2afc: condition scatters

def _2afc_scatters(groupby):
    df_rt = df_all_2afc.pivot_table(index=groupby, columns=['sCongruency', 'sProbe'],
        values='log_correct_rt')
    df_acc = df_all_2afc.pivot_table(index=groupby, columns=['sCongruency', 'sProbe'],
        values='correct')
    df_acc *= 100
    
    fig_scatter, ax_scatter = plt.subplots(2, 2, sharex='row', sharey='row', figsize=(9, 9))
    fig_diffs, ax_diffs = plt.subplots(2, 1, sharex='col', sharey=False, figsize=(3, 9))
    
    if groupby == 'item':
        kwargs = dict(s=14, marker='x')
    else:
        kwargs = dict()
    
    allstat = [
        plots.condition_scatter(ax_scatter[0,0], df_acc,
            title='Accuracy (%)\nProbe-Key', cmap='Oranges', probe='key', tail='less',
            **kwargs),
        plots.condition_scatter(ax_scatter[0,1], df_acc,
            title='Accuracy (%)\nProbe-Other', cmap='Blues', probe='other', tail='greater',
            **kwargs),
        plots.condition_scatter(ax_scatter[1,0], df_rt,
            title='Reaction time (log$_{10}$(RT/s))\nProbe-Key', cmap='Oranges', probe='key',
            tail='greater', **kwargs),
        plots.condition_scatter(ax_scatter[1,1], df_rt,
            title='Reaction time (log$_{10}$(RT/s))\nProbe-Other', cmap='Blues', probe='other',
            tail='less', **kwargs)
    ]
    
    print('2AFC, {}'.format(groupby), file=outfile)
    allstat = pd.concat(allstat)
    print(allstat.to_string() + '\n', file=outfile)
    
    # also output raw (non-transformed) condition means
    print('2AFC, {} raw means RT'.format(groupby), file=outfile)
    df_rt = df_all_2afc.pivot_table(index=groupby, columns=['sCongruency', 'sProbe'],
        values='correct_rt')
    print(df_rt.mean().to_string() + '\n', file=outfile)
    
    plots.ci95_plot(ax_diffs[0], allstat[:2], 'Accuracy difference (%)')
    plots.ci95_plot(ax_diffs[1], allstat[2:], 'Reaction time difference (log$_{10}$(RT/s))')
    
    savefig(fig_scatter, 'scatters-2afc-{}'.format(groupby))
    savefig(fig_diffs, 'diff95ciplots-2afc-{}'.format(groupby))

_2afc_scatters('subnum')
_2afc_scatters('item')


## 2afc: effect of between-subjects factor

df = df_all_2afc.groupby(['subnum', 'con_incon', 'probe_key_other']).mean()[
    ['correct', 'log_correct_rt', 'correct_rt', 'prop_incon_relevant']].reset_index()

incon, probe = map(lambda x: np.asarray(x, dtype='bool'), (df.con_incon, df.probe_key_other))
rt, acc, rawrt = map(np.asarray, (df.log_correct_rt, df.correct, df.correct_rt))

# RT: con-key vs incon-key; acc the other way around
incon_benefit_rt = rt[(~incon) & probe] - rt[incon & probe]
incon_benefit_acc = (acc[incon & probe] - acc[(~incon) & probe])*100

ies = (rawrt/1000) / acc
incon_benefit_ies = (ies[(~incon) & probe] - ies[incon & probe])

prop = np.round(np.asarray(df.prop_incon_relevant)[0::4]*100).astype('int')

df = pd.DataFrame(data=dict(rt=incon_benefit_rt, acc=incon_benefit_acc,
    ies=incon_benefit_ies, p=prop))

# also compute and draw CI on the overall data
cis_overall = [beh_analysis.compute_ci(x) for x in [df.acc, df.rt, df.ies]]

fig, allax = plt.subplots(3, 1, sharex=True, figsize=(5, 7))
allstat = []
for ax, depvar, ci  in zip(allax, ('acc', 'rt', 'ies'), cis_overall):
    stat = pingouin.corr(df.p, df[depvar], tail='two-sided', method='spearman')
    stat.insert(0, 'label', depvar)
    # also compute Ly Bayes factor
    stat.insert(len(stat.columns), 'BF10',
        pingouin.bayesfactor_pearson(stat['r'][0], stat['n'][0], method='ly', kappa=1.0))
    allstat.append(stat)
    
    sns.pointplot(x='p', y=depvar, ax=ax, data=df, alpha=0.3, color='k',
        ci=('parametric', 95), markers='None')
    ax.axhline(ls=':', c='k')
    ax.axhspan(*ci, color=sns.color_palette('Greens')[-2], alpha=0.1, zorder=-1)
    if depvar == 'ies':
        ax.set_xlabel('p(Incongruent = relevant) (%)')
    else:
        ax.set_xlabel('')
    
print('2AFC, between-subjects effect', file=outfile)
allstat = pd.concat(allstat)
print(allstat.to_string() + '\n', file=outfile)

savefig(fig, 'between-subjects-2afc')


## 2afc, control: check congruency effect specifically for p(Key|Incongruent) = 17%

df = df[df.p==17]

allstat = []
for depvar  in ('acc', 'rt', 'ies'):
    # note: tail is always greater here since we code it as Con-Incon for RT/IES but as
    # Incon-Con for Acc (see cell above)
    stat = pingouin.ttest(df[depvar], 0, tail='greater', r=0.33)
    stat.insert(0, 'label', depvar)
    allstat.append(stat)
    
print('2AFC, congruency effect only for p(Key|Incongruent) = 17% subjects', file=outfile)
allstat = pd.concat(allstat)
print(allstat.to_string() + '\n', file=outfile)

## 2afc: predictors of item-level effect

df = df_all_2afc.copy()
df = df[(df.probe_key_other==1) & (df.con_incon==1)] # only probe-key/incongruent
df = df.groupby(['item', 'sCongruency']).mean()

df = df.merge(df_scores, left_index=True, right_index=True)

df = df.reset_index()
df.correct *= 100

fig, allax = plt.subplots(2, 1, sharex='col', sharey='row', figsize=(3,6))

taildict = {
    ('consistency_rating', 'correct'): 'greater',
    ('consistency_rating', 'log_correct_rt'): 'less'
}

allstat = []
xvar = 'consistency_rating'
for ax, yvar in zip(allax, ('correct', 'log_correct_rt')):
    plots.regplot(df[xvar], df[yvar], ax=ax, scatter_kwargs=dict(s=14),
        marker='x')
    
    xlab = labdict[xvar] if np.equal(ax, allax[-1]).all() else None
    ylab = labdict[yvar]
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    stat = pingouin.corr(df[xvar], df[yvar], tail=taildict[(xvar,yvar)])
    stat.insert(0, 'label', xvar + ' X ' + yvar)
    allstat.append(stat)
    
print('2AFC, ratings X items', file=outfile)
allstat = pd.concat(allstat)
print(allstat.to_string() + '\n', file=outfile)

savefig(fig, 'item-correlations-2afc')


## 2afc: relate item-level probe-key effect to probe-other

df = df_all_2afc.copy()
df = df.groupby(['item', 'sCongruency', 'sProbe']).mean()
df.correct *= 100

fig, allax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(8,4))

con = ~np.asarray(df.con_incon, dtype='bool')
key = np.asarray(df.probe_key_other, dtype='bool')

allstat = []
for ax, depvar in zip(allax, ('correct', 'log_correct_rt')):
    xdat = df[depvar][(~con) & key].array - df[depvar][con & key].array
    ydat = df[depvar][(~con) & (~key)].array - df[depvar][con & (~key)].array
    
    plots.regplot(xdat, ydat, ax=ax, scatter_kwargs=dict(s=14),
        marker='x')
    
    ax.set_xlabel('Probe-Key effect')
    ax.set_ylabel('Probe-Other effect')
    ax.set_title(labdict[depvar])
    
    stat = pingouin.corr(xdat, ydat, tail='less')
    stat.insert(0, 'label', depvar)
    allstat.append(stat)
    
print('2AFC, Key X Other effect', file=outfile)
allstat = pd.concat(allstat)
print(allstat.to_string() + '\n', file=outfile)

savefig(fig, 'item-correlations-2afc-key-other')


## cd X 2afc: are the effects related across items?

# per-item congruency effect scores
df_cd = df_all_cd.groupby(['item', 'sCongruency'], sort=True).mean()
df_2afc = df_all_2afc[df_all_2afc.probe_key_other==1].groupby(
    ['item', 'sCongruency'], sort=True).mean()

df_2afc.correct *= 100

# bring effects into same range by zscoring per condition, per experiment
for con_incon in (0,1):
    for depvar in ('log_rt', 'log_loc_error'):
        df_cd.loc[df_cd.con_incon==con_incon,depvar] = sp.stats.zscore(
            df_cd.loc[df_cd.con_incon==con_incon,depvar])
    for depvar in ('log_correct_rt', 'correct'):
        df_2afc.loc[df_2afc.con_incon==con_incon,depvar] = sp.stats.zscore(
            df_2afc.loc[df_2afc.con_incon==con_incon,depvar])

# do a "manual" inner join to account for an item being rejected as outlier
# in one data set, but not in the other
inds_present = df_cd.index.intersection(df_2afc.index)
df_cd = df_cd[df_cd.index.isin(inds_present)]
df_2afc = df_2afc[df_2afc.index.isin(inds_present)]
assert np.all(df_cd.index == df_2afc.index)

labdict = {
    'log_loc_error': 'Localization error (z)\nCongruent - Incongruent',
    'log_rt': 'Reaction time (z)\nCongruent - Incongruent',
    'log_correct_rt': 'Reaction time (z)\nCongruent - Incongruent',
    'correct': 'Accuracy (z)\nCongruent - Incongruent'
}

fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(8, 8))

allstat = []
for k, var_cd in enumerate(('log_rt', 'log_loc_error')):
    for l, var_2afc in enumerate(('correct', 'log_correct_rt')):    
        con = ~np.asarray(df_cd.con_incon, dtype='bool')
        effect_cd = df_cd[var_cd][con].array - df_cd[var_cd][~con].array
        con = ~np.asarray(df_2afc.con_incon, dtype='bool')
        effect_2afc = (df_2afc[var_2afc][con].array -
            df_2afc[var_2afc][~con].array)
        
        plots.regplot(effect_cd, effect_2afc, ax=ax[l,k], marker='x',
            scatter_kwargs=dict(s=14))
        
        if l == 1:
            ax[l,k].set_xlabel(labdict[var_cd])
        if k == 0:
            ax[l,k].set_ylabel(labdict[var_2afc])
        
        tail = 'less' if var_2afc == 'correct' else 'greater'
        stat = pingouin.corr(effect_cd, effect_2afc, tail=tail)
        stat.insert(0, 'label', 'CD-{} X 2AFC-{}'.format(var_cd, var_2afc))
        allstat.append(stat)
    
print('CD X 2AFC', file=outfile)
allstat = pd.concat(allstat)
print(allstat.to_string() + '\n', file=outfile)

savefig(fig, 'item-cdX2afc')


## control analysis: effect of JZS prior scale on BF for t-test of most important effects

# cd, rt
fig, ax = plt.subplots(1, 3, sharex=True, figsize=(14, 5))

def _many_bfs_single_contrast(df, ax, tail):
    stat = pingouin.ttest(df['congruent']-df['incongruent'], 0, tail=tail)
    bfs, scales = beh_analysis.many_bfs(stat['T'][0], len(df), tail=tail)
    ax.plot(scales, bfs)
    ax.set_xlabel('Cauchy scale for effect size prior')
    ax.set_ylabel('Bayes factor for effect')
    ax.axvline(0.33, ls=':', c='k')
    ax.locator_params(nbins=4)

df = df_all_cd.pivot_table(index='subnum', columns=['sCongruency'], values='log_rt')
_many_bfs_single_contrast(df, ax[0], tail='greater')
ax[0].set_title('Experiment 1\nReaction time effect')

df = df_all_2afc[df_all_2afc.sProbe == 'key']
df = df.pivot_table(index='subnum', columns=['sCongruency'], values='correct')
_many_bfs_single_contrast(df, ax[1], tail='less')
ax[1].set_title('Experiment 2\n2AFC accuracy, Probe-Key')

df = df_all_2afc[df_all_2afc.sProbe == 'other']
df = df.pivot_table(index='subnum', columns=['sCongruency'], values='correct')
_many_bfs_single_contrast(df, ax[2], tail='greater')
ax[2].set_title('Experiment 2\n2AFC accuracy, Probe-Other')

savefig(fig, 'control-bf-scale')


## wrap up

outfile.close()
plt.close('all')

elapsed = time.time() - start_time
print('total elapsed time: {} seconds'.format(round(elapsed)))