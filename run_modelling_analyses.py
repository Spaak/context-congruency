import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.ion()
import arviz as az

mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5

# custom modules
import beh_analysis
import plots
import modelling


## set up figures etc.

save_figs = 'paper'

# control whether we should do sampling or not. Note that sampling can take a while, and
# also storing traces takes up quite a bit of space (~3GB).
do_sampling = False
#traces_path = '../traces'
traces_path = '/project/3018045.01/traces-4paper'
try:
    os.makedirs(traces_path)
except:
    pass

# output dir for summary stats and figures
output_dir = '../results/modelling'
try:
    os.makedirs(output_dir)
except:
    pass

# this is optional if you want publication-quality fonts
# if save_figs == 'paper':
#     plots.set_font(font='Helvetica Neue LT Std', stretch='condensed', size=14)
# else:
#     plots.set_font(font='Muli', size=16)

fig_counter = 0
def savefig(fig, title):
    global fig_counter
    fig_counter += 1
    fig.tight_layout()
    if save_figs == 'paper':
        fig.savefig('{}/{:03d}-{}.pdf'.format(output_dir, fig_counter, title))
    elif save_figs == 'png':
        fig.savefig('{}/{:03d}-{}.png'.format(output_dir, fig_counter, title), dpi=300)


outfile = open(output_dir + '/statresults-modelling.txt', 'w', buffering=1)
pd.set_option('display.max_columns', 12)


## load data

df_all_cd = pd.read_pickle('../data/exp1-cd.pkl.gz')
df_all_2afc = pd.read_pickle('../data/exp2-2afc.pkl.gz')


## reject outliers

df_all_cd = beh_analysis.reject_outliers(df_all_cd,
    depvars=['log_rt', 'log_loc_error', 'log_loc_rt'],
    group_columns=['subnum', 'item'])

df_all_2afc = beh_analysis.reject_outliers(df_all_2afc,
    depvars=['log_correct_rt', 'correct'],
    group_columns=['subnum', 'item'])


## cd: sampling

if do_sampling:
    df = df_all_cd.dropna()
    for depvar in ('log_rt', 'log_loc_error'):
        model = modelling.build_bambi_model(df, depvar,
            family='gaussian', factors='1 + con_incon')
        results = modelling.sample_and_get_results(model)

        az.to_netcdf(results, '{}/cd-{}.az'.format(traces_path, depvar))


## 2afc: sampling

if do_sampling:
    for depvar in ('correct', 'log_correct_rt'):
        if depvar == 'log_correct_rt':
            df = df_all_2afc.dropna()
            family = 'gaussian'
        else:
            df = df_all_2afc
            family = 'bernoulli'

        model = modelling.build_bambi_model(df, depvar,
            family=family, factors='1 + con_incon * probe_key_other')
        results = modelling.sample_and_get_results(model)

        az.to_netcdf(results, '{}/2afc-{}.az'.format(traces_path, depvar))


## both tasks: define variables and label them

# variables to inspect/plot (mapping from name to label)
variables_2afc = {'Intercept': 'Intercept',
    'con_incon': 'Congruency',
    'probe_key_other': 'Probe',
    'con_incon:probe_key_other': 'Congruency X Probe',
    '1|subnum_sigma': '$\sigma_{Subject}$(1)',
    'con_incon|subnum_sigma': '$\sigma_{Subject}$(Congruency)',
    'probe_key_other|subnum_sigma': '$\sigma_{Subject}$(Probe)',
    'con_incon:probe_key_other|subnum_sigma': '$\sigma_{Subject}$(Congruency X Probe)',
    '1|item_sigma': '$\sigma_{Item}$(1)',
    'con_incon|item_sigma': '$\sigma_{Item}$(Congruency)',
    'probe_key_other|item_sigma': '$\sigma_{Item}$(Probe)',
    'con_incon:probe_key_other|item_sigma': '$\sigma_{Item}$(Congruency X Probe)'}

variables_cd = {k: v for k, v in variables_2afc.items() if not 'probe' in k}

## cd: stats, plotting

has_refline = ['con_incon']
for depvar in ('log_rt', 'log_loc_error'):
    results = az.from_netcdf('{}/cd-{}.az'.format(traces_path, depvar))
    fig, allax = plt.subplots(3, 2, figsize=(5, 6))

    prop_above = plots.plot_posterior(results, allax.ravel(), variables_cd, has_refline)
    savefig(fig, 'model-cd-{}'.format(depvar))

    print('CD, {}'.format(depvar), file=outfile)
    summary = az.summary(results, kind='stats', var_names=list(variables_cd.keys()),
        round_to=6)
    print(summary.to_string() + '\n', file=outfile)
    print('p(<> 0):\n' + str(prop_above) + '\n', file=outfile)


## 2afc: stats, plotting

# variables to inspect/plot (mapping from name to label)
variables_2afc = {'Intercept': 'Intercept',
    'con_incon': 'Congruency',
    'probe_key_other': 'Probe',
    'con_incon:probe_key_other': 'Congruency X Probe',
    '1|subnum_sigma': '$\sigma_{Subject}$(1)',
    'con_incon|subnum_sigma': '$\sigma_{Subject}$(Congruency)',
    'probe_key_other|subnum_sigma': '$\sigma_{Subject}$(Probe)',
    'con_incon:probe_key_other|subnum_sigma': '$\sigma_{Subject}$(Congruency X Probe)',
    '1|item_sigma': '$\sigma_{Item}$(1)',
    'con_incon|item_sigma': '$\sigma_{Item}$(Congruency)',
    'probe_key_other|item_sigma': '$\sigma_{Item}$(Probe)',
    'con_incon:probe_key_other|item_sigma': '$\sigma_{Item}$(Congruency X Probe)'}

has_refline = ['con_incon', 'probe_key_other', 'con_incon:probe_key_other']

for depvar in ('correct', 'log_correct_rt'):
    results = az.from_netcdf('{}/2afc-{}.az'.format(traces_path, depvar))
    fig, allax = plt.subplots(3, 4, figsize=(9, 6))

    prop_above = plots.plot_posterior(results, allax.ravel(), variables_2afc, has_refline)
    savefig(fig, 'model-2afc-{}'.format(depvar))

    print('2AFC, {}'.format(depvar), file=outfile)
    summary = az.summary(results, kind='stats', var_names=list(variables_2afc.keys()),
        round_to=6)
    print(summary.to_string() + '\n', file=outfile)
    print('p(<> 0):\n' + str(prop_above) + '\n', file=outfile)


## wrap up

outfile.close()
plt.close('all')