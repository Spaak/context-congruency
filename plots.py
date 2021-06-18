import itertools
import itertools as it

import numpy as np
import scipy as sp

import pandas as pd
import pingouin

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator

import seaborn as sns

import arviz as az


## helper functions

def set_font(font='Muli', stretch=None, weight=None, size=16, scan_new=True):
    if scan_new:
        font_dirs = ['/home/predatt/eelspa/.fonts']
        font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
        font_list = font_manager.createFontList(font_files)
        font_manager.fontManager.ttflist.extend(font_list)
    
    mpl.rcParams['font.family'] = font
    
    if stretch is not None:
        mpl.rcParams['font.stretch'] = stretch
    if weight is not None:
        mpl.rcParams['font.weight'] = weight
    
    if size == 'psychscience':
        mpl.rcParams['font.size'] = 9
        mpl.rcParams['axes.labelsize'] = 10
        mpl.rcParams['xtick.labelsize'] = 9
        mpl.rcParams['ytick.labelsize'] = 9
        mpl.rcParams['figure.titlesize'] = 12
    else:
        mpl.rcParams['font.size'] = size

    # this ensures editable text in svg exports
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42


## plots

def distplot(dat, lab=None, mark_outliers=-1, **kwargs):
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
    else:
        ax = plt.gca()
    
    sns.swarmplot(y=dat, ax=ax)
    sns.violinplot(y=dat, ax=ax, bw='silverman', width=0.5)
    if lab is not None:
        ax.set_ylabel(lab)
    # make the violin plot partially transparant and remove the edge
    plt.setp(ax.collections, alpha=.3, edgecolor='none')
    
    # need to mark the outlies after plotting, otherwise seaborn will overwrite
    if mark_outliers > 0:
        facecolors = ax.collections[0].get_facecolors()
        if facecolors.shape[0] < len(dat):
            facecolors = np.tile(facecolors, (len(dat), 1))
        col = facecolors[0,:].copy()
        
        # get the data points from the plot, to match order
        datplt = ax.collections[0].get_offsets()[:,1]
        inds = np.abs(sp.stats.zscore(datplt)) > mark_outliers
        facecolors[inds] = (1, 1, 1, 0)
        edgecolors = [col if m else 'none' for m in inds]
        linewidths = [1 if m else 0 for m in inds]
        plt.setp(ax.collections[0], facecolors=facecolors, edgecolors=edgecolors,
            linewidths=linewidths)
    
    # reduce the number of ticks
    ax.locator_params(nbins=4)


def paired_scatter(datx, daty, ax=None, cmap=None, add_kde=True, **kwargs):
    if ax is None:
        ax = plt.gca()
    if cmap is None:
        cmap = 'Blues'
    
    if add_kde:
        sns.kdeplot(x=datx, y=daty, cmap=cmap, shade=True, thresh=0.05,
            ax=ax, alpha=0.5)
    
    scatter_kwargs = dict(color=sns.color_palette(cmap)[-2], s=4, alpha=0.8)
    scatter_kwargs = {**scatter_kwargs, **kwargs}
    ax.scatter(datx, daty, **scatter_kwargs)
    ax.set_aspect('equal')
    
    dat = np.concatenate((datx, daty))
    lims = np.asarray([dat.min(), dat.max()])
    limrange = lims[1]-lims[0]
    lims[0] -= limrange*0.1
    lims[1] += limrange*0.1
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, 'k:')
    
    # reduce the number of ticks
    ax.locator_params(nbins=4)
    sns.despine(ax=ax)


def regplot(datx, daty, ax=None, cmap=None, add_kde=True, strip_nans=True,
    scatter_kwargs=None, **kwargs):
    ax = plt.gca() if ax is None else ax
    cmap = 'Greys' if cmap is None else cmap
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    
    if strip_nans:
        naninds = np.isnan(datx) | np.isnan(daty)
        datx = datx[~naninds]
        daty = daty[~naninds]
    
    if add_kde:
        sns.kdeplot(x=datx, y=daty, cmap=cmap, shade=True, thresh=0.05,
            ax=ax, alpha=0.2)
    
    scatter_kws = dict(s=4, color=sns.color_palette(cmap)[-2])
    scatter_kws = {**scatter_kws, **scatter_kwargs}
    sns.regplot(x=datx, y=daty, ax=ax, color=sns.color_palette('Greens')[-2],
        scatter_kws=scatter_kws, **kwargs)
    
    # reduce the number of ticks
    ax.locator_params(nbins=4)
    sns.despine(ax=ax)


def plot_posterior(data, allax, variables, has_refline=None):
    prop_above_line = {}
    for ax, (varname, label) in zip(allax.ravel(), variables.items()):
        dat = np.asarray(data.posterior[varname]).ravel()
        
        if 'sigma' in varname:
            clip = (0, np.inf)
        else:
            clip = None
        
        sns.kdeplot(dat, shade=True, ax=ax, clip=clip, edgecolor=None)
        ax.plot(az.hdi(dat), [0,0], c='k', lw=4, solid_capstyle='butt')
        
        ax.set_xlabel(label)
        if has_refline is not None and varname in has_refline:
            ax.axvline(ls=':', c='k', alpha=0.8)
            prop_above_line[varname] = max((np.mean(dat>0), np.mean(dat<0)))
        
        ax.axes.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    return prop_above_line


## specific plots for this study

def condition_scatter(ax, df, title, cmap='Oranges', probe=None, tail='two-sided', **kwargs):
    if probe is not None:
        con_sel = ('congruent', probe)
        incon_sel = ('incongruent', probe)
    else:
        con_sel = 'congruent'
        incon_sel = 'incongruent'
    
    datx = df[con_sel]
    daty = df[incon_sel]
    paired_scatter(datx, daty, ax=ax, cmap=cmap, **kwargs)
    ax.set_xlabel('Congruent')
    ax.set_ylabel('Incongruent')
    ax.set_title(title)
    
    stat = pingouin.ttest(datx-daty, 0, tail=tail, r=0.33)
    stat.insert(0, 'label', title)
    stat.insert(1, 'meanval', np.mean(datx-daty))
    return stat


def ci95_plot(ax, stat, ylabel=None):
    colors = [sns.color_palette('Oranges')[3], sns.color_palette('Blues')[3]]
    ax.set_xticks(range(len(stat)))
    ax.axhline(c='k', ls=':', alpha=0.5)
    
    for k, ((_,s), c) in enumerate(zip(stat.iterrows(), colors)):
        ax.errorbar(k, s.meanval, s['CI95%'][1]-s.meanval, c=c, lw=0, zorder=10,
            elinewidth=2, capsize=5, capthick=2, marker='o', markersize=7,
            fillstyle='full')
    
    ax.set_xlim([-0.5,len(stat)-0.5])
    ax.locator_params(nbins=4)
    if len(stat) == 1:
        ax.set_xticklabels('')
    elif len(stat) == 2:
        ax.set_xticklabels(['Key', 'Other'])
        ax.set_xlabel('Probe')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
