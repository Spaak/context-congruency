import numpy as np
import scipy as sp
import scipy.stats

import matplotlib.pyplot as plt

import pymc3 as pm
import theano.tensor as tt
import theano
import bambi
import arviz as az

import pandas as pd

import pickle
import gzip

import itertools as it
import statsmodels as sm

import logging
import pickle

from fastprogress.fastprogress import progress_bar


def build_bambi_model(df, depvar, family, factors):
    """
    Construct a Bambi model that has factors as fixed factors, as well as subject-level
    random factors, as well as item-level random factors (i.e., this function always
    constructs a model with the full random structure afforded by the design.
    """
    model = bambi.Model(df)
    model.add(factors)
    model.add(random=[factors + '|subnum'], categorical=['subnum'])
    model.add(random=[factors + '|item'], categorical=['item'])
    model.add(depvar + ' ~ 0', family=family)
    model.build('pymc')
    return model


def sample_and_get_results(model):
    results = model.fit(
        samples=3000, chains=4, tune=6000, target_accept=0.9,
        init='advi+adapt_diag', n_init=35000
    )

    data = az.from_pymc3(
        model=model.backend.model,
        trace=model.backend.trace
    )

    return data
