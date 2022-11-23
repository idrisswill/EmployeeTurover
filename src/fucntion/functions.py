import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import os
from os import listdir
from os.path import isfile, join
import datetime
from scipy.stats._continuous_distns import _distn_names


# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []
    # Estimate distribution parameters from data
    for ii, distribution in enumerate([d for d in _distn_names if not d in ['levy_stable', 'studentized_range']]):

        print("{:>3} / {:<3}: {}".format(ii + 1, len(_distn_names), distribution))

        distribution = getattr(st, distribution)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                best_distributions.append((distribution, params, sse))

        except Exception:
            pass

    return sorted(best_distributions, key=lambda x: x[2])


def make_pdf(dist, params: list, size=10000):
    """
    Generate distributions's Probability Distribution Function

    Args:
        dist: Distribution
        params(list): parameter
        size(int): size

    Returns:
        dataframe: Power Distribution Function

    """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def dataset_description(path='/data/AllProjectIA/EmployeesTurnover/src/uploads'):
    '''
    input:
        the absolute path of directory dataset
    return:
        list of description of dataset in path
    '''
    list_dataset = [f for f in listdir(path) if isfile(join(path, f))]
    liste = []
    if len(list_dataset) > 0:
        for dataset in list_dataset:
            data = pd.read_csv(f'{path}/{dataset}')
            c_time = os.path.getctime(f'{path}/{dataset}')
            a = dataset.split(".")
            dictionary = {}
            dictionary['name'] = a[0]
            dictionary['nbr_column'] = len(data.columns)
            dictionary['nbr_row'] = len(data)
            dictionary['memory_size'] = data.memory_usage(deep=True).sum() / 1024
            dictionary['rapport_null'] = (data.isna().sum() / data.shape[0]).sort_values(ascending=True).to_dict()
            dictionary['create_at'] = datetime.datetime.fromtimestamp(c_time)
            liste.append(dictionary)

    return liste
