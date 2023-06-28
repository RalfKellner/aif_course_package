import requests
import pandas as pd
import zipfile
import io
import re
import pickle
import os
import numpy as np
import tensorflow as tf
from scipy.stats import mode
import matplotlib.pylab as plt
from matplotlib import cm

def get_ff_factors(num_factors = 3 , frequency = 'daily', in_percentages = False):

    ''' 
    This function downloades directly the current txt files from the Keneth R. French homepage (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

    Parameters:
    -----------
    num_factors (3 or 5): int
        download either data for the three or the five factor portfolios

    frequency: str
        either daily, weekly or monthly for three factor data or daily or monthly for five factor data
    
    Returns:
    ---------
    pd.DataFrame
    
    '''

    assert num_factors in [3, 5], 'The number of factors must be 3 or 5'

    if num_factors == 3:
        assert frequency in ['daily', 'weekly', 'monthly'], 'frequency for the three factors model must be either daily, weekly or monthly'
        if frequency == 'daily':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_Factors_daily.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] + '-' + date[6:] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)
        elif frequency == 'weekly':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_Factors_weekly.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] + '-' + date[6:] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)
        elif frequency == 'monthly':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_Factors.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if re.search('Annual', string_line):
                    break
                elif string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)
    elif num_factors == 5:
        assert frequency in ['daily', 'monthly'], 'frequency for the five factor model must be either daily or monthly'
        if frequency == 'daily':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_5_Factors_2x3_daily.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] + '-' + date[6:] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)
        elif frequency == 'monthly':
            french_url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_TXT.zip'
            r = requests.get(french_url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open('F-F_Research_Data_5_Factors_2x3.txt') as file:
                out = file.readlines()
            data_lines = []
            for line in out:
                string_line = line.decode('UTF-8')
                if re.search('Annual', string_line):
                    break
                elif string_line[0].isdigit():
                    string_line = string_line.split()
                    data_lines.append(string_line)
            ff_data = pd.DataFrame(data_lines, columns = ['date', 'Mkt-RF', 'SMB', 'HML','RMW', 'CMA', 'RF'])
            ff_data.loc[:, 'date'] = [date[:4] + '-' + date[4:6] for date in ff_data.date]
            ff_data.set_index('date', inplace = True)
            ff_data = ff_data.astype(float)

    if in_percentages == True:
        return ff_data
    else:
        return ff_data / 100
    

def write_pickle(filename, obj):
    '''
    Save an object as a pickle file.

    Parameters:
    -----------
    filename: str   
        location and filename, use .txt ending if you want to read a text file

    Returns:
    ---------
    list of str  
    '''
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    '''
    Read a pickle file.

    Parameters:
    -----------
    filename: str   
        location and filename, use .txt ending if you want to read a text file

    Returns:
    ---------
    object
    '''
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def load_course_data(name):
    '''
    This function loads the course data
    '''
    this_dir, filename = os.path.split(__file__)
    if name == 'sp500_data':
        path_name = os.path.join(this_dir, 'data', 'sp500_prices_2018_2023.pickle')
    if name == 'yfinance':
        path_name = os.path.join(this_dir, 'data', 'yfinance_data.pickle')
    df = read_pickle(path_name)
    df_spy = pd.read_csv(os.path.join(this_dir, 'data', 'spy_data.csv'))
    df_spy.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
    df_spy.set_index('Date', drop = True, inplace = True)
    return df, df_spy


# a function for feature analysis
def feature_analysis(X, nn, feature_names):
    # determine first and second partial derivatives
    with tf.GradientTape() as snd:
        snd.watch(X)
        with tf.GradientTape() as fst:
            fst.watch(X)
            # prediction with the neural network, i.e., f(X)
            pred = nn(X)
        # gradient
        g = fst.gradient(pred, X)
    # jacobian which outputs Hessian matrix
    h = snd.batch_jacobian(g, X)

    # first partial derivatives
    g_np = g.numpy()
    # average squard partial derivatives
    g_mag_sq = (g_np**2).mean(axis = 0)
    # square root of average squard partial derivatives
    g_mag = np.sqrt(g_mag_sq)
    # sign of average partial derivatives
    g_dir = np.sign(g_np.mean(axis = 0))

    # normalizing constant
    C_ = np.sum(g_mag)
    # normalized feature importance with sign
    fi = (g_mag * g_dir) / C_

    # get signs of each sample
    fi_signs = np.sign(g_np)
    # the mode is the sign which can be observed most often among all samples, the counts is how often this sign is observed
    fi_modes, fi_counts = mode(fi_signs, keepdims = True)
    # dividing the count of the sign which is observed most often by the overall sample size gives us a frequency measure
    # which is closer to one, the higher the conformity of the sign
    fi_conformity = fi_counts / g_np.shape[0] #fi_modes * 

    # in analogy to the calculation above, we do the same thing with the second partial derivatives
    h_np = h.numpy()
    # get the square root of average squared direction of curvature and interactions
    h_mag_sq = (h_np**2).mean(axis = 0)
    h_mag = np.sqrt(h_mag_sq)

    # the the sign of average curvature and interactions
    h_dir = np.sign(h_np.mean(axis = 0))

    # normalize the values on the diagonal line to compare the degree of non-linearity
    C_nonlin = np.sum(h_mag.diagonal())
    nonlinearity = (h_dir.diagonal() * h_mag.diagonal()) / C_nonlin

    # normlize the interactions
    lti = np.tril_indices(h_mag.shape[0], k = -1)
    C_ia = np.sum(h_mag[lti])
    interactions = (h_mag[lti] * h_dir[lti]) / C_ia

    # bring curvature and interaction effects back to matrix format
    snd_degree_summary = np.diag(nonlinearity)
    a, b = lti
    inter_iter = iter(interactions)
    for i, j in zip(a, b):
        snd_degree_summary[i, j] = next(inter_iter)
        snd_degree_summary[j, i] = snd_degree_summary[i, j]

    # get the conformity of second order effects
    snd_signs = np.sign(h_np)
    snd_degree_modes, snd_degreee_counts = mode(snd_signs, keepdims = True)
    snd_degree_conformity = snd_degreee_counts / h_np.shape[0] #snd_degree_modes * 

    # finally summarize feature importances and second order effects
    summary = pd.DataFrame(data = snd_degree_summary, index = feature_names, columns = feature_names)
    summary.loc[:, 'feature_importance'] = fi
    # as well as their conformity
    summary_conformity = pd.DataFrame(data = snd_degree_conformity.reshape(h_np.shape[1], h_np.shape[2]), index = feature_names, columns = feature_names)
    summary_conformity.loc[:, 'feature_conformity'] = fi_conformity.flatten()

    return summary, summary_conformity


def plot_weights_over_time(optimal_weights, width = 6, height = 4, ax = None, legend = False):

    x = optimal_weights.index
    labels = optimal_weights.columns

    if not isinstance(optimal_weights, pd.DataFrame):
        raise ValueError("w must be a DataFrame")

    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()
        fig.set_figwidth(width)
        fig.set_figheight(height)
    else:
        fig = ax.get_figure()

    ax.set_title('Optimal portfolio weights over time')

    cmap = 'tab20'
    colormap = cm.get_cmap(cmap)
    colormap = colormap(np.linspace(0, 1, 20))
    nrow = 25

    weights = []
    for col in labels:
        weights.append(optimal_weights.loc[:, col].astype(np.float32).values)
    y = np.vstack(weights)

    n = int(np.ceil(len(labels) / nrow))
    ax.stackplot(x, y, labels=labels, alpha=0.7, edgecolor="black")
    if legend:
        ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), ncol=n)
    
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    try:
        fig.tight_layout()
    except:
        pass

    return ax