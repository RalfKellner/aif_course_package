import requests
import pandas as pd
import zipfile
import io
import re
import pickle
import os




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
    return df