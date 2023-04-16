import os
import zipfile
# read version from installed package
from importlib.metadata import version
__version__ = version("aif_course_package")

path_dir, file_name = os.path.split(__file__)
data_dir = os.path.join(path_dir, 'data')

if 'sp500_data' in os.listdir(data_dir):
    print('Data from the package has already been downloaded')
else:
    with zipfile.ZipFile(os.path.join(data_dir, 'sp500_assetprices.pickle.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_dir, 'sp500_data'))
