from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
from scipy.io.arff import loadarff
import pandas as pd

zfile = ZipFile(r"C:\Users\file-name.arff")
in_mem_fo = TextIOWrapper(BytesIO(zfile.read('Breast.arff')), encoding='utf-8')
data = loadarff(in_mem_fo)
df = pd.DataFrame(data[0])
df.head()
from sklearn import preprocessing
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

normalized_X = preprocessing.normalize(X)
normalized_X
normalized_X.mean(axis=0)
normalized_X.std(axis=0)

standardized_X = preprocessing.scale(X)
standardized_X
standardized_X.mean(axis=0)
standardized_X.std(axis=0)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)
X_train_minmax

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X)
X_train_maxabs

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
transformer  
RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
       with_scaling=True)
transformer.transform(X)

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
johnson_X = PowerTransformer(method='yeo-johnson').fit_transform(X)
johnson_X

Quntile_uniform_X = QuantileTransformer(output_distribution='uniform').fit_transform(X)
Quntile_uniform_X

from sklearn.preprocessing import Normalizer
norm_X = Normalizer().fit_transform(X)
norm_X
