import pandas
import numpy as np
import matplotlib.pyplot as plt
url ="http://bilkav.com/veriler.csv"
veriler =pd.read_csv(url)
#print(veriler)
boy = veriler.iloc[:,1:2].values # veriler[['boy']] olarak da yazabilirdik.
#eksik verileri bulalım (missing values)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan ,strategy='mean')
Yas = veriler.iloc[:,1:4].values
imputer =imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

#birbirinden farklı ülkere 0-1-2 gibi değerler vererek bunları sayısal değere dönüştürelim
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke = veriler.iloc[:,0:1].values
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke2 =ohe.fit_transform(ulke).toarray()
print(ulke2)