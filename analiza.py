#Importowanie bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Prezentacja wszystkich wynikow
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#Wywolanie danych
df = pd.read_csv('american_bankruptcy_dataset.csv')

#Sprawdzenie ilosci danych w poszczegolnych kategoriach
print(df['status_label'].value_counts())

# Usuniecie kolumny company_name
df = df.drop(columns=['company_name'], axis=1)

# Wyswietlenie pierwszych wierszy
print(df.head())


# Zmiana nazwy kolumny 'status_label' na 'bancruptcy'
df['status_label'] = df['status_label'].apply(lambda x : 0 if x == 'alive' else 1)
df = df.rename(columns={'status_label':'bankruptcy'})
print(df['bankruptcy'].value_counts())

#Wyswietlenie podstawowych informacji o danych
print(df.shape)
df.info()
df.isnull().sum()

# zaprezentowanie statystyk
print(df.describe().T.to_string())

# Skalowanie zmiennych
#df.shape
to_scale = df.iloc[:,2:20]
#normalized_df = (to_scale-to_scale.mean())/to_scale.std()
normalized_df=(to_scale-to_scale.min())/(to_scale.max()-to_scale.min())
normalized_df
normalized_df.describe()


#Wykresy wskaznikow finansowych
#Wykres division
bpl = df.groupby(["division","bankruptcy"]).size().unstack()
bpl = bpl.divide(bpl.sum(axis=1), axis=0)
ax = bpl.plot(kind='bar', stacked=True, title='Distribution of the division by bankruptcy flag');

i,j=0,0
PLOTS_PER_ROW = 3
fig, axs = plt.subplots(math.ceil(len(normalized_df.columns)/PLOTS_PER_ROW),PLOTS_PER_ROW, figsize=(20, 60))
for col in normalized_df.columns:
    sns.boxplot(ax=axs[i, j], x="bankruptcy", y=col,  data=df, palette="Set3", hue='bankruptcy')
    axs[i][j].set_ylabel(col)
    j+=1
    if j%PLOTS_PER_ROW==0:
        i+=1
        j=0
plt.show();

df.head()
# dummies for division
df = pd.get_dummies(df, columns=['division'], prefix='', prefix_sep='')
df.head()
df.tail()

                    #pokazuje ktore to kolumny

#Usuniecie danych dla ktorych korelaja jest > 0.7 lub <-0.7
df = df.drop(["depreciation", "net_income", "market_value", "total_current_liabilities", "major_group", "total_assets", "net_sales", "cost_of_goods_sold", "retained_earnings", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],axis=1) #chcilabym to zapisac do pliku
df.head()

#Zapisanie wybranych kolumn
df.to_csv('american_bankruptcy2.csv', sep=';', index=False)