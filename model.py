#Import Libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

#Load data
df=pd.read_csv('delhirain.csv')

#Preprocessing
df.columns = df.columns.str.lstrip(' _')
df = df.drop(columns = ['precipm','wgustm','windchillm','heatindexm','snow','tornado','hail'])
df=df.dropna()

#Mapping for weather,direction
from mappings import weather_mappings,direction_mapping
df['conds']=df['conds'].map(weather_mappings)
df['wdire']=df['wdire'].map(direction_mapping)
df=df.drop(columns=['wdird','datetime_utc'])

X=df.drop(columns='rain')
y=df['rain']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#Model Creation

from xgboost import XGBClassifier
print(df.dtypes)
# Now fit the model
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train, verbose=False)
pickle.dump(model,open("model.pkl","wb"))
y_test_pred = model.predict(X_test)

