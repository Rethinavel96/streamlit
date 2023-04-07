import streamlit as slt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from pathlib import PurePath



df=pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
# print(df.head(4))
# print(df.isna().sum())



df['variety']=LabelEncoder().fit_transform(df['variety'])
# print(df)

rfc=RandomForestClassifier(n_estimators=50,criterion='entropy')

p={'criterion':['gini','entropy'],'n_estimators':[50,100,150,200]}

g=GridSearchCV(rfc,param_grid=p)

x=df.drop('variety',axis=1)
x
y=df['variety']
g.fit(x,y)

print(g.best_score_)
print(g.best_params_)

rfc.fit(x.values,y)





slt.title('Welcome to the iris flower type prediction model by Yogesh')

a=slt.number_input('Enter the sepal length')
b=slt.number_input('Enter the sepal width')
c=slt.number_input('Enter the petal length')
d=slt.number_input('Enter the petal width')

result=rfc.predict([[a,b,c,d]])
if result==[0]:
    result='The flower model is Iris-setosa'
elif result==[1]:
    result='The flower model is Iris-versicolor'
else:
    result='The flower model is Iris-virginica'


if slt.button('click here to find iris model'):
    slt.success(result)
    slt.balloons()