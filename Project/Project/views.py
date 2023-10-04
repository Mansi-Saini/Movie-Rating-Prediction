from django.shortcuts import render
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from django.shortcuts import HttpResponse
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

url = 'https://raw.githubusercontent.com/Mansi-Saini/project1/main/input.csv'
df = pd.read_csv(url)

features = df[['Meta_score', 'No_of_Votes', 'Gross', 'Runtime', 'IMDB_Rating', 'Certificate', 'Genre']].values
df = pd.DataFrame(features,
                      columns=['Meta_score', 'No_of_Votes', 'Gross', 'Runtime', 'IMDB_Rating', 'Certificate', 'Genre'])

# deleting rows having null values
df.dropna(inplace=True)

df = (df.set_index(['Meta_score', 'No_of_Votes', 'Gross', 'Runtime', 'IMDB_Rating', 'Certificate']).apply(
        lambda x: x.str.split(',').explode()).reset_index())

# converting object type into float
df["No_of_Votes"] = df["No_of_Votes"].astype(np.float64)
df["Meta_score"] = df["Meta_score"].astype(np.float64)
df['Gross'] = df['Gross'].str.replace(',', '').astype(np.float64)
df['Runtime'] = df['Runtime'].str.replace('min', '').astype(np.float64)
df["IMDB_Rating"] = df["IMDB_Rating"].astype(np.float64)
df['Genre'] = df['Genre'].str.strip()
# replacing gener values with numbers
df['Genre'] = df['Genre'].replace(['Drama', 'Crime', 'Action', 'Adventure', 'Biography', 'History',
       'Sci-Fi', 'Romance', 'Western', 'Fantasy', 'Comedy', 'Thriller',
       'Animation', 'Family', 'War', 'Mystery', 'Music', 'Horror',
       'Sport', 'Musical', 'Film-Noir'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])

# replacing certificate values with numbers
df['Certificate'] = df['Certificate'].replace(['A', 'UA', 'U', 'R', 'G', 'PG-13', 'PG', 'Passed', 'Approved',
                                                   'TV-PG', 'U/A', 'GP'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
X_train, X_test, y_train, y_test = train_test_split(df.drop('IMDB_Rating', axis=1), df['IMDB_Rating'], test_size=0.25, random_state=40)

def home(request):
    return render(request, 'index.html')

def predict(request):
    result2 = {'resultl': 0, 'resultd': 0, 'resultk': 0,'resultrf': 0, 'm': 0, 'v': 0, 'g': 0, 'r': 0, 'c': 0, 'ge': 0}
    return render(request, 'predict.html', {"result2": result2})

def model(request):
    return render(request, 'model.html')

def result(request):

    # train
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)

    regressorl = LinearRegression()
    regressorl.fit(X_train, y_train)

    regressork = KNeighborsRegressor(n_neighbors=9)
    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                        metric_params=None, n_jobs=1, n_neighbors=8, p=2,
                        weights='uniform')
    regressork.fit(X_train, y_train)

    regressorrf = RandomForestRegressor(n_estimators=1000, random_state=42)
    regressorrf.fit(X_train, y_train);
    # m,v,g,r,c,ge,result1 = 0

    m = float(request.GET['meta'])
    v = float(request.GET['vote'])
    g = float(request.GET['gross'])
    r = float(request.GET['run'])
    c = float(request.GET['cert'])
    ge = float(request.GET['genre'])
    features = [m,v,g,r,c,ge]
    df_movie = pd.DataFrame([features],
                            columns=['Meta_score', 'No_of_Votes', 'Gross', 'Runtime', 'Certificate', 'genre'])
    preddt =regressor.predict(df_movie)
    predl = regressorl.predict(df_movie)
    predk = regressork.predict(df_movie)
    predrf =regressorrf.predict(df_movie)
    # result1 = pred[0]
    result2 = {'resultl': round(predl[0],2), 'resultd': round(preddt[0],2), 'resultk': round(predk[0],2), 'resultrf': round(predrf[0],2), 'm':m, 'v':v, 'g':g, 'r':r, 'c':c, 'ge':ge}
    return render(request, 'predict.html', {"result2": result2})


def train(request):
    result = pd.concat([X_train, y_train], axis=1, join='inner')
    mydict = {
        "df": result.to_html()
    }
    return render(request, 'train.html', context=mydict)

def test(request):
    result = pd.concat([X_test, y_test], axis=1, join='inner')
    mydict = {
        "df": result.to_html()
    }
    return render(request, 'test.html', context=mydict)