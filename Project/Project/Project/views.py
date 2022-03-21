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
    return render(request, 'predict.html')

def model(request):
    return render(request, 'model.html')

def result(request):

    # train
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    print(regressor)

    m = float(request.GET['meta'])
    v = float(request.GET['vote'])
    g = float(request.GET['gross'])
    r = float(request.GET['run'])
    c = float(request.GET['cert'])
    ge = float(request.GET['genre'])
    features = [m,v,g,r,c,ge]
    df_movie = pd.DataFrame([features],
                            columns=['Meta_score', 'No_of_Votes', 'Gross', 'Runtime', 'Certificate', 'genre'])
    pred = regressor.predict(df_movie)
    result1 = pred[0]
    return render(request, 'predict.html', {"result2": result1})

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