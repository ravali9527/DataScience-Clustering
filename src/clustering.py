#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:18:06 2018

@author: ravali
"""
DIR='/Users/ravali/Desktop'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
ratings = pd.read_csv(DIR+'/ratings.csv', delimiter=',')
links = pd.read_csv(DIR+'/links.csv', delimiter=',')
tags = pd.read_csv(DIR+'/tags.csv', delimiter=',')
movies = pd.read_csv(DIR+'/movies.csv', delimiter=',')
print(ratings.head())
print(links.head())
print(tags.head())
print(movies.head())
df_merged121=pd.merge(ratings, tags, on=['userId','movieId'], how='inner')
print(df_merged121)
df_dropped=df_merged121.drop(['timestamp_x', 'timestamp_y'], axis=1)
print(df_dropped)
df_merged3=pd.merge(df_dropped,movies, on=['movieId'], how='inner')
print(df_merged3)
df_mergedfinal=pd.merge(df_merged3,links, on=['movieId'], how='inner')
print(df_mergedfinal)
df_final=df_mergedfinal.drop(['imdbId','tmdbId'],axis=1)
print(df_final)
print(df_final.dtypes)
obj_df = df_final.select_dtypes(include=['object']).copy()
obj_df.head()
obj_df["tag"] = obj_df["tag"].astype('category')
obj_df["tag"] = obj_df["tag"].cat.codes
obj_df["title"] = obj_df["title"].astype('category')
obj_df["title"] = obj_df["title"].cat.codes
obj_df["genres"] = obj_df["genres"].astype('category')
obj_df["genres"] = obj_df["genres"].cat.codes
print(obj_df)
print(df_final)
df_final = pd.concat([df_final, obj_df], axis=1)
print(df_final)
Y=df_final.title #output label
X=df_final.drop('title',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print(X_train)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test, y_train))
predictions = knn.predict(X_test)
score = knn.score(X_test, y_test)
print(score)