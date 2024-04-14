from django.shortcuts import render

from django.shortcuts import render
from .models import Crop
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def predict_yield(request):
    df = pd.read_csv(r"C:\Users\SMILE\Downloads\archive (10)\Crop_recommendation.csv")
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    crop_counts = pd.Series(y_pred).value_counts()
    optimized_values = crop_counts / crop_counts.sum()
    return render(request, 'predict_yield.html', {'prediction': y_pred, 'optimized_values': optimized_values.to_dict()})
