from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics

df = pd.read_csv("BTC-2020min.csv")

train_begin = int(input("eğitim verisi alt sınır?"))
train_end = int(input("eğitim verisi üst sınır?"))

selected_row = int(input("kaçıncı satırı almak istiyorsun?"))

start_index = selected_row
end_index = selected_row + 1

feature_cols = ['unix', 'open']
X = df[feature_cols]
y = df.close

X_train = X[train_begin:train_end]
y_train = y[train_begin:train_end]

clf = LinearRegression()

clf = clf.fit(X_train, y_train)

x_test = X[start_index:end_index]
y_pred = clf.predict(x_test)

print("Gerçek veri: ")
print(X[start_index:end_index], y[start_index:end_index])
print("Tahmin: ")
print(y_pred)
