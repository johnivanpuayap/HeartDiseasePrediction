import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report
import pickle

dataset = pd.read_csv('heart_statlog_cleveland_hungary_final.csv');

X = dataset.drop('target', axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def create_model(layers, activation, neurons):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(units=neurons, activation=activation))
    for _ in range(layers):
        model.add(Dense(units=neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = KerasClassifier(model=create_model, verbose=0)

param_grid = {
    'model__layers': [3, 4, 5],
    'model__activation': ['sigmoid'],
    'model__neurons': [32, 128, 256],
    'batch_size': [10, 20, 30,],
    'epochs': [200, 300, 400],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Pass early stopping callback as a callback in the fit_params parameter
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=2, n_jobs=-1)

fit_params = {
    'callbacks': [[early_stopping]],
}

grid_search.fit(X_train, y_train, **fit_params);


print(grid_search.best_params_)

y_pred = grid_search.predict(X_test)

# Generate and print the classification report
report = classification_report(y_test, y_pred)
print(report)


# After model training and grid search
best_model = grid_search.best_estimator_.model

with open("heart_disease_predictor.pkl", "wb") as f:
    pickle.dump(best_model, f)