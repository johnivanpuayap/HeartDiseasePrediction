import pandas as pd
from keras.layers import Input
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pickle
import matplotlib.pyplot as plt

dataset = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

X = dataset.drop('target', axis=1)
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def create_model(layers, activation, neurons):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(units=64, activation=activation))
    for _ in range(layers):
        model.add(Dense(units=neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = create_model(layers=2, activation='sigmoid', neurons=32)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Define model checkpoint to save the best model
checkpoint = ModelCheckpoint("heart_disease_predictor.keras", monitor='val_loss', save_best_only=True)

# Fit the model
history = model.fit(X_train, y_train, epochs=300, batch_size=10, callbacks=[early_stopping, checkpoint], validation_split=0.2)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


y_pred = model.predict(X_test)

y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Generate and print the classification report
report = classification_report(y_test, y_pred_binary)
print(report)