import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from keras import backend as K
from keras.layers import Dense, Activation, Layer
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Retrieving data from csv file and storing it into variables
data = pd.read_csv('heart.csv')
y = data['target']
X = data.drop(['target'], axis=1)
print(data.describe())
print(data.isnull().sum())

# RBF Layer
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    # Initializing weights
    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer=None,
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 1), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if
             1 < nunique[col] < 50]]  # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()


# Plotting Model Accuracy and Model Loss
def plotlossandaccuracy(history,model):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy ' + model)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss '+ model)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

# Plotting Confusion Matrix and Scores for each model
def calculateconfusionmatrix(y_test, y_pred, model):
    print(model)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5, fmt='g')
    plt.title('Confusion_Matrix ' + model)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print('f1 score of ', model, f1_score(y_test, y_pred) * 100)
    print('precision score of ', model, precision_score(y_test, y_pred) * 100)
    print('accuracy of ', model, accuracy_score(y_test, y_pred) * 100)


# Plotting correlation map of data
plt.subplots(figsize=(15, 10))
sns.heatmap(data=data.corr(), annot=True, cmap='RdYlGn', linewidths=1)
plt.show()
plotPerColumnDistribution(data, 10, 5)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

Scaler = StandardScaler()  # We create a scaling object.
Scaler.fit(x_train)  # We fit this to x_train.
x_train_scaled = Scaler.transform(x_train)
x_test_scaled = Scaler.transform(x_test)
# We have now standardized the sets to be used.

# Single layer perceptron
model4 = Sequential()
model4.add(Dense(1))
model4.add(Activation('sigmoid'))
model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_3 = model4.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), batch_size=32, epochs=200, verbose=0)
model4.summary()
plotlossandaccuracy(history_3,'Slp')
y_pred = model4.predict(x_test_scaled)
y_pred = y_pred > 0.5
calculateconfusionmatrix(y_test, y_pred, 'Slp')

# Multilayer perceptron with one hidden layer
model1 = Sequential()
model1.add(Dense(4))
model1.add(Activation('relu'))
model1.add(Dense(1))
model1.add(Activation('sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history_2 = model1.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), batch_size=32, epochs=200, verbose=0)
model1.summary()
plotlossandaccuracy(history_2,'Mlp1')
y_pred = model1.predict(x_test_scaled)
y_pred = y_pred > 0.5
calculateconfusionmatrix(y_test, y_pred, 'Mlp1')

# Multilayer perceptron with two hidden layer
model2 = Sequential()
model2.add(Dense(4))
model2.add(Activation('relu'))
model2.add(Dense(2))
model2.add(Activation('relu'))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history_1 = model2.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), batch_size=32, epochs=200, verbose=0)
model2.summary()
plotlossandaccuracy(history_1,'Mlp2')
y_pred = model2.predict(x_test_scaled)
y_pred = y_pred > 0.5
calculateconfusionmatrix(y_test, y_pred, 'Mlp2')

# Multilayer perceptron with three hidden layer
model3 = Sequential()
model3.add(Dense(4))
model3.add(Activation('relu'))
model3.add(Dense(2))
model3.add(Activation('relu'))
model3.add(Dense(1))
model3.add(Activation('relu'))
model3.add(Dense(1))
model3.add(Activation('sigmoid'))

model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_3 = model3.fit(x_train_scaled, y_train, validation_data=(x_test_scaled, y_test), batch_size=32, epochs=200, verbose=0)
model3.summary()
plotlossandaccuracy(history_3,'Mlp3')
y_pred = model3.predict(x_test_scaled)
y_pred = y_pred > 0.5
calculateconfusionmatrix(y_test, y_pred, 'Mlp3')

# Radial Basis Function(RBF)
modelR = Sequential()
modelR.add(Dense(4, activation='relu'))
modelR.add(RBFLayer(10, 0.5))
modelR.add(Dense(1, activation='sigmoid'))

modelR.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
history_4=modelR.fit(x_train_scaled, y_train,validation_data=(x_test_scaled,y_test), epochs=200, verbose=0)
modelR.summary()
plotlossandaccuracy(history_4,'Rbf')
y_pred = modelR.predict(x_test_scaled)
y_pred = y_pred > 0.5
calculateconfusionmatrix(y_test, y_pred, 'Rbf')
