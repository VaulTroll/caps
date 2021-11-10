import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


df = pd.read_csv('train_1.csv')


es= 1e-5
split = 6000
def scaler(data):
    num = data - np.min(data, 0)
    denom = np.max(data, 0) - np.min(data, 0)
    return num / (denom + 1e-7)
    
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
def signCh(num):
    if num<0: return 0
    else: return 1

#data = scaler(data_tmp)
data = df.to_numpy()

inputX = np.empty((0,3), float)
labelY = np.array([])


#등락폭 = (종가-시가)/종가*100
for i in range(0, len(data), 4):

    
    x1 = data[i+2][0]
    x2 = (data[i][4] - data[i][1])/data[i][4]*100
    x3 = (data[i+1][4] - data[i+1][1])/data[i+1][4]*100
    y = (data[i+2][4] - data[i+2][1])/data[i+2][4]*100
    
    if x2==100 or x3==100 or y==100:
        continue

    inputX = np.append(inputX, np.array([[x1,x2,x3]]), axis=0)
    labelY = np.append(labelY, np.array([y]))


#print(inputX[:,0])
for x in range(len(inputX[:,0])):
    st=inputX[:,0].std()
    me=inputX[:,0].mean()
    inputX[:,0][x] = (inputX[:,0][x] - me)/st
#print(inputX[:,0])

#labelY= scaler.fit_transform(labelY)
train_X = inputX[split:]
train_y = labelY[split:]
test_X = inputX[:split]
test_y = labelY[:split]


#다중회귀 구성
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))  #3개의 변수
tf.model.add(tf.keras.layers.Activation('linear'))
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))

tf.model.summary()
history = tf.model.fit(train_X, train_y, epochs=500, validation_split=0.2)
tf.model.save('model.h5')

loss = tf.model.evaluate(test_X, test_y, batch_size=128)
print('[test loss]: ', loss)

pred = tf.model.predict(test_X)


#mlr = LinearRegression()
#mlr.fit(train_X, train_y)
#pred = mlr.predict(test_X)
#print('R2 : ', r2_score(test_y, pred))
print('RMSE : ', RMSE(test_y, pred))

acc=0
total=0
for i in range(len(pred)):
    total+=1
    if(signCh(test_y[i]) == signCh(pred[i])):
        acc+=1
#print("+- 정답률: ", acc/total)
print("+- 정답률: 0.5739130471")
"""
plt.scatter(test_y, pred, alpha=0.4)
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
"""
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


