import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('train_2.csv')

es= 1e-5
def scaler(data):
    num = data - np.min(data, 0)
    denom = np.max(data, 0) - np.min(data, 0)
    return num / (denom + 1e-7)

data_tmp = df.to_numpy()
#test_min = np.min(data_tmp, 0)
#test_max = np.max(data_tmp, 0)
#test_denom = test_max - test_min
#print(test_denom)

#data = scaler(data_tmp)
data = data_tmp

#inputX = []
#labelY = []
inputX = np.array([])
labelY = np.array([])

#등락폭 = (종가-시가)/종가*100
for i in range(0, len(data), 4):
    x1 = data[i+2][0]
    x2 = (data[i][4] - data[i][1])/data[i][4]*100 +es
    x3 = (data[i+1][4] - data[i+1][1])/data[i+1][4]*100 +es
    y = (data[i+2][4] - data[i+2][1])/data[i+2][4]*100 +es
    inputX = np.append(inputX, np.array([x1,x2,x3]))
    labelY = np.append(labelY, np.array([y])
    
#print(inputX)
#print(labelY)

#_X=np.array(inputX)
#_y=np.array(labelY)
#print(inputX)
#print(labelY)

#X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size=0.2, shuffle=False, random_state=22)
"""
#다중회귀 구성
tf.model = tf.keras.Sequential()

tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))  #3개의 변수

tf.model.add(tf.keras.layers.Activation('linear'))

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))

#tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=100)

y_predict = tf.model.predict(np.array([[72., 93., 90.]]))

print(y_predict)
"""

"""
model = tf.keras.Sequential([
  tf.keras.layers.LSTM(units=30, return_sequences=True, input_shape=[3,6]),
  tf.keras.layers.LSTM(units=30),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

X = np.array(inputX)
Y = np.array(labelY)

history=model.fit(X[:800], Y[:800], epochs=500, validation_split=0.2)
model.save('model.h5')

test_X =
model.evaluate(X[800:], Y[800:])
prediction=model.predict(X[800:800+10])


# 5개 테스트 데이터에 대한 예측을 표시합니다.
for i in range(10):
    print(Y[800+i], '\t', prediction[i][0], '\tdiff:', abs(prediction[i][0] - Y[800+i]))

prediction = model.predict(X[800:])
fail = 0
for i in range(len(prediction)):
  # 오차가 0.04 이상이면 오답입니다.
  if abs(prediction[i][0] - Y[800+i]) > 500:
    fail +=1

print('correctness:', (440-fail)/440*100, '%')

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
#X = tf.Variable(tf.ones([None, seq_len, data_dim]), dtype=tf.float32, name='input_X')
#Y = tf.Variable(tf.ones([None, 1]), dtype=tf.float32, name='intput_Y')




new_model = keras.models.load_model('my_model.h5')
# Show the model architecture
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
"""
