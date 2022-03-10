# 신경망 기반 방법 실험

# pandas 모듈 다운로드 필요, 명령 프롬프트 창에서 pip3 install pandas
# sklearn 모듈 다운로드 필요, 명령 프롬프트 창에서 pip3 install sklearn
# tensorflow 모듈 다운로드 필요, 명령 프롬프트 창에서 pip3 install tensorflow

import pandas as pd
from pandas import Series, DataFrame
import random

# 문자열 충돌을 막기 위해 csv 파일 1행의 문자열을 지우고 사용하였다.

irisData = pd.read_csv('irisData.csv', names =['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
irisData_y = irisData['species']
irisData_x = irisData.iloc[:,:4]

# irisData_y값을 Labeling

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
irisData_y_integer= encoder.fit_transform(irisData_y)

irisData_y = pd.get_dummies(irisData_y_integer).values

print()
print("<Test1>")
print()

# train_test_split을 활용하여 학습셋과 테스트셋을 9:1로 분류하여 사용하였다.

from sklearn.model_selection import train_test_split

irisData_x_train,irisData_x_test, irisData_y_train, irisData_y_test = train_test_split(irisData_x,irisData_y, test_size = 0.1, random_state = random.randint(1,100)) # randmon_state을 통해 셔플 시드값을 무작위로 설정

from sklearn.neighbors import KNeighborsClassifier

# tensorflow 엔진의 keras 모듈을 이용한 학습

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Neural Network 기반의 model 생성
# Activation function을 선정할 때, sigmoid, tanh, relu, elu, softmax 순으로 선정


model = Sequential()

model.add(Dense(64,input_shape = (4,), activation="sigmoid"))
model.add(Dense(64, activation="tanh"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="elu"))
model.add(Dense(3, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer = "Adam", metrics =["accuracy"])

model.summary()

# 50번 반복하여 학습

hist = model.fit(irisData_x_train, irisData_y_train, validation_data = (irisData_x_test, irisData_y_test), epochs = 50)

# 결과 도출
loss, accuracy = model.evaluate(irisData_x_test, irisData_y_test)
print("Accuracy = {:.5f}".format(accuracy))

print("="*20)
print()
print("<Test2>")
print()

# 학습셋과 테스트셋을 8:2로 분류하여 사용하였다.

irisData_x_train,irisData_x_test, irisData_y_train, irisData_y_test = train_test_split(irisData_x,irisData_y, test_size = 0.2, random_state = random.randint(1,100)) # randmon_state을 통해 셔플 시드값을 무작위로 설정


# Neural Network 기반의 model

model.summary()

# 50번 반복하여 학습

hist = model.fit(irisData_x_train, irisData_y_train, validation_data = (irisData_x_test, irisData_y_test), epochs = 50)

# 결과 도출
loss, accuracy = model.evaluate(irisData_x_test, irisData_y_test)
print("Accuracy = {:.5f}".format(accuracy))

print("="*20)
print()
print("<Test3>")
print()

# 학습셋과 테스트셋을 7:3로 분류하여 사용하였다.

irisData_x_train,irisData_x_test, irisData_y_train, irisData_y_test = train_test_split(irisData_x,irisData_y, test_size = 0.3, random_state = random.randint(1,100)) # randmon_state을 통해 셔플 시드값을 무작위로 설정


# Neural Network 기반의 model 

model.summary()

# 50번 반복하여 학습

hist = model.fit(irisData_x_train, irisData_y_train, validation_data = (irisData_x_test, irisData_y_test), epochs = 50)

# 결과 도출
loss, accuracy = model.evaluate(irisData_x_test, irisData_y_test)
print("Accuracy = {:.5f}".format(accuracy))
