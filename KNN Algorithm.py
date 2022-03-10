# KNN 알고리즘과의 비교 실험

# pandas 모듈 다운로드 필요, 명령 프롬프트 창에서 pip3 install pandas
# sklearn 모듈 다운로드 필요, 명령 프롬프트 창에서 pip3 install sklearn

import pandas as pd
from pandas import Series, DataFrame
import random

# 문자열 충돌을 막기 위해 csv 파일 1행의 문자열을 지우고 사용하였다.

irisData = pd.read_csv('irisData.csv', names =['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
irisData_y = irisData['species']
irisData_x = irisData.iloc[:,:4]

print("<Test1>")
print()

# train_test_split을 활용하여 학습셋과 테스트셋을 9:1로 분류하여 사용하였다.

from sklearn.model_selection import train_test_split

irisData_x_train,irisData_x_test, irisData_y_train, irisData_y_test = train_test_split(irisData_x,irisData_y, test_size = 0.1, random_state = random.randint(1,100)) # randmon_state을 통해 셔플 시드값을 무작위로 설정

from sklearn.neighbors import KNeighborsClassifier

# KNN 알고리즘의 객체 생성

KNN_Algorithm=KNeighborsClassifier()

# train 집단으로 KNN 알고리즘 학습

KNN_Algorithm.fit(irisData_x_train, irisData_y_train)

print("학습 점수: ", KNN_Algorithm.score (irisData_x_train, irisData_y_train))

print()

print("테스트 점수: ", KNN_Algorithm.score(irisData_x_test, irisData_y_test))

print()

compare = pd.DataFrame({'predict':KNN_Algorithm.predict(irisData_x_test), 'answer':irisData_y_test})

print("채점표: ")
print(compare)

print()
print("<Test2>")
print()

# train_test_split을 활용하여 학습셋과 테스트셋을 8:2로 분류하여 사용하였다.

from sklearn.model_selection import train_test_split

irisData_x_train,irisData_x_test, irisData_y_train, irisData_y_test = train_test_split(irisData_x,irisData_y, test_size = 0.2, random_state = random.randint(1,100)) # randmon_state을 통해 셔플 시드값을 무작위로 설정

from sklearn.neighbors import KNeighborsClassifier

# KNN 알고리즘의 객체 생성

KNN_Algorithm=KNeighborsClassifier()

# train 집단으로 KNN 알고리즘 학습

KNN_Algorithm.fit(irisData_x_train, irisData_y_train)

print("학습 점수: ", KNN_Algorithm.score (irisData_x_train, irisData_y_train))

print()

print("테스트 점수: ", KNN_Algorithm.score(irisData_x_test, irisData_y_test))

print()

compare = pd.DataFrame({'predict':KNN_Algorithm.predict(irisData_x_test), 'answer':irisData_y_test})

print("채점표: ")
print(compare)

print()
print("<Test3>")
print()

# train_test_split을 활용하여 학습셋과 테스트셋을 7:3로 분류하여 사용하였다.

from sklearn.model_selection import train_test_split

irisData_x_train,irisData_x_test, irisData_y_train, irisData_y_test = train_test_split(irisData_x,irisData_y, test_size = 0.3, random_state = random.randint(1,100)) # randmon_state을 통해 셔플 시드값을 무작위로 설정
from sklearn.neighbors import KNeighborsClassifier

# KNN 알고리즘의 객체 생성

KNN_Algorithm=KNeighborsClassifier()

# train 집단으로 KNN 알고리즘 학습

KNN_Algorithm.fit(irisData_x_train, irisData_y_train)

print("학습 점수: ", KNN_Algorithm.score (irisData_x_train, irisData_y_train))

print()

print("테스트 점수: ", KNN_Algorithm.score(irisData_x_test, irisData_y_test))

print()

compare = pd.DataFrame({'predict':KNN_Algorithm.predict(irisData_x_test), 'answer':irisData_y_test})

print("채점표: ")
print(compare)
