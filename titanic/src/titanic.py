import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

"""
https://writeren.tistory.com/77?category=757132
클론 코딩
2019년 8월 캐글 공부
"""




"""

Columns Info
---------------------

PassengerId: 승객 번호
Survived: 생존여부 (1: 생존, 0: 사망)
Pclass: 승선권 클래스 (1: 1st, 2: 2nd, 3: 3rd)
Name: 승객이름
Sex: 승객 성별
Age: 승객 나이
SibSp: 동반한 형제자매, 배우자 수
Patch: 동반한 부모, 자식 수
Ticket: 티켓의 고유 넘버
Fare: 티켓의 요금
Cabin: 객실 번호
Embraked: 승선한 항구명(C: Cherbourg, Q: Queenstown, S: Southampton)
"""
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# print(train.head())
"""

train info
--------------------

RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB


train isnull sum
-----------------------

PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
"""
# print(train.info())
# print(train.isnull().sum())

def bar_chart(feature):
    surv_key = 'Survived'
    survived = train[train[surv_key] == 1][feature].value_counts()
    dead = train[train[surv_key] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = [surv_key, 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()


# 성별에 따른 생존여부
# bar_chart('Sex')

# 승선권에 따른 생존여부
# bar_chart('Pclass')

# 형제자매, 배우자 수
# bar_chart('SibSp')

# 부모와 자녀 수
# bar_chart('Parch')

# 승선한 항구
# bar_chart('Embarked')

# train.describe(include='all')

# carbin, ticket은 데이터가 빈값이 많고 연관성이 없기 때문에 제거
train = train.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Cabin'], axis=1)
test = test.drop(['Ticket'], axis=1)

# Embarked 가공
train = train.fillna({'Embarked': 'S'})
# 정수 인코딩
embarked_encoding = {'S': 1, 'C': 2, 'Q': 3}
train['Embarked'] = train['Embarked'].map(embarked_encoding)
test['Embarked'] = test['Embarked'].map(embarked_encoding)
# print(train.head())

# Name 값 가공
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # print(f'{dataset["Name"]} :  {dataset["Title"]}')

# print(pd.crosstab(train['Title'], train['Sex']))
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Laydy', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# print(train.head())

train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
# print(train.head())

sex_mapping = {'male': 0, 'female': 1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# print(train.head())

train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)
# print(train.head())
# bar_chart('AgeGroup')

age_title_mapping = {1: 'Young Adult', 2: 'Student', 3: 'Adult', 4: 'Baby', 5: 'Adult', 6: 'Adult'}
for x in range(len(train['AgeGroup'])):
    if train['AgeGroup'][x] == 'Unknown':
        train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]

for x in range(len(test['AgeGroup'])):
    if test['AgeGroup'][x] == 'Unknown':
        test['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
# print(train.head())

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)

# print(train.head())

train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])

# train = train.drop(train['Fare'], axis=1)
# test = test.drop(test['Fare'], axis=1)
train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)
# print(train.head())





