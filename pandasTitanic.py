import constants as cs
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
traindata = pd.read_csv(cs.trainData)
testdata = pd.read_csv(cs.testData)

def initDF(df):
    df['Gender'] = 4
    df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df['EmbarkedInt'] = df['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2, 'NA': 3} )
    df['AgeFill'] = df['Age']
    median_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
    df['Age*Class'] = df.Age * df.Pclass
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    return df

def quadraticAge(trainData, testData = pd.DataFrame()):
    trainData = initDF(trainData)
    trainData['AgeFill2'] = trainData.AgeFill * trainData.AgeFill

    trainData.dtypes[trainData.dtypes.map(lambda x: x=='object')]


    result = sm.ols(formula="Survived ~ Pclass + Gender + AgeFill2", data=trainData).fit()

    intercept = result.params[0]
    Pclass = result.params[1]
    Gender = result.params[2]
    AgeFill = result.params[3]


    if testData.empty:
        trainData['survivalChance'] = intercept + trainData.Pclass*Pclass + trainData.Gender*Gender + trainData.AgeFill2*AgeFill
        trainData['testSurvival'] = np.where(trainData['survivalChance'] > .5, 1.0, 0.0)
        return trainData


    testData = initDF(testData)
    testData['AgeFill2'] = testData.AgeFill * testData.AgeFill
    testData['survivalChance'] = intercept + testData.Pclass*Pclass + testData.Gender*Gender + testData.AgeFill2*AgeFill
    testData['Survived'] = np.where(testData['survivalChance'] > .5, 1, 0)

    return testData

def linearAge(trainData, testData = pd.DataFrame()):
    trainData = initDF(trainData)
    trainData['AgeFill2'] = trainData.AgeFill

    trainData.dtypes[trainData.dtypes.map(lambda x: x=='object')]


    result = sm.ols(formula="Survived ~ Pclass + Gender + AgeFill2", data=trainData).fit()

    intercept = result.params[0]
    Pclass = result.params[1]
    Gender = result.params[2]
    AgeFill = result.params[3]


    if testData.empty:
        trainData['survivalChance'] = intercept + trainData.Pclass*Pclass + trainData.Gender*Gender + trainData.AgeFill2*AgeFill
        trainData['testSurvival'] = np.where(trainData['survivalChance'] > .5, 1.0, 0.0)
        return trainData


    testData = initDF(testData)
    testData['AgeFill2'] = testData.AgeFill
    testData['survivalChance'] = intercept + testData.Pclass*Pclass + testData.Gender*Gender + testData.AgeFill2*AgeFill
    testData['Survived'] = np.where(testData['survivalChance'] > .5, 1, 0)

    return testData


def cubeAge(trainData, testData = pd.DataFrame()):
    trainData = initDF(trainData)
    trainData['AgeFill2'] = trainData.AgeFill * trainData.AgeFill * trainData.AgeFill

    trainData.dtypes[trainData.dtypes.map(lambda x: x=='object')]


    result = sm.ols(formula="Survived ~ Pclass + Gender + AgeFill2", data=trainData).fit()

    intercept = result.params[0]
    Pclass = result.params[1]
    Gender = result.params[2]
    AgeFill = result.params[3]


    if testData.empty:
        trainData['survivalChance'] = intercept + trainData.Pclass*Pclass + trainData.Gender*Gender + trainData.AgeFill2*AgeFill
        trainData['testSurvival'] = np.where(trainData['survivalChance'] > .5, 1.0, 0.0)
        return trainData


    testData = initDF(testData)
    testData['AgeFill2'] = testData.AgeFill * testData.AgeFill * testData.AgeFill
    testData['survivalChance'] = intercept + testData.Pclass*Pclass + testData.Gender*Gender + testData.AgeFill2*AgeFill
    testData['Survived'] = np.where(testData['survivalChance'] > .5, 1, 0)

    return testData



def testImprovements(traindata):
    traindata['difference'] = abs(traindata.testSurvival - traindata.Survived)
    print traindata.testSurvival
    print traindata.Survived
    print traindata.difference
    return 1 - sum(traindata.difference)/len(traindata.difference)

print testImprovements(cubeAge(traindata))


def writetestfile(predictionfunction):
    predict = predictionfunction(traindata, testdata)
    writeprediction = predict[['PassengerId','Survived']]
    writeprediction.to_csv('/Users/christianholmes/Desktop/Kaggle/Kaggle Titanic/CubeAgeModel.csv', index=False)
    print writeprediction



writetestfile(cubeAge)