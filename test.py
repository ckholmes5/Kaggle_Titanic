import csv as csv
import numpy as np
import constants as cs

testFile = open(cs.testData, 'rb')
testObject = csv.reader(testFile)
header = testFile.next()

predictionFile = open(cs.genderbasedmodel, 'wb')
predictionObject = csv.writer(predictionFile)

predictionObject.writerow(["PassengerId", "Survived"])

for row in testObject:
    if row[3] == 'female':
        predictionObject.writerow([row[0],'1'])
    else:
        predictionObject.writerow([row[0],'0'])


predictionFile.close()
