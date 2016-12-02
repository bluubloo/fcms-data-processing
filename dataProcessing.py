import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Label names
AVG_TIME = 'AvgTime'
SCORE = 'Score'
SLEEP = 'Sleep'
TEST_NAME = 'Test'
NORM_SCORE = 'Normalised Score'

#Test names
APPEARING_OBJECT = 'Appearing Object'
APPEARING_OBJECT_FIXED = 'Appearing Object - Fixed Point'
ARROW_IGNORING = 'Arrow Ignoring Test'
CHANGING_DIRECTIONS = 'Changing Directions'
CHASE_TEST = 'Chase Test'
EVEN_OR_VOWEL = 'Even or Vowel'
FINGER_TAP = 'Finger Tap Test'
MONKEY_LADDER = 'Monkey Ladder'
CARD_LEARNING = 'One Card Learning Test'
PATTERN_RECREATION = 'Pattern Recreation'
STROOP = 'Stroop Test'

#Global maximums for tests
global scoreAO
global scoreAOF
global scoreAI
global scoreCD
global scoreCT
global scoreEOV
global scoreFT
global scoreML
global scoreCL
global scorePR
global scoreStroop

global timeAI
global timeCD
global timeEOV
global timeML
global timePR
global timeStroop

def jawbone(cogCSV, sleepCSV):
	#Read in csv files
	cogData = pd.read_csv(cogCSV)
	sleepData = pd.read_csv(sleepCSV)
	#Format cognitive data so it is better for analysis
	cogData = pd.DataFrame({"Date":pd.to_datetime(cogData.Date), 
								"Test":cogData.Test, "Score":cogData.Score, "ID":cogData.ID})
	scoreSplit = pd.DataFrame(cogData.Score.str.split("|", 1).tolist(), columns=['Score', 'AvgTime'])
	cogData = pd.DataFrame({"Date":pd.to_datetime(cogData.Date), 
								"Test":cogData.Test, "Score":pd.to_numeric(scoreSplit.Score), "AvgTime":pd.to_numeric(scoreSplit.AvgTime), "ID":cogData.ID})
	#Strip uneeded data from jawbone data
	sleepData = pd.DataFrame({"Date":sleepData.DATE, "Sleep":sleepData.s_duration})
	sleepData = sleepData[np.isfinite(sleepData['Sleep'])]
	sleepData = pd.DataFrame({"Date":pd.to_datetime(sleepData.Date, format="%Y%m%d"), "Sleep":sleepData.Sleep})
	#Merge the sleep data and cognitive data
	cogDataIndexed = cogData.reset_index()
	cogDataDate = pd.DataFrame({"Date":pd.DatetimeIndex(cogDataIndexed.Date).normalize(), "index":cogDataIndexed.index})
	cogDataDate = cogDataDate.merge(sleepData, how='left', on='Date')
	mergedData = cogDataDate.merge(cogDataIndexed, how='left', on='index')
	del mergedData['Date_x']
	del mergedData['index']
	mergedData.columns = ['Sleep', 'AvgTime', 'Date', 'ID', 'Score', 'Test']
	mergedData = mergedData.sort_values(['ID', 'Test', 'Date'], ascending=[True, True, True])
	mergedData = mergedData.reset_index(drop=True)
	return mergedData

def addNormScore(data):
	normData = pd.DataFrame({"Date":data.Date, TEST_NAME:data.Test, SCORE:data.Score, AVG_TIME:data.AvgTime,
							 "ID":data.ID, SLEEP:data.Sleep,
							 NORM_SCORE:createNormScoreSeries(data.Score, data.AvgTime, data.Test)})
	return normData

#Plotting methods

#Plots the normalised processing speed score against sleep time in a scatter graph
#Requires data to include normalised score already
def plotProcessNormScatter(data):
	#Define the tests that count towards processing speed score

	#Sort the data by ID then date so it can be processed a day at time
	dataIDS = data.sort_values(['ID', 'Date'], ascending=[True, True]);
	#Dataframe for displaying scatter plot
	dataPScore = pd.DataFrame(columns=('ID', 'Date', NORM_SCORE, SLEEP))
	count = 0;
	print(dataPScore.columns)
	for index, row in dataIDS.iterrows():
		if(count == 0):
			dataPScore.loc[0] = [row.ID, row.Date, row[NORM_SCORE],
									   row[SLEEP]]
			count += 1
		else:
			if(row['ID'] != dataPScore.loc[count - 1]['ID']): #If the loop has gone to the next user in the frame
				dataPScore.loc[count] = [row.ID, row.Date, row[NORM_SCORE],
									   row[SLEEP]]
				count += 1
			else:
				dateDif = row['Date'] - dataPScore.loc[count - 1]['Date']
				if(1 <= dateDif.components.days or 1 <= dateDif.components.hours): #If the loop has gone to the next date in the frame

					dataPScore.loc[count] = [row.ID, row.Date, row[NORM_SCORE],
									   row[SLEEP]]
					count += 1
				else:
					if(isProcressing(row[TEST_NAME])):
						dataPScore.set_value(count - 1, NORM_SCORE, dataPScore.loc[count - 1][NORM_SCORE] + row[NORM_SCORE])
	dataPScore.plot.scatter(x=SLEEP, y=NORM_SCORE)
	print(dataPScore)
	plt.show()
	return

def plotAIScatter(data):
	ai = data.loc[data[TEST_NAME] == ARROW_IGNORING]
	ai.plot.scatter(x=SLEEP, y=AVG_TIME)
	plt.show()
	return

def plotAINormScatter(data):
	ai = data.loc[data[TEST_NAME] == ARROW_IGNORING]
	ai.plot.scatter(x=SLEEP, y=NORM_SCORE)
	plt.show()
	return

def plotCDScatter(data):
	cd = data.loc[data[TEST_NAME] == CHANGING_DIRECTIONS]
	cd.plot.scatter(x=SLEEP, y=AVG_TIME)
	plt.show()
	return

def plotCDNormScatter(data):
	cd = data.loc[data[TEST_NAME] == CHANGING_DIRECTIONS]
	cd.plot.scatter(x=SLEEP, y=NORM_SCORE)
	plt.show()
	return

def plotPatternScatter(data):
	pattern = data.loc[data[TEST_NAME] == PATTERN_RECREATION]
	pattern.plot.scatter(x=SLEEP, y=AVG_TIME)
	plt.show()
	return

def plotStroopScatter(data):
	stroop = data.loc[data[TEST_NAME] == STROOP]
	stroop.plot.scatter(x=SLEEP, y=AVG_TIME)
	plt.show()
	return

#Methods for calculating normalised scores

def createNormScoreSeries(sScore, sTime, sTest):
	sNormScore = pd.Series()
	for index, test in sTest.iteritems():
		sNormScore = sNormScore.set_value(index, calcNormScore(sScore[index], sTime[index], test))
	return sNormScore

def calcNormScore(score, time, test):
	return {
		APPEARING_OBJECT:normScoreAO(score),
		APPEARING_OBJECT_FIXED:normScoreAOF(score),
		ARROW_IGNORING:normScoreAI(score, time),
		CHANGING_DIRECTIONS:normScoreCD(score, time),
		CHASE_TEST:normScoreCT(score),
		EVEN_OR_VOWEL:normScoreEOV(score, time),
		FINGER_TAP:normScoreFT(score),
		MONKEY_LADDER:normScoreML(score, time),
		CARD_LEARNING:normScoreCL(score),
		PATTERN_RECREATION:normScorePR(score, time),
		STROOP:normScoreStroop(score, time)
	}[test]

def normScoreAO(score):
	global scoreAO
	return (-score + scoreAO) / scoreAO

def normScoreAOF(score):
	global scoreAOF
	return (-score + scoreAOF) / scoreAOF

def normScoreAI(score, time):
	#Declare use of global variables
	global scoreAI, timeAI
	normScore = ((-time + timeAI) * score) / (scoreAI * timeAI)
	return normScore

def normScoreCD(score, time):
	#Declare use of global variables
	global scoreCD, timeCD
	normScore = ((-time + timeCD) * score) / (scoreCD * timeCD)
	return normScore

def normScoreCT(score):
	global scoreCT
	return score / scoreCT

def normScoreEOV(score, time):
	#Declare use of global variables
	global scoreEOV, timeEOV
	normScore = ((-time + timeEOV) * score) / (scoreEOV * timeEOV)
	return normScore

def normScoreFT(score):
	global scoreFT
	return score / scoreFT

def normScoreML(score, time):
	#Declare use of global variables
	global scoreML, timeML
	normScore = ((-time + timeML) * score) / (scoreML * timeML)
	return normScore

def normScoreCL(score):
	global scoreCL
	return score / scoreCL

def normScorePR(score, time):
	#Declare use of global variables
	global scorePR, timePR
	normScore = ((-time + timePR) * score) / (scorePR * timePR)
	return normScore

def normScoreStroop(score, time):
	#Declare use of global variables
	global scoreStroop, timeStroop
	normScore = ((-time + timeStroop) * score) / (scoreStroop * timeStroop)
	return normScore

#Methods for setting test maximums from data

def maxInit(data):
	maxAOInit(data)
	maxAOFInit(data)
	maxAIInit(data)
	maxCDInit(data)
	maxCLInit(data)
	maxCTInit(data)
	maxEOVInit(data)
	maxFTInit(data)
	maxMLInit(data)
	maxPRInit(data)
	maxStroopInit(data)
	return

#Sets the maximum Appearing Object tests score from the given DataFrame.
#This assumes there is no current maximum set.
def maxAOInit(data):
	#Declare use of global variables
	global scoreAO
	#Extract Appearing Object rows from data
	dataAO = data.loc[data[TEST_NAME] == APPEARING_OBJECT]
	dataAO = dataAO.reset_index(drop=True)
	for index, row in dataAO.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreAO = row[SCORE]
		else:
			if(scoreAO < row[SCORE]):
				scoreAO = row[SCORE]
	return

#Sets the maximum Appearing Object Fixed Object tests score from the given DataFrame.
#This assumes there is no current maximum set.
def maxAOFInit(data):
	#Declare use of global variables
	global scoreAOF
	#Extract Appearing Object Fixed Object rows from data
	dataAOF = data.loc[data[TEST_NAME] == APPEARING_OBJECT_FIXED]
	dataAOF = dataAOF.reset_index(drop=True)
	for index, row in dataAOF.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreAOF = row[SCORE]
		else:
			if(scoreAOF < row[SCORE]):
				scoreAOF = row[SCORE]
	return

#Sets the maximum Arrow Ignoring tests score from the given DataFrame. 
#This assumes there is no current maximum set.
def maxAIInit(data):
	#Declare use of global variables
	global scoreAI, timeAI
	#Extract Arrow Ignoring rows from data
	dataAI = data.loc[data[TEST_NAME] == ARROW_IGNORING]
	dataAI = dataAI.reset_index(drop=True)
	for index, row in dataAI.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreAI = row[SCORE]
			timeAI = row[AVG_TIME]
		else:
			if(scoreAI < row[SCORE]):
				scoreAI = row[SCORE]
			if(timeAI < row[AVG_TIME]):
				timeAI = row[AVG_TIME]
	return

#Sets the maximum Changing Directions tests score from the given DataFrame.
#This assumes there is no current maximum set.
def maxCDInit(data):
	#Declare use of global variables
	global scoreCD, timeCD
	#Extract Arrow Ignoring rows from data
	dataCD = data.loc[data[TEST_NAME] == CHANGING_DIRECTIONS]
	dataCD = dataCD.reset_index(drop=True)
	for index, row in dataCD.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreCD = row[SCORE]
			timeCD = row[AVG_TIME]
		else:
			if(scoreCD < row[SCORE]):
				scoreCD = row[SCORE]
			if(timeCD < row[AVG_TIME]):
				timeCD = row[AVG_TIME]
	return

#Sets the maximum Chase Tests scores from the given DataFrame.
#This assumes there is no current maximum set.
def maxCTInit(data):
	#Declare use of global variables
	global scoreCT
	#Extract Chase Test rows from data
	dataCT = data.loc[data[TEST_NAME] == CHASE_TEST]
	dataCT = dataCT.reset_index(drop=True)
	for index, row in dataCT.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreCT = row[SCORE]
		else:
			if(scoreCT < row[SCORE]):
				scoreCT = row[SCORE]
	return

#Sets the maximum Even or Vowel tests score from the given DataFrame. 
#This assumes there is no current maximum set.
def maxEOVInit(data):
	#Declare use of global variables
	global scoreEOV, timeEOV
	#Extract Even or Vowel rows from data
	dataEOV = data.loc[data[TEST_NAME] == EVEN_OR_VOWEL]
	dataEOV = dataEOV.reset_index(drop=True)
	for index, row in dataEOV.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreEOV = row[SCORE]
			timeEOV = row[AVG_TIME]
		else:
			if(scoreEOV < row[SCORE]):
				scoreEOV = row[SCORE]
			if(timeEOV < row[AVG_TIME]):
				timeEOV = row[AVG_TIME]
	return

#Sets the maximum Finger Tap Test scores from the given DataFrame.
#This assumes there is no current maximum set.
def maxFTInit(data):
	#Declare use of global variables
	global scoreFT
	#Extract Finger Tap Test rows from data
	dataFT = data.loc[data[TEST_NAME] == FINGER_TAP]
	dataFT = dataFT.reset_index(drop=True)
	for index, row in dataFT.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreFT = row[SCORE]
		else:
			if(scoreFT < row[SCORE]):
				scoreFT = row[SCORE]
	return

#Sets the maximum Monkey Ladder test scores from the given DataFrame.
#This assumes there is no current maximum set.
def maxMLInit(data):
	#Declare use of global variables
	global scoreML, timeML
	#Extract Monkey Ladder rows from data
	dataML = data.loc[data[TEST_NAME] == MONKEY_LADDER]
	dataML = dataML.reset_index(drop=True)
	for index, row in dataML.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreML = row[SCORE]
			timeML = row[AVG_TIME]
		else:
			if(scoreML < row[SCORE]):
				scoreML = row[SCORE]
			if(timeML < row[AVG_TIME]):
				timeML = row[AVG_TIME]
	return

#Sets the maximum One Card Learnind Test scores from the given DataFrame.
#This assumes there is no current maximum set.
def maxCLInit(data):
	#Declare use of global variables
	global scoreCL
	#Extract One Card Leaning Test rows from data
	dataCL = data.loc[data[TEST_NAME] == CARD_LEARNING]
	dataCL = dataCL.reset_index(drop=True)
	for index, row in dataCL.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreCL = row[SCORE]
		else:
			if(scoreCL < row[SCORE]):
				scoreCL = row[SCORE]
	return

#Sets the maximum Pattern Recreation test scores from the given DataFrame.
#This assumes there is no current maximum set.
def maxPRInit(data):
	#Declare use of global variables
	global scorePR, timePR
	#Extract Pattern Recreation rows from data
	dataPR = data.loc[data[TEST_NAME] == PATTERN_RECREATION]
	dataPR = dataPR.reset_index(drop=True)
	for index, row in dataPR.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scorePR = row[SCORE]
			timePR = row[AVG_TIME]
		else:
			if(scorePR < row[SCORE]):
				scorePR = row[SCORE]
			if(timePR < row[AVG_TIME]):
				timePR = row[AVG_TIME]
	return

#Sets the maximum Stroop test scores from the given DataFrame.
#This assumes there is no current maximum set.
def maxStroopInit(data):
	#Declare use of global variables
	global scoreStroop, timeStroop
	#Extract Stroop Test rows from data
	dataStroop = data.loc[data[TEST_NAME] == STROOP]
	dataStroop = dataStroop.reset_index(drop=True)
	for index, row in dataStroop.iterrows():
		#If it's the first row then overrirde current maximums
		if(index == 0):
			scoreStroop = row[SCORE]
			timeStroop = row[AVG_TIME]
		else:
			if(scoreStroop < row[SCORE]):
				scoreStroop = row[SCORE]
			if(timeStroop < row[AVG_TIME]):
				timeStroop = row[AVG_TIME]
	return

#Score calculation dictionaries
def isProcressing(testName):
	pDict = {
		APPEARING_OBJECT:True,
		ARROW_IGNORING:True,
		CHANGING_DIRECTIONS:True,
		CHASE_TEST:True,
		EVEN_OR_VOWEL:True,
		MONKEY_LADDER:True,
		STROOP:True
	}
	return pDict.get(testName, False)