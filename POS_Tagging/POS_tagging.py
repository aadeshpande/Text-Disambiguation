import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# checks if the word contains a number or not
def containsNum(st):
	for i in st:
		if(i.isdigit()):
			return True
	return False

# checks if the first letter is uppercase
def isCapital(st):
	if(len(st)<1):
		return False
	return st[0].isupper()

def getWordFeatures(word_current):
	'''lower case,uppercase,num,captal'''
	ans = np.zeros((4), dtype=int)
	if(word_current.islower()):#lower case
		ans[0] = 1
	if(word_current.isupper()):#uppercase
		ans[1] = 1
	if(containsNum(word_current)):#num
		ans[2] = 1
	if(isCapital(word_current)):#captal
		ans[3] = 1
	return ans

def getPreSufFeature(suffix,prefix,word):
	ans_suffix2 = np.zeros((len(suffix)), dtype=int) #to encode suffixes that are 2 letters long
	ans_prefix2 = np.zeros((len(prefix)), dtype=int) #to encode prefixes that are 2 letters long
	ans_suffix3 = np.zeros((len(suffix)), dtype=int) #to encode suffixes that are 3 letters long
	ans_prefix3 = np.zeros((len(prefix)), dtype=int) #to encode prefixes that are 2 letters long
	if(containsNum(word)==False):
		if(len(word)>=2):
			ans_suffix2[list(suffix).index(word[-2:])] = 1
			ans_prefix2[list(prefix).index(word[:2])] = 1
		if(len(word)>=3):
			ans_suffix3[list(suffix).index(word[-3:])] = 1
			ans_prefix3[list(prefix).index(word[:3])] = 1
	ans_suffix2 = np.append(ans_suffix2,ans_prefix2)
	ans_suffix2 = np.append(ans_suffix2,ans_suffix3)
	ans_suffix2 = np.append(ans_suffix2,ans_prefix3)
	return ans_suffix2

# get accuracy
def accuracy(predicted,groundTruth):
              wrong = 0
              for i in range(len(groundTruth)):
                  if(predicted[i] != groundTruth[i]):
                      wrong = wrong + 1
              return str(((len(groundTruth)-wrong)/(len(groundTruth)*1.0))*100)+"%"

def getFeatures(tag_list,word_list,suffix,prefix,tag_previous,word_previous,word_current):
	ans = np.zeros((len(tag_list)), dtype=int) #to encode the features
	ans[list(tag_list).index(tag_previous)] = 1

	#word vec of word_previous
	ans_word = np.zeros((len(word_list)), dtype=int)
	ans_word[list(word_list).index(word_previous)] = 1
	ans = np.append(ans,ans_word)

	#word vec of word_current
	ans_wordC = np.zeros((len(word_list)), dtype=int)
	ans_wordC[list(word_list).index(word_current)] = 1
	ans = np.append(ans,ans_wordC)

	#word prefix and suffix of len 2 and 3
	ans = np.append(ans,getPreSufFeature(suffix,prefix,word_previous))
	ans = np.append(ans,getPreSufFeature(suffix,prefix,word_current))

	ans = np.append(ans,getWordFeatures(word_previous))
	ans = np.append(ans,getWordFeatures(word_current)).flatten()
	return ans

train = pd.read_csv('train.txt', sep=" ", header=None, names=["word", "pos", "chunk"])
test = pd.read_csv('test.txt', sep=" ", header=None, names=["word", "pos", "chunk"])

#print train.shape
#print test.shape

# creating a list of all possible tags
tag_train = train.pos.unique()
tag_test = test.pos.unique()
tag_list = np.unique(np.append(tag_train,tag_test))

# creating a list of all possible words
word_train = train.word.unique()
word_test = test.word.unique()
word_list = np.unique(np.append(word_train,word_test))

#creating a list of all possible suffixes
suffix = {}
for wordT in word_list:
	if(containsNum(wordT)==False):
		if(len(wordT)>=2):
			if wordT[-2:] not in suffix:
				suffix[wordT[-2:]] = 1
		if(len(wordT)>=3):
			if wordT[-3:] not in suffix:
				suffix[wordT[-3:]] = 1

suffix = suffix.keys()

#creating a list of all possible prefixes
prefix = {}
for wordT in word_list:
	if(containsNum(wordT)==False):
		if(len(wordT)>=2):
			if wordT[:2] not in prefix:
				prefix[wordT[:2]] = 1
		if(len(wordT)>=3):
			if wordT[:3] not in prefix:
				prefix[wordT[:3]] = 1

prefix = prefix.keys()

features = getFeatures(tag_list,word_list,suffix,prefix,'.','.',train.word[0])
for val in range(1,train.shape[0]):
	features = np.vstack((features, getFeatures(tag_list,word_list,suffix,prefix,train.pos[val-1],train.word[val-1],train.word[val])))

labels_temp = np.array(train.pos).flatten()
labels = []
for i in range(len(labels_temp)):
	labels.append(list(tag_list).index(labels_temp[i]))

#print "Start Training"

logisticRegr = LogisticRegression(penalty='l2')
logisticRegr.fit(features, labels)

#print "Done Training"

labels_temp_test = np.array(test.pos).flatten()
labels_test = []
for i in range(len(labels_temp_test)):
        labels_test.append(list(tag_list).index(labels_temp_test[i]))

#Viterbi Algo

viterbi_array = []
for i in range(test.shape[0]):
	tag_array = []
	for j in range(len(tag_list)):
		tag_array.append([None,-1])
	viterbi_array.append(tag_array)

prob_w0 = logisticRegr.predict_log_proba(getFeatures(tag_list,word_list,suffix,prefix,'.','.',test.word[0]).reshape(1, -1))[0]
for i in range(len(tag_list)):
	viterbi_array[0][i][0] = prob_w0[i]
	viterbi_array[0][i][1] = -1

for i in range(1,test.shape[0]):
	for j in range(len(tag_list)):
		prob_wi = logisticRegr.predict_log_proba(getFeatures(tag_list,word_list,suffix,prefix,tag_list[j],test.word[j],test.word[i]).reshape(1, -1))[0]
		for k in range(len(tag_list)):
			if(viterbi_array[i][k][0] == None):
				viterbi_array[i][k][0] = prob_wi[k]+viterbi_array[i-1][j][0]
				viterbi_array[i][k][1] = j
			elif(viterbi_array[i][k][0]<prob_wi[k]+viterbi_array[i-1][j][0]):
				viterbi_array[i][k][0] = prob_wi[k]+viterbi_array[i-1][j][0]
				viterbi_array[i][k][1] = j

ans = []
max_end_val = None
max_end_index = -1
for i in range(len(tag_list)):
	if(max_end_val ==None):
		max_end_val = viterbi_array[test.shape[0]-1][i][0]
                max_end_index = i
	elif(viterbi_array[test.shape[0]-1][i][0]>max_end_val):
		max_end_val = viterbi_array[test.shape[0]-1][i][0]
		max_end_index = i
ans.append(max_end_index)

for i in range(test.shape[0]-1,0,-1):
	#print "pos",ans[len(ans)-1],"val",viterbi_array[i][ans[len(ans)-1]]
	ans.append(viterbi_array[i][ans[len(ans)-1]][1])
ans.reverse()

print "Accuracy:- ",accuracy(ans,labels_test)
