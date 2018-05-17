import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# to find the accuracy
def accuracy(predicted,groundTruth):
              wrong = 0
              for i in range(len(groundTruth)):
                  if(predicted[i] != groundTruth[i]):
                      wrong = wrong + 1
              return str(((len(groundTruth)-wrong)/(len(groundTruth)*1.0))*100)+"%"

def getFeatures(tag_list,ctag_list,word_list,tag_previous,word_current,word_current_tag,word_next_tag,word_prev_ctag):
# pos of previous word
	ans = np.zeros((len(tag_list)), dtype=int) #to encode the features
	ans[list(tag_list).index(tag_previous)] = 1

# pos of current word
	c_tag = np.zeros((len(tag_list)), dtype=int)
	c_tag[list(tag_list).index(word_current_tag)] = 1
	ans = np.append(ans,c_tag)

# pos of next word
	n_tag = np.zeros((len(tag_list)), dtype=int)
	n_tag[list(tag_list).index(word_next_tag)] = 1
	ans = np.append(ans,n_tag)

# chunk of previous word
	p_ctag = np.zeros((len(ctag_list)), dtype=int)
	p_ctag[list(ctag_list).index(word_prev_ctag)] = 1
	ans = np.append(ans,p_ctag).flatten()
	return ans

train = pd.read_csv('train.txt', sep=" ", header=None, names=["word", "pos", "chunk"])
test = pd.read_csv('test.txt', sep=" ", header=None, names=["word", "pos", "chunk"])

# print train.shape
# print test.shape

# creating a list of all possible chunk tags
ctag_train = train.chunk.unique()
ctag_test = test.chunk.unique()
ctag_list = np.unique(np.append(ctag_train,ctag_test))

# creating a list of all possible pos tags
tag_train = train.pos.unique()
tag_test = test.pos.unique()
tag_list = np.unique(np.append(tag_train,tag_test))

# creating a list of all possible words
word_train = train.word.unique()
word_test = test.word.unique()
word_list = np.unique(np.append(word_train,word_test))

features = getFeatures(tag_list,ctag_list,word_list,train.pos[0],train.word[1],train.pos[1],train.pos[2],train.chunk[0])
for val in range(2,train.shape[0]-1):
	features = np.vstack((features, getFeatures(tag_list,ctag_list,word_list,train.pos[val-1],train.word[val],train.pos[val],train.pos[val+1],train.chunk[val-1])))

labels_temp = np.array(train.chunk).flatten()
labels = []
for i in range(1,len(labels_temp)-1):
	labels.append(list(ctag_list).index(labels_temp[i]))

#training
logisticRegr = LogisticRegression(penalty='l2')
logisticRegr.fit(features, labels)

labels_temp_test = np.array(test.chunk).flatten()
labels_test = []
for i in range(1,len(labels_temp_test)-1):
        labels_test.append(list(ctag_list).index(labels_temp_test[i]))

test_features = getFeatures(tag_list,ctag_list,word_list,test.pos[0],test.word[1],test.pos[1],test.pos[2],test.chunk[0])
for val in range(2,test.shape[0]-1):
	test_features = np.vstack((test_features, getFeatures(tag_list,ctag_list,word_list,test.pos[val-1],test.word[val],test.pos[val],test.pos[val+1],test.chunk[val-1])))

predictions = logisticRegr.predict(test_features).flatten()

print "Accuracy:- ",accuracy(predictions,labels_test)
