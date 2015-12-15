__author__ = 'Johnny'





import sys
from collections import defaultdict
import cPickle as pickle
import numpy as np



print('Loading gold training file')



def default_phrase():
    return "unk"

def default_vocab():
    return -1






#Embedding Size
vector_dim = 200

# Number of Words in input
trainphrase_vocab_size = 0


trainphrase_phrase2vocab = {'unk' : '-1'}
trainphrase_vocab2phrase = {'-1' : 'unk'}


trainphrase_embeddings = []

train_size = int(sys.argv[1])
train_file = sys.argv[2]



t_matrix = (train_size,1)
trainphrase_matrix = np.zeros(t_matrix)
for i in range(0,int(train_size)):
    trainphrase_matrix[i] = i


space_delimiter = ' '
print("Reading Gold Train File")
linecount = 0
with open(train_file,'r') as f:
    for line in f.readlines():
        if linecount>= train_size:
            break
        currLine = line.split(space_delimiter)
        trainphrase_phrase2vocab[currLine[0]] = trainphrase_vocab_size
        trainphrase_vocab2phrase[trainphrase_vocab_size] = currLine[0]
        trainphrase_vocab_size+=1
        if currLine[-1] == '\n':
            currLine = currLine[:-1]
        elif currLine[-1][-1] == '\n':
            currLine[-1] = currLine[-1][:-1]
        vector = np.array(currLine[1:], dtype='float32')
        vector /= np.linalg.norm(vector)
        trainphrase_embeddings.append(vector)
        linecount+=1



print(len(trainphrase_embeddings))
print(len(trainphrase_matrix))
print('Saving to pickle')
out = str(train_size)+'.'+'.gold.pickle'

with open(out,'wb') as f:
    pickle.dump([np.asarray(trainphrase_embeddings, np.float32), np.asarray(trainphrase_matrix,np.float32), trainphrase_vocab2phrase, trainphrase_phrase2vocab, trainphrase_vocab_size],f,protocol=1)
