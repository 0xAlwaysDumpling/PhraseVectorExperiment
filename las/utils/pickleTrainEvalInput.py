__author__ = 'Johnny'




import sys
from collections import defaultdict
import cPickle as pickle
import numpy as np



print('Loading input file')


def default_word():
	return "Skip"


def default_vocab():
	return 1


#Maps word to indices
train_word2vocab = {'unk': '-1'}
train_vocab2word = {'-1' :'unk'}
eval_word2vocab = {'unk' : '-1'}
eval_vocab2word = { '-1' : 'unk'}

vector_dim = 200
vocab_size = 0

all_embeddings = []
eval_embeddings = []

train_size = int(sys.argv[1])
eval_size = int(sys.argv[2])
phrase_size = int(sys.argv[3])
train_file = sys.argv[4]
eval_file = sys.argv[5]
eval_vocab_size = 0


t_imatrix = (train_size, phrase_size)
e_imatrix = (eval_size, phrase_size)

train_inputmatrix = np.zeros(t_imatrix)
eval_inputmatrix = np.zeros(e_imatrix)


e_outmatrix = (eval_size, 1)
evalphrase_matrix = np.zeros(e_outmatrix)
for i in range(0, eval_size):
	evalphrase_matrix[i] = i

eval_wordmap = {'-1' :  'unk'}
e_count = 0
train_vocab = 0

print('Reading train file building....')

rowIndex = 0
tab_delimiter = '\t'
space_delimiter = ' '

linecount = 0
count12 = 0


with open(train_file,'r') as f:
	for line in f.readlines():
		currLine = line.split(tab_delimiter)
		words = currLine[0].split(' ')
		m_temp = []
		linecount+=1
		if(len(words)>12): continue
		for word in words:
			train_word2vocab[word] = vocab_size
			train_vocab2word[vocab_size] = word
			m_temp.append(vocab_size)
			vocab_size+=1
		padd_diff = phrase_size - len(m_temp)
		for x in range(0,padd_diff):
			m_temp.append(0)
		train_inputmatrix[rowIndex] = m_temp
		for vector in currLine[1:]:
			arr = vector.split(' ')
			arr = np.array(map(float,arr))
			np.asarray(arr,np.float32)
			arr /= np.linalg.norm(arr)
			all_embeddings.append(arr)


train_vocab = len(all_embeddings)
rowIndex = 0


print('Reading eval file')
with open(eval_file,'r') as e:
	for line in e.readlines():
		if line is None or line == '\n':
			continue
		currLine = line.split(tab_delimiter)
		if '_' in currLine[0]:
			words = currLine[0].split('_')
			eval_wordmap[e_count] = currLine[0]
			e_count+=1
			m_temp = []
			for word in words:
				eval_word2vocab[word] = vocab_size
				eval_vocab2word[vocab_size] = word
				m_temp.append(vocab_size)
				vocab_size+=1
				eval_vocab_size+=1
			padd_diff = phrase_size - len(m_temp)
			for x in range(0,padd_diff):
				m_temp.append(0)
			eval_inputmatrix[rowIndex] = m_temp
			rowIndex+=1
		else:
			word = currLine[0]
			eval_wordmap[e_count] = word
			e_count+=1
			m_temp = []
			eval_word2vocab[word] = vocab_size
			eval_vocab2word[vocab_size] = word
			m_temp.append(vocab_size)
			vocab_size+=1
			padd_diff = phrase_size - len(m_temp)
			for x in range(0,padd_diff):
				m_temp.append(0)
			eval_inputmatrix[rowIndex] = m_temp
			rowIndex+=1
		for vector in currLine[1:]:
			if vector == '\t' or vector == '\n' or vector is None: continue

			vector = vector.strip()
			arr = vector.split(' ')

			arr = np.array(map(float,arr))
			np.asarray(arr,np.float32)
			arr /= np.linalg.norm(arr)
			all_embeddings.append(arr)
			eval_embeddings.append(arr)






print('Saving to pickle')
out = str(train_size)+'.'+'.input.pickle'

with open(out,'wb') as f:
	pickle.dump([
		np.asarray(all_embeddings,np.float32),
		np.asarray(train_inputmatrix,np.float32),
		train_vocab2word,
		train_word2vocab,
		vocab_size,
		train_vocab
	],f,protocol=2)


out = str(eval_size)+'.'+'eval.inputs.pickle'


with open(out,'wb') as f:
	pickle.dump(
		[
			np.asarray(eval_embeddings,np.float32),
			np.asarray(eval_inputmatrix,np.float32),
			np.asarray(evalphrase_matrix,np.float32),
			eval_vocab2word,
			eval_word2vocab,
			eval_vocab_size,
			eval_wordmap
		],
		f, protocol=2)