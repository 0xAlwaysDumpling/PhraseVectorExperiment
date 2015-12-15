__author__ = 'Johnny'

import sys,os,inspect
import cPickle as pickle
import numpy as np
import lasagne

from collections import defaultdict


def default_word():
	return "Skip"


def default_vocab():
	return 1

def default_phrase():
	return "unk"

def default_vocab():
	return -1

class Base(object):
	def __init__(self,name,batch_size,N_hidden,end_epoch=10):
		self.N_Hidden = N_hidden
		self.end_epoch = int(end_epoch)
		self.start_epoch = 0
		self.best_loss = 100
		self.best_bias = 0
		self.model = None
		self.name = name
		self.best_prediction = None
		self.best_model = None
		self.best_params = None


	def save_best(self,loss,predicted,model):
		if loss < self.best_loss:
			self.best_params = lasagne.layers.get_all_param_values(self.model)
			self.best_prediction = predicted
			self.best_loss = loss
			self.best_model = model

	def write_eval(self, predictions):
		print('Saving Eval Predictions')
		o = str(self.eval_vocab_size) + '.prediction'
		arr = np.array(predictions)
		arr.tolist()
		with open('eval.' + str(self.eval_vocab_size)+'.txt', "w") as f:
			for i in range(0,len(arr)):
				vector = arr[i]
				vector /= np.linalg.norm(vector)
				f.write(str(self.eval_wordmap[i])+'\t'+' '.join(map(str,vector))+ '\n')

	def write_best_model(self):
		print('Saving Best Model...gf')
		o = str(self.trainphrase_vocab_size-self.eval_vocab_size)+ '.' + str(self.eval_vocab_size) + '.model'
		with open('../dat/models/'+o, 'wb') as f:
			pickle.dump((self.end_epoch,self.best_loss, self.best_bias, lasagne.layers.get_all_param_values(self.model), self.best_prediction),f,protocol=-1)

	def load_dataset(self, Sampling=False,X_pickle_path=None, Y_pickle_path=None, e_path=None):
		print('Loading data sets')

		#load inputs
		if X_pickle_path is None:
			raise ValueError("Train path not found")

		with open(X_pickle_path,'rb') as f:
			self.all_embedding,self.train_inputmatrix,self.train_vocab2word,self.train_word2vocab,self.vocab_size, self.train_vocab= pickle.load(f)
		with open(Y_pickle_path,'rb') as f:
			self.trainevalphrase_embeddings, self.trainphrase_matrix,self.trainphrase_vocab2word,self.trainphrase_word2vocab,self.trainphrase_vocab_size = pickle.load(f)


		#load eval
		with open(e_path,'rb') as f:
			self.eval_embeddings, self.eval_inputmatrix, self.evalphrase_matrix, self.eval_vocab2word,self.eval_word2vocab, self.eval_vocab_size, self.eval_wordmap = pickle.load(f)
		print("Input and Gold files loaded")



