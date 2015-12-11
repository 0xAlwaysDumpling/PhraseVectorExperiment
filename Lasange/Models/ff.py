__author__ = 'Johnny'


#from generate_input import default_vocab,defaultdict
from collections import defaultdict
import cPickle as pickle

import sys
import os
import time

import numpy as np
import theano
from theano import tensor as T,function,printing
import lasagne


def default_phrase():
	return "unk"

def default_vocab():
	return -1


def default_word():
	return "unk"



def norm(x):
	return (T.sum(T.abs_(x)**2,axis=1))**(1.0/2)

def matrix_dot(x,y):
	return T.dot(x, T.transpose(y))

def vector_matrix_dot(x,y):
	return T.diagonal(matrix_dot(x,y))


class Base(object):
	def __init__(self,name,batch_size,end_epoch=10):
		self.end_epoch = int(end_epoch)
		self.start_epoch = 0
		self.best_loss = 100
		self.best_bias = 0
		self.model = None
		self.name = name
		self.best_prediction = None
		self.best_model = None
		self.best_params =None


	def run(self,Sampling, X_train_path, Y_train_path, eval_path):
		self.load_dataset(Sampling,X_train_path, Y_train_path, eval_path)
		self.build_and_train_model()
		self.write_best_model()
		self.eval()
	def save_best(self,loss,predicted,model):
		if loss < self.best_loss:
			self.best_params = lasagne.layers.get_all_param_values(self.model)
			self.best_prediction = predicted
			self.best_loss = loss
			self.best_model = model


	def write_eval(self, predictions):
		print('Saving Eval Predictions')
		o = str(self.evalphrase_vocab_size) + '.prediction'
		with open('../out/'+o, 'wb') as f:
			pickle.dump((self.eval_matrix,self.eval_vocab2word,predictions, self.trainphrase_embeddings),f,protocol=1)

	def write_best_model(self):
		print('Saving Best Model...gf')
		o = str(self.trainphrase_vocab_size-self.evalphrase_vocab_size)+ '.' + str(self.evalphrase_vocab_size) + '.model'
		with open('../out/'+o, 'wb') as f:
			pickle.dump((self.end_epoch,self.best_loss, self.best_bias, lasagne.layers.get_all_param_values(self.model), self.best_prediction),f,protocol=-1)

	def load_dataset(self, Sampling=False,X_pickle_path=None, Y_pickle_path=None, e_path=None):
		print('Loading data sets')

		#load inputs
		if X_pickle_path is None:
			raise ValueError("Train path not found")

		with open(X_pickle_path,'rb') as f:
			self.train_embedding,self.train_matrix,self.train_vocab2word,self.train_word2vocab,self.train_vocab_size = pickle.load(f)

		with open(Y_pickle_path,'rb') as f:
			self.evalphrase_embeddings, self.evalphrase_matrix,self.evalphrase_vocab2word,self.evalphrase_word2vocab,self.evalphrase_vocab_size, self.trainphrase_embeddings, self.trainphrase_matrix, self.trainphrase_vocab2phrase, self.trainphrase_phrase2vocab, self.trainphrase_vocab_size= pickle.load(f)


		#load eval
		with open(e_path,'rb') as f:
			self.eval_embedding, self.eval_matrix, self.eval_vocab2word, self.eval_word2vocab, self.eval_vocab_size = pickle.load(f)
		print("Input and Gold files loaded")




	def define_loss(self, prediction, labels):
		loss = (T.dot(prediction.T,labels) / (norm(prediction)[0]*norm(labels)[0])).mean()
		return loss


	def build_and_train_model(self):
		print('Building Model')

		input_phrase = T.imatrix('train_matrix')
		labels = T.imatrix('trainphrase_matrix')

		network = self.define_layers(input_phrase, labels)

		print("Defining loss")
		#Prediction or loss
		prediction = []
		prediction.append(T.clip(lasagne.layers.get_output(network[0]),1.0e-7,1.0-1.0e-7))
		prediction.append(T.clip(lasagne.layers.get_output(network[1]),1.0e-7,1.0-1.0e-7))

		loss = self.define_loss(prediction[0],prediction[1])
		self.model = network
		#define params
		params = lasagne.layers.get_all_params(network)
		updates = lasagne.updates.adadelta(loss,params)

		#run test

		train_fn = theano.function([input_phrase,labels],[loss, prediction[0], prediction[1]],updates=updates,allow_input_downcast=True)

		print("Model and params defined now training")
		epoch = 0
		for epoch in range(self.end_epoch):
			train_loss = 0
			train_pred = []
			start_time = time.time()
			loss, predicted, phrase = train_fn(self.train_matrix,self.trainphrase_matrix)
			print('Training Loss: ' + str(loss) + ' Train Epoch ' + str(epoch))
			self.save_best(loss,predicted,network)

		#eval_fn = theano.function([input_phrase,labels],[loss, prediction[0], prediction[1]],allow_input_downcast=True)
		# e_loss, e_predicted, e_phrase = eval_fn(self.eval_matrix,self.evalphrase_matrix)
		# print('Evaluating current loss is ' +str(e_loss))



	def eval(self):
		print('Evaluating '  + str(self.evalphrase_vocab_size))
		input_phrase = T.imatrix('eval_matrix')
		labels = T.imatrix('evalphrase_matrix')
		network = self.define_layers(input_phrase, labels)
		params = self.best_params
		prediction = []
		prediction.append(T.clip(lasagne.layers.get_output(network[0]),1.0e-7,1.0-1.0e-7))
		prediction.append(T.clip(lasagne.layers.get_output(network[1]),1.0e-7,1.0-1.0e-7))
		loss = self.define_loss(prediction[0],prediction[1])

		eval_fn = theano.function([input_phrase,labels],[loss, prediction[0], prediction[1]],updates=None,allow_input_downcast=True)
		e_loss, e_predicted, e_phrase = eval_fn(self.eval_matrix,self.evalphrase_matrix)
		print('Evaluating current loss is ' +str(e_loss))
		self.write_eval(e_predicted)
		pass



	#define layers
	def define_layers(self, input_indices, phrase_indices):
		print('Defining layers')
		N_HIDDEN=200
		MAX_LENGTH=12

		#define embedding inputs
		input_layer = lasagne.layers.InputLayer(shape=(1,MAX_LENGTH), input_var=input_indices)
		embedding_layer = lasagne.layers.EmbeddingLayer(input_layer, input_size=self.train_vocab_size, output_size=200,W=self.train_embedding)

		#Don't train embedding layer
		embedding_layer.params[embedding_layer.W].remove('trainable')

		#Concat or Average

		#Hidden Layer dxd
		hidden_layer = lasagne.layers.DenseLayer(embedding_layer,1.5*N_HIDDEN, W=lasagne.init.GlorotUniform())

		#Outputlayer

		output_layer = lasagne.layers.DenseLayer(hidden_layer,N_HIDDEN,nonlinearity=lasagne.nonlinearities.sigmoid)

		#define phrase output embeddings
		phrase_layer = lasagne.layers.InputLayer(shape=(1,1), input_var=phrase_indices)
		embedding_layer_phrase = lasagne.layers.EmbeddingLayer(phrase_layer, input_size=self.trainphrase_vocab_size, output_size=200, W=self.trainphrase_embeddings)

		#Don't train embedding layer
		embedding_layer_phrase.params[embedding_layer_phrase.W].remove('trainable')

		#Reshape output
		phrase_emb_reshape_layer = lasagne.layers.ReshapeLayer(embedding_layer_phrase, (-1,200))
		return output_layer,phrase_emb_reshape_layer

def main():
	name = sys.argv[1]
	batch_size = sys.argv[2]
	end_epoch = sys.argv[3]
	Sampling = sys.argv[4]
	X_train_path = sys.argv[5]
	Y_train_path = sys.argv[6]
	eval_path = sys.argv[7]
	ff = Base(name,batch_size,end_epoch)
	#ff.run(Sampling,X_train_path,Y_train_path, None)


	ff.run(Sampling,X_train_path,Y_train_path, eval_path)

if __name__ == '__main__':
	main()