__author__ = 'Johnny'


#from generate_input import default_vocab,defaultdict

import time
import cPickle as pickle
import theano
from theano import tensor as T
import lasagne
import sys,os,inspect
base_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../base/")))
cos_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../../loss/cosine")))
if base_subfolder not in sys.path:
     sys.path.insert(0, base_subfolder)
if cos_subfolder not in sys.path:
	sys.path.insert(0,cos_subfolder)

import base as base
import cosinesimilarity as l


#
# def cosine_similarity(y_true, y_pred):
#     return 1-T.abs_(vector_matrix_dot(y_true, y_pred) / norm(y_true) / norm(y_pred))



class ff(base.Base):
	def run(self,Sampling, X_train_path, Y_train_path, eval_path):
		self.load_dataset(Sampling,X_train_path, Y_train_path, eval_path)
		self.build_and_train_model(self.N_Hidden)
		self.write_best_model()
		self.eval()


	def build_and_train_model(self,n_hidden):
		print('Building Model')

		input_phrase = T.imatrix('train_inputmatrix')
		labels = T.imatrix('trainphrase_matrix')

		network = self.define_layers(input_phrase,labels,n_hidden)

		print("Defining loss")
		#Prediction or loss
		prediction = []
		prediction.append(T.clip(lasagne.layers.get_output(network[0]),1.0e-7,1.0-1.0e-7))
		prediction.append(T.clip(lasagne.layers.get_output(network[1]),1.0e-7,1.0-1.0e-7))

		loss = l.define_loss(prediction[0],prediction[1])
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
			loss, predicted, phrase = train_fn(self.train_inputmatrix,self.trainphrase_matrix)
			print('Training Loss: ' + str(loss) + ' Train Epoch ' + str(epoch))
			self.save_best(loss,predicted,network)

		#eval_fn = theano.function([input_phrase,labels],[loss, prediction[0], prediction[1]],allow_input_downcast=True)
		# e_loss, e_predicted, e_phrase = eval_fn(self.eval_matrix,self.evalphrase_matrix)
		# print('Evaluating current loss is ' +str(e_loss))



	def eval(self):
		print('Evaluating '  + str(self.eval_vocab_size))
		input_phrase = T.imatrix('eval_inputmatrix')
		labels = T.imatrix('evalphrase_matrix')
		network = self.define_layers(input_phrase, labels)
		params = self.best_params
		prediction = []
		prediction.append(T.clip(lasagne.layers.get_output(network[0]),1.0e-7,1.0-1.0e-7))
		prediction.append(T.clip(lasagne.layers.get_output(network[1]),1.0e-7,1.0-1.0e-7))
		loss = l(prediction[0],prediction[1])

		eval_fn = theano.function([input_phrase,labels],[loss, prediction[0], prediction[1]],updates=None,allow_input_downcast=True)
		e_loss, e_predicted, e_phrase = eval_fn(self.eval_inputmatrix,self.evalphrase_matrix)
		print('Evaluating current loss is ' +str(e_loss))
		self.write_eval(e_predicted)



	#define layers
	def define_layers(self, input_indices, phrase_indices, N_H = 200):
		print('Defining layers')
		MAX_LENGTH=12
		N_HIDDEN = int(N_H)
		#define embedding inputs
		input_layer = lasagne.layers.InputLayer(shape=(1,MAX_LENGTH), input_var=input_indices)

		embedding_layer = lasagne.layers.EmbeddingLayer(input_layer, input_size=self.vocab_size, output_size=200,W=self.all_embedding)
		#Don't train embedding layer
		embedding_layer.params[embedding_layer.W].remove('trainable')

		#Concat or Average

		#Hidden Layer dxd
		hidden_layer = lasagne.layers.DenseLayer(embedding_layer,N_HIDDEN, W=lasagne.init.GlorotUniform())

		#Outputlayer

		output_layer = lasagne.layers.DenseLayer(hidden_layer,N_HIDDEN,nonlinearity=lasagne.nonlinearities.sigmoid)

		#define phrase output embeddings
		phrase_layer = lasagne.layers.InputLayer(shape=(1,1), input_var=phrase_indices)
		embedding_layer_phrase = lasagne.layers.EmbeddingLayer(phrase_layer, input_size=self.trainphrase_vocab_size, output_size=200, W=self.trainevalphrase_embeddings)

		#Don't train embedding layer
		embedding_layer_phrase.params[embedding_layer_phrase.W].remove('trainable')

		#Reshape output
		phrase_emb_reshape_layer = lasagne.layers.ReshapeLayer(embedding_layer_phrase, (-1,200))
		return output_layer,phrase_emb_reshape_layer

