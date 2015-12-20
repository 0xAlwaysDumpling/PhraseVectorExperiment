__author__ = 'Johnny'


#from generate_input import default_vocab,defaultdict


import lasagne
import sys,os,inspect
base_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../base/")))
if base_subfolder not in sys.path:
     sys.path.insert(0, base_subfolder)

import base as base


#
# def cosine_similarity(y_true, y_pred):
#     return 1-T.abs_(vector_matrix_dot(y_true, y_pred) / norm(y_true) / norm(y_pred))



class ff(base.Base):
	#define layers
	def define_layers(self, input_indices, phrase_indices, N_HU = 200, N_HL=1):
		print('Defining layers')
		MAX_LENGTH=12
		N_HIDDEN_UNITS = int(N_HU)
		#define embedding inputs
		input_layer = lasagne.layers.InputLayer(shape=(1,MAX_LENGTH), input_var=input_indices)

		embedding_layer = lasagne.layers.EmbeddingLayer(input_layer, input_size=self.vocab_size, output_size=200,W=self.all_embedding)
		#Don't train embedding layer
		embedding_layer.params[embedding_layer.W].remove('trainable')

		#Concat or Average

		#Hidden Layer dxd
		hidden_layer = lasagne.layers.DenseLayer(embedding_layer,N_HIDDEN_UNITS, W=lasagne.init.GlorotUniform())

		#Outputlayer

		output_layer = lasagne.layers.DenseLayer(hidden_layer,N_HIDDEN_UNITS,nonlinearity=lasagne.nonlinearities.sigmoid)

		#define phrase output embeddings
		phrase_layer = lasagne.layers.InputLayer(shape=(1,1), input_var=phrase_indices)
		embedding_layer_phrase = lasagne.layers.EmbeddingLayer(phrase_layer, input_size=self.trainphrase_vocab_size, output_size=200, W=self.trainevalphrase_embeddings)

		#Don't train embedding layer
		embedding_layer_phrase.params[embedding_layer_phrase.W].remove('trainable')

		#Reshape output
		phrase_emb_reshape_layer = lasagne.layers.ReshapeLayer(embedding_layer_phrase, (-1,200))
		return output_layer,phrase_emb_reshape_layer

