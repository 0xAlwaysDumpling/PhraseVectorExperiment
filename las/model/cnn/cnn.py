__author__ = 'Johnny'
import lasagne
import sys,os,inspect
base_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../base/")))
if base_subfolder not in sys.path:
	 sys.path.insert(0, base_subfolder)

import base as base




class cnn(base.Base):
	def define_layers(self, input_indices, phrase_indices,N_HU = 200, N_H = 2):
		print('Defining layers')
		MAX_LENGTH=12

		input_layer = lasagne.layers.InputLayer(shape=(None,MAX_LENGTH), input_var=input_indices)
		embedding_layer = lasagne.layers.EmbeddingLayer(input_layer, input_size=self.vocab_size, output_size=200,W=self.all_embedding)
		embedding_layer.params[embedding_layer.W].remove('trainable')


		#Incoming layer, Number of filters, Filter Size
		#Pad = 0 , Same = pads with half the filter size (rounded down) on both sides. When stride=1 this results in an output size equal to the input size. Even filter size is not supported.
		conv_layer = lasagne.layers.Conv1DLayer(embedding_layer,MAX_LENGTH,3,pad='same')


		pooling_layer = lasagne.layers.MaxPool1DLayer(conv_layer,1)

		output_layer = lasagne.layers.DenseLayer(pooling_layer,int(N_HU),nonlinearity=lasagne.nonlinearities.sigmoid)
		#define phrase output embeddings
		phrase_layer = lasagne.layers.InputLayer(shape=(1,1), input_var=phrase_indices)
		embedding_layer_phrase = lasagne.layers.EmbeddingLayer(phrase_layer, input_size=self.trainphrase_vocab_size, output_size=200, W=self.trainevalphrase_embeddings)

		#Don't train embedding layer
		embedding_layer_phrase.params[embedding_layer_phrase.W].remove('trainable')

		#Reshape output
		phrase_emb_reshape_layer = lasagne.layers.ReshapeLayer(embedding_layer_phrase, (-1,200))
		return output_layer,phrase_emb_reshape_layer
