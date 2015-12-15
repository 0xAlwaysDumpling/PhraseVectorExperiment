__author__ = 'Johnny'

import sys,os,inspect
import cPickle as pickle
import numpy as np
from theano import tensor as T,function,printing
import lasagne
import time
import theano
from collections import defaultdict

cos_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../../loss/cosine")))
if cos_subfolder not in sys.path:
	sys.path.insert(0,cos_subfolder)

import cosinesimilarity as l


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
    def run(self,Sampling, X_train_path, Y_train_path, eval_path):
        self.load_dataset(Sampling,X_train_path, Y_train_path, eval_path)
        self.build_and_train_model(self.N_Hidden)
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



