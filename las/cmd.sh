#!/usr/bin/env bash

python pickleInputVectors.py $TRNSIZE $EVALSIZE $PHRASESIZE $TRAINFILE $EVALFILE

python pickleTrainEvalInput.py 19204 215 12 /Users/Johnny/Downloads/dbpedia.input.vectors.txt /Users/Johnny/Downloads/word.sequence.vectors.txt

python pickleGoldTrain.py 19204 /Users/Johnny/Downloads/new.skip.15.txt



#Single layer network
THEANO_FLAGS=floatX=float32  python run.py  sln 1 100 200 False ../dat/train/19204..input.pickle ../dat/train/19204..gold.pickle ../dat/train/215.eval.inputs.pickle


#Multilayer network
THEANO_FLAGS=floatX=float32  python run.py  mln 1 100 200 3 False ../dat/train/19204..input.pickle ../dat/train/19204..gold.pickle ../dat/train/215.eval.inputs.pickle

