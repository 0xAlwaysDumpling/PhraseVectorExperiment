#!/usr/bin/env bash

THEANO_FLAGS=floatX=float32,optimizer=None python ff.py  ff 1 1 False ../dat/preprocessed.inputs.pickle ../dat/preprocessed.phrase.output.pickle

THEANO_FLAGS=floatX=float32,optimizer=None python ff.py  ff 1 1 False ../dat/44999.train.inputs.pickle ../dat/44999.14999.gold.pickle ../dat/14999.eval.inputs.pickle


THEANO_FLAGS=floatX=float32  python ff.py  ff 1 100 False ../dat/44999.train.inputs.pickle ../dat/44999.14999.gold.pickle ../dat/14999.eval.inputs.pickle

python createInput.py 45000 15000 12 ../dat/60000.input.txt

python createGold.py 45000 15000 ../dat/60000.phrase.txt



python pickleInputVectors.py $TRNSIZE $EVALSIZE $PHRASESIZE $TRAINFILE $EVALFILE

python pickleTrainEvalInput.py 19204 215 12 /Users/Johnny/Downloads/dbpedia.input.vectors.txt /Users/Johnny/Downloads/word.sequence.vectors.txt

python pickleGoldTrain.py 19204 /Users/Johnny/Downloads/new.skip.15.txt



THEANO_FLAGS=floatX=float32  python ff.py  ff 1 5 False ../dat/19204..input.pickle ../dat/19204..gold.pickle ../dat/215.eval.inputs.pickle



THEANO_FLAGS=floatX=float32  python ff.py  ff 1 200 False ../dat/19204..input.pickle ../dat/19204..gold.pickle ../dat/215.eval.inputs.pickle

THEANO_FLAGS=floatX=float32  python ff.py  ff 1 100 False ../dat/19204..input.pickle ../dat/19204..gold.pickle ../dat/215.eval.inputs.pickle



THEANO_FLAGS=floatX=float32  python run.py  ff 1 100 200 False ../dat/train/19204..input.pickle ../dat/train/19204..gold.pickle ../dat/train/215.eval.inputs.pickle
