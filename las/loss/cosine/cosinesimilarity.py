__author__ = 'Johnny'


from theano import tensor as T,function,printing
import theano



def norm(x):
	return T.sqrt(T.sum(x*x))

def dot(x,y):
	return T.sum(T.diagonal(T.dot(x, T.transpose(y))))


def define_loss(prediction, labels):
	#loss = (T.dot(prediction.T,labels) / (norm(prediction)[0]*norm(labels)[0])).mean()
	loss = 1 - dot(prediction, labels) / norm(prediction) / norm(labels)
	return loss
