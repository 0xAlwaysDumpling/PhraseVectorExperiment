__author__ = 'Johnny'

import sys,os,inspect
ff_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../model/feedforward/")))
cnn_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../model/cnn/")))
if ff_subfolder not in sys.path:
	 sys.path.insert(0, ff_subfolder)
if cnn_subfolder not in sys.path:
	 sys.path.insert(0, cnn_subfolder)
import singlelayer as sln
import multilayer as mln
import cnn as cn


def main():
	name = sys.argv[1]
	batch_size = sys.argv[2]
	end_epoch = sys.argv[3]
	hidden_units = sys.argv[4]
	hidden_layers = sys.argv[5]
	Sampling = sys.argv[6]
	X_train_path = sys.argv[7]
	Y_train_path = sys.argv[8]
	eval_path = sys.argv[9]
	if str(name) == 'sln':
		feed_forward = sln.ff(name,batch_size,hidden_units,1,end_epoch)
		feed_forward.run(Sampling,X_train_path,Y_train_path, eval_path)
	elif name == 'cnn':
		cnn_net = cn.cnn(name,batch_size,hidden_units,1,end_epoch)
		cnn_net.run(Sampling,X_train_path,Y_train_path, eval_path)
	elif name == 'mln':
		feed_forward = mln.ff(name,batch_size,hidden_units,hidden_layers,end_epoch)
		feed_forward.run(Sampling,X_train_path,Y_train_path, eval_path)




if __name__ == '__main__':
	main()
