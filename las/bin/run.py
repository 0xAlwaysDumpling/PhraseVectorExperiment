__author__ = 'Johnny'

import sys,os,inspect
ff_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../model/feedforward/")))
if ff_subfolder not in sys.path:
     sys.path.insert(0, ff_subfolder)
import ff as f


def main():
    name = sys.argv[1]
    batch_size = sys.argv[2]
    end_epoch = sys.argv[3]
    hidden = sys.argv[4]
    Sampling = sys.argv[5]
    X_train_path = sys.argv[6]
    Y_train_path = sys.argv[7]
    eval_path = sys.argv[8]
    if str(name) == 'ff':
        feed_forward = f.ff(name,batch_size,hidden,end_epoch)
        feed_forward.run(Sampling,X_train_path,Y_train_path, eval_path)
        print('hello')
    elif name == 'cnn':
        pass



if __name__ == '__main__':
    main()
