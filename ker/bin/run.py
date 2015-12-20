__author__ = 'Johnny'

import sys,os,inspect
base_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../model/base/")))
cnn_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../model/cnn/")))
if base_subfolder not in sys.path:
	 sys.path.insert(0, base_subfolder)
if cnn_subfolder not in sys.path:
	 sys.path.insert(0, cnn_subfolder)


def main():
    pass







if __name__ == '__main__':
	main()
