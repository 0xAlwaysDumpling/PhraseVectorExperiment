__author__ = 'Johnny'




import sys



input_file = sys.argv[1]
skip = sys.argv[2]
delimit = sys.argv[3] #if 0 then space if 1 then tab
out_file = 'skip.' + str(skip) +'.vocab.txt'

if int(delimit) == 0:
    delimiter = ' '
else:
    delimiter = '\t'


vocab_size = 0
lineCount = 0

with open(out_file, "w") as w:
    with open(input_file, "r") as f:
        for line in f.readlines():
            if lineCount == int(skip):
                line = line.split(delimiter)
                word = line[0]
                words = word.split('_')
                lineCount = 0
                w.write(word+'\n')
            lineCount+=1
