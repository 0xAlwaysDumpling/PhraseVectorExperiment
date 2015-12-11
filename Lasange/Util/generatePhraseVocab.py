__author__ = 'Johnny'




import sys



input_file = sys.argv[1]
skip = sys.argv[2]
out_file = 'skip.' + str(skip) +'.vocab.txt'

vocab_size = 0
lineCount = 0

with open(out_file, "w") as w:
    with open(input_file, "r") as f:
        for line in f.readlines():
            if lineCount -- int(skip):
                line = line.split(' ')
                word = line[0]
                words = word.split('_')
                lineCount = 0
                w.write(word+'\n')
            lineCount+=1
