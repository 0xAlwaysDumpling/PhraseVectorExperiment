__author__ = 'Johnny'




import sys



input_file = sys.argv[1]
skip = sys.argv[2] #if 0 don't skip
delimit = sys.argv[3] #if 0 then space if >0 then tab
vocab_file = 'skip.' + str(skip) +'.vocab.txt'
vector_file =  'skip.' + str(skip) +'.vector.txt'
if int(delimit) == 0:
    delimiter = ' '
else:
    delimiter = '\t'



lineCount = 0

v = open(vector_file, "w")

with open(vocab_file, "w") as w:
    with open(input_file, "r") as f:
        for line in f.readlines():
            if lineCount >= int(skip):
                tokens = line.split(delimiter)
                words = tokens[0]
                lineCount = 0
                w.write(words+'\n')
                if '\n' in line: v.write(line)
                else: v.write(line+'\n')
            lineCount+=1


v.close()