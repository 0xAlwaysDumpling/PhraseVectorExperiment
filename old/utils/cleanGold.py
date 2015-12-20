__author__ = 'Johnny'





import sys
from collections import defaultdict


def default_word():
	return -1

wordmap  = defaultdict(default_word)

index = 1
with open(sys.argv[1], 'r') as f:
	for line in f.readlines():
		if line == '\n' or line == None:
			continue
		else:
			line = line[:-1]
			wordmap[line] = 1



with open('new.skip.15.txt', 'w') as f:
	with open(sys.argv[2],'r') as r:
		for line in r.readlines():
			l = line.split(' ')
			word = l[0]
			if wordmap[word] > 0:
				f.write(line)




