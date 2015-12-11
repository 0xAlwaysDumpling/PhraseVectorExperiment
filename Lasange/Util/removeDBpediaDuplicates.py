__author__ = 'Johnny'






import sys
from collections import defaultdict


def default_count():
	return -1;



word_dict = defaultdict(default_count)


input_file = sys.argv[1]
out_file = sys.argv[2]

with open(out_file, "w") as w:

	with open(input_file, "r") as f:
		for line in f.readlines():
			v = line
			line = line.split(' ')
			word = line[0]
			if '_' in word:
				if word_dict[word] > -1:
					continue
				else:
					word_dict[word] = 1;
					w.write(v)


