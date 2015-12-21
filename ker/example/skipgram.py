__author__ = 'Johnny'



import numpy as np
import theano
import cPickle

import os, re, json

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Merge


vocab_size = 50000  # vocabulary size: top 50,000 most common words in data
skip_top = 100  # ignore top 100 most common words
nb_epoch = 1
vector_dim = 200  # embedding space dimension


save = True
load_model = False
load_tokenizer = False
train_model = True
save_dir = os.path.expanduser("./out/")
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
model_load_fname = "HN_skipgram_model.pkl"
model_save_fname = "HN_skipgram_model.pkl"
tokenizer_fname = "HN_tokenizer.pkl"

data_path = os.path.expanduser("./dat/")+"HNCommentsAll.1perline.json"


# text preprocessing utils
html_tags = re.compile(r'<.*?>')
to_replace = [('&#x27;', "'")]
hex_tags = re.compile(r'&.*?;')


def clean_comment(comment):
	c = str(comment.encode("utf-8"))
	c = html_tags.sub(' ', c)
	for tag, char in to_replace:
		c = c.replace(tag, char)
	c = hex_tags.sub(' ', c)
	return c


def text_generator(path=data_path):
	f = open(path)
	for i, l in enumerate(f):
		comment_data = json.loads(l)
		comment_text = comment_data["comment_text"]
		comment_text = clean_comment(comment_text)
		if i % 10000 == 0:
			print(i)
		yield comment_text
	f.close()


# model management
if load_tokenizer:
	print('Load tokenizer...')
	tokenizer = cPickle.load(open(os.path.join(save_dir, tokenizer_fname), 'rb'))
else:
	print("Fit tokenizer...")
	tokenizer = text.Tokenizer(nb_words=vocab_size)
	tokenizer.fit_on_texts(text_generator())
	if save:
		print("Save tokenizer...")
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		cPickle.dump(tokenizer, open(os.path.join(save_dir, tokenizer_fname), "wb"))

if train_model:
	if load_model:
		print('Load model...')
		model = cPickle.load(open(os.path.join(save_dir, model_load_fname), 'rb'))
	else:
		print('Build model...')
		word = Sequential()
		word.add(Embedding(vocab_size,vector_dim, init='uniform'))
		context = Sequential()
		context.add(Embedding(vocab_size,vector_dim, init='uniform'))
		model = Sequential()
		model.add(Merge([word, context], mode='dot'))
		model.compile(loss='mse', optimizer='rmsprop')

	sampling_table = sequence.make_sampling_table(vocab_size)


	for e in range(nb_epoch):
		print('-'*40)
		print('Epoch',e)
		print('-'*40)

		progbar = generic_utils.Progbar(tokenizer.document_count)
		samples_seen = 0
		losses = []


		for i, seq in enumerate(tokenizer.texts_to_sequences_generator(text_generator())):
			# get skipgram couples for one text in the dataset
			couples, labels = sequence.skipgrams(seq, vocab_size, window_size=4, negative_samples=1., sampling_table=sampling_table)
			if couples:
				X1,X2 = zip(*couples)
				X1 = np.array(X1,dtype="int32")
				X2 = np.array(X2,dtype="int32")
				loss = model.train_on_batch([X1,X2], labels)
				losses.append(loss)
				if len(losses) % 100 == 0:
					progbar.update(i, values=[("loss", np.mean(losses))])
					losses = []
				samples_seen += len(labels)
		print('Samples seen:', samples_seen)
	print("Training completed!")


	if save:
		print("Saving model...")
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		cPickle.dump(model, open(os.path.join(save_dir, model_save_fname), "wb"))

print("It's test time!")

# recover the embedding weights trained with skipgram:
weights = model.layers[0].get_weights()[0]

# we no longer need this
del model

weights[:skip_top] = np.zeros((skip_top, vector_dim))
norm_weights = np_utils.normalize(weights)

word_index = tokenizer.word_index
reverse_word_index = dict([(v, k) for k, v in list(word_index.items())])



def embed_word(w):
	i = word_index.get(w)
	if (not i) or (i < skip_top) or (i >= vocab_size):
		return None
	return norm_weights[i]


def closest_to_point(point, nb_closest=10):
	proximities = np.dot(norm_weights, point)
	tups = list(zip(list(range(len(proximities))), proximities))
	tups.sort(key=lambda x: x[1], reverse=True)
	return [(reverse_word_index.get(t[0]), t[1]) for t in tups[:nb_closest]]


def closest_to_word(w, nb_closest=10):
	i = word_index.get(w)
	if (not i) or (i < skip_top) or (i >= vocab_size):
		return []
	return closest_to_point(norm_weights[i].T, nb_closest)


''' the resuls in comments below were for:
	5.8M HN comments
	dim_proj = 256
	nb_epoch = 2
	optimizer = rmsprop
	loss = mse
	max_features = 50000
	skip_top = 100
	negative_samples = 1.
	window_size = 4
	and frequency subsampling of factor 10e-5.
'''

words = [
	"article",  # post, story, hn, read, comments
	"3",  # 6, 4, 5, 2
	"two",  # three, few, several, each
	"great",  # love, nice, working, looking
	"data",  # information, memory, database
	"money",  # company, pay, customers, spend
	"years",  # ago, year, months, hours, week, days
	"android",  # ios, release, os, mobile, beta
	"javascript",  # js, css, compiler, library, jquery, ruby
	"look",  # looks, looking
	"business",  # industry, professional, customers
	"company",  # companies, startup, founders, startups
	"after",  # before, once, until
	"own",  # personal, our, having
	"us",  # united, country, american, tech, diversity, usa, china, sv
	"using",  # javascript, js, tools (lol)
	"here",  # hn, post, comments
]

for w in words:
	res = closest_to_word(w)
	print('====', w)
	for r in res:
		print(r)