import numpy as np


def create_word2vec(filename):
	with open(filename, "r", encoding="utf8") as file:
		lines = file.readlines()
		embedding = dict()
		for line in lines:
			parts = line.split()
			embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
		return embedding


def prepare_embeddings_matrix(word2vec_model, EMBEDDING_DIM, VOCAB_SIZE, word2idx):
	# vocab - список слов из текста
	num_words = VOCAB_SIZE
	embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
	for word, i in word2idx.items():
		if i < VOCAB_SIZE:
			try:
				embedding_vector = word2vec_model.get_vector[word]
				embedding_matrix[i] = embedding_vector
			except KeyError:
				# words not found in embedding index will be all zeros.
				pass

