import re
from prepare_data import create_words_labels_pairs_list, prepare_sequence, make_sentences_list, create_full_word_list
import gensim
from model import BiLSTM_CRF
import torch
import torch.optim as optim
import time
import date_regex
from preprocessing_json import preprocess_json


class Solution:

	def __init__(self):
		self.embedding_path = "/embeddings/w2v_size100_window5.txt"
		self.keyedvector_model = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_path)
		self.time = 750
		self.epochs = 10
		self.tag_to_ix = {"B-ORGANIZATION": 0, "I-ORGANIZATION": 1, "B-PERSON": 2, "I-PERSON": 3, "O": 4}
		self.date_regex = re.compile(date_regex.DATE_REGEXP_FULL)

	def train(self, train_data):
		words_labels_pairs_list = create_words_labels_pairs_list(train_data)

		train_x = []
		train_y = []
		for i in range(0, len(words_labels_pairs_list)):
			train_x.append(prepare_sequence(words_labels_pairs_list[i][0], self.keyedvector_model))
			train_y.append([self.tag_to_ix[x] for x in words_labels_pairs_list[i][1]])

		lstm_size = 4
		self.model = BiLSTM_CRF(self.tag_to_ix, lstm_size, self.embedding_path)

		# train_xxx = [torch.tensor(x,  dtype=torch.long) for x in train_x]
		train_yyy = [torch.tensor(y, dtype=torch.long) for y in train_y]

		optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)

		self.model.train()
		start = time.time()

		for j in range(0, self.epochs):
			for i in range(0, len(train_x)):
				# print(f"STEP NUMBER {i} EPOCH NUMBER {j}")
				# print(y_pred[1])
				self.model.zero_grad()
				loss = self.model.neg_log_likelihood(train_x[i], train_yyy[i])
				loss.backward()
				optimizer.step()
				if time.time() - start > self.time:
					return

	def predict(self, texts):
		res = []
		for text in texts:
			word_list = create_full_word_list(make_sentences_list(text))
			vector_of_text = prepare_sequence(word_list, self.keyedvector_model)
			predicted = tag_decoder(self.model(vector_of_text)[1], self.tag_to_ix)
			result_set = labels_decoder(predicted, word_list)

			result = re.finditer(self.date_regex, text)
			for match in result:
				result_set.add((match.start(), match.end(), "DATE"))

			res.append(result_set)

		return res


def labels_decoder(labels_list, word_list):
	result_set = set()
	i = 0
	size = len(labels_list)
	while i < size:
		if labels_list[i] == "B-ORGANIZATION" or labels_list[i] == "B-PERSON":
			tag = labels_list[i].split("-")[1]
			start = word_list[i][1]
			end = word_list[i][2]
			i += 1
			while i < size and labels_list[i] == f"I-{tag}":
				end = word_list[i][2]
				i += 1
			result_set.add((start, end, f"{tag}"))
		else:
			i += 1

	return result_set


def tag_decoder(tags, tag_to_ix):
	result = []
	for tag in tags:
		for key, value in tag_to_ix.items():
			if value == tag:
				result.append(key)
				break
	return result


def test(train_data):
	words_labels_pairs_list = create_words_labels_pairs_list(train_data)
	result = labels_decoder(words_labels_pairs_list[0][1], words_labels_pairs_list[0][0])
	print(result)


if __name__ == "__main__":
	train_data = preprocess_json("tpc-dataset.train_3.json")
	words_labels_pairs_list = create_words_labels_pairs_list(train_data)
	solution = Solution()
	solution.train(train_data)
	solution.predict([train_data[0][0], train_data[1][0]])
	while True:
		string = input("ВВОДИ:")
		solution.predict([string])
