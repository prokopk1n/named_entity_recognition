import re
from prepare_data import create_words_labels_pairs_list, prepare_sequence, make_sentences_list, \
	create_full_word_list, print_text_with_labels
import gensim
import time
from model import BiLSTM_CRF
import torch
import torch.optim as optim
import time
import date_regex
from preprocessing_json import preprocess_json


class Solution:

	def __init__(self):
		self.embedding_path = 'DeepPavlov/rubert-base-cased' # 'rubert_cased_L-12_H-768_A-12_pt/' '/embeddings/ruBert-base/'
		self.time = 1200
		self.epochs = 100
		self.tag_to_ix = {"B-ORGANIZATION": 0, "I-ORGANIZATION": 1, "B-PERSON": 2, "I-PERSON": 3, "O": 4}
		self.date_regex = re.compile(date_regex.DATE_REGEXP_FULL)

	def train(self, train_data):
		lstm_size = 6
		self.model = BiLSTM_CRF(self.tag_to_ix, lstm_size, self.embedding_path)

		words_labels_pairs_list = create_words_labels_pairs_list(train_data)

		"""
		sentences_list = []
		for text, _ in train_data:
			sentences_list.append(make_sentences_list(text))
		"""

		train_x = []
		train_y = []

		res = []
		for i in range(0, len(words_labels_pairs_list)):
			j = 0
			while j < len(words_labels_pairs_list[i][0]):
				buf = []
				while j < len(words_labels_pairs_list[i][0]) and words_labels_pairs_list[i][0][j][0] != ".":
					buf.append((words_labels_pairs_list[i][0][j], words_labels_pairs_list[i][1][j]))
					j+=1
				buf.append((words_labels_pairs_list[i][0][j], words_labels_pairs_list[i][1][j]))
				j+=1
				res.append(buf)    # список предложений, где предложение это список из элементов ((слово, начало, конец), метка))

		for sentence in res:
			labels = ["O"]
			string_sentence = " ".join([word[0][0] for word in sentence])
			string_sentence = "[CLS] " + string_sentence + " [SEP]"
			# print(string_sentence)
			# time.sleep(1)
			tokenized_text = self.model.tokenizer.tokenize(string_sentence)
			i = 1
			j = 0   # Для итерации по предложению
			buf = ""
			while i < len(tokenized_text) - 1:
				if sentence[j][1][0] == "B": # если слово с меткой B-... разиблось на несколько слов, то B-... ставится только самой первой части
					if buf == "":
						labels.append(sentence[j][1])
					else:
						labels.append("I-" + sentence[j][1].split("-")[1])
				else:
					labels.append(sentence[j][1])
				if tokenized_text[i][:2] == '##':
					buf += tokenized_text[i][2:]
				else:
					buf += tokenized_text[i]
				if len(buf) >= len(sentence[j][0][0]):
					buf = ""
					j += 1
				i += 1
			"""
			for tup in zip(tokenized_text, labels):
				print(f'{tup[0]}   {tup[1]}')
			input()
			"""
			labels.append("O")
			train_x.append(tokenized_text)
			train_y.append([self.tag_to_ix[label] for label in labels])
			# test(tokenized_text, labels, sentence)
			# print_text_with_labels(string_sentence, res_buf)
			# input()

		# train_xxx = [torch.tensor(x,  dtype=torch.long) for x in train_x]
		train_yyy = [torch.tensor(y, dtype=torch.long) for y in train_y]

		optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)

		# self.model.train()
		start = time.time()

		for j in range(0, self.epochs):
			print(f"EPOCH NUMBER {j} TIME = {time.time()} TOTAL STEPS = {len(train_x)}")
			self.model.train()
			for i in range(0, len(train_x)):
				if i % 100 == 0:
					print(f"STEP NUMBER {i} EPOCH NUMBER {j} TIME = {time.time()}")
				self.model.zero_grad()
				loss = self.model.neg_log_likelihood(train_x[i], train_yyy[i])
				loss.backward()
				optimizer.step()
				if time.time() - start > self.time:
					return

	def predict(self, texts):
		res = []
		with torch.no_grad():
			for text in texts:
				word_list = create_full_word_list(make_sentences_list(text))
				i = 0
				sentences = []
				while i < len(word_list):
					buf = []
					while i < len(word_list) and word_list[i][0] != ".":
						buf.append(word_list[i])
						i += 1
					buf.append(word_list[i])
					i += 1
					sentences.append(buf)

				result_set = set()
				for sentence in sentences:
					string_sentence = " ".join([word[0] for word in sentence])
					string_sentence = "[CLS] " + string_sentence + " [SEP]"
					tokenized_text = self.model.tokenizer.tokenize(string_sentence)
					predicted = tag_decoder(self.model(tokenized_text)[1], self.tag_to_ix)
					result_set.update(labels_decoder(predicted, sentence, tokenized_text))


				result = re.finditer(self.date_regex, text)
				for match in result:
					result_set.add((match.start(), match.end(), "DATE"))

				res.append(result_set)
		return res


def labels_decoder(labels_list, sentence, tokenized_text):
	result_set = set()
	k = 1    # берем метку от самой первой части слова, если слово разбивается на подслова через ##
	i = 1
	j = 0   # для итерации по предложению
	size = len(tokenized_text) - 1
	while i < size:
		buf = tokenized_text[i]
		i += 1
		while i < size and len(buf) < len(sentence[j][0]):
			if tokenized_text[i][:2] == '##':
				buf += tokenized_text[i][2:]
			else:
				buf += tokenized_text[i]
			i += 1
		# print(f"BUF = {buf} sentence = {sentence[j][0]}")
		label_cur = labels_list[k]
		k = i

		if label_cur == "B-ORGANIZATION" or label_cur == "B-PERSON":
			tag = label_cur.split("-")[1]
			start = sentence[j][1]
			end = sentence[j][2]
			j += 1
			while i < size and labels_list[i] == f"I-{tag}":
				buf = tokenized_text[i]
				i += 1
				while i < size and len(buf) < len(sentence[j][0]):
					if tokenized_text[i][:2] == '##':
						buf += tokenized_text[i][2:]
					else:
						buf += tokenized_text[i]
					i += 1
				end = sentence[j][2]
				j += 1
				k = i

			result_set.add((start, end, f"{tag}"))
		else:
			j += 1

	return result_set


def tag_decoder(tags, tag_to_ix):
	result = []
	for tag in tags:
		for key, value in tag_to_ix.items():
			if value == tag:
				result.append(key)
				break
	return result


def test(train_x, train_y, sentence):
	print(train_x)
	print(train_y)
	print([word[0] for word in sentence])
	result = labels_decoder(train_y, [word[0] for word in sentence], train_x)
	print(result)
	return result


if __name__ == "__main__":
	# print(labels_decoder(["O", "B-ORGANIZATION", "I-ORGANIZATION", "I-ORGANIZATION", "O", "O"],
	                     # [("asdas", 0, 5), ("asdasdas", 6, 13), ("dasdasads", 14, 23)],
	                     # ["[CLS]", "as", "##das", "asdasdas", "dasdasads", "[SEP]"]))
	# print_text_with_labels(train_data[0][0], train_data[0][1]["DAoLAoDRnjHrmkGYw"])


	train_data = preprocess_json("tpc-dataset.train.json")
	words_labels_pairs_list = create_words_labels_pairs_list(train_data)
	solution = Solution()
	solution.train(train_data)
	labels_list = solution.predict([train_data[0][0], train_data[1][0]])
	print_text_with_labels(train_data[0][0], labels_list[0])
	print_text_with_labels(train_data[1][0], labels_list[1])
	while True:
		string = input("ВВОДИ:")
		solution.predict([string])
