import re
from prepare_data import create_words_labels_pairs_list, prepare_sequence, make_sentences_list, \
	create_full_word_list, print_text_with_labels
import matplotlib.pyplot as plt
from model import BiLSTM_CRF
import torch
import torch.optim as optim
import time
import date_regex
from preprocessing_json import preprocess_json
from nerc_quality import QualityNERC
import matplotlib.ticker as ticker
from transformers import BertModel, BertTokenizer


# список предложений всех текстов, где предложение это список из элементов ((слово, начало, конец), метка))
def create_sentences_list_of_word(words_labels_pairs_list):
	res = []
	for i in range(0, len(words_labels_pairs_list)):
		j = 0
		while j < len(words_labels_pairs_list[i][0]):
			buf = []
			while j < len(words_labels_pairs_list[i][0]) and words_labels_pairs_list[i][0][j][0] != ".":
				buf.append((words_labels_pairs_list[i][0][j], words_labels_pairs_list[i][1][j]))
				j += 1
			buf.append((words_labels_pairs_list[i][0][j], words_labels_pairs_list[i][1][j]))
			j += 1
			res.append(buf)
	return res

# формируем обучащие данные
def create_train_x_y(sentences, tokenizer, tag_to_ix):
	train_x = []
	train_y = []
	for sentence in sentences:
		labels = ["O"]
		string_sentence = " ".join([word[0][0] for word in sentence])
		string_sentence = "[CLS] " + string_sentence + " [SEP]"
		# print(string_sentence)
		# time.sleep(1)
		tokenized_text = tokenizer.tokenize(string_sentence)
		if len(sentence) == 0:
			continue
		i = 1
		j = 0  # Для итерации по предложению
		buf = ""
		while i < len(tokenized_text) - 1:
			if sentence[j][1][0] == "B":  # если слово с меткой B-... разиблось на несколько слов, то B-... ставится только самой первой части
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
		if j != len(sentence):
			continue
		labels.append("O")
		train_x.append(tokenized_text)
		train_y.append([tag_to_ix[label] for label in labels])
		# test(tokenized_text, labels, sentence)
		# print_text_with_labels(string_sentence, res_buf)
		# input()

	return train_x, train_y

class Solution:

	def __init__(self):
		self.embedding_path = '/embeddings/rubert-base-cased/'   #'DeepPavlov/rubert-base-cased' # '/embeddings/rubert-base-cased/' # 'rubert_cased_L-12_H-768_A-12_pt/' '/embeddings/ruBert-base/'
		self.time = 28 * 60
		self.epochs = 3
		self.tag_to_ix = {"B-ORGANIZATION": 0, "I-ORGANIZATION": 1, "B-PERSON": 2, "I-PERSON": 3, "O": 4}
		self.date_regex = re.compile(date_regex.DATE_REGEXP_FULL)
		self.tokenizer = BertTokenizer.from_pretrained(self.embedding_path, output_hidden_states=True)
		self.bert_model = BertModel.from_pretrained(self.embedding_path, output_hidden_states=True)
		lstm_size = 6
		self.model = BiLSTM_CRF(self.tag_to_ix, lstm_size, self.embedding_path)

	def _get_embedded(self, sentence):
		tokenized_text = sentence
		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
		segments_ids = [1] * len(tokenized_text)
		tokens_tensor = torch.tensor([indexed_tokens])
		segments_tensors = torch.tensor([segments_ids])
		self.bert_model.eval()
		with torch.no_grad():
			outputs = self.bert_model(tokens_tensor, segments_tensors)
			return outputs[0]

	def train(self, train_data):
		start_train = time.time()
		# print(f"START = {start_train}")
		words_labels_pairs_list = create_words_labels_pairs_list(train_data)

		# список предложений всех текстов, где предложение это список из элементов ((слово, начало, конец), метка))
		sentences = create_sentences_list_of_word(words_labels_pairs_list)

		train_x, train_y = create_train_x_y(sentences, self.tokenizer, self.tag_to_ix)
		train_xxx = [(self._get_embedded(train), len(train)) for train in train_x]
		# преобразуем в tensor, так как модель возвращает tensor
		train_yyy = [torch.tensor(y, dtype=torch.long) for y in train_y]

		optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)

		# self.model.train()

		for j in range(0, self.epochs):
			start_epoch = time.time()
			# print(f"EPOCH NUMBER {j} TIME = {time.time() - start_train} TOTAL STEPS = {len(train_xxx)}")
			self.model.train()
			for i in range(0, len(train_xxx)):
				self.model.zero_grad()
				loss = self.model.neg_log_likelihood(train_xxx[i], train_yyy[i])
				loss.backward()
				optimizer.step()
				if time.time() - start_train > self.time:
					return
			# print(f"OUT OF EPOCH {j} TOTAL TIME = {time.time() - start_epoch}")

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
					tokenized_text = self.tokenizer.tokenize(string_sentence)
					embeds = (self._get_embedded(tokenized_text), len(tokenized_text))
					predicted = tag_decoder(self.model(embeds)[1], self.tag_to_ix)
					result_set.update(labels_decoder(predicted, sentence, tokenized_text))

				result = re.finditer(self.date_regex, text)
				for match in result:
					result_set.add((match.start(), match.end(), "DATE"))

				res.append(result_set)
		return res

	def test(self, train_data):
		test_time = 8 * 60 * 60
		nerc_quality = QualityNERC()
		test = train_data[:len(train_data)//10]
		test_texts = [text for text, _ in test]
		expected = [create_labels_set(labels) for _, labels in test]
		train = train_data[len(train_data)//10:]
		org_plot, person_plot, f_plot, x_plot = [], [], [], []
		start = time.time()
		for j in range(0, self.epochs):
			start_epoch = time.time()
			if start_epoch - start > test_time:
				break
			print(f"EPOCH NUMBER {j} TIME START = {start_epoch - start}")
			self.train(train)
			print(f"OUT OF TRAIN TIME = {time.time() - start_epoch}")
			predicted = self.predict(test_texts)
			f, f_dict = nerc_quality.evaluate(predicted, expected)
			org_plot.append(f_dict["ORGANIZATION"])
			person_plot.append(f_dict["PERSON"])
			f_plot.append(f)
			x_plot.append(j + 1)
			print(f"OUT OF EPOCH {j} TOTAL TIME OF EPOCH = {time.time() - start_epoch}")
			torch.save(self.model, f"model{j}.out")

		print_plot(x_plot, f_plot, person_plot, org_plot, "test.png")
		torch.save(self.model, "model.out")

# вывод графика f-меры по эпохам
def print_plot(x_plot, f_plot, person_plot, org_plot, filename):
	print(f"ORGANIZATION = {org_plot}")
	print(f"PERSON = {person_plot}")
	print(f"FULL = {f_plot}")
	fig, axs = plt.subplots(3)
	fig.suptitle('F-MEASURE')

	axs[0].set_title("ORGANIZATION")
	axs[1].set_title("PERSON")
	axs[2].set_title("F-FULL")

	axs[0].axis([0, x_plot[-1] + 1, -0.2, 1])
	axs[1].axis([0, x_plot[-1] + 1, -0.2, 1])
	axs[2].axis([0, x_plot[-1] + 1, -0.2, 1])

	axs[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
	axs[1].xaxis.set_major_locator(ticker.MultipleLocator(1))
	axs[2].xaxis.set_major_locator(ticker.MultipleLocator(1))

	axs[0].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
	axs[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
	axs[2].yaxis.set_major_locator(ticker.MultipleLocator(0.2))

	axs[0].plot(x_plot, org_plot)
	axs[1].plot(x_plot, person_plot)
	axs[2].plot(x_plot, f_plot)

	fig1 = plt.gcf()
	fig1.savefig(filename, dpi=100)
	plt.show()

def create_labels_set(labels):
	labels_list = [label for _, label in labels.items()]
	if len(labels_list) == 1:
		label = labels_list[0]
	elif len(labels_list) == 2:
		label = labels_list[0] & labels_list[1]
	else:
		label = labels_list[0] & labels_list[1] | labels_list[1] & labels_list[2] | labels_list[0] & labels_list[2]
	return label


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


if __name__ == "__main__":
	train_data = preprocess_json("tpc-dataset.train.json")
	solution = Solution()
	solution.train(train_data)
	# torch.save(solution.model, "model6.out")
	# solution.model = torch.load("model.out")
	# solution.test(train_data)

	labels_list = solution.predict([train_data[0][0], train_data[1][0]])
	print_text_with_labels(train_data[0][0], labels_list[0])
	print_text_with_labels(train_data[1][0], labels_list[1])
	"""
	lstm_size = 6
	model = BiLSTM_CRF({"B-ORGANIZATION": 0, "I-ORGANIZATION": 1, "B-PERSON": 2, "I-PERSON": 3, "O": 4}, 6,
	    'DeepPavlov/rubert-base-cased')
	model.load_state_dict(torch.load("model.bin"))
	model.eval()

	# solution.train(train_data)
	labels_list = solution.predict([train_data[0][0], train_data[1][0]])
	nerc_quality = QualityNERC()
	f, f_dict = nerc_quality.evaluate(labels_list, [train_data[0][1]["DAoLAoDRnjHrmkGYw"], train_data[1][1]["DAoLAoDRnjHrmkGYw"]])
	plt.title("DATE")
	plt.xlabel("EPOCHS")
	plt.ylabel("F1")
	plt.bar(["1", "2"], [f_dict['DATE'], f_dict['DATE']], width=0.1)
	plt.show()
	print(f"F = {f} ORG = {f_dict['ORGANIZATION']} DATE = {f_dict['DATE']} PERSON = {f_dict['PERSON']}")
	print_text_with_labels(train_data[0][0], labels_list[0])
	print_text_with_labels(train_data[1][0], labels_list[1])
	while True:
		string = input("ВВОДИ:")
		solution.predict([string])
	"""