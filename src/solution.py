from src.prepare_data import create_words_labels_pairs_list, make_sentences_list, \
	create_full_word_list, print_text_with_labels
from src.model import BiLSTM_CRF
import torch
import torch.optim as optim
import time
from transformers import BertModel, BertTokenizer
from typing import List, Set, Tuple, Dict
from src.decode import labels_decoder, tag_decoder
from src.prepare_data import create_sentences_list_of_word, create_train_x_y


class Solution:
	"""Класс для обучения модели и последующего предсказания"""

	def __init__(self, model_path=None, bert_embedding_path='sberbank-ai/ruBert-base'):
		self.embedding_path = bert_embedding_path
		self.tag_to_ix = {"B-ORGANIZATION": 0, "I-ORGANIZATION": 1, "B-PERSON": 2, "I-PERSON": 3, "B-DATE": 4,
		                  "I-DATE": 5, "O": 6}
		self.tokenizer = BertTokenizer.from_pretrained(self.embedding_path, output_hidden_states=True,
		                                               do_lower_case=False)
		self.bert_model = BertModel.from_pretrained(self.embedding_path, output_hidden_states=True)
		self.model = BiLSTM_CRF(self.tag_to_ix, lstm_size=6, embedding_path=self.embedding_path)
		if model_path:
			self.model.load_state_dict(torch.load(model_path))

	def _get_embedded(self, sentence):
		"""Получение ембеддинга предложения"""
		tokenized_text = sentence
		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
		segments_ids = [1] * len(tokenized_text)
		tokens_tensor = torch.tensor([indexed_tokens])
		segments_tensors = torch.tensor([segments_ids])
		self.bert_model.eval()
		with torch.no_grad():
			outputs = self.bert_model(tokens_tensor, segments_tensors)
			return outputs[0]

	def create_train_data(self, train_data):
		"""Формируем данные для обучения"""
		words_labels_pairs_list = create_words_labels_pairs_list(train_data)

		# список предложений всех текстов, где предложение это список из элементов ((слово, начало, конец), метка))
		sentences = create_sentences_list_of_word(words_labels_pairs_list)

		train_x, train_y = create_train_x_y(sentences, self.tokenizer, self.tag_to_ix)
		train_xxx = [(self._get_embedded(train), len(train)) for train in train_x]
		# преобразуем в tensor, так как модель возвращает tensor
		train_yyy = [torch.tensor(y, dtype=torch.long) for y in train_y]
		return train_xxx, train_yyy

	def train(self, train_data: List[Tuple[str, Dict[str, Set[Tuple[int, int,str]]]]],
	          train_time=30 * 60, epochs=7, debug=False):
		start_train = time.time()

		train_xxx, train_yyy = self.create_train_data(train_data)
		optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)

		for j in range(0, epochs):
			start_epoch = time.time()
			if debug:
				print(f"EPOCH NUMBER {j} TIME START = {time.ctime(start_epoch)} TOTAL_STEPS = {len(train_xxx)}")
			self.model.train()
			for i in range(0, len(train_xxx)):
				self.model.zero_grad()
				loss = self.model.neg_log_likelihood(train_xxx[i], train_yyy[i])
				loss.backward()
				optimizer.step()
				if time.time() - start_train > train_time:
					return
			if debug:
				print(f"OUT OF EPOCH {j} TOTAL TIME OF EPOCH = {time.time() - start_epoch}")

	def predict(self, texts: List[str], debug = False) -> List[Set[Tuple[int, int, str]]]:
		"""Get list of texts and return list of sets of entities in each text"""
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

				if debug:
					for sentence in sentences:
						print(sentence)

				result_set = set()
				for sentence in sentences:
					string_sentence = " ".join([word[0] for word in sentence])
					string_sentence = "[CLS] " + string_sentence + " [SEP]"
					tokenized_text = self.tokenizer.tokenize(string_sentence)
					embeds = (self._get_embedded(tokenized_text), len(tokenized_text))
					predicted = tag_decoder(self.model(embeds)[1], self.tag_to_ix)
					result_set.update(labels_decoder(predicted, sentence, tokenized_text))

				res.append(result_set)
		return res