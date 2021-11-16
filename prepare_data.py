import numpy as np
import re


def make_sentences_list(text):
	# на вход список текстов, на выходе список троек (предложение, начало, конец)
	SENTENCE_REGEX = r"(?:(?:[иИ]м\.|(?:[А-Я]\.){1,2}|\.[ \n]*[а-я0-9]|[^.!? \n]+)[ \n]*)+(?:[.?!](?:\"|»)|[.?!]|$)"
	sentences = []
	cur_pos = 0
	while True:
		sentence = re.search(SENTENCE_REGEX, text)
		if sentence is None:
			break
		sentences.append((sentence.group(), cur_pos + sentence.start(), cur_pos + sentence.end()))
		text = text[sentence.end():]
		cur_pos += sentence.end()
	return sentences

def make_words_list(sentence):
	# на вход предложение, на выходе список троек (слово, начало, конец)
	WORD_REGEX = r'(?m)(?:(?:[А-ЯA-Z]\.){1,2}|[^$!"«»#$%&()*+,\-/:;<=>?@[\]^_`{|}~ \.\n]+(?=["»]?$)|[^$!"«»#$%&()*+,\-/:;<=>?@[\]^_`{|}~ \n]+(?!["»]?$)|»|«|")'
	words = []
	cur_pos = 0
	while True:
		word = re.search(WORD_REGEX, sentence)
		if word is None:
			break
		words.append((word.group(), cur_pos + word.start(), cur_pos + word.end()))
		sentence = sentence[word.end():]
		cur_pos += word.end()
	return words

def search_pos_in_labels(pos, marks):
	for mark in marks:
		if mark[0] == pos:
			return mark
	return None

def create_labels_list(word_list, marks):
	# marks - множество троек (начало, конец, имя)
	labels_list = []
	i = 0
	size = len(word_list)
	while i < size:
		result = search_pos_in_labels(word_list[i][1], marks)
		if result is not None and result[2] != "DATE":
			labels_list.append(f"B-{result[2]}")
			i += 1
			while i<size and word_list[i][1] < result[1]:
				labels_list.append(f"I-{result[2]}")
				i += 1
		else:
			labels_list.append("O")
			i += 1

	return labels_list


def create_full_word_list(sentences_list):
	word_list = []
	for sentence in sentences_list:
		sentence_words = make_words_list(sentence[0])
		buf = []
		for i in range(0, len(sentence_words)):
			buf.append((sentence_words[i][0], sentence[1] + sentence_words[i][1], sentence[1] + sentence_words[i][2]))

		buf.append((".", sentence[2] - 1, sentence[2]))
		word_list += buf
	return word_list


def create_words_labels_pairs_list(train_data):
	# список пар - список троек слов и список меток
	result = []
	for text, labels in train_data:
		sentences_list = make_sentences_list(text)
		word_list = create_full_word_list(sentences_list)
		for _, label in labels.items():
			labels_list = create_labels_list(word_list, label)
			result.append((word_list, labels_list))
	return result


def prepare_sequence(word_list, keyedvector_model):
	default = keyedvector_model.get_index("неизвестно")
	result = []
	for word, _, _ in word_list:
		if re.fullmatch(r"[ОАЗ]{3}", word):
			# print(f"{word} - Организация")
			result.append(keyedvector_model.get_index("Организация"))
		elif word == '"' or word == "'" or word == "«" or word == "»":
			# print(f"{word} - кавычки")
			result.append(keyedvector_model.get_index("''"))
		elif re.fullmatch(r"(?:\d+(?:\.\d+)?)|[XVI]+", word):
			# print(f"{word} - numrange")
			result.append(keyedvector_model.get_index("numrange"))
		elif re.fullmatch(r"[a-z]+", word):
			# print(f"{word} - английский")
			result.append(keyedvector_model.get_index("английский"))
		elif keyedvector_model.get_index(word, default) == default and re.fullmatch(r"[A-ZА-Я]", word[0]):
			# print(f"{word} - большая буква")
			result.append(keyedvector_model.get_index("заглавной"))
		# добавить еще для имён и фамилий
		else:
			# print(f"{word} - {keyedvector_model.get_index(word, default)}")
			result.append(keyedvector_model.get_index(word, default))

	return result
