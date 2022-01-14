import re


def make_sentences_list(text):
	# на вход список текстов, на выходе список троек (предложение, начало, конец)
	SENTENCE_REGEX = r"(?:(?:[иИ]м\.|(?:[А-Я]\.){1,2}|\.[ \n]*[а-я]|[^.!? \n]+)[ \n]*)+(?:[.?!](?:\"|»)|[.?!]|$)"
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
	WORD_REGEX = r'(?m)(?:(?:[А-ЯA-Z]\.){1,2}|[‑-‑№]|[^$!‑-‑№"«»#$%&()*+,\-/:;<=>?@[\]^_`{|}~ \.\n]+(?=["»]?$)|[^$!"‑-‑№«»#$%&()*+‑,\-/:;<=>?@[\]^_`{|}~ \n]+(?!["»]?$)|»|«|")'
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
		if result is not None:
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
		labels_list = [label for _, label in labels.items()]
		if len(labels_list) == 1:
			label = labels_list[0]
		elif len(labels_list) == 2:
			label = labels_list[0] & labels_list[1]
		else:
			label = labels_list[0] & labels_list[1] | labels_list[1] & labels_list[2] | labels_list[0] & labels_list[2]

		labels_list = create_labels_list(word_list, label)
		result.append((word_list, labels_list))
	return result

def print_text_with_labels(text, labels_set):
	dict = {}
	for label in labels_set:
		dict[label[0]] = (label[1], label[2])

	str_res = ""
	i = 0
	while i < len(text):
		if i in dict:
			str_res += f"({text[i:dict[i][0]]})" + f"|{dict[i][1]}"
			i = dict[i][0]
		else:
			str_res += text[i]
			i+=1

	return str_res

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
def create_train_x_y(sentences, tokenizer, tag_to_ix, debug=False):
	train_x = []
	train_y = []
	for sentence in sentences:
		labels = ["O"]
		string_sentence = " ".join([word[0][0] for word in sentence])
		string_sentence = "[CLS] " + string_sentence + " [SEP]"
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

			if len(buf) == len(sentence[j][0][0]) or tokenized_text[i] == "[UNK]":
				buf = ""
				j += 1
			i += 1
		if j != len(sentence) and debug:
			print(sentence[j - 2][0][0])
			print(sentence[j - 1][0][0])
			print(sentence[j][0][0])
			input()
			continue
		labels.append("O")
		train_x.append(tokenized_text)
		train_y.append([tag_to_ix[label] for label in labels])

	return train_x, train_y

def create_labels_set(labels):
	"""Формирует множество меток, как пересечение большинства"""
	labels_list = [label for _, label in labels.items()]
	if len(labels_list) == 1:
		label = labels_list[0]
	elif len(labels_list) == 2:
		label = labels_list[0] & labels_list[1]
	else:
		label = labels_list[0] & labels_list[1] | labels_list[1] & labels_list[2] | labels_list[0] & labels_list[2]
	return label