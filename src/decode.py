"""
Модуль для BIO декодирования
"""
from typing import List, Set, Tuple, Dict

def labels_decoder(labels_list: List[str], sentence: List[Tuple[str, int, int]],
                   tokenized_text: List[str]) -> Set[Tuple[int, int, str]]:
	"""
	Принимает на вход список меток предложения, предложение как список пар, а также все предложение
	разбитое bert`ом на токены,
	"""
	result_set = set()
	k = 1  # берем метку от самой первой части слова, если слово разбивается на подслова через ##
	i = 1
	j = 0  # для итерации по предложению
	size = len(tokenized_text) - 1
	while i < size:
		buf = tokenized_text[i]
		i += 1
		while len(buf) != len(sentence[j][0]) and tokenized_text[i] != "[UNK]":
			if tokenized_text[i][:2] == '##':
				buf += tokenized_text[i][2:]
			else:
				buf += tokenized_text[i]
			i += 1
		# print(f"BUF = {buf} sentence = {sentence[j][0]}")
		label_cur = labels_list[k]
		k = i

		if label_cur == "B-ORGANIZATION" or label_cur == "B-PERSON" or label_cur == "B-DATE":
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
	"""Преобразует теги в числовом виде обратно с строковый вид"""
	result = []
	for tag in tags:
		for key, value in tag_to_ix.items():
			if value == tag:
				result.append(key)
				break
	return result
