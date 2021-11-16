import re
import time
import torch
from prepare_data import create_words_labels_pairs_list, prepare_sequence
from model import BiLSTM_CRF
from preprocessing_json import preprocess_json
import gensim
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def tag_decoder(tags, tag_to_ix):
	result = []
	for tag in tags:
		for key, value in tag_to_ix.items():
			if value == tag:
				result.append(key)
				break
	return result


def main():
	train_data = preprocess_json("tpc-dataset.train_3.json")
	words_labels_pairs_list = create_words_labels_pairs_list(train_data)
	keyedvector_model = gensim.models.KeyedVectors.load_word2vec_format("w2v_size100_window5.txt")
	tag_to_ix = {"B-ORGANIZATION": 0, "I-ORGANIZATION": 1, "B-PERSON": 2, "I-PERSON": 3, "O": 4}

	train_x = []
	train_y = []
	for i in range(0, len(words_labels_pairs_list)):
		train_x.append(prepare_sequence(words_labels_pairs_list[i][0], keyedvector_model))
		train_y.append([tag_to_ix[x] for x in words_labels_pairs_list[i][1]])
	# вот до сюда вроде все верно

	lstm_size = 4
	embedding_path = "w2v_size100_window5.txt"
	model = BiLSTM_CRF(tag_to_ix, lstm_size, embedding_path)

	# train_xxx = [torch.tensor(x,  dtype=torch.long) for x in train_x]
	train_yyy = [torch.tensor(y,  dtype=torch.long) for y in train_y]

	optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

	model.train()

	with torch.no_grad():
		result = model(train_x[0])
		print(tag_decoder(result[1], tag_to_ix))

	for i in range(0, len(train_x) - 650):
		print(f"STEP NUMBER {i}")
		# print(y_pred[1])
		model.zero_grad()
		loss = model.neg_log_likelihood(train_x[i], train_yyy[i])
		loss.backward()
		optimizer.step()

	with torch.no_grad():
		result = model(train_x[0])
		print(tag_decoder(result[1], tag_to_ix))

main()
