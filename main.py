from prepare_data import prepare_embeddings_matrix, create_word2vec
from model import get_bilstm_lstm_model


def main():
	VOCAB_SIZE = 0
	word2idx = {}
	input_length = 0
	n_tags = 0
	embedding_matrix = prepare_embeddings_matrix(create_word2vec("w2v_size100_window5.txt"),
	                                             EMBEDDING_DIM=100, VOCAB_SIZE=VOCAB_SIZE, word2idx=word2idx)
	model = get_bilstm_lstm_model(embedding_matrix=embedding_matrix, vocab_size=VOCAB_SIZE,
	                              embedding_dim=100, input_length=input_length, n_tags=n_tags)
