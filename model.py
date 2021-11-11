import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional


def get_bilstm_lstm_model(embedding_matrix, vocab_size, embedding_dim, input_length, n_tags):
	model = Sequential()

	# Add Embedding layer
	# vocab_size = input_dim = Размерность one-hot вектора
	embedding = Embedding(input_dim=vocab_size, outputdim=embedding_dim,
	                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), trainable=False, input_length=input_length)

	model.add(embedding)

	# Add bidirectional LSTM
	model.add(Bidirectional(LSTM(units=embedding_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
	                        merge_mode='concat'))

	# Add LSTM
	# почему здесь не умножение на 2?
	model.add(LSTM(units=embedding_dim * 4, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

	# Add timeDistributed Layer
	model.add(TimeDistributed(Dense(n_tags, activation="relu")))

	# Optimiser
	# adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	return model
