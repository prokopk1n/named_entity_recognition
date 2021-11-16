import gensim
import torch.nn as nn
import torch


START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_CRF(nn.Module):
	def __init__(self, tag_to_ix: dict, lstm_size: int, embedding_path: str):
		# lstm_size = размерность выхода lstm
		# num_layers = число последовательных lstm (рекуррентная версия)
		super(BiLSTM_CRF, self).__init__()

		self.hidden_dim = lstm_size

		self.tag_to_ix = {tag: index for tag, index in tag_to_ix.items()}
		self.tag_to_ix[START_TAG] = len(self.tag_to_ix)
		self.tag_to_ix[STOP_TAG] = len(self.tag_to_ix)
		self.tagset_size = len(self.tag_to_ix)

		self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_path)
		weights = torch.FloatTensor(self.word_vectors.vectors)
		self.embedding = nn.Embedding.from_pretrained(weights)

		self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_path)
		self.embedding_dim = self.word_vectors.vector_size

		self.lstm = nn.LSTM(self.embedding_dim, lstm_size // 2, num_layers=1, bidirectional=True)
		self.hidden2tag = nn.Linear(lstm_size, self.tagset_size)

		# transitions[i][j] - вероятность перехода от метки j к метке i
		self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
		self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
		self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

	def _get_embedded(self, sentence):
		sentence = torch.tensor(sentence, dtype=torch.long)
		return self.embedding(sentence)

	def _get_lstm_features(self, sentence):
		self.hidden = self.init_hidden()
		embeds = self._get_embedded(sentence).view(len(sentence), 1, -1) # трехмерная матрица, -1 - неизвестно сколько точно, высчитается по формуле
		lstm_out, self.hidden = self.lstm(embeds, self.hidden)
		lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
		lstm_feats = self.hidden2tag(lstm_out)
		return lstm_feats

	def forward(self, sentence):
		lstm_feats = self._get_lstm_features(sentence)
		score, tag_seq = self._viterbi_decode(lstm_feats)
		return score, tag_seq

	def init_hidden(self):
		# （num_layers*num_directions,minibatch_size,hidden_dim）
		# Собственно инициализированные h0 и c0
		return (torch.randn(2, 1, self.hidden_dim // 2),
		        torch.randn(2, 1, self.hidden_dim // 2))

	def _viterbi_decode(self, feats):
		# Оценка последовательности предсказания, декодирование Витерби, оценка вывода и значение пути
		backpointers = []

		# Initialize the viterbi variables
		init_vvars = torch.full((1, self.tagset_size),
		                        -10000.)  # Это гарантирует, что он должен быть от START до других тегов
		init_vvars[0][self.tag_to_ix[START_TAG]] = 0

		# forward_var at step i holds the viterbi variables for step i-1
		forward_var = init_vvars
		for feat in feats:
			bptrs_t = []  # holds the backpointers for this step
			viterbivars_t = []  # holds the viterbi variables for this step

			for next_tag in range(self.tagset_size):
				# Вероятность использования других тегов (B, I, E, Start, End) для тега next_tag
				next_tag_var = forward_var + self.transitions[
					next_tag]  # forward_var содержит значение предыдущего оптимального пути
				best_tag_id = argmax(next_tag_var)  # Вернуть тег, соответствующий максимальному значению
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
			# Now add in the emission scores, and assign forward_var to the set
			# of viterbi variables we just computed
			# От шага 0 до шага (i-1) максимальная оценка каждой из 5 последовательностей
			forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
			backpointers.append(bptrs_t)  # bptrs_t состоит из 5 элементов

		# Вероятность передачи других тегов в STOP_TAG
		terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
		best_tag_id = argmax(terminal_var)
		path_score = terminal_var[0][best_tag_id]

		# Follow the back pointers to decode the best path.
		best_path = [best_tag_id]
		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
		# Pop off the start tag (we dont want to return that to the caller)
		start = best_path.pop()
		assert start == self.tag_to_ix[START_TAG]  # Sanity check
		best_path.reverse()  # Исправить путь от задней части к передней
		return path_score, best_path

	def neg_log_likelihood(self, sentence, tags):  # loss function
		feats = self._get_lstm_features(sentence)  # Выход после LSTM + Linear используется как вход CRF
		forward_score = self._forward_alg(feats)  # Результат лога части потерь
		gold_score = self._score_sentence(feats, tags)  # Результат второй половины проигрыша S (X, y)
		return forward_score - gold_score  # Loss

	def _score_sentence(self, feats, tags):  # Найдите второй член функции потерь
		# Это то же самое, что и вышеупомянутый def _forward_alg (self, feats) в том, что: Оба являются оценками, вычисленными с помощью матрицы случайных переходов. Разница в том, что вышеуказанная функция вычисляет максимально возможный путь, но на самом деле это может быть неверно Например, значение каждой передачи тега - NVV, но поскольку переходы являются случайными, указанная выше функция на самом деле является NNN, поэтому между ними есть разрыв в оценке. Позже обратное распространение может обновлять переходы, чтобы матрица переходов была близка к реальной «матрице перехода», чтобы получить оценку тега gold_seq, который должен вычислять оценку на основе реальной метки, но поскольку матрица перехода генерируется случайным образом, она вычисляется оценка - не самое идеальное значение
		score = torch.zeros(1)
		# Вставьте тег 3 из START_TAG в верхнюю часть последовательности тегов
		tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
		for i, feat in enumerate(feats):
			# self.transitions [tags [i + 1], tags [i]] Фактически вы получаете вероятность перехода от тега i к тегу i + 1
			# feat [tags [i + 1]], feat - это выходной результат шага i с 5 значениями,
			# Соответствуют B, I, E, START_TAG, END_TAG, принимают значение соответствующего тега
			# transition [j, i] - значение вероятности перехода из i -> j
			score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
		score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
		return score

	def _forward_alg(self, feats):  # Оценка последовательности прогнозов - это первый элемент справа от убытка
		# feats представляет собой оценку эмиссии (emit score), которая фактически является результатом LSTM, что означает, что каждое слово предложения LSTM соответствует количеству оценок каждой метки.
		# Do the forward algorithm to compute the partition function
		init_alphas = torch.full((1, self.tagset_size),
		                         -10000.)  # Используйте -10000., Чтобы заполнить тензор формой [1, tagset_size]

		# START_TAG has all of the score.
		# Поскольку начальный тег равен 4, тензор ([[- 10000., -10000., -10000., 0., -10000.]]),
		# Установите значение start равным нулю, что означает, что распространение сети началось,
		init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

		# Обернуть в переменную для автоматического обратного распространения ошибки
		forward_var = init_alphas  # Forward_var начального состояния изменяется с шагом t

		# Обходите предложение и повторяйте строки подвигов несколько раз
		for feat in feats:
			alphas_t = []  # Положительный тензор текущего временного шага
			for next_tag in range(self.tagset_size):
				# broadcast the emission score: it is the same regardless of the previous tag
				# Генерирующая матрица LSTM - emit_score, размерность 1 * 5
				emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
				# the i_th entry of trans_score is the score of transitioning to
				# next_tag from i
				trans_score = self.transitions[next_tag].view(1, -1)  # Размер 1 * 5
				# Примите во внимание во время первой итерации:
				# trans_score - вероятность того, что все остальные теги попадут в тег B
				# Запускаем от lstm к скрытому слою, а затем к выходному слою, чтобы получить вероятность метки B. Размер emit_score равен 1 * 5, и 5 значений совпадают
				next_tag_var = forward_var + trans_score + emit_score
				# The forward variable for this tag is logsumexp of all the scores.
				alphas_t.append(log_sum_exp(next_tag_var).view(1))
			# Альфа-код t на данный момент имеет длину 5, например <class'list '>:
			# [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
			forward_var = torch.cat(alphas_t).view(1, -1)
		# Наконец, только прямая переменная последнего слова добавляется к вероятности передачи стоп-тега
		# tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
		terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
		alpha = log_sum_exp(terminal_var)  # альфа - это 0-мерный тензор
		return alpha

def log_sum_exp(vec): #vec размерность 1 * 5
	max_score = vec[0, argmax(vec)]#max_score имеет размерность 1
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) # Размер 1 * 5
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
	# Получить индекс наибольшего значения
	_, idx = torch.max(vec, 1) # Возвращает самый большой элемент и индекс самого большого элемента в каждой строке
	return idx.item()
