import numpy as np
import math
import gensim
from gensim.models.word2vec import Word2Vec
import scipy

emb_size = 100


class Measure(object):
	"""docstring for Measure"""
	def __init__(self, embbeding_path_from_chatbot_args_parameter):
		#super(Measure, self).__init__()
		self.model = self.load_model(embbeding_path_from_chatbot_args_parameter)
		
		



	def load_model(self,fpath):

	# TODO make this nice

		embeddings_file = fpath
		return Word2Vec.load(embeddings_file)
		#emb_size = 100


	def perplexity(self,predicted_softmaxes, actual_answer, char_to_id):
		"""
		predicted_softmaxes         matrix of [answer length (in chars) x vocab_size]
		actual_answer               the ground-truth answer in the corpus (a string)
		char_to_id                  maps characters to their ID
		"""

		# Not entirely clear
		VOCAB_SIZE = 42000

		i = 0                       # Word index in current sentence
		perp_sum = 0

		while i < len(actual_answer) and i < len(predicted_softmaxes):
			char_prob = predicted_softmaxes[i][char_to_id[actual_answer[i]]]
			perp_sum += math.log(char_prob, 2)
			i += 1

		# As specified in task description: ./docs/task_description
		# perp = 2^{(-1/n)*\sum^{n}_{t}(log_2(p(w_t | w_1, ... , w_t-1))} -
		average_bpc = (-1/i) * perp_sum
		factor = len(char_to_id) / 30000
		perp = math.pow(2, average_bpc * (len(char_to_id) / float(VOCAB_SIZE)))
		return perp


	# TODO adapt once it's plugged in
	def vector_extrema_dist(self,reference, output):
		"""
		reference       string
		output          string
		"""
		#xemb_size = 100

		#model = load_model(embbeding_path)

		#def normalize():
		def normalize(v):
			norm=np.linalg.norm(v)
			if norm==0: 
				return v
			return v/norm

		def extrema(sentence):
			sentence = sentence.split(" ")
			vector_extrema = np.zeros(shape=(emb_size))
			for i, word in enumerate(sentence):
				if word in self.model.wv.vocab:
					n = self.model[word]
					abs_n = np.abs(n)
					#print("abs")
					abs_v = np.abs(vector_extrema)
					for e in range(emb_size):
						if abs_n[e] > abs_v[e]:
							vector_extrema[e] = n[e]

			return vector_extrema

		ref_ext = extrema(reference)
		out_ext = extrema(output)

		#print(ref_ext)
		#print(normalize(ref_ext))


		return scipy.spatial.distance.cosine(normalize(ref_ext), normalize(out_ext))


