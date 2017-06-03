import gensim, logging

import os

import config

if config.USE_CORNELL:
    import cornell_data as data
else:
    import our_data as data

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


def get_line_file(filename):
	in_path = os.path.join(config.PROCESSED_PATH, filename)

	alllines = []

	with open(in_path, "r") as f:
		for line in f.readlines():
			words = line.strip().split()
			alllines.append(words)

	return alllines


def get_alllines():
	lines = []
	filenames = ['train.enc','train.dec']
	for filename in filenames:
		moreline = get_line_file(filename)
		lines.extend(moreline)
		print(" Finish processing"+filename)

	return lines


def pretrain():
	modelname = "pretrain_model"
	foldername = "pretrained_stuff"
	if os.path.isdir(foldername):
		pass
	else :
		os.mkdir(foldername)

	#totallines = get_alllines()

	#for line in totallines:
		#print line
	print("trying to get a model")

	if os.path.exists(foldername+'/'+modelname):
		print("The model already exists")
		pass
	else:
		print("make the input")
		totallines = get_alllines()
		print("create a model")
		model = gensim.models.Word2Vec(totallines, min_count=1)
		model.save(os.path.join(foldername,modelname))
		print("model saved")


def load_embedding(self, session, vocab, emb, path, dim_embedding):
    '''
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''
    print("Loading external embeddings from %s" % path)
    model = models.Word2Vec.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(FLAGS.vocab_size, dim_embedding))
    matches = 0
    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        
    print("%d words out of %d could be loaded" % (matches, FLAGS.vocab_size))
    
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})



def pretrain_all():
	path = "pretrained_stuff/pretrained_model"
	if os.path.isfile(path):
		pass


if __name__ == '__main__':
	pretrain()