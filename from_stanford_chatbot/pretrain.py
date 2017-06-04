import gensim, logging

import os

import config

if False:
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
		model = gensim.models.Word2Vec(totallines, min_count=cfg['THRESHOLD'])
		model.save(os.path.join(foldername,modelname))
		print("model saved")






def pretrain_all():
	path = "pretrained_stuff/pretrained_model"
	if os.path.isfile(path):
		pass


if __name__ == '__main__':
	pretrain()