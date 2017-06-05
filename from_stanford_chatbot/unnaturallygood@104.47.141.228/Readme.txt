### Description

This directory contains the movie script dataset, which is an extension of the Movie-DiC dataset. More information about the original Movie-DiC dataset can be found in the paper "Movie-DiC: a Movie Dialogue Corpus for Research and Development" by Rafael E. Banchs.

### Preprocessing

Movie-Dic has been expanded to include meta-information for each movie: title, year, genres, user ratings, plot lines etc. This was extracted through the online API service http://www.omdbapi.com. The meta-information was inspected manually to resolve naming issues between movies (e.g. cases where the manuscript differs from the name on www.imdb.com) and to ensure high quality.

The manuscripts were processed to remove duplicate manuscripts. Afterwards, a spelling corrector based on 4051 of Wikipedia's most common English spelling mistakes were applied. The reasoning is that most manuscripts are typed on computers and by people of similar writing skill as Wikipedia editors. Manual inspection showed that this improved the quality of the dataset. 

We also attempted applying the popular spellchecker enchant (http://abisource.com/projects/enchant/), but found that due to novel words (e.g. terms used in sci-fi and adventure stories) and slang found in many movies, this created more mistakes than corrections the mispellings. We also found that many manuscripts contained multiple puntucation signs and spaces between words, which is perhaps a popular method to instruct actors what speed and emotion to use. To standardize this, we attempted to apply word-segmentation from a publicly-available python module of the method described by Segaran and Hammerbacher, 2009. Unfortunately, this too created more mistakes than corrections and we therefore choose not to use it. Instead, we implemented a set of simple regular expressions to remove double puntucation, comma and spacings, and to enforce capitalization at the beginning of each sentence. This appeared to be most effective at standardizing the dataset.

The dataset was then tokenized and named-entity recognition was applied to replace all names and numbers with <person> and <name> tags respectively. We also experimented with replacing organizations and locations (including geopolitical entries) with <organization> and <location> tags respectively, but found that these did not improve the generalization capabilities of our baseline models to any notable extend. The tokenization and named-entity recognition was done with the python-based natural language toolkit NLTK, which uses a a maximum entropy chunker trained on the ACE corpus (http://catalog.ldc.upenn.edu/LDC2005T09). Similar preprocessing has been found to help generalization capabilities significantly in "Developing Non-Goal Dialog System based on Examples of Drama Television" by Lasguido et al, and similar filtering has also been applied to Twitter conversations in "Unsupervised Modeling of Twitter Conversations" by Ritter et al. 

Consecutive dialogue utterances by the same speaker were concatenated together, and all triples of the form A-B-A, i.e. a dialogue where A is making an utterance followed by B making an utterance followed by A again, were extracted. All triples were transformed to lowercase letters.

Finally, the triples were split into training, validation (development) and test sets. This was done in such a way, that triples from the same movie did not overlap into different datasets with the following partitioning:

Dataset Movie Indices
Training	1-484
Validation	485-549
Test		550-615 (excluding movie index 557)

Dataset	Number of triples	Mean triple length	Number of unknowns
Training	196308		53			190588
Validation	24717		53			30069
Test		24271		55			28978

### Experiments

In all our experiments we use a vocabulary of size 10000 tokens (footnote: an additional 3 vocabulary tokens were used to represent start of utterance, end of utterance and end of triple). The start of sentence token was ignored during the evaluation procedure.

### Movie Genres

The original genres categories we extracted contained 25 categories:

Genre	Frequency (movies with that genre)
adult	2
n/a	3
western	3
musical	5
film-noir	6
family	7
animation	8
documentary	8
war	12
sport	13
history	18
music	18
short	22
biography	37
fantasy	38
horror	52
sci-fi	57
mystery	68
adventure	80
romance	97
thriller	109
action	131
crime	144
comedy	193
drama	342

Because many genres only have a few movies, we choose to omit the 8 least frequent ones, together with the generic 'drama' category, which is assigned to more than half of the movies. This left 51 movies without genre labels. The full set of genre labels is still made available to other researchers in the "MetaInfo" files. 

In total we have 16 labels:

Genre	Frequency	Entropy Per Triple
war	12	0.1405114831
sport	13	0.1497134664
history	18	0.1931469538
music	18	0.1931469538
short	22	0.2253636391
biography	37	0.3320208555
fantasy	38	0.3384996992
horror	52	0.4228063884
sci-fi	57	0.4503842044
mystery	68	0.5071030537
adventure	80	0.5635262178
romance	97	0.6350998172
thriller	109	0.6804385035
action	131	0.7537785831
crime	144	0.7917262483
comedy	193	0.9032997199
		
Total	1089	7.2805657873


If we were to add this as input to the model, it would add on average 7.28 bits to the training signal of each triple. Now, suppose there is a strong correlation between movie genre and utterances. On average the entropy of the decoders conditional distribution is log2(28) = 4.80735492206 bits per word. Then adding the genre information could effectively specify 1-2 words of the triple.

### Word Embeddings

The word embeddings based on machine translation were taken from https://www.cl.cam.ac.uk/~fh295/. These were based on the work of Hill et al. They consist of 620-size vectors trained on a 91B words English-German sentence pairs. This corporus was produced from the WMT ’14 parallel data after conducting the data-selection proce-
dure described by Cho et al. (2014). 

The word embeddings based on monologue corpora were taken from http://code.google.com/p/word2vec/. These are based on the work of Mikolov et al. They consist of 300-size vectors constructed by training a n-skipgram mode lon the Google News dataset (about 100 billion words).

For words in the movie scripts vocabulary that could not be directly matched to vocabularies of the other models, regex expressions were applied together with the enchant spellchecker to correct the words and then match them. Similarly to the preprocessing experiments above, this approach does not fix the majority of spelling errors, but manual inspection shows that it often finds very related words which are in the other model vocabularies. In addition to this, all non-word tokens (e.g. placeholders such as <person>, and symbols such as start-of-utterance and end-of-utterance) were initialized randomly. All dimensions were finally rescaled to have mean zero and std deviation 0.01 similar to previous experiments.

For the machine translation pretrained word embeddings, the 620-size vectors were projected down to 300-size vectors using PCA. In total, 135 words (corresponding to 32.70% of the training corpus tokens) were not or could not be matched, and were therefore initialized to random values as in the previous experiments \footnote{Apart from the non-word symbols and palceholders, the majority of these words corresponded to rare slang terms, which we do not believe will influence the performance of the model significantly.}. Although this number may seem high, in fact 32.53% of the training corpus consists of non-word tokens, which implies in fact that only 0.17% of the training corpus tokens, which correspond to actual words, have been initialized randomly.

For the Word2Vec pretrained word embeddings, in total 56 words (corresponding to about 32.68% of the training corpus) in the movie script corpus vocabulary were not matched and so were initialized to random values as in the previous experiments. As before, this implies that only 0.15% of the training corpus tokens, which correspond to actual words, have been initialized randomly.

### SubTle Dataset

The exact same preprocessing was used on the SubTle corpus.

The final SubTle dataset contains 11,007,491 utterances. To make sure we did not have significant overlaps between SubTle and MovieTriples, we compared the SubTle utterances to those of the test set in MovieTriples. The test set contained 72,813 utterances. Of these 7,016 utterances also existed in SubTle, which means an overlap of 7,016/72,813 = 9.64%. In comparison the MovieTriples validation set, which contained 74,151 utterances, has 6,354 utterances in common with the MovieTriples test set, which yields an overlap on a similar scale at 6,354/72,813 = 8.73%. Since the validation set and test set were constructed from a disjoint sets of movies, and since the SubTle corpus, despite being a magnitude larger in size, has a similar utterance overlap as the validation set, we conclude that the overlap between the dialogues in the dataset (as opposed to the individual utterances) is minimal and that any overlap arises from very frequent phrases. Manual inspection of the overlapping phrases confirm this assertion.

### File Descriptions

Dataset.txt: Contains the unshuffled triples extracted from the Movie-DiC dataset. Each row is a triple with three utterances separated by tab characters.
Dataset_Labels.txt: Each row contains label information corresponding to the same row in Dataset.txt. This information includes the movie id, the dialogue id (i.e. the scene as segmented in Movie-DiC), the first speaker and the second speaker as given in the manuscript. Note, utilizing the speaker names may require additional parsing to ensure that mispelled or variants of the same speaker are grouped together correctly.

MetaInfo.txt: Tab-separated file containing meta-information for every move. The first row describes each column. In particular, the first column (id) is tied to the id in each *_Labels.txt file.
MetaInfo.ods: An Open Office Calc version of the meta-information. Same as MetaInfo.csv.
UniqueGenres.txt: File containing all unique movie genres in the dataset.
WordsList.txt: File containing all tokens in the dataset. One token per line.

Shuffled_Dataset.txt: The shuffled version of Dataset.txt, in the order training set, validation set and test set.
Shuffled_Dataset_Labels.txt: Labels for Shuffled_Dataset.txt.

Training_Shuffled_Dataset.txt: Training set extracted by shuffling Dataset.txt. Contains movie indices 1-484.
Training_Shuffled_Dataset_Labels.txt: Labels for Training_Shuffled_Dataset.txt.

Validation_Shuffled_Dataset.txt: Validation set extracted by shuffling Dataset.txt. Contains movie indices 485-549.
Validation_Shuffled_Dataset_Labels.txt: Labels for Validation_Shuffled_Dataset.txt.

Test_Shuffled_Dataset.txt: Test set extracted by shuffling Dataset.txt.
Test_Shuffled_Dataset_Labels.txt: Labels for Test_Shuffled_Dataset.txt. Contains movie indices 550-615 (excluding movie index 557).

Training.dict.pkl: Dictionary with 10000 words extracted from the training set (Training_Shuffled_Dataset.txt). These terms represent 97.97% of the entire training set.
Training.triples.pkl: Triples extracted from the training set (Training_Shuffled_Dataset.txt).
Validation.triples.pkl: Triples extracted from the validation set (Validation_Shuffled_Dataset.txt).
Test.triples.pkl: Triples extracted from the test set (Test_Shuffled_Dataset.txt).

Training.genres.pkl: Genres corresponded to training set (Training.triples.pkl).
Validation.genres.pkl: Genres corresponded to validation set (Validation.triples.pkl).
Test.genres.pkl: Genres corresponded to test set (Test.triples.pkl).

MT_WordEmb.pkl: The MT word embeddings extracted based on Hill et al.
Word2Vec_WordEmb.pkl: The Word2Vec embeddings extracted based on Mikolov et al.

Shuffled_Subtle_Dataset.txt: The shuffled version of the SubTle corpus.
Subtle_Dataset.triples.pkl: Tuples extracted from SubTle corpus (Shuffled_Subtle_Dataset.txt). These are represented as triples with the last utterance being empty.

### Future Work On the Dataset

- Should clean up single character symbols such as "--" "[" etc. This was also done in Joelle and Furgueson paper on bootstrapping.
- Should also remove all conversations with more than 25% <unk> terms.
- Should merge dataset with other datasets.
- Use NLTKs named-entity-recognition together with a comphrensive list of English names. Use same list as was used for SubTle corpus. This is important because NLTK is now taking many nouns as names, yet ignoring other real names.
- The "Don't" is tokenized into "<person> ' t". This is completly wrong. Correct it and check if there are other mistonekization involving "<person>". 


Also, double check the two lines in Process.py:

            t = re.split('([.!?] *)', t)
            t = ''.join([each.capitalize() for each in t])

Make sure all placeholders are surrounded by spaces, e.g. by adding:

            triple = triple.replace('<person>', ' <person> ')
            triple = triple.replace('<number>', ' <number> ')
            triple = triple.replace('<continued_utterance>', ' <continued_utterance> ')
            triple = triple.replace('<location>', ' <location> ')
            triple = triple.replace('<organization>', ' <organization> ')
            triple = triple.replace('  ', ' ')
            triple = triple.replace('  ', ' ')
            triple = triple.replace('  ', ' ')

### References

http://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings, 
Retrieved on February 20th, 2015.

Cite Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O'Reilly Media Inc.  because we use their Named Entity Recognition tool.

http://catalog.ldc.upenn.edu/LDC2005T09

Based on code from the chapter "Natural Language Corpus Data"
    from the book "Beautiful Data" (Segaran and Hammerbacher, 2009)
    http://oreilly.com/catalog/9780596157111/

Bird, Steven, Ewan Klein, and Edward Loper (2009), Natural Language
Processing with Python, O'Reilly Media.

book{BirdKleinLoper09,
  author = {Steven Bird and Ewan Klein and Edward Loper},
  title = {{Natural Language Processing with Python}},
  publisher = {O'Reilly Media},
  year = 2009
}

Hill, F. Cho, KH. Jean, S. Devin, C. Bengio, Y. 2014. Embedding Word Similarity With Neural Machine Translation. Workshop Paper at ICLR 2015 

Cho, Kyunghyun, van Merrienboer, Bart, Gulcehre, Caglar, Bougares, Fethi, Schwenk, Holger,
and Bengio, Yoshua. Learning phrase representations using RNN encoder-decoder for statistical
machine translation. In Proceedings of the Empirical Methods in Natural Language Processing
(EMNLP 2014), October 2014. to appear.

[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013. 
