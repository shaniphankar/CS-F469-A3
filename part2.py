import json
import itertools
from nltk.translate import IBMModel1,IBMModel2,AlignedSent
def get_data():
	'''
	Obtains the source data based on the settings set within the configuration file identified by "langsettings.json". 
	Important Constants:
		"train-data" = The training data to be used within the model
		"source" = The language which is to be translated
		"target" = The language to which the translation must be made.
	Outputs:
		corpus = A list of Alignment Objects, which inturn contain tuples of the source language, the target language and the possible alignment
		source_set = The set of all the words present within the source
		target_set = The set of all the words present within the target
	'''
	# store the corpus for each model
	corpus=[]
	# sets for each type of word
	source_set=set()
	target_set=set()

	# the setting file has all the hardcoded params
	with open('langsettings.json') as f:
		settings=json.load(f)

	# open the file with the training data
	with open(settings['train-data']) as f:
		examples=json.load(f)

	# store the data in the form of alignment objects defined within nltk
	for example in examples:
		source=example[settings['source']].split()
		target=example[settings['target']].split()
		source_set=source_set.union(set(source))
		target_set=target_set.union(set(target))
		corpus.append(AlignedSent(source,target))

	# to account for no mapping of the target
	source_set.add(None)
	
	return corpus,source_set,target_set,settings

def get_examples(settings):
	'''
	Extract the examples of the data from the train-data that has been defined within the "langsettings.json" file
	It holds the training examples in the form of tuples in which they have been tagged in the following form
	"source" = the source or the language to be translated
	"target" = the target or the language in which the translation will be obtained
	'''

	# open the file with the training data
	with open(settings['train-data']) as f:
		examples=json.load(f)

		return examples

def use_IBM1(corpus,settings):
	'''
	Gives back the result on a corpus containing Aligned Objects, on using IBM Model 1
	Inputs:
		corpus = A list of Alignment Objects, which inturn contain tuples of the source language, the target language and the possible alignment
		settings =  The hardcoded options set within the "langsettings.json" file
	Outputs:
		ibm1 = An object containing the mapping for the foreign words and the translated words and the probabilities of each
		corpus = The modified input, which has the alignments for each word.
	'''
	# train the model
	ibm1=IBMModel1(corpus,settings['iterations'])

	return ibm1,corpus

def use_IBM2(corpus,settings):
	'''
	Gives back the result on a corpus containing Aligned Objects, on using IBM Model 2
	Inputs:
		corpus = A list of Alignment Objects, which inturn contain tuples of the source language, the target language and the possible alignment
		settings =  The hardcoded options set within the "langsettings.json" file
	Outputs:
		ibm2 = An object containing the mapping for the foreign words and the translated words and the probabilities of each
		corpus = The modified input, which has the alignments for each word.
	'''
	# train the model
	ibm2=IBMModel2(corpus,settings['iterations'])

	return ibm2,corpus

def print_data(ibm,source_set,target_set,model_num):
	'''
	Prints the data for each mapping of the source word and the destination word.
	Input:
		ibm = The object returned by the ibm model containing all the mapping probabilities
		source_set = The set of words belonging to the source
		target_set = The set of words belonging to the target
		model_num = Tells the number of the IBM Model used
	'''
	product=itertools.product(source_set,target_set)


	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Result for Model %d @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"%(model_num))
	
	for pair in product:
		print("Source:%s\nTarget:%s\nProbability:%s\n"%(pair[0],pair[1],ibm.translation_table[pair[0]][pair[1]]))

if __name__=='__main__':

	corpus,source_set,target_set,settings=get_data()
	ibm,corpus=use_IBM1(corpus,settings)
	print_data(ibm,source_set,target_set,1)

	ibm,corpus=use_IBM2(corpus,settings)
	print_data(ibm,source_set,target_set,2)
