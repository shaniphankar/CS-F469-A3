import json
import itertools
from nltk.translate import IBMModel1,IBMModel2,AlignedSent
def get_data():

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
	'''Extract the examples of the data'''

	# open the file with the training data
	with open(settings['train-data']) as f:
		examples=json.load(f)

		return examples

def use_IBM1(corpus,settings):

	# train the model
	ibm1=IBMModel1(corpus,settings['iterations'])

	return ibm1,corpus

def use_IBM2(corpus,settings):
	
	# train the model
	ibm2=IBMModel2(corpus,settings['iterations'])

	return ibm2,corpus

def print_data(ibm,source_set,target_set,model_num):

	product=itertools.product(source_set,target_set)


	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Result for Model %d @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",model_num)
	
	for pair in product:
		print("Source:%s\nTarget:%s\nProbability:%s\n"%(pair[0],pair[1],ibm.translation_table[pair[0]][pair[1]]))

if __name__=='__main__':

	corpus,source_set,target_set,settings=get_data()
	ibm,corpus=use_IBM1(corpus,settings)
	print_data(ibm,source_set,target_set,1)

	ibm,corpus=use_IBM2(corpus,settings)
	print_data(ibm,source_set,target_set,2)