from nltk.translate import IBMModel1
from nltk.translate import IBMModel2
from nltk.translate import AlignedSent
import json
import pprint
import itertools

pp=pprint.PrettyPrinter(indent=4)
def main():

	# store the corpus for each model
	corpus1=[]
	corpus2=[]
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
		corpus1.append(AlignedSent(source,target))

	# to account for no mapping of the target
	source_set.add(None)
	
	# copying the corpus data
	corpus2=corpus1[:]

	# Train the first model
	ibm1=IBMModel1(corpus1,settings['iterations'])

	# Train the latter model
	ibm2=IBMModel2(corpus2,settings['iterations'])
	

	product=itertools.product(source_set,target_set)


	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Result for Model 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
	
	for pair in product:
		print("Source:%s\nTarget:%s\nProbability:%s\n"%(pair[0],pair[1],ibm1.translation_table[pair[0]][pair[1]]))

	product=itertools.product(source_set,target_set)

	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Result for Model 2 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
	
	for pair in product:
		print("Source:%s\nTarget:%s\nProbability:%s\n"%(pair[0],pair[1],ibm2.translation_table[pair[0]][pair[1]]))

if __name__=='__main__':
	main()