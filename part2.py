from nltk.translate import IBMModel1
from nltk.translate import AlignedSent
import json
import pprint
pp=pprint.PrettyPrinter(indent=4)
def main():

	corpus1=[]
	corpus2=[]
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
		corpus1.append(AlignedSent(source,target))

	# copying the corpus data
	corpus2=corpus1[:]

	# Train the first model
	ibm1=IBMModel1(corpus1,settings['iterations'])

	# Train the latter model
	ibm2=IBMModel2(corpus2,settings['iterations'])
	print(corpus1)
	print(corpus2)


if '__name__'=='__main__':
	main()