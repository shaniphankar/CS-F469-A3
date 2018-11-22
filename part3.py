from nltk.translate import phrase_based
from part2 import use_IBM1,get_examples,get_data
from part1 import drive
if __name__=='__main__':
	corpus,source_set,target_set,settings=get_data()
	examples=get_examples(settings)

	ibm,corpus=use_IBM1(corpus,settings)
	print(examples)
	print("\n")
	print(corpus)
	# phrases=phrase_based.phrase_extraction(examples[0][settings['target']],examples[0][settings['source']],corpus[0].alignment)
	# print(phrases)
