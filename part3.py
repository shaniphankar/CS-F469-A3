"""
This program calculates the scores for each phrase and ranks them in the descending order of probability
"""

from nltk.translate import phrase_based
from part2 import use_IBM1,get_examples,get_data
import pprint
# from part1 import drive
if __name__=='__main__':
	corpus,source_set,target_set,settings=get_data()
	examples=get_examples(settings)
	ibm,corpus=use_IBM1(corpus,settings)

	# for i in range(len(examples)):
	# 	phrases = phrase_based.phrase_extraction(examples[i][settings['source']],examples[i][settings['target']],corpus[i].alignment)
	# 	for i in sorted(phrases):
	# 		print(i)
	# 	print("\n")

	# foreign_eng contains the pairs of foreign and english words and their corresponding count
	# eng contains the english words and their corresponding count
	foreign_eng = {}
	foreign = {}
	for i in range(len(examples)):
		phrases = phrase_based.phrase_extraction(examples[i][settings['source']],examples[i][settings['target']],corpus[i].alignment)
		for i in sorted(phrases):
			t = (i[2],i[3])
			# print(t)
			if t not in foreign_eng: foreign_eng[t] = 1
			else: foreign_eng[t] += 1
			if i[3] not in foreign: foreign[i[3]] = 1
			else: foreign[i[3]] += 1

	# print(foreign_eng)
	# print(eng)

	# scores of the phrases is determined by the formula score(f,e) = count(f,e)/count(f)
	scores = {}

	for phrase in foreign_eng:
		scores[phrase] = foreign_eng[phrase]/foreign[phrase[1]]

	scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)
	pp = pprint.PrettyPrinter(4)
	pp.pprint(scores)
