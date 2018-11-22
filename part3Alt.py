"""
This program calculates the scores for each phrase and ranks them in the descending order of probability
"""

from nltk.translate import phrase_based
from part2 import use_IBM1,get_examples,get_data
import pprint
import json
from main import drive,examples,native_lang,foreign_lang
# from part1 import drive
if __name__=='__main__':
	# corpus,source_set,target_set,settings=get_data()
	# examples=get_examples(settings)
	# ibm,corpus=use_IBM1(corpus,settings)
	# current_probs,corpus=drive(corpus)
	# foreign_eng contains the pairs of foreign and english words and their corresponding count
	# eng contains the english words and their corresponding count
	f=open(examples,'r',encoding='utf-8')
	corpus=json.load(f)
	f.close()
	prob_table,aligned_obj=drive(corpus)
	foreign_eng = {}
	english = {}
	for i in range(len(aligned_obj)):
		print(corpus[i][native_lang])
		print(corpus[i][foreign_lang])
		phrases = phrase_based.phrase_extraction(corpus[i][native_lang],corpus[i][foreign_lang],aligned_obj[i][2])
		for i in sorted(phrases):
			t = (i[2],i[3])
			if t not in foreign_eng: # check for first occurence of phrase pair
				foreign_eng[t] = 1
			else: #If phrase pair has already been encountered, increase count
				foreign_eng[t] += 1
			if i[2] not in english: #Check for first occurence of foreign phrase
				english[i[2]] = 1
			else: #If foreign phrase has already been encountered, increase count
				english[i[2]] += 1

	# scores of the phrases is determined by the formula score(f,e) = count(f,e)/count(f)
	scores = {}

	for phrase in foreign_eng:
		scores[phrase] = foreign_eng[phrase]/english[phrase[0]]

	scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)
	pp = pprint.PrettyPrinter(4)
	pp.pprint(scores)
