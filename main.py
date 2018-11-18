import nltk
import os
import json
import pprint
import itertools
from math import pow
from math import sqrt
from copy import deepcopy

pp=pprint.PrettyPrinter(indent=4)
native_lang="en"
foreign_lang="fr"	
def get_words_corpus(corpus):
	data={}
	data[native_lang]=[]
	data[foreign_lang]=[]
	for dict in corpus:
		for en_word in nltk.word_tokenize(dict[native_lang]):
			data[native_lang].append(en_word)
		for fr_word in nltk.word_tokenize(dict[foreign_lang]):
			data[foreign_lang].append(fr_word)
	return data

def myinit(data):
	 return {en_word: {fr_word: 1/len(data[native_lang]) for fr_word in data[foreign_lang]} for en_word in data[native_lang]}

def distance(data1,data2):
	dist=0
	row_iters=data1.keys()
	row_vals=list(data1.values())
	col_iters=row_vals[0].keys()
	for row_num in row_iters:
		for col_num in col_iters:
			dist+=pow((data1[row_num][col_num]-data2[row_num][col_num]),2)
	return sqrt(dist)

def train(corpus,epsilon):
	data=get_words_corpus(corpus)
	prev_probs=myinit(data)
	current_probs=myinit(data)
	converged=False
	# pp.pprint(prev_probs)
	error=100
	i=0
	en_sents=[]
	fr_sents=[]
	for dict in corpus:
		en_sents.append(dict[native_lang])
		fr_sents.append(dict[foreign_lang])
	combos=list(itertools.product(en_sents,fr_sents))
	pp.pprint(combos)
	while error>epsilon:
		count={en_word:{fr_word:0 for fr_word in data[foreign_lang]} for en_word in data[native_lang]}
		# pp.pprint(count)
		total={fr_word:0 for fr_word in data[foreign_lang]}
		for en_sent,fr_sent in combos:
			s_total={}
			for en_word in nltk.word_tokenize(en_sent):
				s_total[en_word]=0
				for fr_word in nltk.word_tokenize(fr_sent):
					s_total[en_word]+=prev_probs[en_word][fr_word]
			for en_word in nltk.word_tokenize(en_sent):
				for fr_word in nltk.word_tokenize(fr_sent):
					count[en_word][fr_word]+=prev_probs[en_word][fr_word]/s_total[en_word]
					total[fr_word]+=prev_probs[en_word][fr_word]/s_total[en_word]
		for fr_word in data[foreign_lang]:
			for en_word in data[native_lang]:
				current_probs[en_word][fr_word]=count[en_word][fr_word]/total[fr_word]
		error=distance(prev_probs,current_probs)
		pp.pprint(prev_probs)
		pp.pprint(current_probs)
		prev_probs=deepcopy(current_probs)
	return current_probs

def main():
	f=open("data1.json",'r')
	corpus=json.load(f)
	f.close()
	# pp.pprint(corpus)
		# pp.pprint(nltk.word_tokenize(dict[native_lang]))
		# pp.pprint(nltk.word_tokenize(dict[foreign_lang])) 
	# print(data)
	current_probs=train(corpus,0.00001)

if __name__ == '__main__':
	main()
