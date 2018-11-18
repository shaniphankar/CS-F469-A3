import nltk
import os
import json
import pprint
import itertools
from math import pow
from math import sqrt
from copy import deepcopy

pp=pprint.PrettyPrinter(indent=4)
with open('langsettings.json') as f:
		settings=json.load(f)
native_lang=settings['source']
foreign_lang=settings['target']
examples=settings['train-data']
iters=settings['iterations']

def key_maxval(d):
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

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
def myinit_zeros(data):
	 return {en_word: {fr_word: 0 for fr_word in data[foreign_lang]} for en_word in data[native_lang]}
def myinit_invert(data):
	 return {fr_word: {en_word: 1/len(data[foreign_lang]) for en_word in data[native_lang]} for fr_word in data[foreign_lang]}

def distance(data1,data2):
	dist=0
	row_iters=data1.keys()
	row_vals=list(data1.values())
	col_iters=row_vals[0].keys()
	for row_num in row_iters:
		for col_num in col_iters:
			dist+=pow((data1[row_num][col_num]-data2[row_num][col_num]),2)
	return sqrt(dist)

def train(en_sents,fr_sents,corpus,epsilon):
	data=get_words_corpus(corpus)
	prev_probs=myinit(data)
	current_probs=myinit_zeros(data)
	converged=False
	# pp.pprint(prev_probs)
	error=100
	i=0
	combos=list(itertools.product(en_sents,fr_sents))
	pp.pprint(combos)
	while(i<iters):
	# while error>epsilon:
		count={en_word:{fr_word:0 for fr_word in data[foreign_lang]} for en_word in data[native_lang]}
		# pp.pprint(count)
		total={fr_word:0 for fr_word in data[foreign_lang]}
		for dict in corpus:
			en_sent=dict[native_lang]
			fr_sent=dict[foreign_lang]
			print(en_sent,fr_sent)
			s_total={}
			for en_word in nltk.word_tokenize(en_sent):
				s_total[en_word]=0
				for fr_word in nltk.word_tokenize(fr_sent):
					s_total[en_word]+=prev_probs[en_word][fr_word]
			for en_word in nltk.word_tokenize(en_sent):
				for fr_word in nltk.word_tokenize(fr_sent):
					count[en_word][fr_word]+=(prev_probs[en_word][fr_word]/s_total[en_word])
					total[fr_word]+=(prev_probs[en_word][fr_word]/s_total[en_word])
		for fr_word in data[foreign_lang]:
			for en_word in data[native_lang]:
				current_probs[en_word][fr_word]=(count[en_word][fr_word]/total[fr_word])
		error=distance(prev_probs,current_probs)
		prev_probs=deepcopy(current_probs)
		i+=1

	# pp.pprint(prev_probs)
	pp.pprint(current_probs)
	pp.pprint(i)
		
	return current_probs

def align(sents,to,current_probs):
	trans_sents=[]
	for sent in sents:
		# print(sent)
		for word in nltk.word_tokenize(sent):
			pp.pprint(word)
			pp.pprint(key_maxval(current_probs[word]))
			# pp.pprint(current_probs[word])
	# return



def main():
	f=open(examples,'r',encoding='utf-8')
	corpus=json.load(f)
	f.close()
	# pp.pprint(corpus)
		# pp.pprint(nltk.word_tokenize(dict[native_lang]))
		# pp.pprint(nltk.word_tokenize(dict[foreign_lang])) 
	# print(data)
	en_sents=[]
	fr_sents=[]
	for dict in corpus:
		en_sents.append(dict[native_lang])
		fr_sents.append(dict[foreign_lang])
	current_probs=train(en_sents,fr_sents,corpus,0.00000001)
	current_probs_invert=myinit_invert(get_words_corpus(corpus));
	for k1,v1 in current_probs.items():
		current_probs_invert
		for v2 in v1.keys():
			current_probs_invert[v2][k1]=v1[v2]
			# print(v1[v2])
	# pp.pprint(current_probs_invert)
	align(fr_sents,"en",current_probs_invert)
if __name__ == '__main__':
	main()
