import nltk
import os
import json
import pprint
pp=pprint.PrettyPrinter(indent=4)	
def myinit(data):
	 return {en_word: {fr_word: 1/len(data['en']) for fr_word in data['fr']} for en_word in data['en']}

def train(data,epsilon):
	prev_probs=myinit(data)
	# pp.pprint(myinit(data))
	error=100
	i=0
	# while error>epsilon:


def main():
	f=open("data1.json",'r')
	corpus=json.load(f)
	f.close()
	# pp.pprint(corpus)
	data={}
	data['en']=[]
	data['fr']=[]
	for dict in corpus:
		for en_word in nltk.word_tokenize(dict['en']):
			data['en'].append(en_word)
		for fr_word in nltk.word_tokenize(dict['fr']):
			data['fr'].append(fr_word)
		# pp.pprint(nltk.word_tokenize(dict['en']))
		# pp.pprint(nltk.word_tokenize(dict['fr'])) 
	# print(data)
	train(data,0.001)
if __name__ == '__main__':
	main()
