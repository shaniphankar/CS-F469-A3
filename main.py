import nltk
import os
import json
import pprint

def main():
	pp=pprint.PrettyPrinter(indent=4)
	f=open("data1.json",'r')
	corpus=json.load(f)
	f.close()
	# pp.pprint(corpus)
	data={}
	data['en']=[]
	data['fr']=[]
	for dict in corpus:
		data['en'].append(nltk.word_tokenize(dict['en']))
		data['fr'].append(nltk.word_tokenize(dict['fr']))
		# pp.pprint(nltk.word_tokenize(dict['en']))
		# pp.pprint(nltk.word_tokenize(dict['fr']))

if __name__ == '__main__':
	main()
