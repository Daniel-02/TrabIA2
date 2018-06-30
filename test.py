import nltk
import pickle

text = "Socorro! Cai da escada."
tagger = pickle.load(open("tagger.pkl", 'rb'))
portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
sentences = portuguese_sent_tokenizer.tokenize(text)

tags = [tagger.tag(nltk.word_tokenize(sentence)) for sentence in sentences]

print(tags)
