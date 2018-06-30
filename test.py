import nltk
import pickle
from nltk.corpus import wordnet as wn

text = "Falta muito pouco pra excelente. Sugiro melhorar a parte de notícias e incluir informações sobre o BID, por exemplo."
tagger = pickle.load(open("tagger.pkl", 'rb'))
portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
stemmer = nltk.stem.RSLPStemmer()

sentences = portuguese_sent_tokenizer.tokenize(text)
tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

stemmed_sentences = []
for token_s in tokens:
    stemmed_tokens = []
    for token in token_s:
        stemmed_tokens.append(stemmer.stem(token))
    stemmed_sentences.append(stemmed_tokens)

tags = [tagger.tag(token) for token in stemmed_sentences]
print(tags)
