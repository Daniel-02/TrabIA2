import nltk
import pickle
from nltk.stem import WordNetLemmatizer

text = "Falta muito pouco pra excelente. Sugiro melhorar a parte de notícias e incluir informações sobre o BID, por exemplo."

# Tokenização das sentenças
portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
sentences = portuguese_sent_tokenizer.tokenize(text)
tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

# POS tagging
tagger = pickle.load(open("tagger.pkl", 'rb'))
tags = [tagger.tag(token) for token in tokens]
print(tags)


# Stemming usando RSLPStemmer
stemmer = nltk.stem.RSLPStemmer()
stemmed_sentences = []
for tags_s in tags:
    stemmed_tokens = []
    for token in tags_s:
        stemmed_tokens.append([stemmer.stem(token[0]), token[1]])
    stemmed_sentences.append(stemmed_tokens)
print(stemmed_sentences)