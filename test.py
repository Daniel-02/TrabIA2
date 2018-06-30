import nltk
import pickle
from nltk.stem import WordNetLemmatizer

text = "Falta muito pouco pra excelente. Sugiro melhorar a parte de notícias e incluir informações sobre o BID, por exemplo."
tagger = pickle.load(open("tagger.pkl", 'rb'))
portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
stemmer = nltk.stem.RSLPStemmer()

sentences = portuguese_sent_tokenizer.tokenize(text)
tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

tags = [tagger.tag(token) for token in tokens]
print(tags)

stemmed_sentences = []
for tags_s in tags:
    stemmed_tokens = []
    for token in tags_s:
        stemmed_tokens.append([stemmer.stem(token[0]), token[1]])
        print(stemmed_tokens)
    stemmed_sentences.append(stemmed_tokens)
print(stemmed_sentences)