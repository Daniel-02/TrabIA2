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
tags_sentences = [tagger.tag(token) for token in tokens]

# Remoção de stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')
custom_stopwords = ['app', 'por', 'favor', 'conserta', 'conserte']
stopwords.extend(custom_stopwords)
stopwords_sentences = []
for tags_sentence in tags_sentences:
    stopword_tokens = []
    for tag in tags_sentence:
        if tag[0] not in stopwords:
            stopword_tokens.append(tag)
    stopwords_sentences.append(stopword_tokens)

# Stemming usando RSLPStemmer
stemmer = nltk.stem.RSLPStemmer()
stemmed_sentences = []
for stopwords_sentence in stopwords_sentences:
    stemmed_words = []
    for word in stopwords_sentence:
        stemmed_words.append([stemmer.stem(word[0]), word[1]])
    stemmed_sentences.append(stemmed_words)
print(stemmed_sentences)