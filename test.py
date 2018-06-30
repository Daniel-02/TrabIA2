import nltk
import pickle
import pandas as pd

def join_text(text1, text2):
    if type(text1) is str:
        if type(text2) is str:
            return text1 + '.' + text2
        else:
            return text1
    else:
        if type(text2) is str:
            return text2
    return ''

# Inicialização do csv
csv_title = 'brasileirao.csv'
arq = pd.read_csv(csv_title, encoding='utf8')
reviews_title = arq['Title']
reviews_content = arq['Content']
reviews_rating = arq['Rating']
saida = pd.DataFrame(columns=["Review", "Rating"])
for i in range(0, reviews_content.__len__()):
    saida.set_value(i, "Review", join_text(reviews_title[i], reviews_content[i]))
    saida.set_value(i, "Rating", reviews_rating[i])
reviews = saida['Review']
saida.to_csv(csv_title+'_saida.csv', encoding='utf8')

for r in range(0, reviews.__len__()):
# review = "Falta muito pouco pra excelente. Sugiro melhorar a parte de notícias e incluir informações sobre o BID, por exemplo."

# Tokenização das sentenças
    portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
    sentences = portuguese_sent_tokenizer.tokenize(reviews[r])
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

# POS tagging
    tagger = pickle.load(open("tagger.pkl", 'rb'))
    tags_sentences = [tagger.tag(token) for token in tokens]

# Seleção dos ADJ, VERB, NOUN
    tags_removed_sentences = []
    for tags_sentence in tags_sentences:
        tags_tokens = []
        for tag in tags_sentence:
            if tag[1] == 'VERB' or tag[1] == 'ADJ' or tag[1] == 'NOUN':
                tags_tokens.append(tag)
        tags_removed_sentences.append(tags_tokens)

    # Remoção de stopwords
    stopwords = nltk.corpus.stopwords.words('portuguese')
    custom_stopwords = ['app', 'por', 'favor', 'conserta', 'conserte']
    stopwords.extend(custom_stopwords)
    stopwords_sentences = []
    for tags_sentence in tags_removed_sentences:
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
    saida.set_value(r, "Review", stemmed_sentences)
saida.to_csv(csv_title+'_saida_processada.csv', encoding='utf8')
