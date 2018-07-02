import nltk
from nltk.collocations import *
import pickle
import pandas as pd
import pathlib

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
csv_title = 'brasileirao'
arq = pd.read_csv('Data/' + csv_title+'.csv', encoding='utf8')
saida = pd.DataFrame(columns=["Review", "Rating"])
# Cria pasta com o nome do arquivo
pathlib.Path('Data/' + csv_title).mkdir(parents=True, exist_ok=True)
#a ideia é ter um arquivo texto puro daonde a gente faria a leitura para fazer a extração de funcionalidades
saidaTxt = open("Data/" + csv_title + "/" + "Reviews.txt", 'wt', encoding='utf8')
saidaTxt.write('\n')

for i in range(0, arq['Content'].__len__()):
    saida.set_value(i, "Review", join_text(arq['Title'][i], arq['Content'][i]))
    saida.set_value(i, "Rating", arq['Rating'][i])
    try:
       saidaTxt.write(arq['Content'][i] + '\n')
    except:
        pass
saida.to_csv('Data/' + csv_title + "/" +'Saida.csv', encoding='utf8')
saidaTxt.close()

tagger = pickle.load(open("Tagger/tagger.pkl", 'rb'))
portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
stopwords = nltk.corpus.stopwords.words('portuguese')
custom_stopwords = ['app', 'por', 'favor', 'conserta', 'conserte', '&', 'quot']
stopwords.extend(custom_stopwords)
stemmer = nltk.stem.RSLPStemmer()

for r in range(0, saida['Review'].__len__()):
# Tokenização das sentenças
    sentences = portuguese_sent_tokenizer.tokenize(saida['Review'][r])
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]

# POS tagging
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
    stopwords_sentences = []
    for tags_sentence in tags_removed_sentences:
        stopword_tokens = []
        for tag in tags_sentence:
            if tag[0] not in stopwords:
                stopword_tokens.append(tag)
        stopwords_sentences.append(stopword_tokens)
    saida.set_value(r, "Review", stopwords_sentences)

# Stemming usando RSLPStemmer
#     stemmed_sentences = []
#     for stopwords_sentence in stopwords_sentences:
#         stemmed_words = []
#         for word in stopwords_sentence:
#             stemmed_words.append(tuple([stemmer.stem(word[0]), word[1]]))
#         stemmed_sentences.append(stemmed_words)
#     saida.set_value(r, "Review", stemmed_sentences)
saida.to_csv('Data/' + csv_title + "/" +'Saida_processada.csv', encoding='utf8')

# Algoritmo de collocation de bigramas: se não me engano precisamos considerar todas as reviews para encontrar bigramas:
# caso não seja isso colocar o trecho abaixo em um for
# exit_txt = open(csv_title + "texto.txt", 'rt', encoding='utf8')
bigram_measures = nltk.collocations.BigramAssocMeasures()
# print(exit_txt.readlines())
# Pega as reviews processadas e junta todas numa lista para extração das features
reviews = []
for i in range(0,saida["Review"].__len__()):
    for j in range(0,saida["Review"][i].__len__()):
        reviews.extend(saida["Review"][i][j])
finder = BigramCollocationFinder.from_words(reviews, window_size=3)
#frequência mínima pra duas palavras serem um bigrama
finder.apply_freq_filter(3)
features = finder.nbest(bigram_measures.pmi, 10000)

pathlib.Path('Data/' + csv_title + "/Features").mkdir(parents=True, exist_ok=True)
def get_sentence(review):
    sentence = ""
    for word in review:
        sentence = sentence + " " + word[0]
    return sentence
for feature in features:
    feature_name = (feature[0][0] + "_" + feature[1][0])
    # if not os.path.exists(csv_title):
    #     os.makedirs(csv_title)
    saidaTxt = open("Data/" + csv_title + "/Features/" + feature_name + ".txt", 'wt', encoding='utf8')
    saidaTxt.write('\n')
    for i in range(0, saida["Review"].__len__()):
        for j in range(0, saida["Review"][i].__len__()):
            finder = BigramCollocationFinder.from_words(saida["Review"][i][j], window_size=3)
            if feature in finder.nbest(bigram_measures.pmi, 10000):
                saidaTxt.write(get_sentence(saida["Review"][i][j]) + '\n')
    saidaTxt.close()



