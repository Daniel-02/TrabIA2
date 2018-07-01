import nltk
from nltk.collocations import *
import pickle
import pandas as pd
import os, sys

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
#a ideia é ter um arquivo texto puro daonde a gente faria a leitura para fazer a extração de funcionalidades
# saidaTxt = open(csv_title + "texto.txt", 'wt', encoding='utf8')

for i in range(0, arq['Content'].__len__()):
    saida.set_value(i, "Review", join_text(arq['Title'][i], arq['Content'][i]))
    saida.set_value(i, "Rating", arq['Rating'][i])
    # try:
    #    saidaTxt.write(arq['Content'][i] + '\n')
    # except:
    #     pass
reviews = saida['Review']
saida.to_csv('Data/' + csv_title+'_saida.csv', encoding='utf8')
# saidaTxt.close()

tagger = pickle.load(open("Tagger/tagger.pkl", 'rb'))
portuguese_sent_tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
stopwords = nltk.corpus.stopwords.words('portuguese')
custom_stopwords = ['app', 'por', 'favor', 'conserta', 'conserte']
stopwords.extend(custom_stopwords)
stemmer = nltk.stem.RSLPStemmer()

for r in range(0, reviews.__len__()):
# Tokenização das sentenças
    sentences = portuguese_sent_tokenizer.tokenize(reviews[r])
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

# Stemming usando RSLPStemmer
    stemmed_sentences = []
    for stopwords_sentence in stopwords_sentences:
        stemmed_words = []
        for word in stopwords_sentence:
            stemmed_words.append(tuple([stemmer.stem(word[0]), word[1]]))
        stemmed_sentences.append(stemmed_words)
    saida.set_value(r, "Review", stemmed_sentences)
saida.to_csv('Data/' + csv_title+'_saida_processada.csv', encoding='utf8')

# Algoritmo de collocation de bigramas: se não me engano precisamos considerar todas as reviews para encontrar bigramas:
# caso não seja isso colocar o trecho abaixo em um for
# exit_txt = open(csv_title + "texto.txt", 'rt', encoding='utf8')
bigram_measures = nltk.collocations.BigramAssocMeasures()
# print(exit_txt.readlines())
# Pega as reviews processadas e junta todas numa lista para extração das features
collocations = []
for i in range(0,saida["Review"].__len__()):
    for j in range(0,saida["Review"][i].__len__()):
        collocations.extend(saida["Review"][i][j])
finder = BigramCollocationFinder.from_words(collocations, window_size=3)
#frequência mínima pra duas palavras serem um bigrama
finder.apply_freq_filter(3)
print(finder.nbest(bigram_measures.pmi, 10000))


import subprocess
import shlex
def RateSentiment(sentiString):
    path = os.path.dirname(sys.argv[0])
    print("java -jar " + path + "/SentiStrength.jar stdin sentidata " + path + "/SentiStrength/SentiStrength_Data/")
    #open a subprocess using shlex to get the command line string into the correct args list format
    #Modify the location of SentiStrength.jar and D:/SentiStrength_Data/ if necessary
    p = subprocess.Popen(shlex.split("java -jar " + path + "/SentiStrength/SentiStrength2.3Free.jar stdin sentidata " + path + "/SentiStrength/SentiStrength_Data/"),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    #communicate via stdin the string to be rated. Note that all spaces are replaced with +
    #Can't send string in Python 3, must send bytes
    #communicate via stdin the string to be rated. Note that all spaces are replaced with +
    stdout_text, stderr_text = p.communicate(bytes(sentiString.replace(" ","+"), 'utf-8'))
    #remove the tab spacing between the positive and negative ratings. e.g. 1    -5 -> 1-5
    stdout_text = stderr_text.decode("utf-8").rstrip().replace("\t","")
    return stdout_text
#An example to illustrate calling the process.
print(RateSentiment("Muito bom... o ruim que demora a atualizar a classificação. Mas é o melhor pra acompanhar o Brasileirão"))


