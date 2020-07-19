import os
from gensim.models.wrappers import LdaMallet
import gensim
import spacy
import pandas as pd
import string
import re
import emoticons
from pprint import pprint
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


os.environ.update({'MALLET_HOME':r'/Users/samira/Desktop/complex/Complex_Code/mallet-2.0.8'})
def remove_punctuation(from_text):
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in from_text]
    return stripped

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



read_data = pd.read_csv("First_Data_set.csv",  sep=',')
Data_set =pd.DataFrame(read_data)
df = Data_set[['text']]
data = df.values.tolist()
print (data)
data_1 = [remove_punctuation(sent) for sent in data]
print(data_1)
data_2 = [strip_all_entities (str(sent)) for sent in data_1]
print(data_2)
data_3 = [ emoticons.str2emoji(str(sent)) for sent in data_2]
print(data_3)
data_4 = [ re.sub(r'^https?:\/\/.*[\r\n]*', '', sent, flags=re.MULTILINE) for sent in data_3]
print (data_4)

# Remove new line characters
data_4 = [ re.sub(r'\s+', ' ', sent) for sent in data_3]
print (data_4)
# Remove distracting single quotes
data_4 = [ re.sub(r"\'", ' ', sent) for sent in data_3]
print (data_4)
data_5 = [ re.sub(r'\S*@\S*\s?', '', sent) for sent in data_4]
print (data_5)
data_6 = [ re.sub(r'http\S+', '', sent) for sent in data_5]
print (data_6)
data_7 = list(sent_to_words(data_6))
print(data_7)

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_7, min_count=5, threshold=70) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_7], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_7[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data_8 = remove_stopwords(data_7)

print(data_8)

data_9 = make_bigrams(data_8)

print(data_9)

nlp = spacy.load('en', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_9, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])
#id2word[0]

Test = print ([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]) 


mallet_path = '/Users/samira/Desktop/complex/Complex_Code/mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics= 1, id2word=id2word)
pprint(ldamallet.show_topics(formatted=False))
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

#mixture = [dict(ldamallet.show_topics[x]) for x in corpus]
#pd.DataFrame(mixture).to_csv("topic_mixture.csv")

top_words_per_topic = []
for t in range(ldamallet.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in ldamallet.show_topic(t, topn = 1)])

pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_word_1.csv")