import nltk; nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from nltk.corpus import stopwords

class Topic_modeling():
    
    def __init__(self, path, name):
        self.name = name
        self.path = path
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.data = None
        self.import_data()
        self.preprocess()
    
    def import_data(self):
        data = pd.read_excel(self.path)
        data.columns = ['sentence']
        data = data.sentence.values.tolist()
        # Delete repeated quotations
        data = [list(set(text.splitlines())) for text in data]
        self.data = data
    
    def word_correction(self):
        changes = [('ঞ', 'ti'), ('Ć', 'fi'), ('Õ', 'fi'), ('ß', 'fi'),('ø', 'fi'), ('Ą', 'ff'), ('Ö', 'fl'), ('ϐ', 'f'), ('', 'tt'), ('profle', 'profile')]
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                for change in changes:
                    self.data[i][j] = self.data[i][j].replace(change[0],change[1])

    def data_to_words(self):
        for sentence in self.data:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    
    def remove_stopwords(self, texts):
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu'])
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(self,texts):
        data_words = list(self.data_to_words())
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(self,texts):
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(self,texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def preprocess(self):
        self.word_correction()
        data_words = self.data
        # Remove Stop Words
        data_words_nostops =  self.remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = self.make_bigrams(data_words_nostops)
    

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        
        # Create Dictionary (key, word)
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency (Bag of Words)
        corpus = [id2word.doc2bow(text) for text in texts]
        self.corpus = corpus
        self.id2word = id2word
        self.data_lemmatized = data_lemmatized
    
    def modeling(self,limit=20, start=2, step=2, num_topics =None):
        '''Can be initialized with num of topics or let grid search run first to identify the best number of topics for LDA'''
        if num_topics is None:
            num_topics = self.compute_coherence_values(limit=limit, start=start, step=step)
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                   id2word=self.id2word,
                                                   num_topics=num_topics, 
                                                   random_state=100,
                                                   alpha='auto',
                                                   per_word_topics=True)
        self.lda_model = lda_model        
        
    def save_html_lda(self):
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.id2word)
        pyLDAvis.save_html(vis,self.name+'.html')
        
    def compute_coherence_values(self,limit,start,step):
        coherence_values = []
        for num_topics in range(start, limit, step):
            print('Nº topics: '+str(num_topics))
            model=gensim.models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.id2word, num_topics=num_topics)
            coherencemodel = CoherenceModel(model=model, texts=self.data_lemmatized, dictionary=self.id2word, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
            print('Score: '+str(coherence_values[-1]))

        # Show graph
        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

        best_n_topics = (coherence_values.index(max(coherence_values))+start-1)*step
        print('The best model has {} topics'.format(best_n_topics))
        return best_n_topics