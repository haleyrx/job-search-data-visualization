# Some functions inspired by: https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925

from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim import corpora, models
# from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import pos_tag
from nltk.stem.porter import *
# Lemmatize with POS Tag
from nltk.corpus import wordnet
import numpy as np

# Define some variables
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english')) 
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

# Retrieve US job listings from glassdoor data
def isUS(location_txt):
    location_txt = str(location_txt)
    if "united states" in str(location_txt).lower():
        return True
    loc_split = location_txt.split(",")
    if len(loc_split)>1:
        loc_state = loc_split[-1].replace(" ", "")
    else:
        return False
    for state in states:
        if state.lower() == loc_state.lower():
            return True
    return False

# Get US state from job posting
def getState(location_txt):
    location_txt = str(location_txt)

    loc_split = location_txt.split(",")
    if len(loc_split) > 1:
        loc_state = loc_split[-1].replace(" ", "")
    else:
        return False

    return loc_state

# Remove HTML tags from job posting
def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    text= str(text)
    return TAG_RE.sub('', text)

def split_into_passages(data, data2, idx):
    list_from_job = data.split('\n')[0]
    list_of_df_from_job = []
    for i in range(len(list_from_job)):
        if list_from_job[i] != '':
            dict_from_job = data2.iloc[idx,:].to_dict()
            dict_from_job["description"] = list_from_job[i]
            list_of_df_from_job.append(pd.DataFrame.from_dict(dict_from_job, orient="index").T)
    return pd.concat(list_of_df_from_job)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Reduce words to root form
def lemmatize_stemming(text, mode="lemma_only"):
    if mode=="lemma_only":
        return WordNetLemmatizer().lemmatize(text, get_wordnet_pos(text))
    elif mode== "lemma_stem":
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, get_wordnet_pos(text)))
    else:
        print("use lemma_only, or lemma_stem")
        raise 

# Create word tokens, perform lemmatization, remove stopwords. and remove word with <=min_char
def preprocess(text, min_char=3, mode="lemma_only", additional_stop_words=None):
    result = []
    if additional_stop_words is not None:
        for new_word in additional_stop_words:
            stop_words.update([new_word])
    for token in simple_preprocess(text):
        if token not in stop_words and len(token) > min_char:
            result.append(lemmatize_stemming(token, mode))
    return result

def BOW_TFIDF(processed_docs, package="scikit", **kw):
    if package == "gensim":
        # Dictionary of the processed documents
        my_dict = Dictionary(processed_docs)
        # Filter: no_below-> out if less than n_below docs, out if in more than no_above % of docs, keep only keep_n top tokesn
        no_below = kw.pop("no_below", 5)
        no_above = kw.pop("no_above", .5)
        keep_n = kw.pop("keep_n", 100000)
        my_dict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        # Create bag of words
        bow_corpus = [my_dict.doc2bow(doc) for doc in processed_docs]
        # Create TF-IDF scores
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        return my_dict, bow_corpus, corpus_tfidf
    elif package=="scikit":
        text = [' '.join([str(elem) for elem in s]) for s in processed_docs]
        no_below = kw.pop("no_below", 5)
        no_above = kw.pop("no_above", .5)
        keep_n = kw.pop("keep_n", None)
        vectorizer = TfidfVectorizer(max_df = no_above, min_df=no_below, max_features=keep_n, **kw)
        X = vectorizer.fit_transform(text)
        return vectorizer, text, X 
    else:
        print("Use scikit or gensim")
        raise

# Handle  processing for new test text
def assign_topic(doc, vectorizer, model, pre_process="full"):
    if pre_process=="full":
        p_doc = [preprocess(text[i]) for i in range(len(doc))]
        p_doc = [' '.join([str(elem) for elem in s]) for s in p_doc]
        X_ = vectorizer.transform(p_doc)
        val = model.transform(X_)
    elif pre_process=="semi":
        doc = [' '.join([str(elem) for elem in s]) for s in doc]
        val = model.transform(vectorizer.transform(doc))
    else:
        val = model.transform(vectorizer.transform(doc))
    return val, np.argmax(val, axis=1)

def runLDA(corpus, n_topics, dictionary=None, verbose=0, package="scikit", **kw):
    if package =="gensim":
        lda_model = models.LdaMulticore(corpus, num_topics=n_topics, id2word=dictionary, **kw)
        if verbose > 0:
            for idx, topic in lda_model.print_topics(-1):
                print('Topic: {} \nWords: {}\n\n'.format(idx, topic))
        return lda_model
    elif package =="scikit" :
        n_top_words = kw.pop("n_top_words", 10)
        vocab = kw.pop("vocab",None)
        lda = LatentDirichletAllocation(n_components=n_topics,   random_state=0, **kw)
        id_topic = lda.fit_transform(corpus)
        
        if verbose > 0:
            assert vocab != None, "vocab argument is not passed"
            print_lda_top_word(n_top_words, vocab, lda)
        return lda
    else:
        print("Use scikit or gensim")
        raise
        
def print_lda_top_word(n_top_words, vocab, lda):
    topic_words = {}
    for topic, comp in enumerate(lda.components_):
        # for the n-dimensional array "arr":
        # argsort() returns a ranked n-dimensional array of arr, call it "ranked_array"
        # which contains the indices that would sort arr in a descending fashion
        # for the ith element in ranked_array, ranked_array[i] represents the index of the
        # element in arr that should be at the ith index in ranked_array
        # ex. arr = [3,7,1,0,3,6]
        # np.argsort(arr) -> [3, 2, 0, 4, 5, 1]
        # word_idx contains the indices in "topic" of the top num_top_words most relevant
        # to a given topic ... it is sorted ascending to begin with and then reversed (desc. now)    
        word_idx = np.argsort(comp)[::-1][:n_top_words]

        # store the words most relevant to the topic
        topic_words[topic] = [vocab[i] for i in word_idx]
    for topic, words in topic_words.items():
        print('Topic: %d' % topic)
        print('  %s' % ', '.join(words))

# Evaluation of model

