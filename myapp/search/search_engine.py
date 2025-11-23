import random
import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec

from myapp.search.objects import Document

# Import our functions
#from myapp.our_functions.data_preparation import *
from myapp.our_functions.indexing_evaluation import create_index_tfidf, preprocess_text, search_tf_idf
from myapp.our_functions.ranking_filtering import create_index_w2v, search_BM25, search_w2v, RankingG23


def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res


class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query, search_id, corpus):
        # TODO: Apply search_id...
        print("Search query:", search_query)

        results = []
        ### You should implement your search logic here:
        #results = dummy_search(corpus, search_id)  # replace with call to search algorithm

        # --- OUR SEARCH ENGINE --- #
        # Corpus is already imported as a pandas DataFrame and cleaned in the process of loading.
        print("CORPUS:", corpus.head())

        USED_TEXT_COLUMNS = ['title', 'description', 'brand', 'category', 'sub_category', 'seller']
        #tokenized_corpus = corpus.apply(lambda x: ' '.join(x[USED_TEXT_COLUMNS].values).split(' '), axis=1).to_list()
        # TODO: dropdown selector to choose
        # TF-IDF
        inverted_index, tf_index, df_index, idf_index = create_index_tfidf(data=corpus, columns=USED_TEXT_COLUMNS)
        results = search_tf_idf(query=search_query, index=inverted_index, tf=tf_index, idf=idf_index)

        # BM25
        """ 
        BM25 = BM25Okapi(tokenized_corpus)
        results = search_BM25(bm25=BM25, data=corpus, query=search_query, k=20)
        """

        # Word2Vec
        """ 
        w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4, seed=42)
        w2v_doc_vectors = create_index_w2v(model=w2v_model, data=corpus, columns=USED_TEXT_COLUMNS)
        results = search_w2v(model=w2v_model, doc_vectors=w2v_doc_vectors, query=search_query)
        """

        # Our own Ranking
        """ 
        ranker = RankingG23(data=corpus, columns=USED_TEXT_COLUMNS, method='tfidf')
        results = ranker.search(query=search_query)
        """
        # --- End of OUR SEARCH ENGINE --- #

        # results = search_in_corpus(search_query)
        return results
