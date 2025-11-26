import pandas as pd

from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec

#from myapp.search.objects import Document

# Import our functions
from myapp.our_functions.indexing_evaluation import create_index_tfidf, preprocess_text, search_tf_idf
from myapp.our_functions.ranking_filtering import create_index_w2v, search_BM25, search_w2v, RankingG23


""" def dummy_search(corpus: dict, search_id, num_results=20):
    
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res """


class SearchEngine:
    """Class that implements the search engine logic"""

    def search(self, search_query, search_id, og_corpus, corpus, search_method='tfidf'):
        # TODO: Apply search_id...
        print(f"Search query: {search_query} | Method: {search_method}")

        results = []

        # --- OUR SEARCH ENGINE --- #
        # Corpus is already imported as a pandas DataFrame and cleaned in the process of loading.
        print("CORPUS:", corpus.head())

        USED_TEXT_COLUMNS = ['title', 'description', 'brand', 'category', 'sub_category', 'seller']

        # Prepare tokenized corpus for methods that need it (BM25, Word2Vec)
        if search_method in ['bm25', 'word2vec']:
            tokenized_corpus = corpus.apply(lambda x: ' '.join([str(v) for v in x[USED_TEXT_COLUMNS].values if pd.notnull(v)]).split(), axis=1).to_list()

        # Select search method
        if search_method == 'tfidf':
            inverted_index, tf_index, df_index, idf_index = create_index_tfidf(data=corpus, columns=USED_TEXT_COLUMNS)
            results = search_tf_idf(query=search_query, index=inverted_index, tf=tf_index, idf=idf_index, k=250)
        elif search_method == 'bm25':
            BM25 = BM25Okapi(tokenized_corpus)
            results = search_BM25(bm25=BM25, data=corpus, query=search_query, k=250)
        elif search_method == 'word2vec':
            w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4, seed=42)
            w2v_doc_vectors = create_index_w2v(model=w2v_model, data=corpus, columns=USED_TEXT_COLUMNS)
            results = search_w2v(model=w2v_model, doc_vectors=w2v_doc_vectors, query=search_query, k=250)
        elif search_method == 'custom':
            ranker = RankingG23(df=corpus, columns=USED_TEXT_COLUMNS, method='tfidf')
            results = ranker.search(query=search_query, k=250)
        else:
            # Default to TF-IDF
            inverted_index, tf_index, df_index, idf_index = create_index_tfidf(data=corpus, columns=USED_TEXT_COLUMNS)
            results = search_tf_idf(query=search_query, index=inverted_index, tf=tf_index, idf=idf_index, k=250)
        
        # --- End of OUR SEARCH ENGINE --- #

        # --- Format results for the template --- #
        formatted_results = []
        for score, doc_id in results:
            # Find the document in the ORIGINAL corpus DataFrame (og_corpus)
            # We use og_corpus to display the original data (uncleaned)
            matching_rows = og_corpus.loc[og_corpus['pid'] == doc_id]

            if not matching_rows.empty:
                doc_data = matching_rows.iloc[0].to_dict()

                # Add/Transform fields expected by the template
                doc_data['ranking'] = score
                # Keep original URL
                doc_data['original_url'] = doc_data.get('url', '#')
                # Link to the internal details page
                doc_data['url'] = f"/doc_details?pid={doc_id}"
                # Map 'brand' to 'doc_date' as a placeholder for metadata display
                doc_data['doc_date'] = doc_data.get('brand', 'N/A')

                # Use actual_price from the dataset as the previous_price (crossed out)
                doc_data['previous_price'] = doc_data.get('actual_price')

                formatted_results.append(doc_data)

        return formatted_results
