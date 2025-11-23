# File created using copilot, original code is from the part3 notebook

import os
import sys

import pandas as pd
import numpy as np
import numpy.linalg as la

from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec

# --- Import from Part 2 --- #
# Add parent directory to path to allow imports from sibling directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # project_progress
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from indexing_evaluation import (
        create_index_tfidf,
        search_tf_idf,
        preprocess_text
    )
except ImportError:
    # Fallback if running from root or different structure
    try:
        from myapp.our_functions.indexing_evaluation import (
            create_index_tfidf,
            search_tf_idf,
            preprocess_text
        )
    except ImportError:
        print("Warning: Could not import functions from part_2.indexing_evaluation. Ensure the path is correct.")

# --- End of Import from Part 2 --- #




# Global constants
USED_TEXT_COLUMNS = ['title', 'description', 'brand', 'category', 'sub_category', 'seller']


# --- Helper Functions --- #

def print_top_k_results(ranked_documents: list, k: int = 20):
    """
    Print the top K ranked documents in a formatted table.
    """
    print("=" * 42)
    print(f"{'Rank':<6} | {'Document ID':<20} | {'Score':>10}")
    print("=" * 42)

    for i, (score, doc) in enumerate(ranked_documents[:k], 1):
        print(f"{i:<6} | {doc:<20} | {score:>10.3f}")

    print("=" * 42)


def get_top_k_results(data: pd.DataFrame,
                      ranked_documents: list,
                      k: int | str = 'all',
                      text_columns: list[str] = USED_TEXT_COLUMNS,
                      num_columns: list[str] = []) -> pd.DataFrame:
    """
    Retrieve the top K documents with their metadata from the dataframe.

    Args:
        data (pd.DataFrame): The dataset.
        ranked_documents (list): List of [score, pid] tuples.
        k (int | str): Number of documents to retrieve or 'all'.
        text_columns (list): Text columns to include in result.
        num_columns (list): Numeric columns to include in result.

    Returns:
        pd.DataFrame: Dataframe with top K results.
    """
    ranked_documents_df = pd.DataFrame(ranked_documents, columns=['score', 'pid'])
    
    # Ensure sorting
    ranked_documents_df = ranked_documents_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Merge with original data
    cols_to_fetch = ['pid'] + text_columns + num_columns
    merged = ranked_documents_df.merge(data[cols_to_fetch], on='pid', how='left')
    
    if k == 'all':
        return merged
    
    return merged[:int(k)]

# --- End of Helper Functions --- #


# --- BM25 Ranking --- #

def search_BM25(bm25: BM25Okapi, data: pd.DataFrame, query: str, k: int = 10) -> list:
    """
    Perform search using BM25 algorithm.

    Args:
        bm25 (BM25Okapi): Pre-fitted BM25 object.
        data (pd.DataFrame): The dataset.
        query (str): Search query.
        k (int): Number of top results to return.

    Returns:
        list: List of (score, doc_id) tuples.
    """
    # Preprocess query
    query_tokens = preprocess_text(query)

    # Get scores
    scores = np.array(bm25.get_scores(query_tokens))

    # Efficiently get top k indices
    if k >= len(scores):
        idx = np.arange(len(scores))
    else:
        idx = np.argpartition(scores, -k)[-k:]

    # If all scores are 0
    if np.sum(scores[idx]) == 0:
        return []
    
    # Sort top k indices by score descending
    top_indices = idx[np.argsort(-scores[idx])]

    # Build result list
    result = []
    for i in top_indices:
        pid = data.iloc[i]['pid'] if 'pid' in data.columns else i
        result.append((scores[i], pid))

    return result

# --- End of BM25 Ranking --- #


# --- Custom Ranking Class --- #
class RankingG23:
    """
    Custom ranking engine that combines text relevance (TF-IDF or BM25)
    with numerical features (rating, discount, price, availability).
    """
    def __init__(self, df: pd.DataFrame, columns: list = USED_TEXT_COLUMNS, method: str = 'tfidf'):
        self.df = df
        self.columns = columns
        self.method = method
        self.ranked_docs = None
        
        if method == 'tfidf':
            self.index, self.tf, _, self.idf = create_index_tfidf(self.df, self.columns)
        elif method == 'bm25':
            # Prepare corpus for BM25
            corpus = self.df.apply(lambda x: ' '.join(str(v) for v in x[self.columns].values).split(), axis=1).to_list()
            self.bm25 = BM25Okapi(corpus)
        else:
            raise ValueError("Method must be 'tfidf' or 'bm25'")

    def search(self, query: str) -> list:
        """
        Search and rank documents using the combined score.
        """
        if self.method == 'tfidf':
            ranked_docs = search_tf_idf(query, self.index, self.tf, self.idf)
        elif self.method == 'bm25':
            ranked_docs = search_BM25(self.bm25, self.df, query, k=len(self.df))
        else:
            return []

        if not ranked_docs:
            return []
        
        results = []
        
        # Extract scores for normalization
        text_scores = [score for score, pid in ranked_docs]
        max_text_score = max(text_scores) if text_scores else 1

        # Create a dictionary for fast lookup of ranked docs
        ranked_dict = {pid: score for score, pid in ranked_docs}
        
        # Optimization: Bulk fetch relevant rows
        pids = [pid for score, pid in ranked_docs]
        relevant_rows = self.df[self.df['pid'].isin(pids)].set_index('pid')

        for pid, raw_text_score in ranked_dict.items():
            if pid not in relevant_rows.index:
                continue
                
            row = relevant_rows.loc[pid]

            # Normalize text score
            text_score = raw_text_score / max_text_score

            # Numerical features
            # Handle potential NaN values safely
            avg_rating = row.get('average_rating', 0)
            avg_rating = avg_rating if pd.notnull(avg_rating) else 0
            rating_score = avg_rating / 5

            discount = row.get('discount', 0)
            discount = discount if pd.notnull(discount) else 0
            discount_score = discount / 100

            out_of_stock = row.get('out_of_stock', 1)
            availability_score = 1 if out_of_stock == 0 else 0

            selling_price = row.get('selling_price', 0)
            selling_price = selling_price if pd.notnull(selling_price) else 0
            price_score = 1 - (1 + np.log1p(selling_price)) if selling_price > 0 else 0

            # Combine with weights
            combined_score = (0.4 * text_score +
                            0.3 * rating_score +
                            0.2 * discount_score +
                            0.05 * availability_score +
                            0.05 * price_score)
            
            results.append((combined_score, pid))

        results.sort(reverse=True)
        self.ranked_docs = results

        return results
    
    def sort(self, criterion: list[str] = ['average_rating'], ascending: list[bool] = [False]) -> pd.DataFrame:
        """
        Sort the previously retrieved results by specific criteria.
        """
        if self.ranked_docs is None:
            return pd.DataFrame()
        
        # Filter df to only the documents retrieved
        pids = [pid for _, pid in self.ranked_docs]
        df_filtered = self.df[self.df['pid'].isin(pids)].copy()
        
        # Sort by chosen criterion
        df_filtered.sort_values(by=criterion, ascending=ascending, inplace=True)
                
        return df_filtered

# --- End of Custom Ranking Class --- #


# --- Word2Vec Ranking --- #

def text_vector_w2v(model: Word2Vec, tokens: list) -> np.ndarray:
    """
    Computes the vector representation of a text by averaging the word vectors of its tokens.
    """
    # Select only words present in vocabulary
    vectors = [model.wv[word] for word in tokens if word in model.wv]

    if not vectors: # If no words in the text are in the model vocabulary
        return np.zeros(model.vector_size)
    
    # Average all word vectors to get text vector representation
    return np.mean(vectors, axis=0)

def create_index_w2v(model: Word2Vec, data: pd.DataFrame, columns: list = USED_TEXT_COLUMNS) -> dict:
    """
    Creates an index of document vectors for all documents in the dataset.
    Returns: dict(pid -> document vector)
    """
    doc_vectors = {}
    for _, row in data.iterrows():
        pid = row['pid']
        # Tokenize and preprocess text from specified columns
        text_content = ' '.join(str(v) for v in row[columns].values if pd.notnull(v))
        tokens = preprocess_text(text_content)
        # Compute document vector
        doc_vectors[pid] = text_vector_w2v(model, tokens)

    return doc_vectors

def rank_documents_w2v(query_terms: list, model: Word2Vec, doc_vectors: dict) -> list:
    """
    Ranks documents based on cosine similarity between query vector and document vectors.
    Returns: List of (similarity score, pid) tuples.
    """
    # Convert query terms to vector
    query_vector = text_vector_w2v(model, query_terms)
    query_norm = la.norm(query_vector)
    
    doc_scores = []

    for pid, doc_vector in doc_vectors.items():
        doc_norm = la.norm(doc_vector)
        
        if doc_norm == 0 or query_norm == 0: # Avoid division by zero
            similarity = 0.0
        else:
            # Compute cosine similarity
            similarity = np.dot(query_vector, doc_vector) / (query_norm * doc_norm)
            
        doc_scores.append((similarity, pid))
    
    doc_scores.sort(reverse=True)

    return doc_scores

def search_w2v(model: Word2Vec, doc_vectors: dict, query: str) -> list:
    """
    Performs a search by ranking documents based on their similarity to the query using Word2Vec.
    """
    # Tokenize and preprocess the query
    query_terms = preprocess_text(query)
    # Rank documents based on similarity to the query
    return rank_documents_w2v(query_terms, model, doc_vectors)

# --- End of Word2Vec Ranking --- #
