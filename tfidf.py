from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import stop_words
import math
from scipy.sparse import csr_matrix
import pandas as pd
import re


pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns

class Tfidf_Vectorizer_Custom:
    def __init__(self,stop_words=False) -> None:
        self.stop_words = stop_words # Use stop words or not
        self.pattern = r"(?u)\b\w\w+\b" # regex


    # Returns filtered doc list
    def get_filtered_doc(self,doc):
        doc_arr = []
        for word in doc.split():
            if word not in stop_words:
                if re.match(self.pattern,word):
                    doc_arr.append(word)

        doc_arr.sort()
        return " ".join(doc_arr)
        
    # Returns un filtered doc list
    def get_doc(self,doc):
        doc_arr = []
        for word in doc.split():
            if re.match(self.pattern,word):
                doc_arr.append(word)

        doc_arr.sort()
        return " ".join(doc_arr)
    
    # Returns no of terms in a doc
    def get_count(self,term,doc):
        count = 0
        for word in doc.split():
            if word == term:
                count += 1
        return count
        

    def fit_transform(self,corpus,topic_modeling=False):
        n = len(corpus) # No of document
        corpus_list = []
        term_count = {}  # Dictionary containing no of terms in the entire corpus
        term_doc = {}  # term -> [(docno,count)]
        term_set = {} # List of terms
        term_idf = {}
        doc_term_idf = {} # doc_no -> [(term,tfidf)]
        for doc in corpus:
            doc = doc.lower()

            # If stop_words == true
            if(self.stop_words):
                corpus_list.append(self.get_filtered_doc(doc))
            else:
                corpus_list.append(self.get_doc(doc))

        # Count term_count,term_doc,term_set
        for doc_no,doc in enumerate(corpus_list):

            unique_terms = list(dict.fromkeys(doc.split()))
            #unique_terms = set(doc.split())
            for term in unique_terms:
                # Update term count
                term_count[term] = term_count.get(term,0) + 1  
                # Update doc list
                term_occurance = self.get_count(term,doc)
                doc_list = term_doc.get(term,[])
                term_doc[term] = doc_list + [(doc_no,term_occurance)]
                # Update term set
                term_set[term] = term

        # Sort term keys
        term_list = sorted(list(term_set.keys()))
        
        # Find term frequency 
        for term in term_list:
            df = len(term_doc[term])
            idf = math.log((1+n)/(1+df)) + 1
            term_idf[term] = idf

        # Find tfidf per document
        for doc_no,doc in enumerate(corpus_list):
            unique_terms = list(dict.fromkeys(doc.split()))
            #unique_terms = set(doc.split())
            for term in unique_terms:
                term_doc_list = term_doc.get(term,[])
                tf = 0
                idf = term_idf.get(term)
                for tp in term_doc_list:
                    if(tp[0] == doc_no):
                        tf = tp[1] # Count
                tfidf = idf * tf
                doc_term_idf_list = doc_term_idf.get(doc_no,[])
                doc_term_idf[doc_no] = doc_term_idf_list + [(term,tfidf)]

        # Map term to index
        # Create a mapping from terms to column indices
        term_to_index = {term: i for i, term in enumerate(term_list)}
        # Prepare lists to build the sparse matrix
        data = []
        row_indices = []
        col_indices = []

        # Populate the lists with TF-IDF scores
        for doc_id, term_tfidf_list in doc_term_idf.items():
            for term, tfidf in term_tfidf_list:
                if term in term_to_index:  # Ensure the term is in the term_set
                    term_index = term_to_index[term]
                    row_indices.append(doc_id)
                    col_indices.append(term_index)
                    data.append(tfidf)

        # Create the sparse matrix in CSR format
        sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(doc_term_idf), len(term_list)))

        normalized_data = []
        normalized_row_indices = []
        normalized_col_indices = []
        # Create a temporary CSR matrix for the row
        csr_empty_row = csr_matrix((1, len(term_list)))


        # Normalization: a/root(a^2+b^2+c^2)
        for i in range(sparse_matrix.get_shape()[0]):
            row = sparse_matrix.getrow(i)
            sum_of_square = row.power(2).sum() # sum of squares
            root_sum_of_square = math.sqrt(sum_of_square)
            # Avoid division by zero error
            if sum_of_square > 0:
                final_sparse_row = row/root_sum_of_square
            else:
                final_sparse_row =  csr_empty_row
            # Append normalized row data to the final lists
            normalized_data.extend(final_sparse_row.data)
            normalized_row_indices.extend([i] * len(final_sparse_row.data))
            normalized_col_indices.extend(final_sparse_row.indices)

        # Create the final normalized sparse matrix
        normalized_sparse_matrix = csr_matrix((normalized_data, (normalized_row_indices, normalized_col_indices)), shape=(len(doc_term_idf), len(term_list)))

        # print("Corpus list: ",corpus_list)
        # print("Term_count: ",term_count)
        # print("Term_doc: ",term_doc)
        # print("Term_list: ",term_list)
        # print("Term_idf: ",term_idf)
        # print("Doc_term_idf: ",doc_term_idf)
        # print("---------------------------------")
        # print("Sparse matrix: ")
        # print(sparse_matrix)
        # print("----------------------------------")
        # print("Dense matrix: ")
        # print(sparse_matrix.toarray())
        # print('--------------------------------------------')
        # print("Normalized sparse matrix: ")
        # print(normalized_sparse_matrix)
        # print('---------------------------------')
        # print("Normalized dense array :")
        # print(normalized_sparse_matrix.toarray())

        # Story is short enough for efficient operation over dense matrix
        if(topic_modeling == True):
            return normalized_sparse_matrix.todense(),term_list

        return normalized_sparse_matrix


def main():
    df = pd.read_csv('mf_mini.csv')
    features = df['text']
    print(features)
    print("--------------------------------")


    d1="petrol cars are cheaper than diesel cars"
    d2="diesel is cheaper than petrol"
    d3="petrol is cheaper than cars or cars"

    doc_corpus=[d1,d2,d3]
    print("Document: ",doc_corpus)

    vec = Tfidf_Vectorizer_Custom(stop_words=False)
    custom_matrix = vec.fit_transform(features)
    print("Custom: ",custom_matrix.toarray())

    print("----------------------------------")

    # Lib
    vec_lib = TfidfVectorizer()
    matrix = vec_lib.fit_transform(features)
    print("Lib: ", matrix.toarray())

#main()










