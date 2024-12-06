import numpy as np
from tfidf import Tfidf_Vectorizer_Custom
from scipy.sparse.linalg import svds
from nltk.tokenize import sent_tokenize



def sort_eigen(eigenvalues,eigenvectors,term_list):
    eigen_idx = np.argsort(eigenvalues)[::-1] # Returns list of sorted indexes in decending order
    eigenvectors = eigenvectors[:,eigen_idx] # Selects all rows according to specified indices
    term_list = np.array(term_list)[eigen_idx] # Sort he term list according to the sorted eigen idx
    #eigenvalues = sorted(eigenvalues)[::-1] # Sorted eigen values
    eigenvalues = eigenvalues[eigen_idx]      # Align eigenvalues with sorted indices

    return eigenvalues,eigenvectors,term_list

# A is TFIDF matrix and k is no of topic 
def truncated_svd(A,k,term_list):
    AT_A = np.dot(A.T,A)  # Covariance matrix = Transpose(A) * A
    
    # Compute eigen values and associated eigen vectors 
    # eigenvalues, eigenvectors = np.linalg.eigh(AT_A)   # Returns already sorted eigen vectors

    # print("Eigen values: ",eigenvalues)
    # print("Eigen vectors: ",eigenvectors)

    eigenvalues,eigenvectors = np.linalg.eig(AT_A)

    print("Term list before sorting: ",term_list)
    print("Eigen values before sorting: ",eigenvalues)
    print("Eigen vectors before sorting: ",eigenvectors)

    print("-----------------------------------------------------")
    eigenvalues, eigenvectors,term_list = sort_eigen(eigenvalues,eigenvectors,term_list)
    print("-----------------------------------------------------")

    print("Term list after sorting: ",term_list)
    print("Eigen values after sorting: ",eigenvalues)
    print("Eigen vectors after sorting: ",eigenvectors)


    # print("Sorted eigen values: ",eigenvalues)
    # print("Sorted eigen vectors: ",eigenvectors)

    # Fixing numerical instability
    for i in range(len(eigenvalues)):
        if eigenvalues[i] < 0:
            eigenvalues[i] = 0

    # Select top k eigen values and vectors
    top_k_eigenvalues = eigenvalues[:k]
    top_k_eigenvectors = eigenvectors[:,0:k]

    print("--------------------------------")
    print("Top_k eigen values: ",top_k_eigenvalues)
    print("Top_k eigen vectors: ",top_k_eigenvectors)

    # Right singluar matrix V_k 
    V_k = top_k_eigenvectors

    # Sigma_k
    Sigma_k = np.zeros((k,k)) # Initialize empty k*k matrix
    # Fill the diagonal with eigen vectors ie [[sig,0,0],[0,sig,0],[0,0,sig]]
    for i in range(k):
        for j in range(k):
            if i == j:
                Sigma_k[i][j] = np.sqrt(top_k_eigenvalues[i])
                
    # Compute Left singular matrix U_k = A * V_k * Sigma_k^(-1)
    U_k = np.dot(np.dot(A,V_k),np.linalg.inv(Sigma_k))

    # Truncated matrix
    A_k = np.dot(np.dot(U_k,Sigma_k),V_k.T)
    return A_k,U_k,Sigma_k,V_k,term_list


def get_stronger_topic(Sigma_k):
    max_i = -1
    max_j = -1
    maxval = 0
    secondmax_i = -1
    secondmax_j = -1
    secondmaxval = 0

    rows, columns = Sigma_k.shape
    for i in range(rows):
        for j in range(columns):
            if i == j:
                if Sigma_k[i][j] > maxval:
                    # If larger than max

                    #Update second largest
                    secondmax_i = max_i
                    secondmax_j = max_j
                    secondmaxval = maxval
                    # Update the largest
                    max_i = i
                    max_j = j
                    maxval = Sigma_k[i][j]
                else:
                    # Check if larger than secondmax
                    if Sigma_k[i][j] > secondmaxval:
                        #Update second only
                        secondmax_i = i
                        secondmax_j = j
                        secondmaxval = Sigma_k[i][j]

    # print("Largest value: ",Sigma_k[max_i][max_j])
    # print("Largest index (i,j): ",(max_i,max_j))
    # print("Second largest value: ",Sigma_k[secondmax_i][secondmax_j])
    # print("Second Largest index (i,j): ",(secondmax_i,secondmax_j))

    return max_i,max_j,maxval,secondmax_i,secondmax_j,secondmaxval


def get_term_per_topic(V_k):
    # print("VK : ",V_k)

    rows,_= V_k.shape

    max_term_topic_list = []
    
    for i in range(rows):
        # Get max abs value index of a row
        max_column = np.argmax(np.abs(V_k[i]))
        max_value = V_k[i,max_column]
        max_term_topic_list.append([i,max_column,abs(max_value)])

    # print(max_term_topic_list)
    return max_term_topic_list

# Return term and its v_k value
def get_name_val(max_term_topic_list,term_list,topic_index):
    topic_term = max_term_topic_list[topic_index]
    term_val = topic_term[2]
    term_name = term_list[topic_term[1]]
    return term_name,term_val

    


def topic_extraction(story):
    doc_corpus = sent_tokenize(story)

    # d1="cars are mango than apple"
    # d2="apple is cheaper cars mango"
    # d3="apple is cheaper mango"
    # d1 = "The sky was painted in hues of orange and purple as Alex stepped off the train. The small town of Rivermoor greeted him with the scent of fresh rain and the soft hum of distant conversations. He had come here in search of answers, clutching an old photograph of a house by the river."
    # d2 = "As Alex wandered through the cobblestone streets, he noticed how time seemed to stand still in Rivermoor. The locals spoke of the house in hushed tones, calling it the keeper of secrets. Each step brought him closer, and soon, he stood before the weathered gates, overgrown with ivy and shadowed by towering oak trees."
    # d3 = "Inside the house, dust swirled in the beams of sunlight filtering through cracked windows. Alex's heart raced as he uncovered an ornate box hidden beneath the floorboards. Opening it, he found a collection of letters, each one revealing fragments of a forgotten story that connected him to Rivermoor’s enigmatic past."

    # doc_corpus=[d1,d2,d3]
    vec = Tfidf_Vectorizer_Custom(stop_words=True)
    tifdfmatrix,term_list = vec.fit_transform(corpus=doc_corpus,topic_modeling=True)

    # print("TFIDF matrix: ",tifdfmatrix)


    #print("Initial ter:",term_list)
    A_k,U_k,Sigma_k,V_k,term_list = truncated_svd(tifdfmatrix,2,term_list)
    # print("Term_list: ",term_list)
    # print("V_K: ",V_k.T)
    # print("------------------")
    # print("U_K: ",U_k)
    # print("------------------")
    # print("Sigma_K: ",Sigma_k)
    # print("------------------")
    # print("A_K : ",A_k)


    # print("")
    # print("---------------------------SVD LIBRARY-------------------------")
    # print("Term list: ",term_list)

    # u, s, v = svds(tifdfmatrix, k=2)

    # print("U: ",u)
    # print("S: ",s)
    # print("V_lib: ",v)


    # Compute top 2 term for each topic for the corpus

    # Compute the most strongest topic from s
    max_i,max_j,maxval,secondmax_i,secondmax_j,secondmaxval = get_stronger_topic(Sigma_k)

    # Get strongest terms for each topic
    max_term_topic_list = get_term_per_topic(V_k.T)

    term_name_one,term_name_val_one = get_name_val(max_term_topic_list,term_list,max_i)
    term_name_two,term_name_val_two = get_name_val(max_term_topic_list,term_list,secondmax_i)

    return term_name_one,term_name_val_one,term_name_two,term_name_val_two
