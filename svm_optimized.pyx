import numpy as np
import pandas as pd
from random import randrange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse
import joblib
cimport numpy as np
from libc.math cimport exp
from scipy.sparse import csr_matrix,hstack  # for Python code access
cimport scipy.sparse  # for Cython type access
import multiprocessing
import cupyx.scipy.sparse as cusparse
import cupy as cp
from joblib import Parallel, delayed
import cupyx.scipy.sparse.linalg as lnm


pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
# Set print options to show the full array and precision for floating point numbers
np.set_printoptions(precision=6, threshold=np.inf)

cdef float rbf_kernel_plain(x1, x2, float gamma_float):
    distance_squared = (x1-x2).power(2).sum()
    return exp(-gamma_float*distance_squared)


cdef class SVM_Classifier:
    cdef object X
    cdef object c_labels
    cdef float C
    cdef float tol
    cdef object kernel  # Assuming kernel is a function or callable object
    cdef int num_lagrange
    cdef int theta_hat_dim
    cdef object lagrange_muls
    cdef object errors
    cdef float epsilon
    cdef float theta0_hat
    cdef object theta_hat
    cdef float gamma
    cdef float lambda2
    cdef float lambda1
    cdef int c2
    cdef int c1
    cdef float E2
    cdef float E1 
    cdef int attemt_count

    def __init__(self, train_data_features,np.ndarray[np.int64_t] labels,float reg_hyperparameter,float tolerance,str kernel):
        self.X = train_data_features
        self.c_labels = labels
        self.C = reg_hyperparameter
        self.tol = tolerance
        self.kernel = kernel
        self.num_lagrange,self.theta_hat_dim = np.shape(self.X)
        self.lagrange_muls = np.zeros(self.num_lagrange)
        self.errors = np.zeros(self.num_lagrange)
        self.epsilon = 10 ** (-3)
        self.theta0_hat = 0
        #self.theta_hat = np.zeros(self.theta_hat_dim)
        self.theta_hat = csr_matrix((1, self.theta_hat_dim))  # Sparse matrix with 1 row and 'num_columns' columns
        #self.gamma = 1/(self.num_lagrange)
        self.gamma = 0.5
        self.attemt_count = 0


    def __reduce__(self):
        state = {
            'num_lagrange': self.num_lagrange,
            'lagrange_muls': self.lagrange_muls,
            'theta0_hat': self.theta0_hat,
            'gamma': self.gamma
        }
        # Return the constructor arguments as they are required
        return (self.__class__,(self.X, self.c_labels, self.C, self.tol, self.kernel),state)

    def __setstate__(self, state):
        self.num_lagrange = state['num_lagrange']
        self.lagrange_muls = state['lagrange_muls']
        self.theta0_hat = state['theta0_hat']
        self.gamma = state['gamma']
    
    cdef float linear_kernel(self,x1, x2):
        return np.dot(x1,x2)

    cdef float poly_kernel(self, x1, x2, float c=1, int d=2):
        return (np.dot(x1,x2) + c)**d

    cdef float rbf_kernel(self, x1,  x2,float gamma=0.1):
        cdef float gamma_float = gamma
        cdef float distance_squared

        distance_squared = (x1-x2).power(2).sum()
        return exp(-gamma_float*distance_squared)

        # Calculate the squared Euclidean distance between x1 and x2
        #distance_squared_gpu = np.linalg.norm(x1_float - x2_float) ** 2
        #return np.exp(-gamma_float * distance_squared_gpu)
    
    # Compute kernel according to specified type
    cdef float compute_kernel(self, x1, x2,float gamma_float): 
        #cdef float gamma_float
        cdef float distance_squared
        #gamma_float = self.gamma

        if self.kernel == "rbf":
            distance_squared = (x1-x2).power(2).sum()
            return exp(-gamma_float*distance_squared)
        elif self.kernel == "linear":
            return self.linear_kernel(x1,x2)
        elif self.kernel == "poly":
            return self.poly_kernel(x1,x2)
        else:
            return self.linear_kernel(x1,x2)


    # Prediction: (Transpose(Theta_hat) * xi + theta0)
    cdef float discriminant_score(self,int i):
        cdef float dot_product
        dot_product = self.theta_hat.dot(self.X[i].transpose()).sum() # Get only elemnt
        return self.theta0_hat + dot_product

    # Error: Predicted - actual
    cdef float compute_error(self,int i):
        return self.discriminant_score(i) - self.c_labels[i]
    
    # Get all non bound lamda indexes (ie. 0 < lamda < C)
    def get_non_bound_indexes(self): 
        return np.where(np.logical_and(self.lagrange_muls > 0,self.lagrange_muls<self.C))[0]

    # Returns index with maximum step size or -1 if not found
    cdef int find_lagrange_with_maximum_step(self,non_bounded_lagrange_indices):
        cdef int lag_index = -1 # Lagrange index
        cdef float max = 0
        for index in non_bounded_lagrange_indices:
            E1 = self.compute_error(index)
            step = abs(E1 - self.E2)
            # If step is greater than max update
            if step > max:
                max = step
                lag_index = index
        return lag_index
    
    # Take ascent 
    cdef bint take_cordinate_ascent_step(self,int i1,int i2):
        # If the indexes are same
        if i1 == i2:
            return False
        
        # Lamda 1
        self.lambda1 = self.lagrange_muls[i1] # lambda1
        self.c1 = self.c_labels[i1] # label yi or ci
        self.E1 = self.compute_error(i1) # [Transpose(Theta_hat) * xi + theta0] - Ci   

        # i1th and i2th feature vectors
        cdef float L 
        cdef float H
        cdef float K11
        cdef float K22
        cdef float K12
        cdef float double_derivative
        cdef float lambda2_new
        cdef float lambda1_new
        cdef float theta0_final
        cdef float theta0_diff
        cdef int i

        x1 = self.X[i1] 
        x2 = self.X[i2]   

        # Compute L and H 
        if self.c1 != self.c2:
            L = max(0,self.lambda2-self.lambda1)
            H = min(self.C,self.C+self.lambda2-self.lambda1)
        else:
            L = max(0,self.lambda1 + self.lambda2 - self.C)
            H = min(self.C, self.lambda1 + self.lambda2)
        
        if L == H:
            return False
        
        # Compute kernel values   
        K11 = self.compute_kernel(x1,x1,self.gamma)
        K22 = self.compute_kernel(x2,x2,self.gamma)
        K12 = self.compute_kernel(x1,x2,self.gamma)

        double_derivative = 2 * K12 - K11 - K22

        # Skip for edge case
        if double_derivative >= 0:
            return False
        
        # If less than 0 compute lambda2
        lambda2_new = self.lambda2 - (self.c2 * (self.E1 - self.E2))/double_derivative

        # Clip the boundary
        if lambda2_new < L:
            lambda2_new = L
        elif lambda2_new > H:
            lambda2_new = H    

        # If not enough change
        if abs(lambda2_new - self.lambda2)< self.epsilon:
            return False
        
        # Compute lambda1
        lambda1_new = self.lambda1 + self.c1 * self.c2 * (self.lambda2 - lambda2_new)

        # Compute b or theta0
        theta0_final = self.compute_theta0(self.E1,self.E2,self.lambda1,lambda1_new,self.lambda2,lambda2_new,K11,K12,K22,self.c1,self.c2)

        # Theta0 diff
        theta0_diff = theta0_final - self.theta0_hat

        # Updating thetha0 
        self.theta0_hat = theta0_final
        gamma_float = self.gamma

        # Update error for non bounded multipliers
        for i in range(self.num_lagrange):
            if(self.lagrange_muls[i] > 0 and self.lagrange_muls[i] < self.C):
                # Update the errors
                # E_new = E_old + c1(lambda1_new - lambda1_old)* k1i + c2(lambda2_new - lambda2_old)*k2i + theta0_new - theta0_old
                self.errors[i] = self.errors[i] + self.c1*(lambda1_new - self.lambda1)*self.compute_kernel(x1,self.X[i],gamma_float) + self.c2*(lambda2_new - self.lambda2)*self.compute_kernel(x2,self.X[i],gamma_float) + theta0_diff

        # Updating langrange multipliers
        self.lagrange_muls[i1] = lambda1_new
        self.lagrange_muls[i2] = lambda2_new

        # For optimized non bound multipliers set errors to 0
        if(lambda1_new > 0 and lambda1_new < self.C):
            self.errors[i1] = 0
        if(lambda2_new > 0 and lambda2_new < self.C):
            self.errors[i2] = 0

        return True

    cdef float compute_theta0(self,float E1,float E2,float lambda1,float lambda1_final,float lambda2,float lambda2_final,float k11,float k12,float k22,int c1,int c2):
        cdef float theta0_1
        cdef float theta0_2
        theta0_1 = self.theta0_hat - E1 - c1*(lambda1_final-lambda1)*k11 - c2*(lambda2_final - lambda2)*k12
        theta0_2 = self.theta0_hat - E2 - c1*(lambda1_final-lambda1)*k12 - c2*(lambda2_final - lambda2)*k22
        if lambda1_final > 0 and lambda1_final < self.C:
            theta0_hat_final = theta0_1
        elif lambda2_final > 0 and lambda2_final < self.C:
            theta0_hat_final = theta0_2
        else:
            theta0_hat_final = (theta0_1 + theta0_2)/2.0
        return theta0_hat_final

    # Tries to find a successful ascent over entire training set at random point
    cdef int attempt_entire_training_set(self,int i2):
        cdef int i1 = 0
        cdef int num_lagrange
        num_lagrange = self.num_lagrange
        self.attemt_count = self.attemt_count + 1

        # If ascent is not found iterate over entire training set from random point and try to take ascent
        random_range = randrange(num_lagrange)
        all_indices = list(range(num_lagrange))
        shuffled_training_list = all_indices[random_range:] + all_indices[:random_range]
        for i1 in shuffled_training_list:
            if self.take_cordinate_ascent_step(i1,i2) == True:
                return 1 # Else try another index

        # If still not satisfied skip the choosen lambda2
        return 0
    
    # Tries to find a successful ascent over non bounded set at random point if not then entire training set at random point
    cdef int attempt_non_bounded_set(self,non_bounded_lagrange_indices,int i2):
        cdef int i1 = 0
         # Randomly select non bound indices and take ascent and find one that satisfies ascent
        random_range = randrange(len(non_bounded_lagrange_indices))
        shuffled_non_bounded_list = non_bounded_lagrange_indices[random_range:] + non_bounded_lagrange_indices[:random_range]
        for i1 in shuffled_non_bounded_list:
            # Try to take ascent, if succesful then return 1 
            # If ascent is not found iterate over entire training set from random point and try to take ascent
            if self.take_cordinate_ascent_step(i1,i2) == True:
                return 1
        
        # If not succesful try entire set
        return self.attempt_entire_training_set(i2)

    # Check lamda pairs and optimize them
    def examine_lagrange(self,int i2):
        self.lambda2 = self.lagrange_muls[i2] # lambda2
        self.c2 = self.c_labels[i2] # label yi or ci
        self.E2 = self.compute_error(i2) # [Transpose(Theta_hat) * xi + theta0] - Ci
        constraint = self.c_labels[i2] * self.E2 # Ci *[Transpose(Theta_hat) * xi + theta0] - 1  as squared(Ci) = 1
        cdef int lag_index = 0
        # print(f"Constraint: {constraint}")
        # Check if the lamda violates KKT conditions
        if ((constraint < -self.tol and self.lambda2 < self.C) or (constraint > self.tol and self.lambda2 > 0)):
            
            # Search for another lamda from non bouding set (ie. 0 < lamda < C)
            non_bounded_lagrange_indices = list(self.get_non_bound_indexes())

            # print("Non bounded indices: ",non_bounded_lagrange_indices)

            # If there are no non bounded lagrange indices
            if len(non_bounded_lagrange_indices) == 0:
                #print("Inside non bounded = 0")
                # Attempt over entire training set
                return self.attempt_entire_training_set(i2)

            # print("Non bound != 0")
            # If there are lagrange indices find the lagrange multiplier with largest step size/change
            lag_index = self.find_lagrange_with_maximum_step(non_bounded_lagrange_indices)

            # If such valid lagrange multiplier is found 
            if(lag_index >= 0 and lag_index != i2):
                if self.take_cordinate_ascent_step(lag_index,i2) == True:
                    return 1
                else: 
                    return self.attempt_non_bounded_set(non_bounded_lagrange_indices,i2)

            # If no such lamda is found or found lamda is the same 
            elif lag_index == -1 or lag_index == i2:
                return self.attempt_non_bounded_set(non_bounded_lagrange_indices,i2)
        else:
            # If doesn't violate return 0 
            return 0


    # SMO outer loop
    def fit(self):
        print(f"Dataset size: {self.num_lagrange}")

        # Keep iterating until no lagrange multiplier can be changed
        cdef int num_optimized = 0
        cdef bint checkAll = True
        cdef int i = 0
    
        while num_optimized > 0 or checkAll:
            num_optimized = 0 # Reset

            if checkAll == True:
                # Loop over entire training example for finding lamda that violates KKT conditions
                for i in range(self.num_lagrange):
                    num_optimized = num_optimized + self.examine_lagrange(i)
                    #print("Num_optimized inside allset: ",num_optimized)

                print(f"Num optimized (All set): {num_optimized}")

            
            elif checkAll == False:
                # Search for another lamda from non bouding set (ie. 0 < lamda < C)
                non_bounded_lagrange_indices = list(self.get_non_bound_indexes())

                # If there are bounded lagrange indices
                if len(non_bounded_lagrange_indices) > 0:
                    # Loop over non bounded set for finding lamda that violates KKT conditions
                    for i in range(len(non_bounded_lagrange_indices)):
                        num_optimized = num_optimized + self.examine_lagrange(i)
                
                print(f"Num optimized (Num optimized (Non bounded set)): {num_optimized}")

            # Loop
            if checkAll == True:
                checkAll = False
            elif num_optimized == 0:
                checkAll = True

    def compute_score(self,x, X, lagrange_muls, c_labels,float gamma_float,float theta0_hat):
        cdef float score
        score = 0

        print("X_train shape: ",X.shape[0])
        threshold = 1e-3


        # Iterate over Lagrange multipliers (support vectors)
        for i in range(len(lagrange_muls)):
            if lagrange_muls[i] > threshold and lagrange_muls[i] <= self.C:  # Only use support vectors
                #print(f"Test data shape at {i}:", x.shape[1])
                #print(f"Training data Shape at {i}: ", X[i].shape[1])
                #score += lagrange_muls[i] * c_labels[i]
                #continue

                #row_data = X.getrow(i)
                #print(f"{i} data access success")


                #distance_squared = lnm.norm(X[i]-x,axis=1)
                # Compute squared distance
                diff = X[i]-x
                distance_squared = (diff).multiply(diff).sum(axis=1)
                #distance_squared = cp.sum(cp.power(X[i] - x, 2), axis=1)

                val = exp(-gamma_float * distance_squared)
                score += lagrange_muls[i] * c_labels[i] * val

        print("Score: ",score)
        return score # Return score of the chunk


    def predict(self, X):
        cdef float score
        cdef int i
        cdef float gamma_float
        cdef float val
        gamma_float = self.gamma
        score = 0

        print("Evaluating...")
        predictions = []

        print(f"Test data shape whole:",X.shape[1])
        print(f"Training data Shape whole: ",self.X.shape[1])


        # Set a threshold for small Lagrange multipliers
        threshold = 1e-3

        support_vector_indices = np.where((self.lagrange_muls > threshold) & (self.lagrange_muls < self.C) )[0]
        print("Support vector count: ",len(support_vector_indices))
        print("Regularization hyper parameter: ",self.C)

        # Padding the sparse test data with zeros
        if X.shape[1] < self.X.shape[1]:
            missing_features = self.X.shape[1] - X.shape[1]
            X_test_padded = csr_matrix((X.shape[0], missing_features)).tolil()
            X_test = hstack([X, X_test_padded])
            X = X_test.tocsr()

        X_one = X[0]

        print("Copying...")
        X_cupy_test = cusparse.csr_matrix(X)  # Convert to CuPy sparse format
        X_cupy_trained = cusparse.csr_matrix(self.X)
        print("Copied...")

        #Split into chunks to parallelize
        cores = 3
        #chunk_size = X_cupy_trained.shape[0] // cores
        #chunks = [X_cupy_trained[i:i + chunk_size] for i in range(0, X_cupy_trained.shape[0], chunk_size)]
        #chunks = [X_cupy_trained[i * chunk_size:(i + 1) * chunk_size] for i in range(cores)]
        # Compute the chunk size for each worker
        print("X_cupy_trained shape: ",X_cupy_trained.shape[0])

        chunk_size = X_cupy_trained.shape[0] // cores

        print("Chunk size: ",chunk_size)

        # Create chunks
        chunks = []  # X_train chunk
        lagrange_chunks = []
        c_labels_chunks = []

        # Loop to create chunks
        for i in range(cores):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size - 1 if i < cores - 1 else X_cupy_trained.shape[0] - 1
            
            print(f"Start index: {start_index} and end index: {end_index} for chunk {i} ")
            
            chunks.append(X_cupy_trained[start_index:end_index+1])  # end_index + 1 to include the last row
            lagrange_chunks.append(self.lagrange_muls[start_index:end_index+1])  # end_index + 1 to include the last row
            c_labels_chunks.append(self.c_labels[start_index:end_index+1])  # end_index + 1 to include the last row

        for i, chunk in enumerate(chunks):
            print(f"Chunk {i} range: {chunk.shape}")  # Debug chunk sizes
            print("Lagrange chunk [{i}]: ",len(lagrange_chunks[i]))
            print("Clabel chunks: [{i}] ",len(c_labels_chunks[i]))
        #print(chunks[0][13631])
        #print(chunks[1][13631])
        #print(chunks[2][13631])



            # Parallelize for each test sample
        for x in X_cupy_test:
            #if len(predictions) != 0:
             #   break
            # Use joblib to parallelize the prediction process
            scores_list = Parallel(n_jobs=cores)(delayed(self.compute_score)(
                x, chunk, lagrange_chunk, c_labels_chunk, self.gamma, self.theta0_hat) for chunk, lagrange_chunk, c_labels_chunk in zip(chunks, lagrange_chunks, c_labels_chunks))

            score = sum(scores_list)
            score += self.theta0_hat
            predictions.append(np.sign(score))

        # Return final prediction list
        return np.array(predictions)


        for x in X:
            if score != 0:
                break
            score = 0
            for i in range(self.num_lagrange):
                if self.lagrange_muls[i] > 0.9:  # Only use support vectors
                    print(f"Test data shape at {i}:",x.shape[1])
                    print(f"Training data Shape at {i}: ",self.X[i].shape[1])


                    #distance_squared = (self.X[i]-x).multiply(self.X[i]-x).sum()
                    #distance_squared = X_trained*x.T
                    #distance_squared = (X_cupy_trained[i]-x).power(2).sum()
                    #val = exp(-gamma_float*distance_squared)
                    #score += self.lagrange_muls[i] * self.c_labels[i] * val

                    # Assuming X_cupy_trained is a CuPy sparse matrix, x is a sparse vector
                    distance_squared = (self.X[i] - x).multiply(self.X[i] - x).sum()
                    val = cp.exp(-gamma_float * distance_squared)

                    score += self.lagrange_muls[i] * self.c_labels[i] * val

            score += self.theta0_hat
            predictions.append(np.sign(score))
        return np.array(predictions)

def evaluate_svm(svm, X_train, X_test, y_train, y_test):
    train_predictions = svm.predict(X_train)
    test_predictions = svm.predict(X_test)
    
    metrics = {}
    
    metrics['train_accuracy'] = accuracy_score(y_train, train_predictions)
    metrics['train_precision'] = precision_score(y_train, train_predictions, zero_division=0)
    metrics['train_recall'] = recall_score(y_train, train_predictions)
    metrics['train_f1'] = f1_score(y_train, train_predictions)
    
    metrics['test_accuracy'] = accuracy_score(y_test, test_predictions)
    metrics['test_precision'] = precision_score(y_test, test_predictions, zero_division=0)
    metrics['test_recall'] = recall_score(y_test, test_predictions)
    metrics['test_f1'] = f1_score(y_test, test_predictions)
    
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"Training Metrics:")
    print(f"Accuracy:  {metrics['train_accuracy']:.4f}")
    print(f"Precision: {metrics['train_precision']:.4f}")
    print(f"Recall:    {metrics['train_recall']:.4f}")
    print(f"F1 Score:  {metrics['train_f1']:.4f}")
    
    print("\nTesting Metrics:")
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall:    {metrics['test_recall']:.4f}")
    print(f"F1 Score:  {metrics['test_f1']:.4f}")
    
    # Plot confusion matrices
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # train_cm = confusion_matrix(y_train, train_predictions)
    # sns.heatmap(train_cm, annot=True, fmt='d', ax=ax1)
    # ax1.set_title('Training Confusion Matrix')
    # ax1.set_xlabel('Predicted')
    # ax1.set_ylabel('Actual')
    
    # test_cm = confusion_matrix(y_test, test_predictions)
    # sns.heatmap(test_cm, annot=True, fmt='d', ax=ax2)
    # ax2.set_title('Testing Confusion Matrix')
    # ax2.set_xlabel('Predicted')
    # ax2.set_ylabel('Actual')
    
    # plt.tight_layout()
    # plt.show()
    
    return metrics

def main():
    # Load dataset
    # 1. Load the CSV file
    csv_path = './micro_gender.csv'
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. Separate features and target
    #features = df.drop(columns=['class'])  # Features
    #target = df['class']  # Target
    # 2. Separate features and target
    features = df['text']
    #features = df.drop(columns=['Outcome'])  # Features
    target = df['gender_label']  # Target

    print("Converting labels from {'male', 'female'} to {1, -1}...")
    target = target.replace({'male': 1, 'female': -1})

    # 3. Convert labels to {-1, 1} if they're {0, 1}
    #if set(target.unique()) == {0, 1}:
    #   print("Converting labels from {0, 1} to {-1, 1}...")
    #  target = 2 * target - 1

    #scaler = StandardScaler()
    #scaler.fit(features)
    #standardized_data = scaler.transform(features)
    #print(standardized_data)

    # Now `tfidf_features` contains the TF-IDF representation of the text data

    # Update features as standardized data
    #features = standardized_data

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    # Transform the text data into TF-IDF features
    tfidf_features = tfidf_vectorizer.fit_transform(features)
    features = tfidf_features
    print("TFID : ",features)

    # 4. Split the data
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=0.2,
        random_state=2,
        shuffle=False
    )

    # Transfer to gpu

    print("X_train: ",X_train)


    y_train_np = y_train.to_numpy()  # Convert pandas Series to numpy ndarray
    print("Normal Y: ",y_train)
    print("Yafter to numpy(): ",y_train_np)
    print(y_train_np.dtype)  # Should be np.float64


    print("Instantiating model....")
    model = SVM_Classifier(X_train,y_train_np,10,0.001,"rbf")
    print("Fitting model...")
    model.fit()
    print("Model fitted...")

    print("Saving model...")
    # Saving the model

    joblib.dump(model, 'micro_custom_gender.joblib')
    #with open('gender_svm_custom.dill', 'wb') as f:
    #   dill.dump(model, f)

    #Loading model
    print("Loading the model")
    model_loaded = joblib.load("micro_custom_gender.joblib")
    #with open('gender_svm_custom.dill', 'rb') as f:
        #model_loaded = dill.load(f)

    print("Evaluating accuracy...")
    metrics = evaluate_svm(model_loaded, X_train, X_test, y_train, y_test)

    # Save the model
    # Assuming 'model' is your trained SVM model instance
    #joblib.dump(model, 'gender_svm_custom.joblib')
