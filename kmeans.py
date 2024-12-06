import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

def cluster_dataset(X):
    # df = pd.read_csv('clustering.csv')
    # df.head()


    # X = df[["topic_two_val","topic_one_val"]]


    #Visualise data points
    # plt.scatter(X["topic_one_val"],X["topic_two_val"],c='black')
    # plt.xlabel('Topic 1 Distribution')
    # plt.ylabel('Topic 2 Distribution')
    # plt.show()


    K=3

    # Select random observation as centroids
    Centroids = (X.sample(n=K))
    # plt.scatter(X["topic_one_val"],X["topic_two_val"],c='black')
    # plt.scatter(Centroids["topic_one_val"],Centroids["topic_two_val"],c='red')
    # plt.xlabel('Topic 1 Distribution')
    # plt.ylabel('Topic 2 Distribution')
    # plt.show()


    # Step 3 - Assign all the points to the closest cluster centroid
    # Step 4 - Recompute centroids of newly formed clusters
    # Step 5 - Repeat step 3 and 4

    diff = 1
    j=0

    while(diff!=0):

        # print("X before: ")
        # print(X.head())
        XD=X
        i=1

        # Compute distance form each point to centroids
        for index1,row_c in Centroids.iterrows():
            ED=[]
            for index2,row_d in XD.iterrows():
                d1=(row_c["topic_one_val"]-row_d["topic_one_val"])**2
                d2=(row_c["topic_two_val"]-row_d["topic_two_val"])**2
                d=np.sqrt(d1+d2)
                ED.append(d)
            X[i]=ED # Set distance list for i'th centroid column
            #print(f"{i}'th centroid distance list: ",X[i])
            i=i+1
        
        # print("After distance :")
        # print(X.head())


        # Assign clusters  (Kunai row ko minimum distance kun cluster ma cha 1,2,3)
        C=[]
        for index,row in X.iterrows():
            min_dist=row[1] # Distance of from the first centroid of the particular row
            pos=1  # Cluster

            # Find minimum distance among K cluster for a row
            for i in range(K): # 0->K-1
                if row[i+1] < min_dist:
                    min_dist = row[i+1]
                    pos=i+1 # Current cluster no
            # Append the cluster no with minimum distance for particular row
            C.append(pos)
        # Add the cluster column
        X["cluster"]=C

        # Calculate new centroid per clusters 
        Centroids_new = X.groupby(["cluster"]).mean()[["topic_two_val","topic_one_val"]]

        # Convergence check
            # First iteration
        if j == 0:
            # Initialize diff at 1 and increment j
            diff=1
            j=j+1
        else:
            # Other iteration
            diff = (Centroids_new['topic_two_val'] - Centroids['topic_two_val']).sum() + (Centroids_new['topic_one_val'] - Centroids['topic_one_val']).sum()
            #print("Diff: ",diff)

        # Update the new centroids
        Centroids = Centroids_new


    # # Visualize the clusters
    # color=['blue','green','cyan']
    # for k in range(K):
    #     # Get rows of k cluster
    #     data=X[X["cluster"]==k+1]
    #     # Plot with cluster color
    #     plt.scatter(data["topic_one_val"],data["topic_two_val"],c=color[k])
    # plt.scatter(Centroids["topic_one_val"],Centroids["topic_two_val"],c='red')
    # plt.xlabel('Topic 1 Distribution')
    # plt.ylabel('Topic 2 Distribution')
    # plt.show()


    print("Data set after clustering: ",X)
    return X

