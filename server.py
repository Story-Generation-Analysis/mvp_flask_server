# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request 
from svd import topic_extraction
from pymongo import MongoClient
from bson.objectid import ObjectId
from threading import Thread
import pandas as pd 
from runner import predict_sentiment,predict_gender
from kmeans import cluster_dataset

# creating a Flask app 
app = Flask(__name__) 

# Mongodb
client = MongoClient("mongodb+srv://sakshamthapa010:helloworld@cluster0.greotwa.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["story"]  
collection = db["storycollection"]

# Analyze story in background
def analyze_story(story,story_id,term_name_one,term_name_two,term_name_val_one,term_name_val_two):
     
    # Analyze sentiment and gender
    print("Analyzing gender....")
    gender = predict_gender(story)
    print("Gender analyzed...")
    print("Analyzing sentiment....")
    sentiment = predict_sentiment(story)
    print("Sentiment analyzed...")

    # Save to mongodb
    result = collection.update_one(
        {"_id": ObjectId(story_id)},  # Find document by its ID
        {"$set": {"gender":gender,"sentiment":sentiment,"topic_one": term_name_one,"topic_two":term_name_two,"topic_one_val":term_name_val_one,"topic_two_val":term_name_val_two}}  # Update Topic and values
    )

    # Check if the update was successful
    if result.matched_count == 0:
        print("Story not found with given id")
    else:
        print("Clustering stories...")

        # Fetch all rows from MongoDB collection
        cursor = collection.find() 
        df = pd.DataFrame(list(cursor))
        X = df[["topic_two_val", "topic_one_val"]]

        # Cluster stories using kmeans
        clusteredData = cluster_dataset(X)

        print("Saving to database...")

        # Loop over the rows and update each document in the MongoDB collection
        for index, row in clusteredData.iterrows():
            cluster = row["cluster"]
            story_id = df.iloc[index]["_id"]  # Getting the _id of the document
            
            # Update the document with the new cluster value
            res = collection.update_one(
                {"_id": ObjectId(story_id)},  # Find the document by _id
                {"$set": {"cluster": cluster}}  # Set the new cluster value
            )
            
            # Check if update was successful
            if res.matched_count == 0:
                print(f"{index} story not found with given id")
            else:
                print("Saved to database...")
    
    

@app.route('/analyze', methods = ['POST']) 
def home(): 

    try: 
        # Accessing the JSON data from the request body
        data = request.get_json()

        # Extract the story from the JSON payload
        story = data.get('story')
        story_id = data.get('story_id')

        # If not story is passed
        if not story or not story_id:
            return jsonify({'msg': 'Story is required'}), 400
        

        print("Story: ",story)


        # Extract common themes
        term_name_one,term_name_val_one,term_name_two,term_name_val_two = topic_extraction(story)

        print("------------Strongest--------------")
        print("Strongest_term_name: ",term_name_one)
        print("Strongest_term_val: ",term_name_val_one)

        
        print("------------Second strongest--------------")
        print("Strongest_term_name: ",term_name_two)
        print("Strongest_term_val: ",term_name_val_two)


        # Process story in a background thread
        Thread(target=analyze_story, args=(story,story_id,term_name_one,term_name_two,term_name_val_one,term_name_val_two)).start()

        return jsonify({'msg': "Analysis request successful"}), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500



# driver function 
if __name__ == '__main__': 

	app.run(debug = True) 
