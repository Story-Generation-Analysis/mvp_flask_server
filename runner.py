#import svm_optimized
from svm_optimized import SVM_Classifier,evaluate_svm,main
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def test():
    # Load dataset
    # 1. Load the CSV file
    csv_path = './mf_mini_shuffled.csv'
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. Separate features and target
    features = df['text']
    target = df['gender_label']  # Target

    print("Converting labels from {'male', 'female'} to {1, -1}...")
    target = target.replace({'male': 1, 'female': -1})

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    # Transform the text data into TF-IDF features
    tfidf_features = tfidf_vectorizer.fit_transform(features)
    features = tfidf_features

    # 4. Split the data
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=0.01,
        random_state=53,
        shuffle=False
    )

    print("Test shape: ",X_test.shape)

    #Loading model
    print("Loading the model")
    model_loaded = joblib.load("gender_svm_custom_final.joblib")

    print("Prediction..")
    test_predictions = model_loaded.predict(X_test)
    print("Test prediction: ")
    print(test_predictions)


    print("Real outcomes: ")
    print(y_test)

    print("Evaluating accuracy...")
    #metrics = evaluate_svm(model_loaded, X_train, X_test, y_train, y_test)

def test_model():

    csv_path = './mf_mini_shuffled.csv'
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)


    features = df['text']
    target = df['gender_label']  # Target

    print("Converting labels from {'male', 'female'} to {1, -1}...")
    target = target.replace({'male': 1, 'female': -1})



    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    # Transform the text data into TF-IDF features
    tfidf_features = tfidf_vectorizer.fit_transform(features)
    features = tfidf_features

    # 4. Split the data
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=0.2,
        random_state=2,
        shuffle=False
    )



    #Loading model
    print("Loading the model")
    model_loaded = joblib.load("gender_svm_custom.joblib")

    print("Evaluating accuracy...")
    metrics = evaluate_svm(model_loaded, X_train, X_test, y_train, y_test)


# Predict story 
def predict_gender(story):

    #story = "A Colorado doctor recently published his thoughts on this medieval figure with its grotesque beak with head covering , covered eye holes , waxed robe and cane before and after his first approach to a patient dressed in similar garb of hazmat suit , gloves , goggles , and respirator mask "
    df = pd.DataFrame({'text': [story]})  # For a single story
    # df = pd.DataFrame({'text': stories})  # For multiple stories

    story_text = df['text']

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    # Transform the text data into TF-IDF features
    tfidf_features = tfidf_vectorizer.fit_transform(story_text)

    #Loading model
    print("Loading the model")
    model_loaded = joblib.load("gender_svm_custom_final.joblib")

    print("Prediction..")
    prediction = model_loaded.predict(tfidf_features)
    print("Prediction: ",prediction)
    
    if(prediction[0] == -1):
        return "Female"
    else:
        return "Male"

# Predict sentiment
def predict_sentiment(story):
    story = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."
    
    #story = "A Colorado doctor recently published his thoughts on this medieval figure with its grotesque beak with head covering , covered eye holes , waxed robe and cane before and after his first approach to a patient dressed in similar garb of hazmat suit , gloves , goggles , and respirator mask "
    df = pd.DataFrame({'text': [story]})  # For a single story
    # df = pd.DataFrame({'text': stories})  # For multiple stories

    story_text = df['text']

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    # Transform the text data into TF-IDF features
    tfidf_features = tfidf_vectorizer.fit_transform(story_text)

    #Loading model
    print("Loading the model")
    model_loaded = joblib.load("sentiment_svm_custom_final.joblib")

    print("Prediction..")
    prediction = model_loaded.predict(tfidf_features)
    print("Prediction: ",prediction)
    
    if(prediction[0] == -1):
        return "Negative"
    else:
        return "Positive"

#main()
#test()
#predict_gender("")
#predict_sentiment("")

