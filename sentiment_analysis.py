from extract_comment import fetch_and_save_comments
import os
import csv
import re
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import googleapiclient.discovery
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema import Document
from langchain_community.llms import Ollama

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def read_csv_file(csv_filename):
    """Read the CSV file and return a DataFrame."""
    with open(csv_filename, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        comment = []
        for row in reader:
            comment.append(row['comment'])
    return comment

# ---------------------------------- PREPROCESSING ----------------------------------#
# Emoticon Conversion Function
def find_emoticons(string):
    happy_faces = [' :D',' :)',' (:',' =D',' =)',' (=',' ;D',' ;)',' :-)',' ;-)',' ;-D',' :-D']
    sad_faces = [' D:',' :(',' ):',' =(',' D=',' )=',' ;(',' D;', ' )-:',' )-;',' D-;',' D-:',' :/',' :-/', ' =/']
    neutral_faces = [' :P',' :*','=P',' =S',' =*',' ;*',' :-|',' :-*',' =-P',' =-S']
    for face in happy_faces:
        if face in string:
            string = string.replace(face, ' happy_face ')
    for face in sad_faces:
        if face in string:
            string = string.replace(face, ' sad_face ')
    for face in neutral_faces:
        if face in string:
            string = string.replace(face, ' neutral_face ')  
    return string

def preprocess_for_sentiment(comment_list):
    processed_comments = []

    for comment in comment_list:
        # Deconvert emojis
        comment = emoji.demojize(comment)

        # Replace emoticons like :) or :( with label
        comment = find_emoticons(comment)

        # Remove URLs
        comment = re.sub(r'http\S+|www\S+|https\S+|t\.co\S+', '', comment)

        # Optional: remove accents (helps standardize encoding)
        comment = unidecode(comment)

        # Clean excessive whitespace
        comment = re.sub(r'\s+', ' ', comment).strip()

        # Keep sentence as-is for VADER
        processed_comments.append(comment)

    return processed_comments

def preprocess_comment_for_wordcloud (comment_list):
    """Preprocess comments: only remove stopwords and apply lemmatization."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_comments = []

    for comment in comment_list:
        # Tokenize
        words = word_tokenize(comment)

        # Remove stopwords and lemmatize
        cleaned = [
            lemmatizer.lemmatize(word)
            for word in words
            if word.lower() not in stop_words and word.isalpha()
        ]

        # Join tokens back
        processed_comments.append(' '.join(cleaned))

    return processed_comments
# ---------------------------------- SENTIMENT ANALYSIS ----------------------------------#

def analyze_sentiment(comments):
    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Read in the YouTube comments from the CSV file
    comments_positive = []
    comments_negative = []
    comments_neutral = []
    # Count the number of neutral, positive, and negative comments
    num_neutral = 0
    num_positive = 0
    num_negative = 0

    for comment in comments:
        sentiment_scores = analyzer.polarity_scores(comment)
        if sentiment_scores['compound'] > 0.0:
            num_positive += 1
            comments_positive.append(comment)
        elif sentiment_scores['compound'] < 0.0:
            num_negative += 1
            comments_negative.append(comment)
        else:
            comments_neutral.append(comment)
            num_neutral += 1
    results = [num_neutral, num_positive, num_negative]
    # Return the results as a dictionary
    return results, comments_positive, comments_negative, comments_neutral

def plot_sentiment_pie_chart(results):
    # Get the counts for each sentiment category
    num_neutral = results[0]
    num_positive = results[1]
    num_negative = results[2]
    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [num_positive, num_negative, num_neutral]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)  # explode 1st slice (Positive)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')    
    return fig

# ---------------------------------- WORD CLOUD ----------------------------------#
def generate_wordcloud(all_comments):
    # Preprocess the entire list for word cloud
    processed_comments = preprocess_comment_for_wordcloud(all_comments)

    # Combine into a single string
    text_all = ' '.join(processed_comments)

    # Generate the word cloud
    wc_all = WordCloud(width=1000, height=500, background_color='white').generate(text_all)

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_all, interpolation='bilinear')
    ax.set_title("☁️ Word Cloud of Video's Comments", fontsize=18)
    ax.axis('off')
    plt.tight_layout()

    return fig

# ------------------ Summarize positive and negative comment -----------------------#

# Models and Vector DBs
EMBEDDING_MODEL = OllamaEmbeddings(model="mxbai-embed-large")
POSITIVE_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
NEGATIVE_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="llama3.2")

def chunk_documents(raw_documents):
    document_objects = [Document(page_content=doc) for doc in raw_documents]
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        add_start_index=True
    )
    return text_processor.split_documents(document_objects)

def index_positive_documents(documents):
    POSITIVE_VECTOR_DB.add_documents(chunk_documents(documents))

def index_negative_documents(documents):
    NEGATIVE_VECTOR_DB.add_documents(chunk_documents(documents))

def find_related_positive(query, k=200):
    return POSITIVE_VECTOR_DB.similarity_search(query, k=k)

def find_related_negative(query, k=200):
    return NEGATIVE_VECTOR_DB.similarity_search(query, k=k)

def generate_positive_summary_from_vector(query="Summarize main point of these comments."):
    PROMPT_TEMPLATE_POSITIVE = """
    You are a YouTube sentiment analysis assistant. Your task is to summarize YouTube video comments.

    Below are the positive comments:
    ---
    {positive_comments}
    ---
    
    Please summarize the main points expressed in these positive comments.
    Start your response with: "Positive Aspects:", followed by a bullet-point list of the main takeaways.
    """
    docs = find_related_positive(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_POSITIVE)
    chain = prompt | LANGUAGE_MODEL
    return chain.invoke({
        "user_query": query,
        "positive_comments": context
    })


def generate_negative_summary_from_vector(query="Summarize main point of the negative comments."):
    
    PROMPT_TEMPLATE_NEGATIVE = """
    You are a YouTube sentiment analysis assistant. Your task is to analyze and summarize YouTube video comments.

    Below are the negative comments:
    ---
    {negative_comments}
    ---
    Please summarize the main points expressed in these negative comments
    Start your response with: "Negative Aspects:", followed by a bullet-point list of the main takeaways.
   """
     
    docs = find_related_negative(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_NEGATIVE)
    chain = prompt | LANGUAGE_MODEL
    return chain.invoke({
        "user_query": query,
        "negative_comments": context
    })
    
def summarize_both_sentiments(positive_comments, negative_comments):
    # Index each into their own vector store
    index_positive_documents(positive_comments)
    index_negative_documents(negative_comments)

    # Summarize each
    pos_summary = generate_positive_summary_from_vector()
    neg_summary = generate_negative_summary_from_vector()
    
    return pos_summary, neg_summary  


# ---------------------------------- MAIN FUNCTION ----------------------------------#
def main(youtube_url, api_key):
    # Fetch and save comments
    comments, csv_filename = fetch_and_save_comments(youtube_url, api_key)

    # Read the CSV file and preprocess comments
    comments = read_csv_file(csv_filename)
    comments = preprocess_for_sentiment(comments)

    # Analyze sentiment
    results, comments_positive, comments_negative, comments_neutral = analyze_sentiment(comments)

    # Plot sentiment pie chart
    sentiment_fig = plot_sentiment_pie_chart(results)
    # sentiment_fig.savefig("sentiment_pie_chart.png")

    # Generate word cloud
    comments_cloud = preprocess_comment_for_wordcloud(comments)
    wordcloud_fig = generate_wordcloud(comments_cloud)
    # wordcloud_fig.savefig("sentiment_wordclouds.png")

    # Summarize positive and negative comments
    pos_summary, neg_summary = summarize_both_sentiments(comments_positive, comments_negative)
    return pos_summary, neg_summary, sentiment_fig, wordcloud_fig

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=Y5XnWtLsK5E&ab_channel=Econ"
    API_KEY = 'AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0'
    pos_summary, neg_summary, sentiment_fig, wordcloud_fig = main(youtube_url, API_KEY)
    print("Positive Summary:", pos_summary)
    print("\nNegative Summary:", neg_summary)






