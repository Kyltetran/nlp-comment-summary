import googleapiclient.discovery
import argparse
import os
import shutil
import re
import sys
import matplotlib.pyplot as plt
import emoji
import nltk
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Global constants
CHROMA_PATH = "chroma"

# Models and Vector DBs for sentiment analysis
EMBEDDING_MODEL = OllamaEmbeddings(model="mxbai-embed-large")
POSITIVE_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
NEGATIVE_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="llama3.2")


def get_embedding_function():
    """Returns the embedding function for vector database."""
    return OllamaEmbeddings(model="mxbai-embed-large")


def get_comments(video_id, api_key):
    """Fetch comments and replies from YouTube."""
    # Create a YouTube API client
    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey=api_key)

    # Call the API to get the comments
    comments = []
    next_page_token = None

    while True:
        # Request comments
        request = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat='plainText'
        )
        response = request.execute()

        # Extract top-level comments and replies
        for item in response.get('items'):
            # Top-level comment
            top_level_comment = item['snippet']['topLevelComment']['snippet']
            comment = top_level_comment['textDisplay']
            author = top_level_comment['authorDisplayName']
            # No 'replied_to' for top-level comment
            comments.append({'author': author, 'comment': comment})

            # Replies (if any)
            if 'replies' in item:
                for reply in item['replies']['comments']:
                    reply_author = reply['snippet']['authorDisplayName']
                    reply_comment = reply['snippet']['textDisplay']
                    # Include the 'replied_to' field only for replies
                    comments.append({
                        'author': reply_author,
                        'comment': reply_comment,
                        'replied_to': author  # The reply is to the top-level comment's author
                    })

        # Check for more comments (pagination)
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break  # No more pages, exit the loop

    return comments


def save_comments_to_chroma(comments):
    """Populate comments into Chroma database, clearing previous data."""
    # Remove the existing Chroma directory (this will clear all data)
    if os.path.exists("chroma"):
        # This will remove the entire 'chroma' directory
        shutil.rmtree("chroma")

    # Prepare the Chroma database
    db = Chroma(persist_directory="chroma",
                embedding_function=get_embedding_function())

    # Create Document objects for each comment
    documents = []
    for idx, comment in enumerate(comments, start=1):
        content = f"{comment['author']}:\n{comment['comment']}"

        # Add metadata
        metadata = {"source": f"Comment {idx}"}
        if 'replied_to' in comment:
            # Add 'replied_to' for replies
            metadata['replied_to'] = comment['replied_to']

        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    # Add new documents to Chroma
    db.add_documents(documents)
    print(f"Added {len(documents)} comments to Chroma.")


def read_comments_from_chroma():
    """Read comments from the Chroma database."""
    # Connect to the existing Chroma database
    db = Chroma(persist_directory="chroma",
                embedding_function=get_embedding_function())

    # Get all documents from the database
    results = db.get()

    # Extract comments from the documents
    comments = []
    for doc in results['documents']:
        # Each document has format "Author:\nComment"
        # Split to get just the comment part
        parts = doc.split('\n', 1)
        if len(parts) > 1:
            comments.append(parts[1])  # Just the comment text, not the author

    return comments


def generate_comment_summary():
    """Generate a general summary of all comments."""
    # Load the Chroma vector store
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Define the prompt template
    PROMPT_TEMPLATE = """
    You are a YouTube comment summarizer. Below is a collection of user comments extracted from a video.

    {context}

    ---

    Please write a summary highlighting the key points and general sentiment expressed in these comments.
    Focus on providing a well-rounded overview in less than 5 paragraphs.
    """

    # Retrieve relevant documents
    results = db.similarity_search_with_score(
        "summarize youtube comments", k=2000)

    # Build context string from retrieved documents
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])

    # Format prompt with context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text)

    # Use OllamaLLM model
    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)

    # Save the output to a file
    with open("overall_summary.txt", "w", encoding="utf-8") as f:
        f.write(response_text)

    print("Overall summary saved to overall_summary.txt")
    return response_text

# ---------------------------------- PREPROCESSING ----------------------------------#


def find_emoticons(string):
    """Convert emoticons to text descriptions."""
    happy_faces = [' :D', ' :)', ' (:', ' =D', ' =)',
                   ' (=', ' ;D', ' ;)', ' :-)', ' ;-)', ' ;-D', ' :-D']
    sad_faces = [' D:', ' :(', ' ):', ' =(', ' D=', ' )=', ' ;(',
                 ' D;', ' )-:', ' )-;', ' D-;', ' D-:', ' :/', ' :-/', ' =/']
    neutral_faces = [' :P', ' :*', '=P', ' =S',
                     ' =*', ' ;*', ' :-|', ' :-*', ' =-P', ' =-S']
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
    """Preprocess comments for sentiment analysis."""
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


def preprocess_comment_for_wordcloud(comment_list):
    """Preprocess comments: remove stopwords and apply lemmatization."""
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
    """Analyze sentiment of comments using VADER."""
    # Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

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
    # Return the results and categorized comments
    return results, comments_positive, comments_negative, comments_neutral


def plot_sentiment_pie_chart(results):
    """Create a pie chart of sentiment distribution."""
    # Get the counts for each sentiment category
    num_neutral = results[0]
    num_positive = results[1]
    num_negative = results[2]

    labels = ['😊 Positive', '😠 Negative', '😐 Neutral']
    sizes = [num_positive, num_negative, num_neutral]
    colors = ['#DFF0D8', '#F2DEDE', '#EAEAEA']
    explode = (0.1, 0, 0)  # explode 1st slice (Positive)

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels,
           colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    return fig

# ---------------------------------- WORD CLOUD ----------------------------------#


def generate_wordcloud(all_comments):
    """Generate a word cloud from comments."""
    # Preprocess the entire list for word cloud
    processed_comments = preprocess_comment_for_wordcloud(all_comments)

    # Combine into a single string
    text_all = ' '.join(processed_comments)

    # Generate the word cloud
    wc_all = WordCloud(width=1000, height=500,
                       background_color='white').generate(text_all)

    # Create the figure and plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc_all, interpolation='bilinear')
    ax.set_title("☁️ Word Cloud of Video's Comments", fontsize=18)
    ax.axis('off')
    plt.tight_layout()
    return fig

# ------------------ Summarize positive and negative comment -----------------------#


def chunk_documents(raw_documents):
    """Split documents into chunks for better vector search."""
    document_objects = [Document(page_content=doc) for doc in raw_documents]
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        add_start_index=True
    )
    return text_processor.split_documents(document_objects)


def index_positive_documents(documents):
    """Add positive comments to vector database."""
    POSITIVE_VECTOR_DB.add_documents(chunk_documents(documents))


def index_negative_documents(documents):
    """Add negative comments to vector database."""
    NEGATIVE_VECTOR_DB.add_documents(chunk_documents(documents))


def find_related_positive(query, k=200):
    """Find related positive comments for a query."""
    return POSITIVE_VECTOR_DB.similarity_search(query, k=k)


def find_related_negative(query, k=200):
    """Find related negative comments for a query."""
    return NEGATIVE_VECTOR_DB.similarity_search(query, k=k)


def generate_positive_summary_from_vector(query="Summarize main point of these comments."):
    PROMPT_TEMPLATE_POSITIVE = """
    You are a YouTube sentiment analysis assistant. Your task is to summarize YouTube video comments.
    Below are the positive comments:
    ---
    {positive_comments}
    ---
    Please summarize the main points expressed in these positive comments.
    Start your response with: "Positive Aspects:", followed by a bullet-point list of the main takeaways, with the layout of 1 line break for each point.
    Note that just include 2-3 main points.
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
    Start your response with: "Negative Aspects:", followed by a bullet-point list of the main takeaways, with the layout of 1 line break for each point.
    Note that just include 2-3 main points.
    """
    docs = find_related_negative(query)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_NEGATIVE)
    chain = prompt | LANGUAGE_MODEL
    return chain.invoke({
        "user_query": query,
        "negative_comments": context
    })


def summarize_both_sentiments(positive_comments, negative_comments, output_file="sentiment_summary.txt"):
    # Index each into their own vector store
    index_positive_documents(positive_comments)
    index_negative_documents(negative_comments)

    # Summarize each
    pos_summary = generate_positive_summary_from_vector()
    neg_summary = generate_negative_summary_from_vector()

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")
        f.write(pos_summary if isinstance(
            pos_summary, str) else str(pos_summary))
        f.write("\n\n")
        f.write(neg_summary if isinstance(
            neg_summary, str) else str(neg_summary))

    return pos_summary, neg_summary

# ---------------------------------- MAIN FUNCTION ----------------------------------#


def analyze_youtube_comments(youtube_url, api_key="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0"):
    """
    Main function to analyze YouTube comments from a URL.

    Args:
        youtube_url: URL or video ID of the YouTube video
        api_key: YouTube API key (uses default if not provided)

    Returns:
        Dictionary with summaries and analysis results
    """
    print(f"Analyzing comments for: {youtube_url}")

    # Extract video ID if full URL is provided
    if "v=" in youtube_url:
        video_id = youtube_url.split("v=")[-1]
        # Remove any additional URL parameters
        if "&" in video_id:
            video_id = video_id.split("&")[0]
    else:
        # Assume it's already a video ID
        video_id = youtube_url

    print(f"Extracted video ID: {video_id}")

    # Step 1: Get comments from YouTube API
    print("Fetching comments from YouTube...")
    comments = get_comments(video_id, api_key)

    # Step 2: Save comments to Chroma vector database
    print("Saving comments to vector database...")
    save_comments_to_chroma(comments)

    # Step 3: Read comments from Chroma
    raw_comments = read_comments_from_chroma()

    # Step 4: Generate overall comment summary
    print("Generating overall comment summary...")
    overall_summary = generate_comment_summary()

    # Step 5: Preprocess comments for sentiment analysis
    print("Analyzing sentiment...")
    processed_comments = preprocess_for_sentiment(raw_comments)

    # Step 6: Perform sentiment analysis
    sentiment_results, positive_comments, negative_comments, neutral_comments = analyze_sentiment(
        processed_comments)

    # Step 7: Create sentiment visualization
    print("Creating visualizations...")
    sentiment_chart = plot_sentiment_pie_chart(sentiment_results)
    sentiment_chart.savefig("sentiment_pie_chart.png")

    # Step 8: Generate word cloud
    wordcloud = generate_wordcloud(processed_comments)
    wordcloud.savefig("comment_wordcloud.png")

    # Step 9: Summarize positive and negative comments
    print("Generating sentiment-specific summaries...")
    pos_summary, neg_summary = summarize_both_sentiments(
        positive_comments, negative_comments)

    # Return results
    results = {
        "video_id": video_id,
        "comment_count": len(comments),
        "overall_summary": overall_summary,
        "sentiment_counts": {
            "positive": sentiment_results[1],
            "negative": sentiment_results[2],
            "neutral": sentiment_results[0]
        },
        "positive_summary": pos_summary,
        "negative_summary": neg_summary,
        "output_files": {
            "sentiment_chart": "sentiment_pie_chart.png",
            "wordcloud": "comment_wordcloud.png",
            "overall_summary": "overall_summary.txt",
            "sentiment_summary": "sentiment_summary.txt"
        }
    }

    print("\nAnalysis complete! Results saved to output files.")
    return results


# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube Comment Analysis Tool")
    parser.add_argument(
        "youtube_url", help="YouTube URL or video ID to analyze")
    parser.add_argument("--api-key", default="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0",
                        help="YouTube API key (optional)")

    args = parser.parse_args()

    results = analyze_youtube_comments(args.youtube_url, args.api_key)

    # Print a summary of results
    print("\n===== ANALYSIS RESULTS =====")
    print(f"Video ID: {results['video_id']}")
    print(f"Total comments analyzed: {results['comment_count']}")
    print(f"Sentiment distribution: {results['sentiment_counts']['positive']} positive, "
          f"{results['sentiment_counts']['negative']} negative, "
          f"{results['sentiment_counts']['neutral']} neutral")
    print("Output files:")
    for name, path in results['output_files'].items():
        print(f"- {name}: {path}")
