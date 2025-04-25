import argparse
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import os
import shutil
import googleapiclient.discovery
import sys
import time

# Global constants
CHROMA_PATH = "chroma_qa_test"


def get_embedding_function():
    """Returns the embedding function for vector database."""
    return OllamaEmbeddings(model="mxbai-embed-large")


def get_comments(video_id, api_key, max_results=None):
    """
    Fetch comments and replies from YouTube.

    Args:
        video_id: YouTube video ID
        api_key: YouTube API key
        max_results: Maximum number of comments to fetch (None for all available)

    Returns:
        List of comment dictionaries
    """
    # Create a YouTube API client
    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey=api_key)

    # Call the API to get the comments
    comments = []
    next_page_token = None
    total_comments = 0

    print("Fetching comments from YouTube API...")

    try:
        while True:
            # Request comments
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=next_page_token,
                maxResults=100,  # Maximum allowed by API
                textFormat='plainText'
            )
            response = request.execute()

            # Handle potential API errors
            if 'error' in response:
                print(f"API Error: {response['error']['message']}")
                break

            # Extract top-level comments and replies
            items_count = len(response.get('items', []))
            if items_count == 0:
                print("No comments found or all comments processed.")
                break

            print(f"Processing batch of {items_count} comment threads...")

            for item in response.get('items', []):
                # Top-level comment
                top_level_comment = item['snippet']['topLevelComment']['snippet']
                comment = top_level_comment['textDisplay']
                author = top_level_comment['authorDisplayName']
                likes = top_level_comment.get('likeCount', 0)

                # No 'replied_to' for top-level comment
                comments.append({
                    'author': author,
                    'comment': comment,
                    'likes': likes
                })
                total_comments += 1

                # Replies (if any)
                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_author = reply['snippet']['authorDisplayName']
                        reply_comment = reply['snippet']['textDisplay']
                        reply_likes = reply['snippet'].get('likeCount', 0)

                        # Include the 'replied_to' field only for replies
                        comments.append({
                            'author': reply_author,
                            'comment': reply_comment,
                            'replied_to': author,
                            'likes': reply_likes
                        })
                        total_comments += 1

                # Check if we've reached the maximum requested comments
                if max_results and total_comments >= max_results:
                    print(f"Reached maximum requested comments: {max_results}")
                    return comments[:max_results]

            # Print progress
            print(f"Fetched {total_comments} comments so far...")

            # Check for more comments (pagination)
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break  # No more pages, exit the loop

            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)

    except Exception as e:
        print(f"Error fetching comments: {str(e)}")

    print(f"Completed fetching {total_comments} comments.")
    return comments


def save_comments_to_chroma(comments):
    """Populate comments into Chroma database, clearing previous data."""
    # Always remove the existing Chroma directory to ensure fresh data
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing Chroma database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    # Prepare the Chroma database
    print("Creating new Chroma vector database...")
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Create Document objects for each comment
    documents = []
    for idx, comment in enumerate(comments, start=1):
        # Format the comment text to include author and likes
        if comment.get('likes', 0) > 0:
            content = f"{comment['author']} [👍 {comment['likes']}]:\n{comment['comment']}"
        else:
            content = f"{comment['author']}:\n{comment['comment']}"

        # Add metadata
        metadata = {
            "source": f"Comment {idx}",
            "author": comment['author'],
            "likes": comment.get('likes', 0)
        }

        if 'replied_to' in comment:
            # Add 'replied_to' for replies
            metadata['replied_to'] = comment['replied_to']

        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    # Add documents to Chroma in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        db.add_documents(batch)
        print(f"Added batch of {len(batch)} comments to Chroma (total {i+len(batch)})")

    print(f"Successfully added all {len(documents)} comments to Chroma database.")
    return len(documents)


def answer_question(question, k=30):
    """
    Answer a question based on the YouTube comments data.

    Args:
        question: The user's question about the video comments
        k: Number of relevant comments to retrieve for context

    Returns:
        Answer generated by the LLM
    """
    # Load the Chroma vector store
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Get the total number of documents in the database
    doc_count = len(db.get()['ids'])

    # Adjust k if it's larger than the number of available documents
    if k > doc_count:
        print(f"Adjusting k from {k} to {doc_count} (total available documents)")
        k = doc_count

    # Define the optimized prompt template
    PROMPT_TEMPLATE = """
    You are an expert YouTube comment analyzer focusing on direct question answering.

    QUESTION: {question}

    Below are relevant comments from the video:
    {context}

    Answer the specific question asked in the most appropriate format. Your response format should match the type of question:

    - For questions about preferences or counts (how many, what percentage, etc.):
    • Provide a direct numerical answer if possible
    • Explain how you arrived at this number
    • Include relevant quotes as evidence

    - For comparison questions (pros/cons, differences, etc.):
    • Use clear headers to separate items being compared
    • Use bullet points for listing multiple points
    • Structure information logically by item or category

    - For open-ended or analytical questions:
    • Organize by key themes or findings
    • Present insights in a logical progression
    • Include specific examples from comments

    Always begin your answer by directly addressing the question asked. Be specific and concise.

    DO NOT invent information not present in the comments.
    DO NOT include follow-up questions or recommendations.
    FOCUS only on answering exactly what was asked: {question}
    """

    print(f"Retrieving {k} most relevant comments for the question...")
    start_time = time.time()

    # Retrieve relevant documents
    results = db.similarity_search_with_score(question, k=k)

    retrieval_time = time.time() - start_time
    print(f"Retrieved {len(results)} comments in {retrieval_time:.2f} seconds")

    # Build context string from retrieved documents
    context_text = "\n\n---\n\n".join(
        [f"Comment: {doc.page_content}" for doc, _score in results])

    # Format prompt with context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=question, context=context_text)

    # Use OllamaLLM model to generate the answer
    print("Generating answer with language model...")
    model = OllamaLLM(model="llama3.2")

    generation_start = time.time()
    response_text = model.invoke(prompt)
    generation_time = time.time() - generation_start

    print(f"Answer generated in {generation_time:.2f} seconds")

    return response_text


def extract_video_id(youtube_url):
    """Extract video ID from a YouTube URL."""
    if "v=" in youtube_url:
        video_id = youtube_url.split("v=")[-1]
        # Remove any additional URL parameters
        if "&" in video_id:
            video_id = video_id.split("&")[0]
    else:
        # Assume it's already a video ID
        video_id = youtube_url

    return video_id


def display_help():
    """Display help information about available commands."""
    print("\n=== AVAILABLE COMMANDS ===")
    print("exit, quit, q       - End the Q&A session")
    print("help                - Show this help information")
    print("k N                 - Change the number of comments retrieved to N")
    print("clear               - Clear the screen")
    print("info                - Show current settings")
    print("========================\n")


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Comment Q&A Test Script")
    parser.add_argument(
        "youtube_url", help="YouTube URL or video ID to analyze")
    parser.add_argument("--api-key", default="AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0",
                        help="YouTube API key (optional)")
    parser.add_argument("--k", type=int, default=100,
                        help="Number of comments to retrieve for context (default: 100)")
    parser.add_argument("--max-comments", type=int, default=None,
                        help="Maximum number of comments to fetch (default: all available)")
    parser.add_argument("--reuse-db", action="store_true",
                        help="Reuse existing Chroma database if available (default: asks for confirmation)")

    args = parser.parse_args()

    # Extract video ID
    video_id = extract_video_id(args.youtube_url)
    print(f"Using video ID: {video_id}")

    # Check if Chroma database exists
    db_exists = os.path.exists(CHROMA_PATH)

    # Determine whether to reuse or refresh the database
    refresh_db = True

    if db_exists:
        # If --reuse-db flag is used, automatically reuse the database
        if args.reuse_db:
            refresh_db = False
            print(f"Reusing existing Chroma database at {CHROMA_PATH}")
        else:
            # Ask the user whether to reuse or refresh
            response = input(f"Chroma database already exists at {CHROMA_PATH}. Reuse it? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                refresh_db = False
                print(f"Reusing existing Chroma database")
            else:
                print(f"Will fetch fresh comments and create new database")

    # Get comment count and either fetch fresh comments or use existing database
    if refresh_db:
        print("Fetching comments from YouTube...")
        try:
            comments = get_comments(video_id, args.api_key, args.max_comments)
            total_comments = save_comments_to_chroma(comments)
        except Exception as e:
            print(f"Error fetching comments: {str(e)}")
            sys.exit(1)
    else:
        # Get number of documents in existing database
        db = Chroma(persist_directory=CHROMA_PATH,
                    embedding_function=get_embedding_function())
        total_comments = len(db.get()['ids'])
        print(f"Using existing database with {total_comments} comments")

    # Interactive Q&A loop
    print("\n=== YouTube Comment Q&A Test ===")
    print(f"Video ID: {video_id}")
    print(f"Total comments: {total_comments}")
    print(f"Using optimized prompt with k={args.k}")
    print("Type 'help' for available commands or 'exit' to end the session.\n")

    current_k = args.k

    while True:
        user_input = input("\nAsk a question (or type 'help'/'exit'): ").strip()

        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Ending Q&A test session.")
            break

        if user_input.lower() == 'help':
            display_help()
            continue

        if user_input.lower() == 'info':
            print(f"\nCurrent settings:")
            print(f"- k value: {current_k}")
            print(f"- Total comments: {total_comments}")
            continue

        if user_input.lower() == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue



        if user_input.lower().startswith('k '):
            try:
                new_k = int(user_input.split()[1])
                if new_k > 0:
                    current_k = new_k
                    print(f"Changed k value to {current_k}")
                else:
                    print("k value must be positive")
            except (IndexError, ValueError):
                print("Invalid k command. Use 'k N' where N is a positive number")
            continue

        if user_input.strip():
            print("\nFinding answer...")
            try:
                answer = answer_question(user_input, k=current_k)
                print("\n" + "=" * 50)
                print("ANSWER:")
                print(answer)
                print("=" * 50)
            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
