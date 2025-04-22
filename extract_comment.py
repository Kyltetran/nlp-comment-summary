import googleapiclient.discovery
import argparse
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import shutil
import os


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

    # Create Document objects for each top-level comment
    documents = []
    for idx, comment in enumerate(comments, start=1):
        content = f"{comment['author']}:\n{comment['comment']}"

        # # No parent_id needed since we are only dealing with top-level comments
        # document = Document(page_content=content, metadata={
        #                     "source": f"Comment {idx}"})
        # documents.append(document)

        # Add replied comment if it exists
        metadata = {"source": f"Comment {idx}"}
        if 'replied_to' in comment:
            # Add 'replied_to' for replies
            metadata['replied_to'] = comment['replied_to']

        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    # Add new documents to Chroma
    db.add_documents(documents)
    print(f"Added {len(documents)} comments to Chroma.")


def main():
    """Main function to fetch comments and save them to Chroma."""
    parser = argparse.ArgumentParser()
    parser.add_argument("youtube_url", type=str,
                        help="YouTube URL to extract comments from")
    args = parser.parse_args()

    # Extract the video ID from the URL
    youtube_url = args.youtube_url
    video_id = youtube_url.split("v=")[-1]

    # Replace with your actual API key
    API_KEY = 'AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0'
    comments = get_comments(video_id, API_KEY)

    # Save comments to Chroma
    save_comments_to_chroma(comments)


if __name__ == "__main__":
    main()
