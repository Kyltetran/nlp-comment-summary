import googleapiclient.discovery
import argparse
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import shutil
import os
import csv

API_KEY = 'AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0'

def get_comments(video_id, api_key):
    """Fetch comments and replies from YouTube."""
    # Create a YouTube API client
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

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
            comments.append({'author': author, 'comment': comment})  # No 'replied_to' for top-level comment

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

def clear_chroma_directory():
    """Clears the chroma directory before adding new data."""
    folder_path = "chroma"
        
    if os.path.exists(folder_path):
        # Loop through all the files and subdirectories
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Change file/folder permissions to make sure it can be deleted
                os.chmod(file_path, 0o777)  # Grant read, write, and execute permissions
                
                # Remove non-empty directories
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path) 
                else:
                    os.remove(file_path)  # Remove file
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"The folder '{folder_path}' does not exist.")


def save_comments_to_chroma(comments):
    """Populate comments into Chroma database, clearing previous data."""
    clear_chroma_directory()
    # Prepare the Chroma database
    db = Chroma(persist_directory="chroma", embedding_function=get_embedding_function())

    # Create Document objects for each comment 
    documents = []
    for idx, comment in enumerate(comments, start=1):
        content = f"{comment['author']}:\n{comment['comment']}"

        # Add replied comment if it exists
        metadata = {"source": f"Comment {idx}"}
        if 'replied_to' in comment:
            metadata['replied_to'] = comment['replied_to']  # Add 'replied_to' for replies

        document = Document(page_content=content, metadata=metadata)
        documents.append(document)

    # Add new documents to Chroma
    db.add_documents(documents)
    print(f"Added {len(documents)} comments (including replies) to Chroma.")    
    
def save_comments_to_csv(comments, filename="youtube_comments.csv"):
    """Save comments to a CSV file."""
    fieldnames = ['author', 'comment', 'replied_to']
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for comment in comments:
            row = {
                'author': comment.get('author', ''),
                'comment': comment.get('comment', ''),
                'replied_to': comment.get('replied_to', '')  # empty string if not a reply
            }
            writer.writerow(row)
    print(f"Saved {len(comments)} comments to {filename}")
    return filename

# Function to fetch comments and save them to Chroma
def fetch_and_save_comments(youtube_url, api_key):
    """Fetch comments from YouTube and save them to Chroma."""
    # Extract the video ID from the URL
    video_id = youtube_url.split("v=")[-1].split("&")[0]
    # Get comments from YouTube
    comments = get_comments(video_id, api_key)
    # Save comments to Chroma
    # save_comments_to_chroma(comments)
    # Save comments to CSV
    csv_filename = save_comments_to_csv(comments)
    
    return comments, csv_filename

# youtube_url = "https://www.youtube.com/watch?v=TtLJ2ocislk&ab_channel=JoshOnTheMove" 
# csv_filename, comments = fetch_and_save_comments(youtube_url, API_KEY)
# print(comments)

# def main():
#     """Main function to fetch comments and save them to Chroma."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("youtube_url", type=str,
#                         help="YouTube URL to extract comments from")
#     args = parser.parse_args()

#     # Extract the video ID from the URL
#     youtube_url = args.youtube_url
#     video_id = youtube_url.split("v=")[-1]

#     # Replace with your actual API key
#     API_KEY = 'AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0'
#     comments = get_comments(video_id, API_KEY)

#     # Save comments to Chroma
#     save_comments_to_chroma(comments)


# if __name__ == "__main__":
#     main()
