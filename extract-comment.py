import googleapiclient.discovery

# Replace with your own API Key and Video ID
API_KEY = 'AIzaSyDj7I12G6kpxEt4esWYXh2XwVAOXu7mbz0'
VIDEO_ID = 'LAS33IPqhJE'
OUTPUT_FILE = 'youtube_comments_6_LAS33IPqhJE.txt'


def get_comments(video_id, api_key):
    # Create a YouTube API client
    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey=api_key)

    # Call the API to get the comments
    comments = []
    next_page_token = None

    while True:
        # Request comments
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat='plainText'
        )
        response = request.execute()

        # Extract comments from the response
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comments.append({'author': author, 'comment': comment})

        # Check for more comments
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments


def save_comments_to_file(comments, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for idx, comment in enumerate(comments, start=1):
            file.write(
                f"{idx}. {comment['author']}:\n{comment['comment']}\n\n")
    print(f"Comments saved to {filename}")


# Get comments
comments = get_comments(VIDEO_ID, API_KEY)

# Save to file
save_comments_to_file(comments, OUTPUT_FILE)
