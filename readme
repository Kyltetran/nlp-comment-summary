# YouTube Comment Summarizer

Transform YouTube comment sections into structured insights using AI-powered analysis. Extract meaningful patterns, sentiment, and answers from thousands of comments automatically.

## Features

### Core Analysis
- **Comment Extraction**: Fetches all comments and replies using YouTube Data API
- **AI Summarization**: Generate comprehensive summaries of discussion themes
- **Sentiment Analysis**: Classify comments as positive, negative, or neutral
- **Visual Analytics**: Sentiment charts and word clouds
- **Vector Search**: Semantic search through comment databases

### Interactive Q&A System
- **Natural Language Queries**: Ask questions about the video comments
- **Context-Aware Responses**: Answers based on actual comment content
- **Adaptive Sampling**: Intelligently selects relevant comments for accuracy
- **Multi-video Support**: Maintains separate databases per video

## Performance

### Initial Analysis Time
- **Small videos** (< 500 comments): 2-3 minutes
- **Medium videos** (500-2000 comments): 3-5 minutes  
- **Large videos** (2000+ comments): 5-10+ minutes

### Q&A Response Time
- **Comment retrieval**: Milliseconds (vector search)
- **Answer generation**: 30-60 seconds (LLM processing)
- **Total Q&A time**: ~1 minute per question

Performance depends heavily on hardware specifications (GPU vs CPU).

## Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- YouTube Data API key ([Get one here](https://developers.google.com/youtube/v3/getting-started))
- **Hardware Recommendations**:
  - 8GB+ RAM (16GB for large comment sets)
  - GPU with CUDA support (optional but significantly faster)
  - 5GB+ free storage for models and vector databases

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/youtube-comment-summarizer
cd youtube-comment-summarizer

# Install dependencies
pip install -r requirements.txt

# Download required models
ollama pull llama3.2
ollama pull mxbai-embed-large

# Set up your API key in youtube_summary_tool_copy.py
# Replace: api_key="YOUR_API_KEY_HERE"

# Launch the application
python app.py
```

## Usage

### Web Interface
1. Start the server: `python app.py`
2. Navigate to `http://localhost:5000`
3. Enter YouTube URL and wait 3-5 minutes for analysis
4. Explore results in Summary, Sentiment, Word Cloud, and Q&A tabs

### Example Questions
- "What do people think about the music quality?"
- "Are there any complaints about the video?"
- "What are the most common suggestions?"
- "How many people mentioned the graphics?"

### Command Line
```bash
# Analyze a video
python youtube_summary_tool_copy.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Ask specific questions
python youtube_summary_tool_copy.py "VIDEO_ID" --question "What are the main criticisms?"

# Custom sampling
python youtube_summary_tool_copy.py "VIDEO_ID" --question "Summarize reactions" --k 100
```

## API Endpoints

### Analyze Video
```http
POST /api/analyze
Content-Type: application/json

{
  "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### Ask Questions
```http
POST /api/ask
Content-Type: application/json

{
  "question": "What do people think about this?",
  "k": 50
}
```

### Check Status
```http
GET /api/status
```

## How It Works

### Initial Processing
When you submit a YouTube URL:

1. **Comment Extraction** (30-60s): Fetches all comments via YouTube API
2. **Embedding Generation** (1-2 min): Converts comments to vector representations  
3. **Database Storage** (30s): Saves to ChromaDB for fast retrieval
4. **Sentiment Analysis** (1-2 min): AI analyzes emotional tone
5. **Visualization** (30s): Generates charts and word clouds
6. **Summary Creation** (1-2 min): LLM writes comprehensive summaries

### Q&A Process
When you ask a question:

1. **Vector Search** (milliseconds): Finds comments most similar to your question
2. **Context Building** (seconds): Assembles relevant comments
3. **LLM Processing** (30-60s): Llama 3.2 synthesizes an answer
4. **Response** delivered with context and insights

## Architecture

```
Web Interface ◄──► Flask Backend ◄──► YouTube API
     │                   │
     └───────────────────┼──────────────────┐
                         ▼                  ▼
                 Analysis Engine      ChromaDB
                         │           (Vector DB)
                    ┌────┼────┐
                    ▼    ▼    ▼
                Ollama HuggingFace NLTK
                (LLM) (Sentiment) (Text)
```

## Configuration

### Sentiment Analysis Models
- **Default**: `AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual`
- **Alternative**: VADER (available in `youtube_summary_tool.py`)

### Sample Size Optimization
The system automatically calculates optimal sample sizes:
- **<50 comments**: 60-70% of total
- **50-200 comments**: 30-40% of total  
- **200-1000 comments**: 20-30% of total
- **1000+ comments**: 10-20% with caps at 600

## Output Files

Each analysis creates:
```
chroma/
└── VIDEO_ID/
    ├── sentiment_pie_chart.png
    ├── comment_wordcloud.png
    ├── overall_summary.txt
    ├── sentiment_summary.txt
    └── video_metadata.json
```

## Limitations

- **Processing Time**: Initial analysis takes 3-10 minutes depending on hardware
- **API Limits**: YouTube API has daily quotas (10,000 requests/day by default)
- **Hardware Dependency**: Performance heavily depends on GPU/CPU specifications
- **Memory Usage**: Large videos (5k+ comments) require significant RAM
- **Language**: Optimized for English, supports multilingual content
- **Dependencies**: Requires local Ollama installation and model downloads

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request
