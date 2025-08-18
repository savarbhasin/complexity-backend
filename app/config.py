import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
EXA_API_KEY = os.getenv("EXA_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_RESULTS = int(os.getenv("DEFAULT_NUM_RESULTS"))

# External API URLs
YOUTUBE_SUMMARY_API_BASE = "https://yt-fastapi-backend.onrender.com/summary"
FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v1/scrape"

# Search Configuration
EXCLUDED_DOMAINS = ["youtube.com", "twitter.com", "x.com", "arxiv.org"]
SOCIAL_MEDIA_DOMAINS = ["twitter.com", "x.com"]
YOUTUBE_DOMAINS = ["youtube.com", "youtu.be"]
