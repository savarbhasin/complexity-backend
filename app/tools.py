import re
import requests
from typing import Dict, Any
from exa_py import Exa
from langchain.tools import tool
from langchain_community.utilities import ArxivAPIWrapper

from .config import (
    EXA_API_KEY, 
    FIRECRAWL_API_KEY, 
    DEFAULT_NUM_RESULTS, 
    YOUTUBE_SUMMARY_API_BASE,
    FIRECRAWL_API_BASE,
    EXCLUDED_DOMAINS,
    SOCIAL_MEDIA_DOMAINS,
    YOUTUBE_DOMAINS
)

# Initialize external services
exa = Exa(api_key=EXA_API_KEY)
arxiv = ArxivAPIWrapper()


@tool
def summarize_youtube_video(url: str) -> Dict[str, Any]:
    """Summarize a YouTube video given the url"""
    regex = (r"(?:https?:\/\/(?:www\.|m\.)?youtube\.com\/(?:watch\?v=|v\/)|https?:\/\/youtu\.be\/)([a-zA-Z0-9_-]{11})("
             r"?:[&?][^\s]*)?")
    video_id = re.search(regex, url).group(1)
    headers = {"Content-Type": "application/json"}
    response = requests.get(f"{YOUTUBE_SUMMARY_API_BASE}/{video_id}", headers=headers)
    return {
        "contents": response.json(),
        "urls": [url],
        "tool_name": "summarize_youtube_video"
    }


@tool
def retrieve_web_content(query: str) -> Dict[str, Any]:
    """Find latest web content based on a query"""
    data = exa.search_and_contents(
        query=query, 
        num_results=DEFAULT_NUM_RESULTS, 
        text=True, 
        use_autoprompt=True,
        exclude_domains=EXCLUDED_DOMAINS
    )
    formatted_data = [
        {
            "title": res.url,
            "author": res.author,
            "content": res.text,
            "url": res.url,
            "date": res.published_date
        }
        for res in data.results
    ]
    return {
        "contents": formatted_data,
        "urls": [res.url for res in data.results],
        "tool_name": "retrieve_web_content"
    }


@tool
def webscrape(url_to_scrape: str) -> Dict[str, Any]:
    """Webscrape a given URL and return a list of matching webpages"""
    payload = {
        "url": url_to_scrape,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "headers": {},
        "waitFor": 0,
        "mobile": False,
        "skipTlsVerification": False,
        "timeout": 30000,
        "location": {
            "country": "US",
            "languages": ["en-US"]
        },
        "removeBase64Images": True
    }
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.request("POST", FIRECRAWL_API_BASE, json=payload, headers=headers)
    
    # Find similar results
    similar_results = [result.url for result in exa.find_similar(
        url=url_to_scrape,
        num_results=DEFAULT_NUM_RESULTS,
        exclude_source_domain=True
    ).results]
    
    return {
        "content": response.json()["data"]["markdown"],
        "urls": [url_to_scrape] + similar_results,
        "tool_name": "webscrape"
    }


@tool
def arxiv_search(query: str) -> Dict[str, Any]:
    """Search for research papers on arxiv"""
    return {
        "contents": arxiv.run(query=query),
        "urls": [],
        "tool_name": "arxiv_search"
    }


@tool
def get_twitter_posts(query: str) -> Dict[str, Any]:
    """Get twitter posts based on a query"""
    result = exa.search_and_contents(
        query,
        type="auto",
        text=True,
        include_domains=SOCIAL_MEDIA_DOMAINS,
        num_results=DEFAULT_NUM_RESULTS
    )
    contents = [{"url": res.url, "content": res.text} for res in result.results]
    return {
        "contents": contents,
        "urls": [res.url for res in result.results],
        "tool_name": "get_twitter_posts"
    }


@tool
def get_youtube_videos(query: str) -> Dict[str, Any]:
    """Get youtube videos based on a query"""
    result = exa.search_and_contents(
        query,
        type="auto",
        text=True,
        include_domains=YOUTUBE_DOMAINS
    )
    contents = [{"url": res.url, "content": res.text} for res in result.results]
    return {
        "contents": contents,
        "urls": [res.url for res in result.results],
        "tool_name": "get_youtube_videos"
    }


@tool
def search_on_any_website(query: str, domain: str) -> Dict[str, Any]:
    """
    This tool allows for searching content related to a specific website. 
    Only call this tool if the user has specified a domain to search on. 
    (Those domain should not be youtube, twitter, arxiv)
    """
    result = exa.search_and_contents(
        query,
        type="auto",
        text=True,
        include_domains=[domain],
        num_results=DEFAULT_NUM_RESULTS
    )
    contents = [{"url": res.url, "content": res.text} for res in result.results]
    return {
        "contents": contents,
        "urls": [res.url for res in result.results],
        "tool_name": "search_on_any_website"
    }




# List of all available tools
ALL_TOOLS = [
    retrieve_web_content,
    webscrape,
    arxiv_search,
    get_twitter_posts,
    get_youtube_videos,
    summarize_youtube_video,
    search_on_any_website
]
