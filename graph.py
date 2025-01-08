from langgraph.graph import StateGraph, END, START
from typing import List, Optional
from pydantic import BaseModel
from exa_py import Exa
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
# from summarize import summarize_text
from langgraph.types import Command
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from copilotkit.langchain import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
import re
from langchain_community.utilities import ArxivAPIWrapper
import requests
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langgraph.graph.message import add_messages
from copilotkit.langchain import copilotkit_emit_message
import os
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI

load_dotenv()


# openai = OpenAI()
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

arxiv = ArxivAPIWrapper()


class ComplexityState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages] = []
    
    # get_youtube_videos
    youtube_urls: Optional[List[str]] = None

    # retrieve_web_content
    web_urls: Optional[List[str]] = None

    # webscrape
    similar_urls: Optional[List[str]] = None

    # arxiv_search



# youtube summarize tool
@tool
def summarize_youtube_video(url:str)->str:
    """Summarize a youtube video given the url"""
    regex = r"(?:https?:\/\/(?:www\.|m\.)?youtube\.com\/(?:watch\?v=|v\/)|https?:\/\/youtu\.be\/)([a-zA-Z0-9_-]{11})(?:[&?][^\s]*)?"
    video_id = re.search(regex, url).group(1)

    headers = { "Content-Type": "application/json" }
    
    response = requests.get(f"https://yt-fastapi-backend.onrender.com/summary/{video_id}", headers=headers)
    return response.json()

# exa search tool
@tool
def retrieve_web_content(query: str) -> List[str]:
    """Find latest web content based on a query"""
    data = exa.search_and_contents(query=query, num_results=5, text=True, use_autoprompt=True, exclude_domains=["youtube.com", "twitter.com", "x.com", "arxiv.org"])
    formatted_data = [ 
        {   
            "title": res.url, 
            "author": res.author,
            "content": res.text, 
            "url": res.url,
            "date": res.published_date
        } 
        for res in data.results]
    urls = [res.url for res in data.results]
    return formatted_data

# webscrape tool
@tool
def webscrape(url_to_scrape:str):
    """Webscrape a given URL and return a list of matching webpages"""

    url = "https://api.firecrawl.dev/v1/scrape"

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
        "Authorization": f"Bearer {os.getenv('FIRECRAWL_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    
    similar_results = [result.url for result in exa.find_similar(
        url=url_to_scrape,
        num_results=4,
        exclude_source_domain=True
    ).results]
 
    return response.json()["data"]["markdown"], similar_results

# arxiv tool
@tool
def arxiv_search(query: str):
    """Search for research papers on arxiv"""
    return arxiv.run(query=query)

# x/twitter posts
@tool
def get_twitter_posts(query:str)->List[str]:
    """Get twitter posts based on a query"""
    result = exa.search_and_contents(
        query,
        type="auto",
        text=True,
        include_domains=["twitter.com", "x.com"]
    )
    contents = [ {"url": res.url, "content": res.text} for res in result.results]
    return contents

# youtube videos
@tool
def get_youtube_videos(query:str)->List[str]:
    """Get youtube videos based on a query"""
    result = exa.search_and_contents(
        query,
        type="auto",
        text=True,
        include_domains=["youtube.com", "youtu.be"]
    )
    contents = [ {"url": res.url, "content": res.text} for res in result.results]
    return contents


prompt = PromptTemplate.from_template("""
    ## SYSTEM INSTRUCTIONS:
    You are complexity.ai, a versatile AI assistant equipped with advanced tools to answer user questions effectively. 
    Your responses must:
    - Provide comprehensive, narrative-style summaries that weave together information from multiple sources
    - Naturally integrate citations and references within the flow of the text
    - Focus on synthesizing key insights and connections across sources
    - Maintain a cohesive story-like structure rather than numbered lists or source-by-source breakdowns
    - Support claims with evidence while keeping the narrative engaging
    - Adapt the technical depth based on the user's apparent knowledge level
    - Use clear, professional language that emphasizes clarity and readability
                                      
    ## TOOL BEHAVIOR INSTRUCTIONS:
    You have the following tools available to assist users: retrieve_web_content, webscrape, arxiv_search, get_twitter_posts, get_youtube_videos, summarize_youtube_video
    
    For all tools, focus on creating a flowing narrative that:
    - Synthesizes information across sources into a unified story
    - Embeds citations naturally within sentences and paragraphs
    - Highlights connections and patterns between different sources
    - Avoids numbered lists or source-by-source summaries
    - Creates a coherent reading experience
    
    retrieve_web_content:
    Weave together insights from multiple documents into a unified narrative, citing sources naturally within the text. Focus on key themes and connections rather than listing individual sources.
    
    webscrape:
    Provide a flowing summary of the main content and similar sites, integrating comparisons naturally rather than as separate points. Citations should feel organic within the narrative.
    
    arxiv_search:
    Create a cohesive story about the research landscape, connecting papers' findings and implications while citing naturally. Avoid isolating individual paper summaries.
    
    get_twitter_posts:
    Craft a narrative about the social conversation, weaving together trends and notable posts while citing handles/hashtags within the flow of discussion.
    
    get_youtube_videos:
    Tell a story about the video content landscape, naturally incorporating video details and creators while maintaining narrative flow rather than listing videos separately.
    
    summarize_youtube_video:
    return the summary as it is

    # USER CHAT: {messages}
""")


def format_messages(messages)->str:
    text = ""
    for i in messages:
        if isinstance(i, HumanMessage):
            text += f"Human: {i.content}\n"
        elif isinstance(i, AIMessage):
            text += f"AI: {i.content}\n"
        elif isinstance(i, ToolMessage):
            text += f"Tool: {i.content}\n"
    return text

def chatbot(state:ComplexityState, config: RunnableConfig):
     
    config = copilotkit_customize_config(
        config,
        emit_tool_calls=["retrieve_web_content", "webscrape", "arxiv_search", "get_twitter_posts", "get_youtube_videos", "summarize_youtube_video"],
    )
    
    messages = format_messages(state.messages)
    chain = prompt | openai_final
    answer = chain.invoke(messages, config)
    return {"messages": [answer]}

   

tools = [retrieve_web_content, webscrape, arxiv_search, get_twitter_posts, get_youtube_videos, summarize_youtube_video]
tool_node = ToolNode(tools)


openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True).bind_tools(tools=tools);
openai_final = openai.with_config(tags=["final_node"])



graph = StateGraph(ComplexityState)

graph.add_node("chat", chatbot)
graph.add_node("tools", tool_node)
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")


graph.set_entry_point("chat")


memory = MemorySaver()
workflow = graph.compile(checkpointer=memory)



# async def process_events():
#     async for event in workflow.astream_events({"messages": [
#                 {"role": "human", "content": "latest x posts on openai"}
#         ]}, version='v1'):
#         kind = event["event"]
#         tags = event.get("tags", [])
        
#         if kind == "on_chat_model_stream" and "final_node" in event.get("tags", []):
#             data = event["data"]
#             if data["chunk"].content:
#                 print(data["chunk"].content, end="", flush=True)

# async def main():
#     await process_events()

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

# x = workflow.invoke({"messages": [
#     {"role": "human", "content": "latest news on open ai"},
# ]}, debug=True)
# print(x["messages"][-1].content)





