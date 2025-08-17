"""Data models and state definitions for the Perplexity backend."""

from typing import Annotated, Sequence
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ComplexityState(BaseModel):
    """State model for the complexity AI graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages] = []


class SearchResult(BaseModel):
    """Model for search result data."""
    title: str
    author: str = ""
    content: str
    url: str
    date: str = ""


class ToolResponse(BaseModel):
    """Standard response format for tools."""
    contents: list | str
    urls: list[str]
    tool_name: str
