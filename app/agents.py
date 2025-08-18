from datetime import datetime
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.models import ComplexityState
from app.tools import ALL_TOOLS
from app.utils import format_messages
from app.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


# Prompt template for the AI assistant
COMPLEXITY_PROMPT = PromptTemplate.from_template("""
You are complexity.ai, an AI assistant with access to advanced tools. Your job is to answer user questions by:
- Synthesizing information from multiple sources into a clear, narrative summary
- Citing sources naturally within the text (not as a list at the end)
- Adapting technical depth to the user's apparent knowledge
- Using Markdown formatting for clarity (headings, lists, links)
- Only providing information, not code

# TOOLS INSTRUCTIONS:
- Use tools as needed to gather and synthesize information
- After receiving the result of a tool, analyze the result: if you need more information, you may call another tool. 
- DO NOT RUSH to answer the user if more tool calls are needed.
- After using all necessary tools, integrate their results into your narrative answer
- ONLY GIVE THE ANSWER AFTER USING ALL NECESSARY TOOLS.
                                                 
Directly start with the answer. Dont use terms like 'Here is the answer' or 'Here is the information' or 'Here is the result'.
                                                                        
# USER CHAT: {messages}
""")


def create_llm():
    """Create and configure the language model."""
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL, 
        temperature=DEFAULT_TEMPERATURE,
    )
    return llm.bind_tools(tools=ALL_TOOLS)


async def chatbot(state: ComplexityState, config: RunnableConfig):
    """Main chatbot function that processes user messages."""

    reduced_tokens_messages = format_messages(state.messages)
    date = datetime.now().strftime("%Y-%m-%d")

    llm = create_llm()
    chain = COMPLEXITY_PROMPT | llm
    
    answer = await chain.ainvoke({
        "messages": reduced_tokens_messages,
        "date": date
    }, config)

    return {"messages": [answer]}


async def streaming_chatbot(state: ComplexityState, config: RunnableConfig):
    """Streaming version of the chatbot function."""
    
    reduced_tokens_messages = format_messages(state.messages)
    date = datetime.now().strftime("%Y-%m-%d")

    llm = create_llm()
    chain = COMPLEXITY_PROMPT | llm
    
    # Stream the response
    full_content = ""
    chunks = []
    
    async for chunk in chain.astream({
        "messages": reduced_tokens_messages,
        "date": date
    }, config):
        chunks.append(chunk)
        if hasattr(chunk, 'content'):
            full_content += chunk.content
    
    # Return the final complete message for the state
    if chunks:
        # Use the last chunk as the final message but with complete content
        final_chunk = chunks[-1]
        if hasattr(final_chunk, 'content'):
            final_chunk.content = full_content
        return {"messages": [final_chunk]}
    
    return {"messages": []}


def create_workflow():
    """Create and configure the LangGraph workflow."""
    # Create tool node
    tool_node = ToolNode(ALL_TOOLS)
    
    # Create the graph
    graph = StateGraph(ComplexityState)
    graph.add_node("chat", chatbot)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    graph.set_entry_point("chat")
    
    # Create memory saver and compile workflow
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def create_streaming_workflow():
    """Create and configure the LangGraph workflow with streaming support."""
    # Create tool node
    tool_node = ToolNode(ALL_TOOLS)
    
    # Create the graph with streaming chatbot
    graph = StateGraph(ComplexityState)
    graph.add_node("chat", streaming_chatbot)
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    graph.set_entry_point("chat")
    
    # Create memory saver and compile workflow
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
