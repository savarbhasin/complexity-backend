import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def format_messages(messages) -> str:
    """Format messages for the prompt"""
    text = ""
    for i in messages:
        if isinstance(i, HumanMessage):
            text += f"Human: {i.content}\n"
        elif isinstance(i, AIMessage):
            text += f"AI: {i.content}\n"
        elif isinstance(i, ToolMessage):
            tool_content = json.loads(i.content)
            text += f"Tool ({tool_content.get('tool_name', 'unknown')}): {tool_content}\n"
    return text