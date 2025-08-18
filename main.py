
import os
import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agents import create_streaming_workflow
from app.models import ComplexityState
from langchain_core.messages import HumanMessage, AIMessageChunk


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


app = FastAPI(title="Complexity SSE API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create streaming workflow instance
workflow = create_streaming_workflow()


def format_sse(data: dict) -> str:
    """Format data for Server-Sent Events"""
    return f"data: {json.dumps(data)}\n\n"


async def stream_workflow_response(message: str, thread_id: str) -> AsyncGenerator[str, None]:
    """Stream the workflow response"""
    
    try:

        # Create initial state with user message
        initial_state = ComplexityState(messages=[
            HumanMessage(content=message)
        ])
        
        # Stream the workflow
        async for step in workflow.astream(initial_state, {"thread_id": thread_id}, stream_mode=['messages', 'updates']):

            if step[0] == 'messages' and isinstance(step[1][0], AIMessageChunk):
                yield format_sse({
                    "type": "chunk",
                    "content": step[1][0].content,
                    "timestamp": datetime.now().isoformat()
                })
            elif step[0] == 'updates':
                step = step[1]
                # Extract and send any chat messages
                if "chat" in step:
                    messages = step["chat"].get("messages", [])
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            yield format_sse({
                                "type": "content",
                                "content": msg.content,
                                "timestamp": datetime.now().isoformat()
                            })
                        if hasattr(msg, 'tool_calls'):
                            for tool_call in msg.tool_calls:
                                yield format_sse({
                                    "type": "tool_call",
                                    "tool_call": tool_call["name"],
                                    "timestamp": datetime.now().isoformat()
                                })
                        

                if "tools" in step:
                    messages = step["tools"]["messages"]
                    for message in messages:
                        if hasattr(message, 'content') and message.content:
                            contents = json.loads(message.content)
                            urls = contents['urls']
                            yield format_sse({
                                "type": "tool_result",
                                "urls": urls,
                                "timestamp": datetime.now().isoformat()
                            })
                
        
    except Exception as e:
        yield format_sse({
            "type": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/chat/stream")
async def stream_chat(chat_request: ChatRequest):
    """Stream chat responses using Server-Sent Events"""
    
    return StreamingResponse(
        stream_workflow_response(chat_request.message, chat_request.thread_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )



def main():
    """Run the SSE server"""
    import uvicorn
    uvicorn.run(
        "main:app",
        host="localhost",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
