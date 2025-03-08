from copilotkit import CopilotKitSDK, LangGraphAgent 
from updated_graph import workflow 
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from fastapi import FastAPI
 
app = FastAPI()
 

sdk = CopilotKitSDK(
  agents=[ 
    LangGraphAgent(
      name="complexity_ai",
      description="Agent that knows everything",
      graph=workflow,
    )
  ],
)
 
 
add_fastapi_endpoint(app, sdk, "/copilotkit")

def main():
    """Run the uvicorn server."""
    import uvicorn
    uvicorn.run("server:app", host="localhost", port=8000, reload=True)
 
if __name__ == "__main__":
    main()