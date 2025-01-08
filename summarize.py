from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from typing import List, Annotated, TypedDict, Literal
from langchain_core.documents import Document
import operator
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from dotenv import load_dotenv
from langgraph.constants import Send
from langchain.chains.combine_documents.reduce import (acollapse_docs,split_list_of_docs,)
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

token_max = 1000

map_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer, having an indepth knowledge of youtube videos. Given an excerpt of a youtube video, you are able to identify the key points and summarize them in a clear and concise manner."),
    ("human", "Generate a summary of the given text. Ensure the summary is clear, concise, and captures the key points in a professional manner. {input}")
])

map_prompt = hub.pull("rlm/map-prompt")
reduce_prompt = hub.pull("rlm/reduce-prompt")


llms = {
    "llama3-70b-8192": ChatGroq(model = 'llama3-70b-8192', temperature=0.2),
    "llama3-8b-8192": ChatGroq(model = 'llama3-8b-8192', temperature=0.2),
    "llama-3.1-8b-instant": ChatGroq(model = 'llama-3.1-8b-instant', temperature=0.2),
    "llama-guard-3-8b": ChatGroq(model = 'llama-guard-3-8b', temperature=0.2),
    "mixtral-8x7b-32768": ChatGroq(model = 'mixtral-8x7b-32768', temperature=0.2),
    "gemma2-9b-it": ChatGroq(model = 'gemma2-9b-it', temperature=0.2),
    "gemma-7b-it": ChatGroq(model = 'gemma-7b-it', temperature=0.2),
}

llm = ChatGroq(model = 'llama3-70b-8192', temperature=0.2)
reduce_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


def text_splitter(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=250)
    splitted =  splitter.split_text(text)
    return splitted

class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

class SummaryState(TypedDict):
    content: str


graph = StateGraph(OverallState)

def get_tokens_length(documents: List[Document])->int:
    """get the total number of tokens in a list of texts"""    
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

async def generate_summary(state: SummaryState):
    prompt = map_prompt.invoke(state["content"])
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}

def map_summaries(state: OverallState):
    return [Send("generate_summary", {"content": content}) for content in state["contents"]]

def collect_summaries(state: OverallState):
    # only adds the summaries to the state as documents
    return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}

async def _reduce(input: dict) -> str:
    prompt = reduce_prompt.invoke(input)
    response = await reduce_llm.ainvoke(prompt)
    return response.content

async def collapse_summaries(state: OverallState):
    # Split the list of summary documents into smaller lists based on token length
    doc_lists = split_list_of_docs(state["collapsed_summaries"], get_tokens_length, token_max)
    results = []
    for doc_list in doc_lists:
        # Collapse each sub-list into a single summary using _reduce function
        # acollapse_docs handles the async reduction of multiple docs into one
        results.append(await acollapse_docs(doc_list, _reduce))
        
    # Return the collapsed summaries as a state update
    return {"collapsed_summaries": results}


def should_collapse(state: OverallState,) -> Literal["collapse_summaries", "generate_final_summary"]:
    # if length of all summaries is greater than token_max, collapse them
    num_tokens = get_tokens_length(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


async def generate_final_summary(state: OverallState):
    response = await _reduce(state["collapsed_summaries"])
    print(response)
    return {"final_summary": response}


graph.add_node("generate_summary", generate_summary)
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)


graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()

async def summarize_text(text):
    text = text_splitter(text)
    async for step in app.astream(
        {"contents": text},
        {"recursion_limit": 10},
    ):
        print(list(step.keys()))
   



