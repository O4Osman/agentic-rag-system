from dotenv import load_dotenv

load_dotenv()

from typing import Dict, Any

from langchain_core.tracers import langchain

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state['question']
    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({'query': question})
    joined_tavily_result = "\n".join(
        result["content"]
        for result in tavily_results["results"]
    )
    web_results = Document(joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {'question': question, 'documents': documents}


if __name__ == "__main__":
    web_search(state={'question': 'agent memory', 'documents': None})
