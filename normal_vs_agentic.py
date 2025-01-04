from duckduckgo_search import DDGS
from tavily import TavilyClient


def traditional_search(query):
    ddg = DDGS()
    results = ddg.text(query, max_results=5)
    return results[0]['body']


def agentic_search(query):
    client = TavilyClient(api_key="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")
    
    # Perform an analysis-focused search
    results = client.search(
        query=query,
        search_depth="advanced",
        include_answer=True,
        analyze_results=True
    )
    
    return results.get('answer'), results.get('analysis')

# Combining both approaches for comprehensive research
def comprehensive_research(topic):
    # Traditional search for broad coverage
    basic_results = traditional_search(topic)
    
    # Agentic search for deeper analysis
    answer, _ = agentic_search(f"Analyze latest developments in {topic}")
    
    return {
        "basic_results": basic_results,
        "tavily_answer": answer
    }

print(comprehensive_research('Is AI going to take over the world?'))