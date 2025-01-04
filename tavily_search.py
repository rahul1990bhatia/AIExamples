from tavily import TavilyClient

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

# Example usage
answer, _ = agentic_search(
    "Compare the environmental impact of electric vs gas cars"
)
print(f'answer: {answer}')