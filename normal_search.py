from duckduckgo_search import DDGS


def traditional_search(query):
    ddg = DDGS()
    results = ddg.text(query, max_results=5)
    print(results)
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Link: {result['href']}")
        print(f"Snippet: {result['body']}\n")

# Example usage
traditional_search("What is artificial intelligence?")