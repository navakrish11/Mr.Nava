import requests

def search_duckduckgo(query):
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1"
        response = requests.get(url)
        data = response.json()

        results = []
        if data.get("Abstract"):
            results.append({
                "title": "DuckDuckGo Abstract",
                "link": data.get("AbstractURL"),
                "snippet": data.get("Abstract")
            })
        if data.get("RelatedTopics"):
            for topic in data["RelatedTopics"][:3]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text").split("-")[0],
                        "link": topic.get("FirstURL"),
                        "snippet": topic.get("Text")
                    })
        return results
    except Exception as e:
        return [{"title": "Error", "link": "", "snippet": str(e)}]
