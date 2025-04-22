import wikipedia

def search_wikipedia(query):
    try:
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        return {
            "title": "Wikipedia",
            "link": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "snippet": summary
        }
    except:
        return None
