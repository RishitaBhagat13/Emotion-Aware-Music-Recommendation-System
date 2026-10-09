from recommender.spotify_api import search_songs
from recommender.search_optimizer import optimize_query

# ==========================================================
# RECOMMENDATION ENGINE
# ==========================================================

def recommend(parsed, limit=5):

    query = optimize_query(parsed)

    print()
    print("=" * 60)
    print("Optimized Query")
    print(query)
    print("=" * 60)

    # Search Spotify
    songs = search_songs(query, limit)

    return songs


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    parsed = {

        "emotion": "Happy",

        "actions": ["study"],

        "objects": ["computer"]

    }

    recommendations = recommend(parsed)

    print()
    print("Recommendations")
    print("-" * 60)

    for song in recommendations:

        print(song["title"])
        print(song["artist"])
        print(song["album"])
        print(song["url"])
        print()