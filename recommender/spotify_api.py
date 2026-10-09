# ==========================================================
# SPOTIFY API
# ==========================================================

import os

from dotenv import load_dotenv

import spotipy

from spotipy.oauth2 import SpotifyClientCredentials


# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()


CLIENT_ID = os.getenv(
    "SPOTIFY_CLIENT_ID"
)

CLIENT_SECRET = os.getenv(
    "SPOTIFY_CLIENT_SECRET"
)

REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI"
)

# ==========================================================
# CREATE SPOTIFY CLIENT
# ==========================================================

spotify = spotipy.Spotify(

    auth_manager=SpotifyClientCredentials(

        client_id=CLIENT_ID,

        client_secret=CLIENT_SECRET

    )

)

# ==========================================================
# SEARCH SONGS
# ==========================================================

def search_songs(query, limit=5):

    results = spotify.search(

        q=query,

        type="track",

        limit=limit

    )

    songs = []

    for item in results["tracks"]["items"]:

        songs.append({

            "title": item["name"],

            "artist": ", ".join(
                artist["name"]
                for artist in item["artists"]
            ),

            "album": item["album"]["name"],

            "thumbnail": item["album"]["images"][0]["url"],

            "url": item["external_urls"]["spotify"]

        })

    return songs

