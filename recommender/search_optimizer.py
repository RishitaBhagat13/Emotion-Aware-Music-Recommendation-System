# ==========================================================
# SEARCH QUERY OPTIMIZER
# ==========================================================

def optimize_query(parsed):

    emotion = parsed["emotion"]

    actions = parsed["actions"]

    objects = parsed["objects"]

    query = []

    # ------------------------------------------------------
    # EMOTION
    # ------------------------------------------------------

    if emotion == "Happy":

        query.append("happy")

    elif emotion == "Sad":

        query.append("calm")

    elif emotion == "Angry":

        query.append("focus")

    elif emotion == "Fear":

        query.append("peaceful")

    elif emotion == "Surprise":

        query.append("energetic")

    # ------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------

    if "study" in actions:

        query.append("study playlist")

    elif "work" in actions:

        query.append("work playlist")

    elif "dance" in actions:

        query.append("dance hits")

    elif "walk" in actions:

        query.append("walking playlist")

    elif "play" in actions:

        query.append("party songs")

    # ------------------------------------------------------
    # OBJECTS
    # ------------------------------------------------------

    if "computer" in objects:

        query.append("coding lofi")

    elif "book" in objects:

        query.append("reading music")

    elif "bed" in objects:

        query.append("sleep music")

    # ------------------------------------------------------

    if len(query) == 0:

        return "trending songs"

    return " ".join(query)

