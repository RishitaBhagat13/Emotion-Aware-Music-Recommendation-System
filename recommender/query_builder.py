# ==========================================================
# QUERY BUILDER
# ==========================================================

def build_query(parsed):

    # ======================================================
    # EXTRACT PARSED INFORMATION
    # ======================================================

    emotion = parsed["emotion"]

    actions = parsed["actions"]
    objects = parsed["objects"]
    foods = parsed["foods"]
    animals = parsed["animals"]
    people = parsed["people"]
    places = parsed["places"]
    sports = parsed["sports"]
    colors = parsed["colors"]
    health = parsed["health"]
    descriptions = parsed["descriptions"]
    feelings = parsed["feelings"]
    time = parsed["time"]

    # Final search query
    query = []

    # ======================================================
    # EMOTION PREFIX
    # ======================================================

    emotion_prefix = {
        "Happy": "happy",
        "Sad": "calm",
        "Angry": "relaxing",
        "Fear": "peaceful",
        "Surprise": "energetic",
        "Disgust": "soft",
        "Neutral": ""
    }.get(emotion, "")

        # ======================================================
    # STAGE 1 : HIGH LEVEL INTENT RULES
    # ======================================================

    # ------------------------------------------------------
    # STUDY
    # ------------------------------------------------------

    if "study" in actions and "book" in objects:
        return f"{emotion_prefix} lofi study beats".strip()

    if "study" in actions and "computer" in objects:
        return f"{emotion_prefix} coding study music".strip()

    if "study" in actions and "paper" in objects:
        return f"{emotion_prefix} exam preparation music".strip()

    if "study" in actions and "language" in objects:
        return f"{emotion_prefix} language learning music".strip()

    if "study" in actions and "school" in places:
        return f"{emotion_prefix} deep focus study playlist".strip()

    # ------------------------------------------------------
    # COMPUTER
    # ------------------------------------------------------

    if "play" in actions and "computer" in objects:
        return f"{emotion_prefix} coding music".strip()

    if "work" in actions and "computer" in objects:
        return f"{emotion_prefix} programming focus playlist".strip()

    # ------------------------------------------------------
    # FOOD
    # ------------------------------------------------------

    if "eat" in actions and "pizza" in foods:
        return f"{emotion_prefix} pizza party playlist".strip()

    if "cook" in actions and foods:
        return f"{emotion_prefix} cooking playlist".strip()

    if "drink" in actions:
        return f"{emotion_prefix} cafe music".strip()

    # ------------------------------------------------------
    # SPORTS
    # ------------------------------------------------------

    if "play" in actions and "basketball" in sports:
        return f"{emotion_prefix} sports motivation".strip()

    if "basketball" in sports:
        return f"{emotion_prefix} basketball playlist".strip()

    if "bowling" in sports:
        return f"{emotion_prefix} fun sports songs".strip()

    # ------------------------------------------------------
    # FAMILY
    # ------------------------------------------------------

    if "mother" in people and "birthday" in time:
        return f"{emotion_prefix} mother birthday songs".strip()

    if "family" in people and "birthday" in time:
        return f"{emotion_prefix} family celebration songs".strip()

    if "mother" in people:
        return f"{emotion_prefix} emotional family songs".strip()

    if "family" in people:
        return f"{emotion_prefix} family playlist".strip()

    # ------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------

    if "doctor" in health and "medicine" in health:
        return f"{emotion_prefix} healing instrumental".strip()

    if "doctor" in health:
        return f"{emotion_prefix} healing music".strip()

    if "accident" in health:
        return f"{emotion_prefix} stress relief music".strip()

    # ------------------------------------------------------
    # TRAVEL
    # ------------------------------------------------------

    if "go" in actions and "city" in places:
        return f"{emotion_prefix} travel playlist".strip()

    if "go" in actions and "africa" in places:
        return f"{emotion_prefix} african travel music".strip()
    
        # ======================================================
    # STAGE 2 : FALLBACK RULES
    # ======================================================

    # ------------------------------------------------------
    # EMOTION
    # ------------------------------------------------------

    if emotion_prefix:
        query.append(emotion_prefix)

    # ------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------

    action_map = {

        "study": "study music",

        "work": "focus music",

        "dance": "dance songs",

        "walk": "walking playlist",

        "play": "songs",

        "eat": "dinner music",

        "drink": "cafe music",

        "cook": "cooking songs",

        "paint": "creative music",

        "meet": "friends playlist",

        "tell": "story songs",

        "give": "motivational songs",

        "forget": "relaxing music",

        "change": "uplifting music",

        "decide": "thinking music",

        "finish": "celebration songs",

        "go": "travel music",

        "pull": "gym motivation"

    }

    for action in actions:

        if action in action_map:

            query.append(
                action_map[action]
            )

        # ======================================================
    # OBJECTS
    # ======================================================

    object_map = {

        "computer": "coding music",

        "book": "reading music",

        "paper": "study playlist",

        "language": "language learning music",

        "chair": "lofi beats",

        "table": "background music",

        "bed": "sleep music",

        "shirt": "fashion music",

        "jacket": "fashion music",

        "clothes": "fashion music",

        "hat": "fashion music",

        "letter": "instrumental music"

    }

    for obj in objects:

        if obj in object_map:

            query.append(
                object_map[obj]
            )

        # ======================================================
    # FOOD
    # ======================================================

    if foods:
        query.append("restaurant music")

    # ======================================================
    # ANIMALS
    # ======================================================

    animal_map = {

        "dog": "happy pet songs",

        "bird": "nature sounds",

        "cow": "country music"

    }

    for animal in animals:

        if animal in animal_map:

            query.append(
                animal_map[animal]
            )

    # ======================================================
    # HEALTH
    # ======================================================

    health_map = {

        "doctor": "healing music",

        "medicine": "calming music",

        "accident": "stress relief music",

        "deaf": "instrumental music",

        "hearing": "relaxing instrumental"

    }

    for item in health:

        if item in health_map:

            query.append(
                health_map[item]
            )

    # ======================================================
    # SPORTS
    # ======================================================

    sports_map = {

        "basketball": "sports motivation",

        "bowling": "fun songs"

    }

    for sport in sports:

        if sport in sports_map:

            query.append(
                sports_map[sport]
            )

    # ======================================================
    # PLACES
    # ======================================================

    place_map = {

        "school": "student study playlist",

        "city": "city pop",

        "africa": "african music"

    }

    for place in places:

        if place in place_map:

            query.append(
                place_map[place]
            )

    # ======================================================
    # COLORS
    # ======================================================

    if colors:

        query.append(
            "colorful vibes"
        )

    # ======================================================
    # DESCRIPTIONS
    # ======================================================

    if descriptions:

        query.append(
            "mood music"
        )

    # ======================================================
    # FEELINGS
    # ======================================================

    feeling_map = {

        "help": "comfort songs",

        "enjoy": "feel good songs",

        "fine": "positive music",

        "like": "popular songs",

        "want": "trending songs",

        "need": "motivational music",

        "yes": "happy hits",

        "no": "relaxing music"

    }

    for feeling in feelings:

        if feeling in feeling_map:

            query.append(
                feeling_map[feeling]
            )

        # ======================================================
    # REMOVE DUPLICATES
    # ======================================================

    query = list(dict.fromkeys(query))

    # ======================================================
    # REMOVE EMPTY STRINGS
    # ======================================================

    query = [q for q in query if q.strip()]

    # ======================================================
    # DEFAULT QUERY
    # ======================================================

    if len(query) == 0:

        return "recommended songs"

    # ======================================================
    # RETURN FINAL QUERY
    # ======================================================

    return " ".join(query)