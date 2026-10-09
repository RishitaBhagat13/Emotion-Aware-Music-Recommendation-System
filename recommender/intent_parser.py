# ==========================================================
# ACTIONS
# ==========================================================

ACTIONS = {
    "go",
    "walk",
    "finish",
    "study",
    "dance",
    "eat",
    "drink",
    "play",
    "work",
    "cook",
    "paint",
    "pull",
    "give",
    "meet",
    "tell",
    "decide",
    "change",
    "forget",
}

# ==========================================================
# OBJECTS
# ==========================================================

OBJECTS = {
    "computer",
    "book",
    "chair",
    "table",
    "bed",
    "hat",
    "shirt",
    "jacket",
    "clothes",
    "paper",
    "letter",
    "language"
}

# ==========================================================
# FOOD
# ==========================================================

FOOD = {
    "apple",
    "pizza",
    "corn",
    "candy",
    "fish",
    "orange"
}

# ==========================================================
# ANIMALS
# ==========================================================

ANIMALS = {
    "dog",
    "bird",
    "cow"
}

# ==========================================================
# PEOPLE
# ==========================================================

PEOPLE = {
    "mother",
    "woman",
    "man",
    "doctor",
    "secretary",
    "son",
    "cousin",
    "family"
}

# ==========================================================
# PLACES
# ==========================================================

PLACES = {
    "school",
    "city",
    "africa"
}

# ==========================================================
# SPORTS / HOBBIES
# ==========================================================

SPORTS = {
    "basketball",
    "bowling"
}

# ==========================================================
# COLORS
# ==========================================================

COLORS = {
    "black",
    "blue",
    "white",
    "pink",
    "purple",
    "brown",
    "orange",
    "color"
}

# ==========================================================
# TIME
# ==========================================================

TIME = {
    "before",
    "later",
    "last",
    "now",
    "year",
    "time",
    "birthday",
    "thanksgiving",
    "thursday"
}

# ==========================================================
# QUESTIONS
# ==========================================================

QUESTIONS = {
    "who",
    "what",
    "how",
    "can"
}

# ==========================================================
# HEALTH
# ==========================================================

HEALTH = {
    "doctor",
    "medicine",
    "deaf",
    "hearing",
    "accident"
}

# ==========================================================
# DESCRIPTIONS
# ==========================================================

DESCRIPTIONS = {
    "thin",
    "tall",
    "short",
    "hot",
    "cool",
    "dark",
    "full",
    "same",
    "right",
    "wrong",
    "many",
    "all"
}

# ==========================================================
# EMOTIONS / PREFERENCES
# ==========================================================

EMOTIONS = {
    "help",
    "enjoy",
    "fine",
    "like",
    "want",
    "need",
    "yes",
    "no"
}

# ==========================================================
# MISCELLANEOUS
# ==========================================================

MISC = {
    "kiss",
    "graduate",
    "cheat",
    "but"
}

# ==========================================================
# CLASSIFY A SINGLE WORD
# ==========================================================

def classify_word(word):

    word = word.lower()

    if word in ACTIONS:
        return "action"

    elif word in OBJECTS:
        return "object"

    elif word in FOOD:
        return "food"

    elif word in ANIMALS:
        return "animal"

    elif word in PEOPLE:
        return "person"

    elif word in PLACES:
        return "place"

    elif word in SPORTS:
        return "sport"

    elif word in COLORS:
        return "color"

    elif word in TIME:
        return "time"

    elif word in QUESTIONS:
        return "question"

    elif word in HEALTH:
        return "health"

    elif word in DESCRIPTIONS:
        return "description"

    elif word in EMOTIONS:
        return "emotion"

    elif word in MISC:
        return "misc"

    else:
        return "unknown"
    
# ==========================================================
# PARSE COMPLETE SENTENCE
# ==========================================================

def parse(sentence, emotion):

    words = sentence.lower().split()

    parsed = {

        "source": "video",

        "query": sentence,

        "emotion": emotion,

        "actions": [],

        "objects": [],

        "foods": [],

        "animals": [],

        "people": [],

        "places": [],

        "sports": [],

        "colors": [],

        "time": [],

        "questions": [],

        "health": [],

        "descriptions": [],

        "feelings": [],

        "misc": [],

        "keywords": []
    }

    for word in words:

        category = classify_word(word)

        if category == "action":
            parsed["actions"].append(word)

        elif category == "object":
            parsed["objects"].append(word)

        elif category == "food":
            parsed["foods"].append(word)

        elif category == "animal":
            parsed["animals"].append(word)

        elif category == "person":
            parsed["people"].append(word)

        elif category == "place":
            parsed["places"].append(word)

        elif category == "sport":
            parsed["sports"].append(word)

        elif category == "color":
            parsed["colors"].append(word)

        elif category == "time":
            parsed["time"].append(word)

        elif category == "question":
            parsed["questions"].append(word)

        elif category == "health":
            parsed["health"].append(word)

        elif category == "description":
            parsed["descriptions"].append(word)

        elif category == "emotion":
            parsed["feelings"].append(word)

        elif category == "misc":
            parsed["misc"].append(word)

        parsed["keywords"].append(word)

    return parsed

    # ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    sentence = "play computer"

    emotion = "Neutral"

    result = parse(sentence, emotion)

    from pprint import pprint

    pprint(result)