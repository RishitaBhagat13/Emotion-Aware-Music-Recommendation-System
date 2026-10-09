from typing import List


class SentenceBuilder:
    """
    Builds a sentence from predicted sign words.

    Features
    --------
    ✓ Prevents duplicate consecutive words
    ✓ Allows removing last word
    ✓ Returns complete sentence
    ✓ Can clear sentence
    """

    def __init__(self):

        self.words: List[str] = []

    # -----------------------------------------
    # Add a new word
    # -----------------------------------------

    def add_word(self, word: str):

        word = word.strip().lower()

        if word == "":
            return

        # Prevent duplicates
        if len(self.words) > 0:

            if self.words[-1] == word:
                return

        self.words.append(word)

    # -----------------------------------------
    # Remove last word
    # -----------------------------------------

    def remove_last_word(self):

        if len(self.words) > 0:
            self.words.pop()

    # -----------------------------------------
    # Clear sentence
    # -----------------------------------------

    def clear(self):

        self.words = []

    # -----------------------------------------
    # Return sentence
    # -----------------------------------------

    def get_sentence(self):

        return " ".join(self.words)

    # -----------------------------------------
    # Number of words
    # -----------------------------------------

    def word_count(self):

        return len(self.words)

    # -----------------------------------------
    # Check if empty
    # -----------------------------------------

    def is_empty(self):

        return len(self.words) == 0

    # -----------------------------------------
    # Return list of words
    # -----------------------------------------

    def get_words(self):

        return self.words.copy()