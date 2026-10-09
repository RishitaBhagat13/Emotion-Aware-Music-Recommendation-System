class SentenceBuilder:
    """
    Builds a sentence from predicted words.

    Features
    --------
    ✔ Prevents duplicate consecutive words
    ✔ Remove last word
    ✔ Clear sentence
    ✔ Return complete sentence
    """

    def __init__(self):
        self.words = []

    # ----------------------------------
    # Add a new word
    # ----------------------------------
    def add_word(self, word):

        word = word.strip().lower()

        if word == "":
            return

        # Ignore consecutive duplicates
        if len(self.words) > 0 and self.words[-1] == word:
            return

        self.words.append(word)

    # ----------------------------------
    # Remove last word
    # ----------------------------------
    def remove_last_word(self):

        if len(self.words) > 0:
            self.words.pop()

    # ----------------------------------
    # Clear sentence
    # ----------------------------------
    def clear(self):
        self.words.clear()

    # ----------------------------------
    # Return sentence
    # ----------------------------------
    def get_sentence(self):
        return " ".join(self.words)

    # ----------------------------------
    # Return list of words
    # ----------------------------------
    def get_words(self):
        return self.words.copy()

    # ----------------------------------
    # Check if sentence is empty
    # ----------------------------------
    def is_empty(self):
        return len(self.words) == 0

    # ----------------------------------
    # Number of words
    # ----------------------------------
    def word_count(self):
        return len(self.words)