from collections import deque, Counter


class EmotionTracker:
    """
    Tracks stable emotions over time.

    Features
    --------
    ✔ Removes emotion flickering
    ✔ Majority voting
    ✔ Returns stable emotion
    ✔ Stores confidence
    """

    def __init__(self, window_size=10, stable_threshold=6):

        self.window = deque(maxlen=window_size)

        self.current_emotion = "Neutral"

        self.current_confidence = 0.0

        self.stable_count = 0

        self.last_prediction = ""

        self.stable_threshold = stable_threshold

    # =====================================

    def update(self, emotion, confidence):

        self.window.append(emotion)

        stable_emotion = Counter(
            self.window
        ).most_common(1)[0][0]

        if stable_emotion == self.last_prediction:

            self.stable_count += 1

        else:

            self.last_prediction = stable_emotion

            self.stable_count = 1

        if self.stable_count >= self.stable_threshold:

            self.current_emotion = stable_emotion

            self.current_confidence = confidence

        return self.current_emotion

    # =====================================

    def get_emotion(self):

        return self.current_emotion

    # =====================================

    def get_confidence(self):

        return self.current_confidence

    # =====================================

    def reset(self):

        self.window.clear()

        self.current_emotion = "Neutral"

        self.current_confidence = 0.0

        self.stable_count = 0

        self.last_prediction = ""