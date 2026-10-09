from sentence_builder import SentenceBuilder

builder = SentenceBuilder()

builder.add_word("help")
builder.add_word("help")
builder.add_word("computer")
builder.add_word("mouse")

print(builder.get_sentence())