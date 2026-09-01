import unittest

from app.sentence_splitter import SentenceSplitter


class SentenceSplitterTests(unittest.TestCase):
    def test_first_sentence_cuts_on_comma(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("Hey Nelson, I'm ready. Let's go!")
        self.assertEqual(sentences, ["Hey Nelson,", "I'm ready.", "Let's go!"])

    def test_subsequent_sentences_cut_on_strong_punctuation(self):
        splitter = SentenceSplitter()
        splitter.feed("First part, then more. ")
        sentences = splitter.feed("Second sentence! Third?")
        self.assertEqual(sentences, ["Second sentence!", "Third?"])

    def test_flush_returns_remaining_text(self):
        splitter = SentenceSplitter()
        splitter.feed("No punctuation here")
        self.assertEqual(splitter.flush(), "No punctuation here")

    def test_flush_returns_none_when_empty(self):
        splitter = SentenceSplitter()
        self.assertIsNone(splitter.flush())

    def test_empty_feed_returns_nothing(self):
        splitter = SentenceSplitter()
        self.assertEqual(splitter.feed(""), [])

    def test_multiple_sentences_in_one_feed(self):
        splitter = SentenceSplitter()
        sentences = splitter.feed("One. Two. Three.")
        self.assertEqual(sentences, ["One.", "Two.", "Three."])


if __name__ == "__main__":
    unittest.main()
