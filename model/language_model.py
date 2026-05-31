"""
Language Model for OCR Post-Processing
========================================

This module provides language model integration to improve OCR output quality.
It corrects recognition errors by leveraging linguistic context — something
the raw CTC decoder cannot do since it treats each character independently.

Two complementary approaches:

1. N-GRAM LANGUAGE MODEL (Character-level)
   - Scores character sequences based on how likely they are in real language
   - Integrates directly into beam search decoding (joint CTC + LM scoring)
   - Helps the decoder prefer linguistically plausible character sequences
   - No external dependencies (self-contained bigram/trigram model)

2. DICTIONARY-BASED POST-PROCESSING
   - Corrects individual words after decoding using edit distance
   - Uses a word dictionary to find the closest valid word
   - Handles substitution, insertion, and deletion errors
   - Supports custom dictionaries for domain-specific text

Why Language Models Help OCR:
    The CTC decoder only looks at character probabilities from the vision model.
    It has no concept of language — it might output "th3" when "the" was written,
    because '3' and 'e' look similar in handwriting. A language model knows that
    "the" is a common English word and "th3" is not, biasing the decoder toward
    the correct output.

Usage:
    # Standalone post-processing (correct already-decoded text)
    from model.language_model import LanguageModel
    lm = LanguageModel(language='en')
    corrected = lm.correct_text("helo wrld")  # → "hello world"

    # Integrated with beam search (joint CTC + LM decoding)
    from model.ocr_utils import ctc_decode_batch
    from model.language_model import LanguageModel
    lm = LanguageModel(language='en')
    texts = ctc_decode_batch(log_probs, charset, method='beam_search',
                             beam_width=10, language_model=lm, lm_weight=0.3)

    # Custom dictionary
    lm = LanguageModel(language='en', custom_words=['pytorch', 'mnist', 'tensorflow'])
"""

import os
import re
import math
from collections import Counter, defaultdict


# =============================================================================
# WORD DICTIONARIES
# =============================================================================

# Common English words (top ~2000 frequency words)
ENGLISH_COMMON_WORDS = [
    # Articles, prepositions, conjunctions
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "he", "she", "they", "we", "you", "I", "me", "him",
    "her", "us", "them", "my", "your", "his", "our", "their", "this",
    "that", "these", "those", "what", "which", "who", "whom", "where",
    "when", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "not", "only", "same", "than",
    "too", "very", "just", "because", "if", "then", "so", "about", "up",
    "out", "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "once",
    # Common nouns
    "time", "year", "people", "way", "day", "man", "woman", "child", "world",
    "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "member", "law", "car", "city",
    "community", "name", "president", "team", "minute", "idea", "body",
    "information", "back", "parent", "face", "others", "level", "office",
    "door", "health", "person", "art", "war", "history", "party", "result",
    "change", "morning", "reason", "research", "girl", "guy", "moment",
    "air", "teacher", "force", "education",
    # Common verbs
    "get", "make", "go", "know", "take", "see", "come", "think", "look",
    "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel",
    "try", "leave", "call", "keep", "let", "begin", "show", "hear", "play",
    "run", "move", "like", "live", "believe", "hold", "bring", "happen",
    "write", "provide", "sit", "stand", "lose", "pay", "meet", "include",
    "continue", "set", "learn", "change", "lead", "understand", "watch",
    "follow", "stop", "create", "speak", "read", "allow", "add", "spend",
    "grow", "open", "walk", "win", "offer", "remember", "love", "consider",
    "appear", "buy", "wait", "serve", "die", "send", "expect", "build",
    "stay", "fall", "cut", "reach", "kill", "remain",
    # Common adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "old",
    "right", "big", "high", "different", "small", "large", "next", "early",
    "young", "important", "few", "public", "bad", "same", "able", "free",
    "true", "real", "best", "better", "sure", "clear", "recent", "hard",
    "full", "special", "easy", "strong", "possible", "whole", "local",
    # Tech/ML words (domain-specific for this project)
    "model", "neural", "network", "training", "learning", "deep", "machine",
    "data", "image", "digit", "recognition", "accuracy", "loss", "epoch",
    "batch", "layer", "weight", "bias", "gradient", "optimizer", "function",
    "input", "output", "prediction", "classification", "convolutional",
    "pooling", "activation", "dropout", "normalization", "tensor", "matrix",
    "algorithm", "parameter", "feature", "kernel", "filter", "stride",
    "padding", "flatten", "dense", "softmax", "sigmoid", "relu",
    "python", "pytorch", "numpy", "mnist", "dataset", "validation", "test",
    # Numbers as words
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "thousand", "million",
]

# Common Arabic words
ARABIC_COMMON_WORDS = [
    "مرحبا", "عالم", "برمجة", "تعلم", "ذكاء", "بيانات", "شبكة",
    "حاسوب", "علم", "نموذج", "تدريب", "اختبار", "صورة", "نص",
    "كلمة", "جديد", "قديم", "كبير", "صغير", "سريع", "بطيء",
    "جيد", "ممتاز", "افضل", "مساعدة", "خطأ", "نجاح", "تحذير",
    "معلومات", "استيراد", "واجهة", "خادم", "عميل", "قاعدة",
    "مستخدم", "اسم", "بريد", "دخول", "صفحة", "رئيسية",
]


# =============================================================================
# CHARACTER N-GRAM LANGUAGE MODEL
# =============================================================================

class CharNGramModel:
    """
    Character-level N-gram language model.

    Scores character sequences based on their frequency in a training corpus.
    Uses smoothed bigram/trigram probabilities to estimate how likely a
    character sequence is in a given language.

    This is used to bias the CTC beam search toward linguistically
    plausible character sequences.

    Args:
        n (int): N-gram order (2=bigram, 3=trigram). Default: 3.
        smoothing (float): Laplace smoothing factor. Default: 0.01.
    """

    def __init__(self, n=3, smoothing=0.01):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)
        self.vocab = set()
        self._trained = False

    def train(self, texts):
        """
        Train the n-gram model on a list of text strings.

        Args:
            texts (list[str]): Training texts.
        """
        for text in texts:
            # Add start/end markers
            padded = '^' * (self.n - 1) + text + '$'
            self.vocab.update(text)

            for i in range(len(padded) - self.n + 1):
                ngram = padded[i:i + self.n]
                context = ngram[:-1]
                char = ngram[-1]

                self.ngram_counts[context][char] += 1
                self.context_totals[context] += 1

        self.vocab.add('^')
        self.vocab.add('$')
        self._trained = True

    def score_char(self, context, char):
        """
        Get the log-probability of a character given its context.

        Uses Laplace (add-k) smoothing to handle unseen n-grams.

        Args:
            context (str): The preceding (n-1) characters.
            char (str): The character to score.

        Returns:
            float: Log-probability of the character given context.
        """
        if not self._trained:
            return 0.0

        # Ensure context is the right length
        context = context[-(self.n - 1):]
        if len(context) < self.n - 1:
            context = '^' * (self.n - 1 - len(context)) + context

        count = self.ngram_counts[context][char]
        total = self.context_totals[context]
        vocab_size = len(self.vocab) + 1  # +1 for unseen chars

        # Laplace smoothing
        prob = (count + self.smoothing) / (total + self.smoothing * vocab_size)

        return math.log(prob) if prob > 0 else -20.0

    def score_text(self, text):
        """
        Score an entire text string (sum of character log-probs).

        Args:
            text (str): Text to score.

        Returns:
            float: Total log-probability (more negative = less likely).
        """
        if not self._trained:
            return 0.0

        padded = '^' * (self.n - 1) + text + '$'
        score = 0.0

        for i in range(self.n - 1, len(padded)):
            context = padded[i - self.n + 1:i]
            char = padded[i]
            score += self.score_char(context, char)

        return score

    def perplexity(self, text):
        """
        Compute perplexity of a text (lower = more likely).

        Args:
            text (str): Text to evaluate.

        Returns:
            float: Perplexity score.
        """
        score = self.score_text(text)
        n_chars = len(text) + 1  # +1 for end token
        return math.exp(-score / n_chars) if n_chars > 0 else float('inf')


# =============================================================================
# LANGUAGE MODEL (Main Interface)
# =============================================================================

class LanguageModel:
    """
    Language model for OCR post-processing.

    Provides two capabilities:
    1. Character-level scoring for beam search integration
    2. Word-level correction using dictionary + edit distance

    Args:
        language (str): Language code ('en', 'ar', or 'mixed'). Default: 'en'.
        custom_words (list[str], optional): Additional words for the dictionary.
        ngram_order (int): Order of character n-gram model. Default: 3.
    """

    def __init__(self, language='en', custom_words=None, ngram_order=3):
        self.language = language
        self.ngram_order = ngram_order

        # Build word dictionary
        self.dictionary = set()
        if language in ('en', 'mixed'):
            self.dictionary.update(w.lower() for w in ENGLISH_COMMON_WORDS)
        if language in ('ar', 'mixed'):
            self.dictionary.update(ARABIC_COMMON_WORDS)
        if custom_words:
            self.dictionary.update(w.lower() for w in custom_words)

        # Train character n-gram model on the dictionary
        self.char_model = CharNGramModel(n=ngram_order)
        training_texts = list(self.dictionary)
        if training_texts:
            self.char_model.train(training_texts)

    def score_char(self, context, char):
        """
        Score a character given its context (for beam search integration).

        Args:
            context (str): Preceding characters.
            char (str): Character to score.

        Returns:
            float: Log-probability from the language model.
        """
        return self.char_model.score_char(context, char)

    def score_text(self, text):
        """
        Score a full text string.

        Args:
            text (str): Text to score.

        Returns:
            float: Log-probability (higher = more likely).
        """
        return self.char_model.score_text(text)

    def correct_text(self, text, max_edit_distance=2):
        """
        Correct OCR output text using dictionary-based spell correction.

        Splits text into words, corrects each word that isn't in the
        dictionary using minimum edit distance, and reassembles.

        Args:
            text (str): Raw OCR output text.
            max_edit_distance (int): Maximum allowed edits per word (default: 2).

        Returns:
            str: Corrected text.
        """
        if not text or not self.dictionary:
            return text

        words = text.split()
        corrected_words = []

        for word in words:
            corrected = self._correct_word(word, max_edit_distance)
            corrected_words.append(corrected)

        return ' '.join(corrected_words)

    def _correct_word(self, word, max_edit_distance=2):
        """
        Correct a single word using the dictionary.

        Strategy:
        1. If word is in dictionary → keep it
        2. If word is short (≤2 chars) → keep it (likely abbreviation)
        3. Find closest dictionary word within max_edit_distance
        4. If no match found → keep original

        Args:
            word (str): Word to correct.
            max_edit_distance (int): Maximum edits allowed.

        Returns:
            str: Corrected word (or original if no correction found).
        """
        word_lower = word.lower()

        # Already correct
        if word_lower in self.dictionary:
            return word

        # Too short to correct reliably
        if len(word) <= 2:
            return word

        # Find closest dictionary word
        best_word = word
        best_distance = max_edit_distance + 1

        for dict_word in self.dictionary:
            # Quick filter: skip if length difference > max_edit_distance
            if abs(len(dict_word) - len(word_lower)) > max_edit_distance:
                continue

            distance = _edit_distance(word_lower, dict_word)

            if distance < best_distance:
                best_distance = distance
                best_word = dict_word

            # Early exit on perfect match
            if distance == 1:
                break

        if best_distance <= max_edit_distance:
            # Preserve original casing pattern
            if word[0].isupper() and len(best_word) > 0:
                best_word = best_word[0].upper() + best_word[1:]
            return best_word

        return word

    def get_candidates(self, word, max_edit_distance=2, max_candidates=5):
        """
        Get correction candidates for a word, sorted by edit distance.

        Useful for showing the user multiple correction options.

        Args:
            word (str): Word to find candidates for.
            max_edit_distance (int): Maximum edit distance.
            max_candidates (int): Maximum number of candidates to return.

        Returns:
            list[tuple]: List of (candidate_word, edit_distance) tuples.
        """
        word_lower = word.lower()
        candidates = []

        for dict_word in self.dictionary:
            if abs(len(dict_word) - len(word_lower)) > max_edit_distance:
                continue
            distance = _edit_distance(word_lower, dict_word)
            if distance <= max_edit_distance:
                candidates.append((dict_word, distance))

        candidates.sort(key=lambda x: x[1])
        return candidates[:max_candidates]

    def __repr__(self):
        return (f"LanguageModel(language='{self.language}', "
                f"dict_size={len(self.dictionary)}, "
                f"ngram_order={self.ngram_order})")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _edit_distance(s1, s2):
    """
    Compute Levenshtein edit distance between two strings.

    Operations: insertion, deletion, substitution (each costs 1).

    Args:
        s1, s2 (str): Strings to compare.

    Returns:
        int: Minimum number of edits to transform s1 into s2.
    """
    m, n = len(s1), len(s2)

    # Optimize for common cases
    if m == 0:
        return n
    if n == 0:
        return m
    if s1 == s2:
        return 0

    # Use two-row DP (space optimization)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]


def load_dictionary_file(filepath):
    """
    Load a word dictionary from a text file (one word per line).

    Args:
        filepath (str): Path to dictionary file.

    Returns:
        list[str]: List of words.
    """
    if not os.path.exists(filepath):
        return []

    words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith('#'):
                words.append(word)
    return words
