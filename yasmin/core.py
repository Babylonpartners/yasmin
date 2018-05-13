import logging

from spacy.matcher import Matcher
from spacy.attrs import LOWER

from .helpers import (
    validate_types, hash_types, predict_output_word
)

logger = logging.getLogger('yasmin')
logger.setLevel(logging.INFO)


class WSD:

    def __init__(self, nlp, model, model_types, type_cache):
        self.nlp = nlp
        self.model = model
        self.type_cache = type_cache
        self.model_types = model_types

    def disambiguate(
            self, sent, word, types, model_types=None, context_window=3
    ):
        """
        Disambiguates the occurrences of a word/phrase in a sentence based on
        a set of categories represented as sets of words.

        :param sent: sentence
        :type sent: str
        :param word: word to disambiguate
        :type word: str
        :param types: type names
        :type types: list
        :param model_types: all model types type_name->[word_member, ...]
        :type model_types: dict
        :param context_window: context window size
        :type context_window: int
        :return: disambiguation results
        :rtype: list
        """
        model_types = model_types or self.model_types
        validate_types(selected=types, available=model_types.keys())
        matcher = self._make_phrase_matcher(word)
        doc = self.nlp(sent)
        results = []
        matches = matcher(doc)
        new_type_hash = hash_types(model_types)
        type_matrix = self.type_cache.get(new_type_hash)
        if type_matrix is None:
            type_matrix = self._make_type_matrix(
                model_types=model_types, model=self.model
            )
        filtered = self._filter_types(matrix=type_matrix, names=types)

        for _, start, end in matches:
            context, match = self._get_context(
                doc=doc, start=start, end=end, win=context_window
            )
            _type, prob = self._find_sense(
                context=context, type_matrix=filtered
            )
            results.append({
                'type': _type,
                'prob': prob,
                'offsets': [match.start_char, match.end_char]
            })

        return results

    def _find_sense(self, *, context, type_matrix):
        """
        Finds the most likely sense of a word given a context.

        :param context: context words
        :type context: list
        :param type_matrix: mapping types->one-hot vectors of type vocabulary
        :type type_matrix: dict
        :return: selected type, selection probability
        """
        probs = predict_output_word(model=self.model, context=context)
        type_probs = {}
        for t, type_marks in type_matrix.items():
            ln = type_marks.sum()
            type_probs[t] = (probs * type_marks).sum() / ln
        choice_type, raw_prob = max(type_probs.items(), key=lambda x: x[1])
        choice_prob = raw_prob / sum(type_probs.values())

        return choice_type, choice_prob

    def _get_context(self, *, doc, start, end, win):
        # get the phrase
        match = doc[start:end]

        # get token indices for the context
        context_start = start - win if start - win > 0 else 0
        context_end = end + win if end + win < len(doc) else len(doc)

        # slice the context
        context = [t.orth_.lower() for t in doc[context_start:context_end]
                   if self._validate_context(t, match.orth_)]

        return context, match

    @staticmethod
    def _filter_types(*, matrix, names):
        return {typ: vec for typ, vec in matrix.items() if typ in names}

    @staticmethod
    def _validate_context(context_token, target_word):
        return (
            not context_token.is_punct
            and not context_token.orth_ == target_word
        )

    def _make_phrase_matcher(self, phrase):
        lowered = phrase.lower()
        m = Matcher(self.nlp.vocab)
        name = phrase.replace(' ', '_')
        m.add(name, None, [{LOWER: w.orth_} for w in self.nlp(lowered)])
        return m
