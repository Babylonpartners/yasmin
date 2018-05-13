from hashlib import md5

import numpy as np
import spacy
from spacy.tokenizer import Tokenizer

from .exceptions import ValidationException, OutOfVocabException


def predict_output_word(*, model, context):
    """
    Report the probability distribution of the center word given the context
    words as input to the trained `model`.

    NOTE: This method is a version of the `predict_output_word` method in
    `gensim.models.word2vec.Word2Vec`.

    :param model: the trained word2vec model
    :type model: gensim.models.word2vec.Word2Vec
    :param context: list of context words
    :type context: list
    """
    if not model.negative:
        raise RuntimeError(
            "We have currently only implemented predict_output_word for the "
            "negative sampling scheme, so you need to have run word2vec with "
            "negative > 0 for this to work."
        )

    if not hasattr(model.wv, 'syn0') or not hasattr(model, 'syn1neg'):
        raise RuntimeError(
            "Parameters required for predicting the output words not found.")

    word_vocabs = [model.wv.vocab[w] for w in context if
                   w in model.wv.vocab]
    if not word_vocabs:
        raise OutOfVocabException(
            "All the input context words are out-of-vocabulary for the "
            "current model.")

    word2_indices = [word.index for word in word_vocabs]

    l1 = np.sum(model.wv.syn0[word2_indices], axis=0)
    if word2_indices and model.cbow_mean:
        l1 /= len(word2_indices)

    # propagate hidden -> output and take softmax to get probabilities
    prob_values = np.exp(np.dot(l1, model.syn1neg.T))
    prob_values /= sum(prob_values)

    return prob_values


def parse_custom_types(types):
    """
    Parses curstom types format as sent through the service.

    :param types: curstom types JSON
    :type types: list
    :return: custom types dictionary
    :rtype: dict
    """
    model_types = {}
    for typ in types:
        name = typ['name']
        model_types[name] = [
            x.lower().strip() for x in typ['keywords'].split(',')
        ]
    return model_types


def validate_types(*, selected, available):
    """
    Makes sure the selected types for a particular phrase/word are in the list
    of declared types.

    :param selected: types selected for disambiguation
    :type selected: iterable
    :param available: all model types
    :type available: iterable
    :raises ValidationException: if one or more of the selected types is not
        in the list of model types
    """
    spurious_types = set(selected) - set(available)
    if spurious_types:
        raise ValidationException(f'Type(s): "{spurious_types}" unknown')


def hash_types(types):
    m = md5()
    types_str = '+'.join(f"{k}:{sorted(set(v))}" for k, v in types.items())
    m.update(types_str.encode('utf-8'))
    return m.digest()


def make_type_matrix(*, model_types, model):
    """
    Produces a dictionary mapping of type names to one-hot vectors of the
    size of the `model` vocabulary, representing the terms present in the
    type according to `model_types`.

    :param model_types: dictionary mapping of type names and term members
    :type model_types: dict
    :param model: gensim word2vec model
    :type model: gensim.models.word2vec.Word2Vec
    :return: type one-hot matrix
    :rtype: dict
    """
    matrix = {}
    vocab = model.wv.vocab
    ln = len(vocab)
    for typ, terms in model_types.items():
        indices = [vocab[term].index for term in terms if term in vocab]
        v = np.zeros(ln)
        np.put(v, indices, np.ones(len(indices)))
        matrix[typ] = v
    return matrix


def custom_tokenizer(nlp):  # pragma: no cover
    """
    Creates a custom spaCy tokenizer.

    :param nlp: a spaCy Language object
    :return:
    """
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
    infix_re = spacy.util.compile_infix_regex(nlp.Defaults.infixes)
    tokenizer = Tokenizer(nlp.vocab, nlp.Defaults.tokenizer_exceptions,
                          prefix_re.search, suffix_re.search,
                          infix_re.finditer, token_match=None)

    def make_doc(text): return tokenizer(text)
    return make_doc
