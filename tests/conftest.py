import json
import logging
import mock
import os
import pytest

import gensim.downloader as api
from gensim.models import Word2Vec
import spacy

from yasmin import WSD
from yasmin.constants import SPACY_MODEL_NAME
from yasmin.helpers import hash_types, make_type_matrix
from .constants import TESTS_DIR, MODEL_FILE


def pytest_logger_config(logger_config):
    """
    pytest-logger method
    """
    logger_config.add_loggers(['test_utils'], stdout_level='info')
    logger_config.set_log_option_default('test_utils')


logger = logging.getLogger(__name__)


if os.path.isfile(MODEL_FILE):
    logger.info('Loading a cached model...')
    _model = Word2Vec.load(str(MODEL_FILE))
else:
    logger.info('Downloading text...')
    _dataset = api.load("text8")
    logger.info('Building the model...')
    _model = Word2Vec(_dataset)
    logger.info('Saving the model...')
    _model.save(str(MODEL_FILE))

_nlp = spacy.load(SPACY_MODEL_NAME)
# raw types
fp = os.path.join(TESTS_DIR / 'fixtures' / 'data' / 'raw_types.json')
with open(fp) as fh:
    _raw_types = json.load(fh)

# types
fp = os.path.join(TESTS_DIR / 'fixtures' / 'data' / 'types.json')
with open(fp) as fh:
    _types = json.load(fh)

# type matrix
_type_matrix = make_type_matrix(
    model_types=_types, model=_model
)

# type cache
_type_cache = {hash_types(_types): _type_matrix}


@pytest.fixture
def wsd_instance():
    wsd = WSD(_nlp, _model, _types, _type_cache)

    return wsd


@pytest.fixture
def raw_types():
    return _raw_types


@pytest.fixture
def types():
    return _types


@pytest.fixture(scope='session')
def mini_model():
    model_ = mock.Mock()
    model_.wv = mock.Mock()
    model_.wv.vocab = {}
    for i, term in enumerate(['term1', 'term2', 'term3']):
        word = mock.Mock()
        word.index = i
        model_.wv.vocab[term] = word
    return model_


@pytest.fixture(scope='session')
def mini_model_types():
    return {'type1': ['term1'], 'type2': ['term2']}


@pytest.fixture(scope='session')
def mini_type_matrix():
    return {'type1': [1, 0, 0], 'type2': [0, 1, 0]}
