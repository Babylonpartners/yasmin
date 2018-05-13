from hashlib import md5
import json
import logging
import pytest
import spacy

from .constants import TESTS_DIR
from yasmin.exceptions import ValidationException
from yasmin.helpers import (
    custom_tokenizer, hash_types, validate_types, parse_custom_types
)
from yasmin.constants import SPACY_MODEL_NAME


logger = logging.getLogger(__name__)


@pytest.fixture
def _types():
    return json.load(open(TESTS_DIR / 'fixtures' / 'data' / 'types.json'))


@pytest.mark.parametrize('name, keywords', _types().items())
def test_parse_custom_types(name, keywords, types, raw_types):
    parsed = parse_custom_types(raw_types)
    assert set(parsed.keys()) == set(types.keys()), 'Type names'
    assert set(parsed[name]) == set(keywords), '{} keywords not matching.'\
        .format(name)


def test_validate_types():
    selected_types = ['foo', 'bar']
    available_types = ['foo', 'bar', 'moo']
    assert not validate_types(selected=selected_types,
                              available=available_types)
    selected_types.append('bla')
    with pytest.raises(ValidationException):
        validate_types(selected=selected_types, available=available_types)


def test_type_hash():
    types = {
        'A': ['a', 'b', 'c'],
        'B': ['e', 'f', 'g']
    }
    types_str = "A:['a', 'b', 'c']+B:['e', 'f', 'g']"
    generated = hash_types(types)
    m = md5()
    m.update(types_str.encode('utf-8'))
    real = m.digest()
    assert generated == real, 'Type hashing not working as expected.'


def test_custom_tokeniser():
    nlp = spacy.load(SPACY_MODEL_NAME, create_make_doc=custom_tokenizer)
    tokens = nlp('"test?')
    assert len(tokens) == 3
    assert str(tokens[0]) == '"'
    assert str(tokens[1]) == 'test'
    assert str(tokens[2]) == '?'
