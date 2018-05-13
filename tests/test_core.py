import mock
import pytest

from yasmin.core import predict_output_word
from yasmin.exceptions import ValidationException, OutOfVocabException
from yasmin.helpers import make_type_matrix


def test_type_validation(wsd_instance):
    sent = "We areived in Berlin yesterday."
    word = "Berlin"
    types = ['Person', 'FakeType']
    with pytest.raises(ValidationException):
        wsd_instance.disambiguate(sent=sent, word=word, types=types)


@pytest.mark.xfail()
def test_context_repetition(wsd_instance):
    # This is a bit of an edge case where the context are practically the same.
    # There isn't much that can be done without preprocessing.
    sent = "When we arrived in Berlin, Berlin were performing at the stadium."
    word = "Berlin"
    types = ['Person', 'GeographicArea']
    res = wsd_instance.disambiguate(sent=sent, word=word, types=types)
    assert len(res) == 2
    city_sense, band_sense = res
    assert city_sense['type'] == 'GeographicArea'
    assert band_sense['type'] == 'Person'


def test_context_punctuation(wsd_instance):
    sent = "We arrived at Berlin's old Nazi airported in Tempelhoff."
    word = "Berlin"
    types = ['Person', 'GeographicArea']
    res = wsd_instance.disambiguate(sent=sent, word=word, types=types)
    assert len(res) == 1
    sense = res[0]
    assert sense['type'] == 'GeographicArea'


def test_type_matrix(wsd_instance):
    types = {
        'weekdays': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'],
        'digits': ['one', 'two', 'three', 'four', 'five', 'six', 'seven']
    }
    matrix = make_type_matrix(model_types=types, model=wsd_instance.model)

    assert sum(matrix['weekdays']) == len(types['weekdays'])
    assert sum(matrix['digits']) == len(types['digits'])


def test_negative_model():
    model = mock.Mock()
    model.negative = False
    with pytest.raises(RuntimeError):
        predict_output_word(model=model, context=[])


def test_missing_params_of_model():
    model = mock.Mock()
    model.negative = True
    model.wv = None
    with pytest.raises(RuntimeError):
        predict_output_word(model=model, context=[])


def test_model_warning():
    model = mock.Mock()
    model.negative = True
    model.wv = mock.Mock()
    model.wv.syn0 = True
    model.syn1neg = True
    with pytest.raises(OutOfVocabException):
        predict_output_word(model=model, context=[])
