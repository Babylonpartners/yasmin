import os
from pathlib import Path

import gensim.downloader as api
from gensim.models import Word2Vec
import spacy

from yasmin.constants import SPACY_MODEL_NAME
from yasmin.core import WSD
from yasmin.helpers import custom_tokenizer, hash_types, make_type_matrix

model_path = str(Path(__file__).parents[1] /
                 'tests' / 'fixtures' / 'data' / 'text8.model')

if os.path.isfile(model_path):
    model = Word2Vec.load(model_path)
else:
    dataset = api.load("text8")
    model = Word2Vec(dataset)
model_types = {
    'furniture': ['sofa', 'desk', 'chair', 'stool', 'bed', 'table', 'cabinet'],
    'data': ['figure', 'diagram', 'chart', 'illustration', 'image', 'table']
}
nlp = spacy.load(SPACY_MODEL_NAME, create_make_doc=custom_tokenizer)
type_matrix = make_type_matrix(
    model_types=model_types, model=model
)
type_cache = {hash_types(model_types): type_matrix}

wsd = WSD(nlp, model, model_types, type_cache)

sentences = {
    'I sit at the table': 'furniture',
    'I ate breakfast on the kitchen table in the morning': 'furniture',
    'The table above shows the results of our study': 'data',
    'The wrote the results in the table on page 4': 'data',
}

for sent, type_ in sentences.items():
    predicted_type = wsd.disambiguate(
        sent=sent, word='table', types=list(model_types.keys())
    )
    print(f'{sent}\n'
          f'Real type: {type_}\n'
          f'Predicted type: {predicted_type[0]["type"]}\n'
          f'Prediction probability: {predicted_type[0]["prob"]}\n'
          f'--------------------')
