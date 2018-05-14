# yasmin

_Lightweight Word Sense Disambiguation_

[![CircleCI](https://circleci.com/gh/Babylonpartners/yasmin/tree/master.svg?style=svg&circle-token=cad1fd5882bff5b56bb44673067f8a7f641ef53b)](https://circleci.com/gh/Babylonpartners/yasmin/tree/master)
[![Maintainability](https://api.codeclimate.com/v1/badges/fe246ee26f1d4a3da9e1/maintainability)](https://codeclimate.com/repos/5af841f0b813595a630009bf/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/fe246ee26f1d4a3da9e1/test_coverage)](https://codeclimate.com/repos/5af841f0b813595a630009bf/test_coverage)
[![PyPI version](https://badge.fury.io/py/yasmin.svg)](https://badge.fury.io/py/yasmin)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Yasmin approaches WSD as a classification problem where a CBOW model is used to
predict the probabilities of each class based on the mean probability of each
of their members given a context.

### Installation

```bash
$ pip install yasmin
```

You can also checkout this repository and install from the root folder
```bash
$ pip install .
```

Note: make sure you download the `en_core_web_sm` spaCy model.

```bash
$ spacy download en_core_web_sm
```


### Examples

Check out a usage example based on the `text8` corpus in [examples/text8.py](https://github.com/Babylonpartners/yasmin/blob/master/examples/text8.py). 


### Why Yasmin?

WSD is a well-know issue in the clinical domain where diseases often bear the
name of the person that "discovered" them. However, it turns out that 
contraceptive pills are mentioned much more often and appear to be commonly
named using common feminine names, namely [Yasmin](https://en.wikipedia.org/wiki/Drospirenone).

### Maintainers
* [@savkov](https://github.com/savkov)
* [@fran-babylon](https://github.com/fran-babylon)
* [@jack-babylon](https://github.com/jack-babylon)


### Acknowledgements

A special shoutout to [@ironvital](https://github.com/ironvital) for coming up 
with the idea.


Even though this work was developed independently, we should state that
similar approaches have been used in the past. See, for example, [this paper](https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2018-ustalovetal-lrec18-unsupwsd.pdf) from the 
group that built [SenseGram](https://github.com/tudarmstadt-lt/sensegram).