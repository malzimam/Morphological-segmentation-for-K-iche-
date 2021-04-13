# Morphological-segmentation-for-K-iche-

## requirements
* scikit-learn>=0.23.2
* numpy>=1.18.5
* pandas>=1.0.5
* docopts>=0.6.2
* scipy>=1.5.0

## Usage
python kiche.py [(--train=foo.tsv) (--predict=bar.tsv) (--output=baz.tsv)] [--version] [--help]
Currently .tsv format is hardcoded for all files
Contents of the second column in the --predict file don't matter, they get overridden by predictions

## Current stats
* P: 0.8581078431372547
* R: 0.8582450980392154
* F: 0.857680672268908
