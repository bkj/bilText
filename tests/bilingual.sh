#!/bin/bash

time fasttext bilingual \
	-input ~/projects/clf-test/data/ft-train.txt \
    -input-mono1 ~/projects/clf-test/data/ft-train.txt \
    -input-mono2 ~/projects/clf-test/data/ft-train.txt \
    -input-par1 ~/projects/clf-test/data/ft-train.txt \
    -input-par2 ~/projects/clf-test/data/ft-train.txt \
    -output ./.bil-dev \
    -dim 10 -lr 0.075

echo '- testing -'
time fasttext test ./.bil-dev-sup.bin ~/projects/clf-test/data/ft-test.txt
