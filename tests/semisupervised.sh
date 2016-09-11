#!/bin/bash

echo '- training -'
fasttext semisupervised -input ~/projects/clf-test/data/ft-semi.txt \
    -output ./delete-me \
    -dim 10 -lr 0.075 -lr_wv 0.2

echo '- testing -'
fasttext test ./delete-me-sup.bin ~/projects/clf-test/data/ft-test.txt

rm delete-me-sup.bin
rm delete-me-sup.vec
rm delete-me-wv.bin
rm delete-me-wv.vec
