#!/bin/bash

fasttext bilingual \
	-input ~/projects/clf-test/data/ft-train.txt \
    -input-mono1 ~/projects/clf-test/data/ft-train.txt \
    -input-mono2 ~/projects/clf-test/data/ft-train.txt \
    -input-par1 ~/projects/clf-test/data/ft-train.txt \
    -input-par2 ~/projects/clf-test/data/ft-train.txt \
    -output ./delete-me
