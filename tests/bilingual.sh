#!/bin/bash

time fasttext bilingual \
	-input ~/projects/clf-test/data/ft-train-s.txt \
    -input-mono1 ~/projects/clf-test/data/ft-train-s-u.txt \
    -input-mono2 ~/projects/clf-test/data/ft-train-t-u.txt \
    -input-par1 ~/projects/clf-test/data/ft-train-s-u.txt \
    -input-par2 ~/projects/clf-test/data/ft-train-t-u.txt \
    -output ./.bil-dev \
    -dim 10 -lr 0.075

# >>

# --
# Single language 

./make.sh
rm .bil-dev-*
time fasttext bilingual \
    -input ~/projects/clf-test/data/ft-train-s-u.txt \
    -input-mono1 ~/projects/clf-test/data/ft-train-s-u.txt \
    -output ./.bil-dev \
    -dim 10 -lr 0.075

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-mono1.vec \
    ~/projects/clf-test/data/ft-train-s.txt \
    ~/projects/clf-test/data/ft-test-s.txt

# 0.85

# --
# Two language (untied)

./make.sh
rm .bil-dev-*
time fasttext bilingual \
    -input ~/projects/clf-test/data/ft-train-s-u.txt \
    -input-mono1 ~/projects/clf-test/data/ft-train-s-u.txt \
    -input-mono2 ~/projects/clf-test/data/ft-train-t-u.txt \
    -output ./.bil-dev \
    -dim 10 -lr 0.075

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-sup.vec \
    ~/projects/clf-test/data/ft-train-s.txt \
    ~/projects/clf-test/data/ft-test-s.txt

# 0.85

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-sup.vec \
    ~/projects/clf-test/data/ft-train-t.txt \
    ~/projects/clf-test/data/ft-test-t.txt

# 0.85

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-sup.vec \
    ~/projects/clf-test/data/ft-train-s.txt \
    ~/projects/clf-test/data/ft-test-t.txt

# 0.50

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-sup.vec \
    ~/projects/clf-test/data/ft-train-t.txt \
    ~/projects/clf-test/data/ft-test-s.txt

# 0.50

# --
# Two language (tied)
./make.sh
rm .bil-dev-*

time fasttext bilingual \
    -input ./tmp.txt \
    -input-mono1 ~/projects/clf-test/data/ft-train-s-u.txt \
    -input-mono2 ~/projects/clf-test/data/ft-train-t-u.txt \
    -input-par1 ~/projects/clf-test/data/ft-train-s-u.txt \
    -input-par2 ~/projects/clf-test/data/ft-train-t-u.txt \
    -output ./.bil-dev \
    -dim 10 -lr 0.075

cat .bil-dev-par.vec | head -n 20

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-par.vec \
    ~/projects/clf-test/data/ft-train-s.txt \
    ~/projects/clf-test/data/ft-test-s.txt # 0.85

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-par.vec \
    ~/projects/clf-test/data/ft-train-t.txt \
    ~/projects/clf-test/data/ft-test-t.txt # 0.85

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-par.vec \
    ~/projects/clf-test/data/ft-train-s.txt \
    ~/projects/clf-test/data/ft-test-t.txt # 0.85

