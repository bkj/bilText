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

# --------------------------------------------------------

./make.sh && rm .bil-dev-*

# Sup
time fasttext bilingual \
    -input ~/projects/clf-test/data/ft-train-s.txt \
    -output ./.bil-dev \
    -dim 2 -lr 0.01

fasttext test ./.bil-dev-sup.bin ~/projects/clf-test/data/ft-train-s.txt 



# Mono
./make.sh && rm .bil-dev-*

time fasttext bilingual \
    -input ./empty \
    -input-mono1 ~/projects/clf-test/data/ft-train-s.txt \
    -input-mono2 ~/projects/clf-test/data/ft-train-t.txt \
    -output ./.bil-dev \
    -dim 2 -lr 0.075

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-mono1.vec \
    ~/projects/clf-test/data/ft-train-s.txt \
    ~/projects/clf-test/data/ft-test-s.txt

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-mono1.vec \
    ~/projects/clf-test/data/ft-train-t.txt \
    ~/projects/clf-test/data/ft-test-t.txt

cp .bil-dev-mono1.vec .bil-unconstrained.vec 

# Adding constraint
./make.sh

time fasttext bilingual \
    -input ./empty \
    -input-mono1 ~/projects/clf-test/data/ft-train-s.txt \
    -input-mono2 ~/projects/clf-test/data/ft-train-t.txt \
    -input-par1 ~/projects/clf-test/data/ft-train-s.txt \
    -input-par2 ~/projects/clf-test/data/ft-train-t.txt \
    -output ./.bil-dev \
    -dim 2 -lr 0.075 -lr_wv 0.1

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-mono1.vec \
    ~/projects/clf-test/data/ft-train-s.txt \
    ~/projects/clf-test/data/ft-test-s.txt

python ~/projects/clf-test/vec-logreg.py \
    ~/projects/cpp/fasttext/.bil-dev-mono1.vec \
    ~/projects/clf-test/data/ft-train-t.txt \
    ~/projects/clf-test/data/ft-test-t.txt

cp .bil-dev-mono1.vec .bil-constrained.vec 

# Without mono
./make.sh

time fasttext bilingual \
    -input ./empty \
    -input-par1 ~/projects/clf-test/data/ft-train-s.txt \
    -input-par2 ~/projects/clf-test/data/ft-train-t.txt \
    -output ./.bil-dev \
    -dim 2 -epoch 1

R
options(stringsAsFactors = FALSE)
x <- read.csv('./.bil-dev-par.vec', sep = ' ', skip=1, header=F, quote='')
plot(x[,2], x[,3], cex=0.2, col = 1 + grepl('_s$', x[,1]))


# --
# Parallel modeling
tail -n 5000 ~/projects/clf-test/data/ft-train-s.txt > s-5000-2.txt
tail -n 5000 ~/projects/clf-test/data/ft-train-t.txt > t-5000-2.txt

./make.sh

# -- 
# Grid

./make.sh

cat s-5000.txt | sed 's/__label__[^ ]* *//' > s-5000-u.txt
cat t-5000.txt | sed 's/__label__[^ ]* *//' > t-5000-u.txt

# Small amount of target training data
time fasttext-orig skipgram \
    -input t-5000-u.txt \
    -output ./models/orig \
    -dim 10 -lr 0.5

python ~/projects/clf-test/vec-logreg.py \
    ./models/orig.vec \
    t-100.txt \
    t-5000-2.txt

# 0.81

# Traditional transgram
time fasttext bilingual-u \
    -input ./empty \
    -input-par1 s-100-u.txt \
    -input-par2 t-100-u.txt \
    -output ./models/dev \
    -dim 10 -lr_wv 0.5

python ~/projects/clf-test/vec-logreg.py \
    ./models/dev-par-u.vec \
    s-5000.txt \
    t-5000-2.txt

# 0.70

# Supervised transgram
time fasttext bilingual \
    -input s-5000.txt \
    -input-par1 s-100-u.txt \
    -input-par2 t-100-u.txt \
    -output ./models/dev \
    -dim 10 -lr 0.5 -lr_wv 0.01 # Optimised (sortof)

fasttext test ./models/dev-sup.bin t-5000-2.txt

python ~/projects/clf-test/vec-logreg.py \
    ./models/dev-sup.vec \
    s-5000.txt \
    t-5000-2.txt

# 0.78


