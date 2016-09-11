/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "args.h"

#include <stdlib.h>
#include <string.h>

#include <iostream>

Args::Args() {
  ws = 5;
  epoch = 5;
  neg = 5;
  wordNgrams = 1;
  loss = loss_name::ns;
  model = model_name::sg;
  bucket = 2000000;
  thread = 12;
  lrUpdateRate = 100;
  t = 1e-4;
  label = "__label__";
  verbose = 2;

  // Customized
  lr = 0.01;
  lr_wv = 0.05;
  dim = 1;
  minCount = 1;
  minn = 0;
  maxn = 0;
}

// --
// fasttext semisupervised -input ./data/ft-semi.txt -output ./models/semi -dim 10
// 0.1, 0.01 -> 0.848
// 0.1, 0.011 -> 0.849
// 0.1, 0.013 -> 0.850
// 0.1, 0.015 -> 0.851
// 0.1, 0.02 -> 0.852
// 0.1, 0.05 -> 0.853
// 0.1, 0.075 -> 0.853
// 0.2, 0.075 -> 0.856

// Semisupervised
void Args::toggleWV() {
  model = model_name::cbow;
  loss = loss_name::ns;
  lr = lr_wv;
}


// Bilingual
void Args::toggleSup() {
  model = model_name::sup;
  loss = loss_name::softmax;
  
  input_par1.clear();
  input_par2.clear();
  input_mono2.clear();
  input_mono1.clear();
}

void Args::toggleMono(const int i) {
  model = model_name::cbow;
  loss = loss_name::ns;
  lr = lr_wv;
  
  input.clear();
  input_par1.clear();
  input_par2.clear();
  if (i == 1) {
    input_mono2.clear();
  } else if (i == 2) {
    input_mono1.clear();
  }
}

void Args::togglePar() {
  model = model_name::bil;
  loss = loss_name::ns;
  lr = lr_wv;

  input.clear();
  input_mono1.clear();
  input_mono2.clear();
}


void Args::parseArgs(int argc, char** argv) {
  std::string command(argv[1]);
  if (command == "semisupervised") {
    model = model_name::sup;
    loss = loss_name::softmax;
    lr = 0.1;
  } else if (command == "bilingual") {
    model = model_name::sup;
    loss = loss_name::softmax;
    lr = 0.1;
  } else if (command == "cbow") {
    model = model_name::cbow;
  }
  
  int ai = 2;
  while (ai < argc) {
    if (argv[ai][0] != '-') {
      std::cout << "Provided argument without a dash! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    if (strcmp(argv[ai], "-h") == 0) {
      std::cout << "Here is the help! Usage:" << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    } else if (strcmp(argv[ai], "-input") == 0) {
      input = std::string(argv[ai + 1]);

    // Bilingual
    } else if (strcmp(argv[ai], "-input-mono1") == 0) {
      input_mono1 = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-input-mono2") == 0) {
      input_mono2 = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-input-par1") == 0) {
      input_par1 = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-input-par2") == 0) {
      input_par2 = std::string(argv[ai + 1]);

    } else if (strcmp(argv[ai], "-test") == 0) {
      test = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-output") == 0) {
      output = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lr") == 0) {
      lr = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lr_wv") == 0) {
      lr_wv = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-lrUpdateRate") == 0) {
      lrUpdateRate = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-dim") == 0) {
      dim = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-ws") == 0) {
      ws = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-epoch") == 0) {
      epoch = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minCount") == 0) {
      minCount = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-neg") == 0) {
      neg = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-wordNgrams") == 0) {
      wordNgrams = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-loss") == 0) {
      if (strcmp(argv[ai + 1], "hs") == 0) {
        loss = loss_name::hs;
      } else if (strcmp(argv[ai + 1], "ns") == 0) {
        loss = loss_name::ns;
      } else if (strcmp(argv[ai + 1], "softmax") == 0) {
        loss = loss_name::softmax;
      } else {
        std::cout << "Unknown loss: " << argv[ai + 1] << std::endl;
        printHelp();
        exit(EXIT_FAILURE);
      }
    } else if (strcmp(argv[ai], "-bucket") == 0) {
      bucket = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-minn") == 0) {
      minn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-maxn") == 0) {
      maxn = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-thread") == 0) {
      thread = atoi(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-t") == 0) {
      t = atof(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-label") == 0) {
      label = std::string(argv[ai + 1]);
    } else if (strcmp(argv[ai], "-verbose") == 0) {
      verbose = atoi(argv[ai + 1]);
    } else {
      std::cout << "Unknown argument: " << argv[ai] << std::endl;
      printHelp();
      exit(EXIT_FAILURE);
    }
    ai += 2;
  }
  if (input.empty() || output.empty()) {
    std::cout << "Empty input or output path." << std::endl;
    printHelp();
    exit(EXIT_FAILURE);
  }
  if (wordNgrams <= 1 && maxn == 0) {
    bucket = 0;
  }
}

void Args::printHelp() {
  std::cout
    << "\n"
    << "The following arguments are mandatory:\n"
    << "  -input        training file path\n"
    << "  -output       output file path\n\n"
    << "The following arguments are optional:\n"
    << "  -lr           learning rate [" << lr << "]\n"
    << "  -lrUpdateRate change the rate of updates for the learning rate [" << lrUpdateRate << "]\n"
    << "  -dim          size of word vectors [" << dim << "]\n"
    << "  -ws           size of the context window [" << ws << "]\n"
    << "  -epoch        number of epochs [" << epoch << "]\n"
    << "  -minCount     minimal number of word occurences [" << minCount << "]\n"
    << "  -neg          number of negatives sampled [" << neg << "]\n"
    << "  -wordNgrams   max length of word ngram [" << wordNgrams << "]\n"
    << "  -loss         loss function {ns, hs, softmax} [ns]\n"
    << "  -bucket       number of buckets [" << bucket << "]\n"
    << "  -minn         min length of char ngram [" << minn << "]\n"
    << "  -maxn         max length of char ngram [" << maxn << "]\n"
    << "  -thread       number of threads [" << thread << "]\n"
    << "  -t            sampling threshold [" << t << "]\n"
    << "  -label        labels prefix [" << label << "]\n"
    << "  -verbose      verbosity level [" << verbose << "]\n"
    << std::endl;
}

void Args::save(std::ostream& out) {
  out.write((char*) &(dim), sizeof(int));
  out.write((char*) &(ws), sizeof(int));
  out.write((char*) &(epoch), sizeof(int));
  out.write((char*) &(minCount), sizeof(int));
  out.write((char*) &(neg), sizeof(int));
  out.write((char*) &(wordNgrams), sizeof(int));
  out.write((char*) &(loss), sizeof(loss_name));
  out.write((char*) &(model), sizeof(model_name));
  out.write((char*) &(bucket), sizeof(int));
  out.write((char*) &(minn), sizeof(int));
  out.write((char*) &(maxn), sizeof(int));
  out.write((char*) &(lrUpdateRate), sizeof(int));
  out.write((char*) &(t), sizeof(double));
}

void Args::load(std::istream& in) {
  in.read((char*) &(dim), sizeof(int));
  in.read((char*) &(ws), sizeof(int));
  in.read((char*) &(epoch), sizeof(int));
  in.read((char*) &(minCount), sizeof(int));
  in.read((char*) &(neg), sizeof(int));
  in.read((char*) &(wordNgrams), sizeof(int));
  in.read((char*) &(loss), sizeof(loss_name));
  in.read((char*) &(model), sizeof(model_name));
  in.read((char*) &(bucket), sizeof(int));
  in.read((char*) &(minn), sizeof(int));
  in.read((char*) &(maxn), sizeof(int));
  in.read((char*) &(lrUpdateRate), sizeof(int));
  in.read((char*) &(t), sizeof(double));
}
