/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_ARGS_H
#define FASTTEXT_ARGS_H

#include <istream>
#include <ostream>
#include <string>

enum class model_name : int {cbow=1, sg, sup, bil};
enum class loss_name : int {hs=1, ns, softmax};

class Args {
  public:
    Args();
    std::string input;
    std::string name;
    
    // Bilingual
    std::string input_mono1;
    std::string input_mono2;
    std::string input_par1;
    std::string input_par2;
    
    std::string test;
    std::string output;
    double lr;
    double lr_wv;
    double lr_mono;
    double lr_par;
    
    int lrUpdateRate;
    int dim;
    int ws;
    int epoch;
    int minCount;
    int neg;
    int wordNgrams;
    loss_name loss;
    model_name model;
    int bucket;
    int minn;
    int maxn;
    int thread;
    int32_t threadOffset;
    double t;
    std::string label;
    int verbose;

    void parseArgs(int, char**);
    void printHelp();
    void save(std::ostream&);
    void load(std::istream&);
    
    void toggleSup();
    void toggleMono(const int);
    void togglePar();
};

#endif
