/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef FASTTEXT_FASTTEXT_H
#define FASTTEXT_FASTTEXT_H

#include <time.h>

#include <atomic>
#include <memory>

#include "matrix.h"
#include "vector.h"
#include "dictionary.h"
#include "model.h"
#include "utils.h"
#include "real.h"
#include "args.h"

class FastText {
  private:
    clock_t start;
    std::vector<std::ifstream> ifs;
    
  public:
    std::atomic<int64_t> tokenCount;
    real progress;
    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;
    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;
    std::shared_ptr<Model> model_;
    
    void getVector(Vector&, const std::string&);
    void saveVectors(const std::string);
    void printVectors();
    void saveModel(const std::string);
    void loadModel(const std::string&);
    void printInfo(real, real);

    void supervised(Model&, real, const std::vector<int32_t>&, const std::vector<int32_t>&);
    void cbow(Model&, real, const std::vector<int32_t>&);
    void skipgram(Model&, real, const std::vector<int32_t>&);
    void bilingual_cbow(Model&, real, const std::vector<int32_t>&, const std::vector<int32_t>&);
    void bilingual_skipgram(Model&, real, const std::vector<int32_t>&, const std::vector<int32_t>&);
    
    void test(const std::string&, int32_t);
    void predict(const std::string&, int32_t, bool);
    
    void setup(std::shared_ptr<Args>, std::shared_ptr<Dictionary>, std::shared_ptr<Matrix>, std::shared_ptr<Matrix>);
    void close(std::string);
    void train();
    void step();
};

#endif
