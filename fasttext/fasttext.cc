/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "fasttext.h"

#include <fenv.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include <iostream>
#include <iomanip>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>


void printUsage() {
  std::cout
  << "usage: fasttext <command> <args>\n\n"
  << "The commands supported by fasttext are:\n\n"
  << "  semisupervised   train a semisupervised classifier (experimental)\n"
  << "  bilingual        train a bilingual classifier (experimental)\n"
  << "  test             evaluate a supervised classifier\n"
  << "  predict          predict most likely labels\n"
  << "  predict-prob     predict most likely labels with probabilities\n"
  << "  skipgram         train a skipgram model\n"
  << "  cbow             train a cbow model\n"
  << "  print-vectors    print vectors given a trained model\n"
  << std::endl;
}

void printTestUsage() {
  std::cout
  << "usage: fasttext test <model> <test-data> [<k>]\n\n"
  << "  <model>      model filename\n"
  << "  <test-data>  test data filename\n"
  << "  <k>          (optional; 1 by default) predict top k labels\n"
  << std::endl;
}

void printPredictUsage() {
  std::cout
  << "usage: fasttext predict[-prob] <model> <test-data> [<k>]\n\n"
  << "  <model>      model filename\n"
  << "  <test-data>  test data filename\n"
  << "  <k>          (optional; 1 by default) predict top k labels\n"
  << std::endl;
}

void printPrintVectorsUsage() {
  std::cout
  << "usage: fasttext print-vectors <model>\n\n"
  << "  <model>      model filename\n"
  << std::endl;
}

void test(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 1;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printTestUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.test(std::string(argv[3]), k);
  exit(0);
}

void predict(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 1;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  bool print_prob = std::string(argv[1]) == "predict-prob";
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.predict(std::string(argv[3]), k, print_prob);
  exit(0);
}

void printVectors(int argc, char** argv) {
  if (argc != 3) {
    printPrintVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.printVectors();
  exit(0);
}

void FastText::getVector(Vector& vec, const std::string& word) {
  const std::vector<int32_t>& ngrams = dict_->getNgrams(word);
  vec.zero();
  for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
    vec.addRow(*input_, *it);
  }
  if (ngrams.size() > 0) {
    vec.mul(1.0 / ngrams.size());
  }
}

void FastText::saveVectors(std::string suffix) {
  std::ofstream ofs(args_->output + suffix + ".vec");
  if (!ofs.is_open()) {
    std::cout << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords() << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords(); i++) {
    std::string word = dict_->getWord(i);
    getVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::printVectors() {
  std::string word;
  Vector vec(args_->dim);
  while (std::cin >> word) {
    getVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
}

void FastText::saveModel(std::string suffix) {
  std::ofstream ofs(args_->output + suffix + ".bin");
  if (!ofs.is_open()) {
    std::cerr << "Model file cannot be opened for saving!" << std::endl;
    exit(EXIT_FAILURE);
  }
  args_->save(ofs);
  dict_->save(ofs);
  input_->save(ofs);
  output_->save(ofs);
  ofs.close();
}

void FastText::loadModel(const std::string& filename) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  args_ = std::make_shared<Args>();
  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  
  args_->load(ifs);
  dict_->load(ifs);
  input_->load(ifs);
  output_->load(ifs);
  
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  
  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label), dict_);
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word), dict_);
  }
  ifs.close();
}

void FastText::printInfo(real progress, real loss) {
  real t = real(clock() - start) / CLOCKS_PER_SEC;
  real wst = real(tokenCount) / t;
  real lr = args_->lr * (1.0 - progress);
  int eta = int(t / progress * (1 - progress) / args_->thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;
  std::cout << std::fixed;
  std::cout << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::cout << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::cout << "  lr: " << std::setprecision(6) << lr;
  std::cout << "  loss: " << std::setprecision(6) << loss;
  std::cout << "  eta: " << etah << "h" << etam << "m ";
  std::cout << std::flush;
}

void FastText::test(const std::string& filename, int32_t k) {
  int32_t nexamples = 0, nlabels = 0;
  double precision = 0.0;
  std::vector<int32_t> line, labels;
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Test file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  while (ifs.peek() != EOF) {
    dict_->getLine(ifs, line, labels, args_->model, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    if (labels.size() > 0 && line.size() > 0) {
      std::vector<std::pair<real, int32_t>> predictions;
      model_->predict(line, k, predictions);
      for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
        if (std::find(labels.begin(), labels.end(), it->second) != labels.end()) {
          precision += 1.0;
        }
      }
      nexamples++;
      nlabels += labels.size();
    }
  }
  ifs.close();
  std::cout << std::setprecision(3);
  std::cout << "P@" << k << ": " << precision / (k * nexamples) << std::endl;
  std::cout << "R@" << k << ": " << precision / nlabels << std::endl;
  std::cout << "Number of examples: " << nexamples << std::endl;
}

void FastText::predict(const std::string& filename, int32_t k, bool print_prob) {
  std::vector<int32_t> line, labels;
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Test file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  while (ifs.peek() != EOF) {
    dict_->getLine(ifs, line, labels, args_->model, model_->rng);
    dict_->addNgrams(line, args_->wordNgrams);
    if (line.empty()) {
      std::cout << "n/a" << std::endl;
      continue;
    }
    std::vector<std::pair<real, int32_t>> predictions;
    model_->predict(line, k, predictions);
    for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
      if (it != predictions.cbegin()) {
        std::cout << ' ';
      }
      std::cout << dict_->getLabel(it->second);
      if (print_prob) {
        std::cout << ' ' << exp(it->first);
      }
    }
    std::cout << std::endl;
  }
  ifs.close();
}

// --------
// vv Train


void FastText::supervised(Model& model, real lr, const std::vector<int32_t>& line, const std::vector<int32_t>& labels) {
  if (labels.size() == 0 || line.size() == 0) return;
  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
  int32_t i = uniform(model.rng);
  model.update(line, labels[i], lr);
}

void FastText::cbow(Model& model, real lr, const std::vector<int32_t>& line) {
  std::vector<int32_t> bow;
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    bow.clear();
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w + c]);
        bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
      }
    }
    model.update(bow, line[w], lr);
  }
}

void FastText::skipgram(Model& model, real lr, const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(model.rng);
    const std::vector<int32_t>& ngrams = dict_->getNgrams(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model.update(ngrams, line[w + c], lr);
      }
    }
  }
}

void FastText::bilingual_skipgram(Model& model, real lr, const std::vector<int32_t>& x, const std::vector<int32_t>& y) {
//  real lr_x = lr * (args_->ws) / y.size();
  real lr_x = lr;
  
  // Forwards
  for (int32_t w = 0; w < x.size(); w++) {
    const std::vector<int32_t>& ngrams_x = dict_->getNgrams(x[w]);
    for (int32_t i = 0; i < y.size(); i++) {
      model.update(ngrams_x, y[i], lr_x);
    }
  }
}

void FastText::step() {
  std::vector<int32_t> line1, line2, labels;
  
  progress = real(tokenCount) / (args_->epoch * dict_->ntokens());
  real lr = args_->lr * (1.0 - progress);
  
  std::uniform_real_distribution<> uniform(0, 1);
  real u = uniform(model_->rng);
  tokenCount += dict_->getLine2(ifs[0], line1, labels, args_->model, u);
  
  if (args_->model == model_name::sup) {
    dict_->addNgrams(line1, args_->wordNgrams);
    supervised(*model_, lr, line1, labels);
  } else if (args_->model == model_name::cbow) {
    cbow(*model_, lr, line1);
  } else if (args_->model == model_name::sg) {
    skipgram(*model_, lr, line1);
  } else if (args_->model == model_name::bil) {
    tokenCount += dict_->getLine2(ifs[1], line2, labels, args_->model, u);

//    skipgram(*model_, lr, line1);
//    skipgram(*model_, lr, line2);
    bilingual_skipgram(*model_, lr, line1, line2);
    bilingual_skipgram(*model_, lr, line2, line1);
  }

  if (tokenCount % args_->lrUpdateRate == 0) {
    if (args_->verbose > 1) {
      printInfo(progress, model_->getLoss());
    }
  }
}

void FastText::setup(std::shared_ptr<Args> args, std::shared_ptr<Dictionary> dict, std::shared_ptr<Matrix> input,
                     std::shared_ptr<Matrix> output) {
  
  std::cout << "setup : " << args->name << std::endl;
  
  start = clock();
  tokenCount = 0;
  args_ = args;
  dict_ = dict;
  input_ = input;
  output_ = output;
  
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label), dict_);
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word), dict_);
  }
  
  std::vector<std::string> possible_inputs = {args->input, args->input_mono1, args->input_mono2, args->input_par1, args->input_par2};
  for(auto possible_input : possible_inputs) {
    if(!possible_input.empty()) {
      std::cout << "adding : " << possible_input << std::endl;
      ifs.push_back(std::ifstream(possible_input));
    }
  }
}

void FastText::close(std::string suffix) {
  for(int i = 0; i < ifs.size(); i++) {
    ifs[i].close();
  }
  model_ = std::make_shared<Model>(input_, output_, args_, 0);
  saveModel(suffix);
  saveVectors(suffix);
}

void lockTrain(std::vector<FastText*> models, real progress) {
  while(progress < 1) {
    FastText* min_model_ = models[0];
    real min_progress_ = min_model_->progress;
    for(auto model : models) {
      if(model->progress < min_progress_) {
        min_model_ = model;
        min_progress_ = model->progress;
      }
    }
    min_model_->step();
    progress = min_progress_;
  }
}

void trainBilingualUnsupervisedMono(int argc, char** argv) {
  FastText ft_par, ft_mono1, ft_mono2;
  
  std::cout << "--\nParsing arguments" << std::endl;
  std::shared_ptr<Args> args = std::make_shared<Args>();
  args->parseArgs(argc, argv);
  
  std::cout << "--\nCreating input matrix" << std::endl;
  std::shared_ptr<Dictionary> dict = std::make_shared<Dictionary>(args);
  std::shared_ptr<Matrix> input = std::make_shared<Matrix>(dict->nwords()+args->bucket, args->dim);
  input->uniform(1.0 / args->dim);
  
  std::shared_ptr<Matrix> output_word;
  output_word = std::make_shared<Matrix>(dict->nwords(), args->dim);
  output_word->zero();

  std::shared_ptr<Args> args_par = std::make_shared<Args>(*args);
  args_par->togglePar();
  ft_par.setup(args_par, dict, input, output_word);
  
  std::shared_ptr<Args> args_mono1 = std::make_shared<Args>(*args);
  args_mono1->toggleMono(1);
  ft_mono1.setup(args_mono1, dict, input, output_word);
  
  std::shared_ptr<Args> args_mono2 = std::make_shared<Args>(*args);
  args_mono2->toggleMono(2);
  ft_mono2.setup(args_mono2, dict, input, output_word);
  
  // Train
  real progress(0);
  std::vector<FastText*> models = {&ft_par, &ft_mono1, &ft_mono2};
  lockTrain(models, progress);
  ft_par.close("");
}

void trainBilingualMono(int argc, char** argv) {
  FastText ft_par, ft_mono1, ft_mono2;
  
  std::cout << "--\nParsing arguments" << std::endl;
  std::shared_ptr<Args> args = std::make_shared<Args>();
  args->parseArgs(argc, argv);
  
  std::cout << "--\nCreating input matrix" << std::endl;
  std::shared_ptr<Dictionary> dict = std::make_shared<Dictionary>(args);
  std::shared_ptr<Matrix> input = std::make_shared<Matrix>(dict->nwords()+args->bucket, args->dim);
  input->uniform(1.0 / args->dim);
  
  std::shared_ptr<Matrix> output_word;
  output_word = std::make_shared<Matrix>(dict->nwords(), args->dim);
  output_word->zero();

  std::shared_ptr<Args> args_mono1 = std::make_shared<Args>(*args);
  args_mono1->toggleMono(1);
  ft_mono1.setup(args_mono1, dict, input, output_word);
  
  std::shared_ptr<Args> args_mono2 = std::make_shared<Args>(*args);
  args_mono2->toggleMono(2);
  ft_mono2.setup(args_mono2, dict, input, output_word);
  
  // Train
  real progress(0);
  std::vector<FastText*> models = {&ft_mono1, &ft_mono2};
  lockTrain(models, progress);
  ft_mono1.close("");
}

int main(int argc, char** argv) {
//  srand(time(NULL));
  
  utils::initTables();
  if (argc < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(argv[1]);
  if (command == "bilingual-um") {
    trainBilingualUnsupervisedMono(argc, argv);
  } else if (command == "bilingual-m") {
    trainBilingualMono(argc, argv);
  
  } else if (command == "test") {
    test(argc, argv);
  } else if (command == "print-vectors") {
    printVectors(argc, argv);
  } else if (command == "predict" || command == "predict-prob" ) {
    predict(argc, argv);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  utils::freeTables();
  return 0;
}
