#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x
OBJS = args.o dictionary.o matrix.o vector.o model.o utils.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops
opt: fasttext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: fasttext

args.o: fasttext/args.cc fasttext/args.h
	$(CXX) $(CXXFLAGS) -c fasttext/args.cc

dictionary.o: fasttext/dictionary.cc fasttext/dictionary.h fasttext/args.h
	$(CXX) $(CXXFLAGS) -c fasttext/dictionary.cc

matrix.o: fasttext/matrix.cc fasttext/matrix.h fasttext/utils.h
	$(CXX) $(CXXFLAGS) -c fasttext/matrix.cc

vector.o: fasttext/vector.cc fasttext/vector.h fasttext/utils.h
	$(CXX) $(CXXFLAGS) -c fasttext/vector.cc

model.o: fasttext/model.cc fasttext/model.h fasttext/args.h
	$(CXX) $(CXXFLAGS) -c fasttext/model.cc

utils.o: fasttext/utils.cc fasttext/utils.h
	$(CXX) $(CXXFLAGS) -c fasttext/utils.cc

fasttext : $(OBJS) fasttext/fasttext.cc
	$(CXX) $(CXXFLAGS) $(OBJS) fasttext/fasttext.cc -o ft

clean:
	rm -rf *.o ft
