//
// Created by ltc on 2021/9/9.
//

#ifndef TINYDL_LINEAR_H
#define TINYDL_LINEAR_H

#include <vector>
#include "Tensor.h"
#include "model/Model.h"

using std::vector;

class Linear {
public:
	Linear(Model& m, int inputNum, int outputNum, bool base=true);
	Tensor operator()(Tensor& input);
private:
	bool isBase;
	Tensor weight;
	Tensor base;
};


#endif //TINYDL_LINEAR_H
