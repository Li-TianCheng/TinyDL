//
// Created by ltc on 2021/9/9.
//

#ifndef TINYDL_LINEAR_H
#define TINYDL_LINEAR_H

#include "Tensor.h"
#include "model/Model.h"

class Linear {
public:
	Linear(Model& m, int inputNum, int outputNum, bool bias=true);
	Tensor operator()(const Tensor& input);
private:
	bool isBias;
	Tensor weight;
	Tensor bias;
};


#endif //TINYDL_LINEAR_H
