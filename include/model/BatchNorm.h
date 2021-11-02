//
// Created by ltc on 2021/9/13.
//

#ifndef TINYDL_BATCHNORM_H
#define TINYDL_BATCHNORM_H

#include "Tensor.h"
#include "model/Model.h"

class BatchNorm {
public:
	BatchNorm(Model& m, int inputNum, double eps=1e-5);
	Tensor operator()(const Tensor& input);
private:
	double eps;
	Tensor weight;
	Tensor bias;
};


#endif //TINYDL_BATCHNORM_H
