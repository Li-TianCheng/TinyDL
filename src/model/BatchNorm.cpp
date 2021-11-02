//
// Created by ltc on 2021/9/13.
//

#include "model/BatchNorm.h"

BatchNorm::BatchNorm(Model &m, int inputNum, double eps) : weight(1, inputNum, false), bias(1, inputNum, false), eps(eps){
	weight.setRandom();
	bias.setRandom();
	m.parameters.push_back(&weight);
	m.parameters.push_back(&bias);
}

Tensor BatchNorm::operator()(const Tensor &input) {
	Tensor t(1, input.row(), input.isCuda());
	t.setOnes();
	Tensor m = t * input / Tensor(input.row(), input.isCuda());
	Tensor v = t * (input - t.transpose() * m).pow(2) / Tensor(input.row(), input.isCuda());
	Tensor output = input;
	output -= t.transpose() * m;
	v = (v+Tensor(eps, input.isCuda())).pow(-2);
	output = output.dot(t.transpose() * v);
	output = output.dot(t.transpose() * weight);
	output += t.transpose() * bias;
	return output;
}
