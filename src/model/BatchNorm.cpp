//
// Created by ltc on 2021/9/13.
//

#include "model/BatchNorm.h"

BatchNorm::BatchNorm(Model &m, int inputNum, double eps) : weight(1, inputNum), base(1, inputNum), eps(eps){
	weight.setRandom();
	base.setRandom();
	m.parameters.push_back(weight);
	m.parameters.push_back(base);
}

Tensor BatchNorm::operator()(const Tensor &input) {
	Tensor t(1, input.row());
	t.setOnes();
	Tensor m = t * input / input.row();
	Tensor v = t * (input - t.transpose() * m).pow(2) / input.row();
	Tensor output = input;
	output -= t.transpose() * m;
	v = (v+eps).pow(-2);
	output = output.dot(t.transpose() * v);
	output = output.dot(t.transpose() * weight);
	output += t.transpose() * base;
	return output;
}
