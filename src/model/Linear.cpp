//
// Created by ltc on 2021/9/9.
//

#include "model/Linear.h"

Linear::Linear(Model& m, int inputNum, int outputNum, bool base) : weight(inputNum, outputNum), base(1, outputNum), isBase(base) {
	weight.setRandom();
	this->base.setRandom();
	m.parameters.push_back(weight);
	if (isBase) {
		m.parameters.push_back(this->base);
	}
}

Tensor Linear::operator()(const Tensor &input) {
	if (isBase) {
		Tensor t(input.row(), 1);
		t.setOnes();
		return input * weight + t * base;
	}
	return input * weight;
}

