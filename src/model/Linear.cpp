//
// Created by ltc on 2021/9/9.
//

#include "model/Linear.h"

Linear::Linear(Model& m, int inputNum, int outputNum, bool bias) : weight(inputNum, outputNum, false), bias(1, outputNum, false), isBias(bias) {
	weight.setRandom();
	this->bias.setRandom();
	m.parameters.push_back(&weight);
	if (isBias) {
		m.parameters.push_back(&this->bias);
	}
}

Tensor Linear::operator()(const Tensor &input) {
	if (isBias) {
		Tensor t(input.row(), 1, input.isCuda());
		t.setOnes();
		return input * weight + t * bias;
	}
	return input * weight;
}

