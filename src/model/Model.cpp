//
// Created by ltc on 2021/9/9.
//

#include "model/Model.h"

Tensor Model::operator()(const Tensor &input) {
	return forward(input);
}

void Model::cuda() {
	for (auto& p : parameters) {
		p->cuda();
	}
}

void Model::cpu() {
	for (auto& p : parameters) {
		p->cpu();
	}
}
