//
// Created by ltc on 2021/9/10.
//

#include "optimizer/SGDOptimizer.h"

SGDOptimizer::SGDOptimizer(const vector<Tensor*> &parameters, double lr, double rho) : Optimizer(parameters), lr(lr), rho(rho) {
	for (auto& p : parameters) {
		v.emplace_back(p->row(), p->col(), p->isCuda());
		v.back().setZero();
	}
}

void SGDOptimizer::step() {
#pragma omp parallel for
	for (int i = 0; i < parameters.size(); ++i) {
		v[i] = v[i] * rho + parameters[i]->grad();
		**parameters[i] -= v[i] * lr;
	}
}

void SGDOptimizer::cuda() {
	Optimizer::cuda();
	for (auto& c : v) {
		c.cuda();
	}
}

void SGDOptimizer::cpu() {
	Optimizer::cpu();
	for (auto& c : v) {
		c.cpu();
	}
}
