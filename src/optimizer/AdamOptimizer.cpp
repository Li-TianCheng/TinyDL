//
// Created by ltc on 2021/9/10.
//

#include "optimizer/AdamOptimizer.h"

AdamOptimizer::AdamOptimizer(const vector<Tensor> &parameters, double lr, double beta1, double beta2, double eps) : Optimizer(parameters), lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {
	for (auto& p : parameters) {
		v.emplace_back(p.row(), p.col());
		m.emplace_back(p.row(), p.col());
	}
}

void AdamOptimizer::step() {
	++t;
#pragma omp parallel for
	for (int i = 0; i < parameters.size(); ++i) {
		*m[i] = *m[i] * beta1 + parameters[i].grad() * (1-beta1);
		*v[i] = *v[i] * beta2 + parameters[i].grad().array().pow(2).matrix() * (1-beta2);
		auto _m = *m[i] / (1-pow(beta1, t));
		auto _v = (((*v[i]).array() / (1-pow(beta2, t))).pow(0.5) + eps).pow(-1).matrix();
		*parameters[i] -= (_m.array() * _v.array()).matrix() * lr;
	}
}
