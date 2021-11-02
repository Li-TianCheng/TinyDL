//
// Created by ltc on 2021/9/10.
//

#include "optimizer/AdamOptimizer.h"

AdamOptimizer::AdamOptimizer(const vector<Tensor*> &parameters, double lr, double beta1, double beta2, double eps) : Optimizer(parameters), lr(lr), beta1(beta1), beta2(beta2), eps(eps), t(0) {
	for (auto& p : parameters) {
		v.emplace_back(p->row(), p->col(), p->isCuda());
		m.emplace_back(p->row(), p->col(), p->isCuda());
		v.back().setZero();
		m.back().setZero();
	}
}

void AdamOptimizer::step() {
	++t;
#pragma omp parallel for
	for (int i = 0; i < parameters.size(); ++i) {
		m[i] = m[i] * beta1 + parameters[i]->grad() * (1-beta1);
		v[i] = v[i] * beta2 + parameters[i]->grad().pow(2) * (1-beta2);
		CuMatrix _m = m[i] / (1-pow(beta1, t));
		CuMatrix tmp((*v[i]).rows(), (*v[i]).cols(), v[i].isCuda());
		tmp.setOnes();
		auto _v = ((v[i] / (1-pow(beta2, t))).pow(0.5) + tmp*eps).pow(-1);
		**parameters[i] -= _m.dot(_v) * lr;
	}
}

void AdamOptimizer::cuda() {
	Optimizer::cuda();
	for (auto& c : m) {
		c.cuda();
	}
	for (auto& c : v) {
		c.cuda();
	}
}

void AdamOptimizer::cpu() {
	Optimizer::cpu();
	for (auto& c : m) {
		c.cpu();
	}
	for (auto& c : v) {
		c.cpu();
	}
}
