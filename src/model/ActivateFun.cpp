//
// Created by ltc on 2021/9/10.
//

#include "model/ActivateFun.h"

Tensor sigmoid(const Tensor& t) {
	return (t.exp().pow(-1)+1).pow(-1);
}

Tensor tanh(const Tensor& t) {
	return sigmoid(t*2)*2-1;
}

Tensor relu(const Tensor& t) {
	Tensor tmp(t.row(), t.col());
#pragma omp parallel
	for (int i = 0; i < tmp.row(); ++i) {
		for (int j = 0; j < tmp.col(); ++j) {
			if (t(i, j) > 0) {
				(*tmp)(i, j) = 1;
			}
		}
	}
	return t.dot(tmp);
}
