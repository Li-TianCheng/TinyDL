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
	return (*shared_ptr<Operator>(new ReluOperator(t)))();
}
