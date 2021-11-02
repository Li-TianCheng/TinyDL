//
// Created by ltc on 2021/9/10.
//

#include "model/ActivateFun.h"

Tensor sigmoid(const Tensor& t) {
	return (t.exp().pow(-1)+Tensor(1, t.isCuda())).pow(-1);
}

Tensor tanh(const Tensor& t) {
	return sigmoid(t*Tensor(2, t.isCuda()))*Tensor(2, t.isCuda())-Tensor(1, t.isCuda());
}

Tensor relu(const Tensor& t) {
	return (*shared_ptr<Operator>(new ReluOperator(t)))();
}
