//
// Created by ltc on 2021/9/8.
//

#include "operator/TransposeOperator.h"

TransposeOperator::TransposeOperator(const Tensor &tensor1, bool isNew) : Operator(tensor1, 0), isNew(isNew) {

}

Tensor TransposeOperator::operator()() {
	auto value = std::make_shared<CuMatrix>((*tensor1).transpose());
	return Tensor(value, shared_from_this());
}

void TransposeOperator::backward(Tensor &result) {
	if (!isNew) {
		tensor1.grad() += result.grad().transpose();
	}
}
