//
// Created by ltc on 2021/9/8.
//

#include "operator/DotOperator.h"

DotOperator::DotOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor DotOperator::operator()() {
	auto value = std::make_shared<CuMatrix>(((*tensor1).dot(*tensor2)));
	return Tensor(value, shared_from_this());
}

void DotOperator::backward(Tensor &result) {
	tensor1.grad() += (*tensor2).dot(result.grad());
	tensor2.grad() += (*tensor1).dot(result.grad());
}
