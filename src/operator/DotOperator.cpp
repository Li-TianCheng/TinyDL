//
// Created by ltc on 2021/9/8.
//

#include "operator/DotOperator.h"

DotOperator::DotOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor DotOperator::operator()() {
	auto value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(((*tensor1).array() * (*tensor2).array()).matrix());
	return Tensor(value, shared_from_this());
}

void DotOperator::backward(Tensor &result) {
	tensor1.grad() += ((*tensor2).array() * result.grad().array()).matrix();
	tensor2.grad() += ((*tensor1).array() * result.grad().array()).matrix();
}
