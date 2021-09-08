//
// Created by ltc on 2021/9/8.
//

#include "operator/DotOperator.h"

DotOperator::DotOperator(const Tensor &tensor1, const Tensor &tensor2) : Operator(tensor1, tensor2) {

}

Tensor DotOperator::operator()() {
	auto value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(((*tensor1.value).array() * (*tensor2.value).array()).matrix());
	return Tensor(value, shared_from_this());
}

void DotOperator::grad(Tensor &result) {
	*tensor1.gradient += ((*tensor2.value).array() * (*result.gradient).array()).matrix();
	*tensor2.gradient += ((*tensor1.value).array() * (*result.gradient).array()).matrix();
}
