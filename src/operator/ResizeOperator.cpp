//
// Created by ltc on 2021/9/8.
//

#include "operator/ResizeOperator.h"

ResizeOperator::ResizeOperator(const Tensor &tensor1, int rowNum, int colNum, bool isNew) : Operator(tensor1, 0), rowNum(rowNum), colNum(colNum), isNew(isNew) {

}

Tensor ResizeOperator::operator()() {
	auto value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1);
	value->conservativeResize(rowNum, colNum);
	return Tensor(value, shared_from_this());
}

void ResizeOperator::backward(Tensor &result) {
	if (!isNew) {
		result.grad().conservativeResize(tensor1.grad().rows(), tensor1.grad().cols());
		tensor1.grad() += result.grad();
		result.grad().conservativeResize(rowNum, colNum);
	}
}
