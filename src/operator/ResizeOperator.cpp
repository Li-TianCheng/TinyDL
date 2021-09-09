//
// Created by ltc on 2021/9/8.
//

#include "operator/ResizeOperator.h"

ResizeOperator::ResizeOperator(const Tensor &tensor1, int rowNum, int colNum, bool isNew) : Operator(tensor1, 0), rowNum(rowNum), colNum(colNum), isNew(isNew) {

}

Tensor ResizeOperator::operator()() {
	auto value = std::make_shared<Matrix<double, Dynamic, Dynamic, RowMajor>>(*tensor1.value);
	value->resize(rowNum, colNum);
	return Tensor(value, shared_from_this());
}

void ResizeOperator::backward(Tensor &result) {
	if (!isNew) {
		result.gradient->resize(tensor1.gradient->rows(), tensor1.gradient->cols());
		*tensor1.gradient += *result.gradient;
		result.gradient->resize(rowNum, colNum);
	}
}
