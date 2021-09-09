//
// Created by ltc on 2021/9/8.
//

#ifndef TINYDL_RESIZEOPERATOR_H
#define TINYDL_RESIZEOPERATOR_H

#include "operator/Operator.h"

class ResizeOperator : public Operator {
public:
	ResizeOperator(const Tensor& tensor1, int rowNum, int colNum, bool isNew);
	Tensor operator()() override;
	void backward(Tensor& result) override;
private:
	bool isNew;
	int rowNum;
	int colNum;
};


#endif //TINYDL_RESIZEOPERATOR_H
