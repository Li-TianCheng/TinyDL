//
// Created by ltc on 2021/9/2.
//

#ifndef TINYDL_LOGOPERATOR_H
#define TINYDL_LOGOPERATOR_H

#include "operator/Operator.h"

class LogOperator : public Operator {
public:
	LogOperator(const Tensor& tensor1, const Tensor& tensor2);
	Tensor operator()() override;
	void grad(Tensor& result) override;
};


#endif //TINYDL_LOGOPERATOR_H
