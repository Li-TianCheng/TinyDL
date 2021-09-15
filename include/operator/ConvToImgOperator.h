//
// Created by ltc on 2021/9/13.
//

#ifndef TINYDL_CONVTOIMGOPERATOR_H
#define TINYDL_CONVTOIMGOPERATOR_H

#include "operator/Operator.h"

class ConvToImgOperator : public Operator {
public:
	ConvToImgOperator(const Tensor& tensor1, int num);
	Tensor operator()() override;
	void backward(Tensor& result) override;
private:
	int num;
};


#endif //TINYDL_CONVTOIMGOPERATOR_H
