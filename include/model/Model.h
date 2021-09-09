//
// Created by ltc on 2021/9/9.
//

#ifndef TINYDL_MODEL_H
#define TINYDL_MODEL_H

#include <vector>
#include "Tensor.h"

using std::vector;

class Model {
public:
	Tensor operator()(Tensor& input);
	virtual Tensor forward(Tensor& input) = 0;
public:
	vector<Tensor*> parameters;
};


#endif //TINYDL_MODEL_H
