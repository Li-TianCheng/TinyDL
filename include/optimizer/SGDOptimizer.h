//
// Created by ltc on 2021/9/10.
//

#ifndef TINYDL_SGDOPTIMIZER_H
#define TINYDL_SGDOPTIMIZER_H

#include "optimizer/Optimizer.h"

class SGDOptimizer : public Optimizer {
public:
	SGDOptimizer(const vector<Tensor*>& parameters, double lr, double rho);
	void step() override;
	void cuda() override;
	void cpu() override;
	~SGDOptimizer() override = default;
private:
	vector<CuMatrix> v;
	double lr;
	double rho;
};


#endif //TINYDL_SGDOPTIMIZER_H
