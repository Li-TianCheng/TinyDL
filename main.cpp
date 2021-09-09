#include <iostream>
#include "Tensor.h"
#include "model/Model.h"
#include "model/Linear.h"

using namespace std;

class Example : public Model {
public:
	Tensor forward(Tensor &input) override {
		auto out = fc1(input);
		out = fc2(out);
		out = fc3(out);
		return out;
	}
private:
	Linear fc1 = Linear(*this, 1, 400);
	Linear fc2 = Linear(*this, 400, 200);
	Linear fc3 = Linear(*this, 200, 1);
};

int main() {
	Eigen::initParallel();
	Tensor t(1, 1);
	t.setRandom();
	cout << "t: " << t << endl;
	Example e;

	for (int i = 0; i < 100; i++) {
		Tensor out = e(t);
		cout << "out: " << out << endl;
		out.backward();
		for (auto& j : e.parameters) {
			*j -= j->grad() * 0.01;
			j->clearGradient();
		}
	}
    return 0;
}
