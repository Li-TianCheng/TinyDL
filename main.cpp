#include <iostream>
#include "Tensor.h"

using namespace std;

int main() {
	Matrix<double, 1, 2> a;
	Matrix<double, 2, 1> b;
	a << 1,2;
	b << 2,1;
	Tensor t1(a);
	Tensor t2(b);
	Tensor t3 = (t1 * t2).exp();
//	cout << a << endl;
//	cout << b << endl;
	cout << *t3 << endl;
	t3.backward();
	cout << t1.grad() << endl;
	cout << t2.grad() << endl;
    return 0;
}
