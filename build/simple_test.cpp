#include "ml_engine/tensor.h"
#include <iostream>
using namespace ml_engine;
int main() {
    Tensor32f t1({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor32f t2({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    auto result = t1 + t2;
    std::cout << "Basic tensor operations work!" << std::endl;
    return 0;
}
