#include <iostream>

#define CECCO_ERASURE_SUPPORT
#include "cecco.hpp"
using namespace CECCO;
using F2 = Fp<2>;

int main(void) {
    F2 a(1);
    F2 b = 0;
    auto c = F2(1);

    std::cout << "Show textual info about the field:" << std::endl;
    std::cout << F2::get_info() << std::endl;

    c = a + b;
    auto d = c * c;
    auto e = -d / F2(1);
    e *= c;

    F2 f;
    f.randomize();
    auto g = F2().randomize();

    std::cout << "(Random) value of g is " << g << std::endl;

    std::cout << "Additive order of g: " << g.get_additive_order() << std::endl;
    if (!g.is_zero()) std::cout << "Multiplicative order of g: " << g.get_multiplicative_order() << std::endl;

    g.erase();
    std::cout << "Value of erased g is " << g << std::endl;
    if (g.is_erased()) std::cout << "g is now erased" << std::endl;

    g.unerase();
    std::cout << "Value of unerased g is " << g << std::endl;
    if (!g.is_erased()) std::cout << "g is not erased any more" << std::endl;

    return 0;
}