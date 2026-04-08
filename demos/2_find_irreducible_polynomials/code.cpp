#include <iostream>

#define CECCO_ERASURE_SUPPORT
#include "cecco.hpp"
using namespace CECCO;

int main(void) {
    using F = Fp<2>;
    // using F = Ext<Fp<2>, {1, 1, 1}>;                                                     // F4
    // using F = Ext<Fp<2>, {1, 0, 1, 1}>;                                                  // F8
    // using F = Iso<Ext<Fp<2>, {1, 0, 0, 1, 1}>, Ext<Ext<Fp<2>, {1, 1, 1}>, {2, 2, 1}> >;  // F16

    // using F = Ext<Fp<3>, {1, 2, 0, 1}>;                                                  // F27

    constexpr size_t m = 6;
    auto P = find_irreducible<F>(m);
    std::cout << "Monic irreducible polynomial: " << P << std::endl;
    std::cout << "Coefficient vector (to be used as template parameter of Ext<>): " << Vector(P) << std::endl;

    auto Q = ConwayPolynomial<p, m>();
    std::cout << "Conway polynomial: " << Q << std::endl;
    std::cout << "Coefficient vector (to be used as template parameter of Ext<>): " << Vector(Q) << std::endl;

    return 0;
}