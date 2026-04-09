#include <iostream>

#include "cecco.hpp"
using namespace CECCO;

int main(void) {
    using F4 = Ext<Fp<2>, {1, 1, 1}>;
    using F8 = Ext<Fp<2>, {1, 0, 1, 1}>;
    using F16 = Iso<Ext<Fp<2>, {1, 0, 0, 1, 1}>, Ext<Ext<Fp<2>, {1, 1, 1}>, {2, 2, 1}> >;

    using F27 = Ext<Fp<3>, {1, 2, 0, 1}>;

    constexpr size_t q = 2;
    constexpr size_t m = 6;
    using Fq = Fp<q>;
    auto P = find_irreducible<Fq>(m);
    std::cout << "Monic irreducible polynomial: " << P << std::endl;
    std::cout << "Coefficient vector (to be used as template parameter of Ext<>): " << P.get_coefficients() << std::endl;

    auto Q = ConwayPolynomial<q, m>();
    std::cout << "Conway polynomial: " << Q << std::endl;
    std::cout << "Coefficient vector (to be used as template parameter of Ext<>): " << Q.get_coefficients() << std::endl;

    return 0;
}
