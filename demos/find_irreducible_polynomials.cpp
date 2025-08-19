#include <iostream>

// Use CECCO library, compile with -std=c++20 -I../
// Adapt -Ipath_to_hpp_files if necessary, append -O3 for performance
#include "cecco.hpp"
using namespace CECCO;

int main(void) {
    // Start with any finite field (use only one of the following three lines)
    using F = Fp<2>;
    // using F = Ext<Fp<2>, {1, 1, 1}>;
    // using F = Ext<Fp<2>, {1, 0, 1, 1}>;                                                               // F2
    // using F = Iso<Ext<Fp<2>, {1, 0, 0, 1, 1}>, Ext<Ext<Fp<2>, {1, 1, 1}>, {2, 2, 1}> >;  // F16
    // using F = Ext<Fp<3>, {1, 2, 0, 1}>;                                                  // F27

    // Find random monic irreducible polynomial with coefficients from F of given degree m (here: m = 3)
    size_t m = 6;
    auto p = find_irreducible<F>(m);
    std::cout << "Monic irreducible polynomial: " << p << std::endl;
    std::cout << "Coefficient vector (to be used as template parameter of Ext<>): " << Vector(p) << std::endl;

    /*
     * Use coefficient vector in order to construct field with |F|^m elements (as a superfield of F), cf. demo
     * field_extensions.cpp. See https://christiansenger.github.io/ecc/classCECCO_1_1Ext.html for documentation.
     */

    return 0;
}
