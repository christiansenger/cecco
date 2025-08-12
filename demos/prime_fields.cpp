#include <iostream>

// Use ECC library, compile with -std=c++20 -I../
// Adapt -Ipath_to_hpp_files if necessary, append -O3 for performance
#include "ecc.hpp"
using namespace ECC;

int main(void) {
    // Make prime field available under convenient name (any prime is allowed, here: p = 2):
    using F2 = Fp<2>;  // block scope, might have to put this in global scope

    // Define a few elements (using different methods):
    F2 a(1);
    F2 b = 0;
    auto c = F2(1);

    std::cout << "Show textual info about the field:" << std::endl;
    std::cout << F2::get_info() << std::endl;

    // Perform basic calculations:
    c = a + b;
    auto d = c * c;  // for convenience: let compiler figure out the type of the result
    auto e = -d / F2(1);
    e *= c;

    // Get uniformly random field element (using different methods):
    F2 f;
    f.randomize();
    auto g = F2().randomize();

    // Output field elements
    std::cout << "(Random) value of g is " << g << std::endl;

    // Get order of field elements
    std::cout << "Additive order of g: " << g.get_additive_order() << std::endl;
    if (!g.is_zero()) std::cout << "Multiplicative order of g: " << g.get_multiplicative_order() << std::endl;

    // Erase field elements (make them have an invalid value)
    g.erase();
    std::cout << "Value of erased g is " << g << std::endl;
    if (g.is_erased()) std::cout << "g is now erased" << std::endl;

    // Unerase field elements (return them back into the field, set value to zero)
    g.unerase();
    std::cout << "Value of unerased g is " << g << std::endl;
    if (!g.is_erased()) std::cout << "g is not erased any more" << std::endl;

    /*
     * Can also query characteristic of field, get a generator of the multiplicative group, etc.
     * See https://christiansenger.github.io/ecc/classECC_1_1Fp.html for documentation
     *
     * Prime fields with three, five, seven, etc. elements work in the same manner:
     * using F3 = Fp<3>;
     * using F5 = Fp<5>;
     * using F7 = Fp<7>;
     * ...
     */

    return 0;
}