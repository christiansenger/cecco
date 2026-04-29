#include <iostream>

#include "cecco.hpp"
using namespace CECCO;

int main(void) {
    using F = Ext<Fp<2>, {1, 1, 1}, LutMode::CompileTime>;

    const size_t n = 17;
    const size_t k = 9;

    auto gamma = Polynomial<F>();
    do {
        gamma.randomize(n - k);
    } while (gamma[0].is_zero());

    auto C = LinearCode<F>(k, gamma);
    std::cout << showall << C << std::endl;

    auto Cd = C.get_dual();
    std::cout << showall << Cd << std::endl;

    if (C.is_cyclic()) {
        assert(Cd.is_polynomial());
        assert(Cd.is_cyclic());
    }

    // TX
    auto u = Vector<F>(k).randomize();
    std::cout << "Random message:               " << u << std::endl;
    auto c = C.enc(u);

    const double pe = 0.05;
    SDMC<F> channel(pe);

    // RX
    try {
        auto c_est = C.dec_BD(r);
        if (C.is_cyclic()) assert(c_est == C.dec_Meggitt(r));
        auto u_est = C.encinv(c_est);
        std::cout << "BD decoding message estimate: " << u_est << std::endl;
        }
    } catch (const decoding_failure& e) {
        std::cout << "BD decoding failure!" << std::endl;
    }

    auto Cp = SubfieldSubcode(C);
    std::cout << showall << Cp << std::endl;
    assert(Cp.is_polynomial());
    if (C.is_cyclic()) assert(Cp.is_cyclic());

    return 0;
}
