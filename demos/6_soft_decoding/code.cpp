#include <iostream>

#include "cecco.hpp"
using namespace CECCO;

int main(void) {
    using F2 = Fp<2>;
    using F4 = Ext<F2, MOD{1, 1, 1}>;

    DEMUX<F4, F2> demux;
    BI_AWGN bi_awgn(2.0);
    LLRCalculator<F4> llrcalculator(bi_awgn);  // field aware LLR calculator

    std::cout << F4::get_info() << std::endl;

    auto Cpp = HammingCode<F4>(3);
    auto Cp = extend(Cpp);
    auto C = Cp.get_equivalent_code_in_standard_form();
    std::cout << showall << C << std::endl;

    const size_t k = C.get_k();

    auto u = Vector<F4>(k).randomize();
    Matrix<F2> u_binary = demux(u);  // only needed for counting bit errors
    std::cout << "Random message over F4:        " << u << std::endl;

    auto c = C.enc(u);
    std::cout << "Codeword over F4:              " << c << std::endl;

    auto llrs = llrcalculator(bi_awgn(demux(c)));

    auto c_hat_Viterbi = C.dec_Viterbi_soft(llrs);
    auto u_hat_Viterbi = C.encinv(c_hat_Viterbi);
    std::cout << "Viterbi message estimate:      " << u_hat_Viterbi << std::endl;
    std::cout << "Viterbi message bit errors:    " << dH(u_binary, demux(u_hat_Viterbi)) << std::endl;

    /*
    auto c_hat_ML = C.dec_ML_soft(llrs);  // highly inefficient, not for larger codes!
    auto u_hat_ML = C.encinv(c_hat_ML);
    std::cout << "ML message estimate:           " << u_hat_ML << std::endl;
    std::cout << "ML message bit errors:         " << dH(u_binary, demux(u_hat_ML)) << std::endl;

    assert(c_hat_ML == c_hat_Viterbi);  // Viterbi _is_ ML
    */

    auto c_hat_BCJR = C.dec_BCJR(llrs);
    auto u_hat_BCJR = C.encinv(c_hat_BCJR);
    std::cout << "BCJR message estimate:         " << u_hat_BCJR << std::endl;
    std::cout << "BCJR message bit errors:       " << dH(u_binary, demux(u_hat_BCJR)) << std::endl;

    auto c_hat_BP = C.dec_BP(llrs, 20);  // max. 20 iterations of BP decoding
    auto u_hat_BP = C.encinv(c_hat_BP);
    std::cout << "BP message estimate:           " << u_hat_BP << std::endl;
    std::cout << "BP message bit errors:         " << dH(u_binary, demux(u_hat_BP)) << std::endl;

    {
        /* alternative realization: simulation chain (BCJR example) */

        Enc enc(C);                  // encoder block
        Dec dec(C, method_t::BCJR);  // (BCJR) decoder block
        Encinv encinv(C);            // (systematic, standard form) encoder inverse block

        Vector<F4> u_hat;

        u >> enc >> demux >> bi_awgn >> llrcalculator >> dec >> encinv >> u_hat;

        std::cout << "Sim. chain message estimate:   " << u_hat << std::endl;
        std::cout << "Sim. chain message bit errors: " << dH(u_binary, demux(u_hat)) << std::endl;
    }

    return 0;
}
