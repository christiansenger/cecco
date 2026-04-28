#include <iostream>

#define CECCO_ERASURE_SUPPORT
#include "cecco.hpp"
using namespace CECCO;

int main(void) {
    using F = Fp<2>;

    const size_t n = 17;
    const size_t k = 9;
    auto G = ZeroMatrix<F>(k, n);
    do {
        G.randomize();
    } while (G.rank() < k);

    auto Cp = LinearCode(n, k, G);
    // std::cout << showbasic << Cp << std::endl;
    // std::cout << showmost << Cp << std::endl;
    std::cout << showall << Cp << std::endl;

    auto C = Cp.get_equivalent_code_in_standard_form();
    assert(C.is_equivalent(Cp));
    std::cout << showall << C << std::endl;

    auto Cd = C.get_dual();
    std::cout << showall << Cd << std::endl;

    const auto A = C.get_weight_enumerator();
    const size_t dmin = C.get_dmin();
    const bool is_perfect = C.is_perfect();

    if (C.is_polynomial()) {
        std::cout << "C is polynomial ";
        if (C.is_cyclic()) std::cout << " and cyclic ";
        std::cout << " with generator polynomial " << C.get_gamma() << std::endl;
    } else {
        std::cout << "C is not polynomial" << std::endl;
    }

    if constexpr (std::is_same_v<F, Fp<2>>) {
        const double pe = 0.1;
        BSC channel(pe);
        std::cout << "Word error probability for BSC with pe=" << pe << " is at most "
                  << C.Bhattacharyya_bound(channel.get_Bhattacharyya_param()) << std::endl;
    }

    auto T = C.get_minimal_trellis();
    T.export_as_tikz("trellis.tex");

    // TX
    auto u = Vector<F>(k).randomize();
    std::cout << "Random message:               " << u << std::endl;
    auto c = C.enc(u);

    {
        const double pe = 0.05;
        SDMC<F> channel(pe);

        // RX
        auto r = channel(c);
        Vector<F> c_est, u_est;
        try {
            c_est = C.dec_BD(r);
            u_est = C.encinv(c_est);
            std::cout << "BD decoding message estimate: " << u_est << std::endl;
            if (u_est == u) {
                std::cout << "... this is correct decoding." << std::endl;
            } else {
                std::cout << "... this is wrong decoding/a word error with " << dH(u, u_est)
                          << " (message) symbol errors" << std::endl;
            }
        } catch (const decoding_failure& e) {
            std::cout << "BD decoding failure!" << std::endl;
        }
        c_est = C.dec_ML(r);
        u_est = C.encinv(c_est);
        std::cout << "ML decoding message estimate: " << u_est << std::endl;
        if (u_est == u) {
            std::cout << "... this is correct decoding." << std::endl;
        } else {
            std::cout << "... this is wrong decoding/a word error with " << dH(u, u_est) << " (message) symbol errors"
                      << std::endl;
        }
        assert(dH(r, c_est) == dH(r, C.dec_Viterbi(r)));
    }

    {
        const double pe = 0.025;
        const double px = 0.05;
        SDMEC<F> channel(pe, px);

        // RX
        auto r = channel(c);
        Vector<F> c_est, u_est;
        try {
            c_est = C.dec_BD_EE(r);
            u_est = C.encinv(c_est);
            std::cout << "BD EE decoding message estimate: " << u_est << std::endl;
            if (u_est == u) {
                std::cout << "... this is correct decoding." << std::endl;
            } else {
                std::cout << "... this is wrong decoding/a word error with " << dH(u, u_est)
                          << " (message) symbol errors" << std::endl;
            }
        } catch (const decoding_failure& e) {
            std::cout << "BD EE decoding failure!" << std::endl;
        }
        c_est = C.dec_ML_EE(r);
        u_est = C.encinv(c_est);
        std::cout << "ML EE decoding message estimate: " << u_est << std::endl;
        if (u_est == u) {
            std::cout << "... this is correct decoding." << std::endl;
        } else {
            std::cout << "... this is wrong decoding/a word error with " << dH(u, u_est) << " (message) symbol errors"
                      << std::endl;
        }
        assert(dH(r, c_est) == dH(r, C.dec_Viterbi_EE(r)));
    }

    if constexpr (std::is_same_v<F, Fp<2>>) {
        const double EbNodB = -1;
        BI_AWGN channel(EbNodB);

        // RX
        auto y = channel(c);
        auto LLRs = LLRCalculator(channel)(y);

        {
            auto c_est = C.dec_Viterbi_soft(LLRs);
            auto u_est = C.encinv(c_est);
            std::cout << "Soft ML decoding message estimate: " << u_est << std::endl;
            if (u_est == u) {
                std::cout << "... this is correct decoding." << std::endl;
            } else {
                std::cout << "... this is wrong decoding/a word error with " << dH(u, u_est)
                          << " (message) symbol errors" << std::endl;
            }
        }

        {
            auto output_LLRs = C.dec_BCJR(LLRs);
            auto c_est = Vector<F>(C.get_n());
            for (size_t i = 0; i < C.get_n(); ++i) c_est.set_component(i, output_LLRs[i] < 0 ? 1 : 0);
            auto u_est = c_est.get_subvector(0, k);
            std::cout << "Soft s/s MAP decoding message estimate: " << u_est << std::endl;
            if (u_est == u) {
                std::cout << "... this is correct decoding." << std::endl;
            } else {
                std::cout << "... this is wrong decoding/a word error with " << dH(u, u_est)
                          << " (message) symbol errors" << std::endl;
            }
        }

        {
            auto r = Vector<F>(C.get_n());
            for (size_t i = 0; i < C.get_n(); ++i) r.set_component(i, LLRs[i] < 0 ? 1 : 0);
            std::vector<double> reliabilities(n);
            for (size_t i = 0; i < C.get_n(); ++i) reliabilities[i] = abs(LLRs[i]);

            try {
                auto c_est = C.dec_GMD(r, reliabilities);
                auto u_est = C.encinv(c_est);
                std::cout << "Pseudo-soft GMD decoding message estimate: " << u_est << std::endl;
                if (u_est == u) {
                    std::cout << "... this is correct decoding." << std::endl;
                } else {
                    std::cout << "... this is wrong decoding/a word error with " << dH(u, u_est)
                              << " (message) symbol errors" << std::endl;
                }
            } catch (const decoding_failure& e) {
                std::cout << "GMD decoding failure: " << e.what() << std::endl;
            }
        }
    }

    return 0;
}
