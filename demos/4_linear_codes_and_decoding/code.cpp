#include <iostream>

#define CECCO_ERASURE_SUPPORT
#include "cecco.hpp"
using namespace CECCO;

int main(void) {
    using F = Ext<Fp<2>, MOD{1, 1, 1}>;

    const size_t n = 13;
    const size_t k = 6;
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

    // note: this only works if F has characteristic 2

    DEMUX<F, Fp<2>> demux;
    const double EbNodB = 0;
    BI_AWGN channel(EbNodB);
    LLRCalculator<F> llrcalculator(channel);

    // RX
    auto y = channel(demux(c));
    auto llrs = llrcalculator(y);
    std::cout << "Per-symbol LLR matrix: " << std::endl << llrs << std::endl;

    auto r = Vector<F>(n);
    auto reliabilities = Vector<double>(n);

    for (size_t i = 0; i < n; ++i) {
        size_t best_label = 0; // initialization, F(0) is best
        double best = 0.0;
        double second = std::numeric_limits<double>::infinity();
        for (size_t a = 1; a < F::get_q(); ++a) {
            const double cost = llrs(a - 1, i);
            if (cost < best) {
                second = best;
                best = cost;
                best_label = a;
            } else if (cost < second) {
                second = cost;
            }
        }

        // hard decision: most probable symbol
        r.set_component(i, F(best_label));

        // reliability in [0, 1]: best-vs-second through tanh
        const double lambda = second - best;
        reliabilities.set_component(i, std::tanh(lambda / 2.0));
    }

    std::cout << "Hard decision r: " << std::endl << r << std::endl;
    std::cout << "Reliabilities: " << std::endl << reliabilities << std::endl;

    try {
        auto c_est = C.dec_GMD(r, reliabilities);
        auto u_est = C.encinv(c_est);
        std::cout << "GMD decoding message estimate: " << u_est << std::endl;
        if (u_est == u) {
            std::cout << "... this is correct decoding." << std::endl;
        } else {
            std::cout << "... this is wrong decoding/a word error with " << dH(u, u_est) << " (message) symbol errors"
                      << std::endl;
        }
    } catch (const decoding_failure& e) {
        std::cout << "GMD decoding failure: " << e.what() << std::endl;
    }

    return 0;
}
