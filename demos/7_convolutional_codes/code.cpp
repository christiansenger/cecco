#include <iostream>

#include "cecco.hpp"
using namespace CECCO;

using F2 = Fp<2>;
using F4 = Ext<F2, MOD{1, 1, 1}>;

// converts a polynomial in customary octal notation into the corresponding polynomial,
// e.g., from_octal(013) = 1 + x^2 + x^3; the MSB is the constant term, so the width is
// derivable from the bit width alone whenever p(0) != 0
Polynomial<F2> from_octal(unsigned g) {
    Polynomial<F2> p(0);
    const size_t width = std::bit_width(g);
    for (size_t e = 0; e < width; ++e)
        if ((g >> (width - 1 - e)) & 1u) p.set_coefficient(e, F2(1));
    return p;
}

int main(void) {
    // classic rate-1/2 encoder (5, 7) with memory 2, given in octal notation
    auto C = ConvolutionalCode<F2>({05, 07}, 8, termination_t::zero_terminated);
    std::cout << showall << C << std::endl;

    const auto& T1 = C.get_minimal_trellis();
    std::cout << "Trellis state complexity: " << T1.get_maximum_depth() << std::endl;
    T1.export_as_tikz("trellis1.tikz");
    auto T2 = T1.merge_segments<F4>();
    T2.export_as_tikz("trellis2.tikz");

    for (size_t L : {2, 4, 8, 16}) {
        auto CL = ConvolutionalCode<F2>({05, 07}, L, termination_t::zero_terminated);
        std::cout << "zero-terminated, L = " << std::setw(2) << L << ": [" << CL.get_n() << ", " << CL.get_k()
                  << "], R = " << static_cast<double>(CL.get_k()) / CL.get_n() << ", dmin = " << CL.get_dmin()
                  << std::endl;
    }

    for (size_t L : {3, 4, 8, 16}) {
        auto CL = ConvolutionalCode<F2>({05, 07}, L, termination_t::tailbitten);
        std::cout << "tailbitten,      L = " << std::setw(2) << L << ": [" << CL.get_n() << ", " << CL.get_k()
                  << "], R = " << static_cast<double>(CL.get_k()) / CL.get_n() << ", dmin = " << CL.get_dmin()
                  << std::endl;
    }

    {
        // TX
        auto u = Vector<F2>(C.get_k()).randomize();
        std::cout << "Random message:                      " << u << std::endl;
        auto c = C.enc(u);

        const double pe = 0.05;
        BSC channel(pe);

        // RX
        auto r = channel(c);
        auto c_est = C.dec_Viterbi(r);
        auto u_est = C.encinv(c_est);
        std::cout << "Hard-input Viterbi message estimate: " << u_est << std::endl;
    }

    {
        // TX
        auto u = Vector<F2>(C.get_k()).randomize();
        std::cout << "Random message:                      " << u << std::endl;

        BI_AWGN channel(2.0);  // BPSK at Eb/N0 = 2 dB
        LLRCalculator llrcalculator(channel);

        // RX
        auto llrs = llrcalculator(channel(C.enc(u)));
        auto c_est = C.dec_Viterbi_soft(llrs);
        auto u_est = C.encinv(c_est);
        std::cout << "Soft-input Viterbi message estimate: " << u_est << std::endl;

        // alternative realization: simulation chain
        Enc enc(C);
        Dec dec(C, method_t::Viterbi_soft);
        Encinv encinv(C);

        Vector<F2> u_hat;
        u >> enc >> channel >> llrcalculator >> dec >> encinv >> u_hat;
        std::cout << "Simulation chain message estimate:   " << u_hat << std::endl;
    }

    // 1 + x and 1 + x^2 = (1 + x)^2 share the factor 1 + x, which divides x^L - 1 for every L
    try {
        auto C = ConvolutionalCode<F2>({06, 05}, 8, termination_t::tailbitten);
    } catch (const std::invalid_argument& e) {
        std::cout << e.what() << std::endl;
    }

    // k_cc = 2, n_cc = 3 encoder with row degrees 1 and 2: row widths are derived from the
    // largest entry per row, so (2, 1, 3) reads as (1, x, 1 + x) and (1, 4, 7) as
    // (x^2, 1, 1 + x + x^2)
    {
        auto C = ConvolutionalCode<F2>({{2, 1, 3}, {1, 4, 7}}, 6, termination_t::zero_terminated);
        std::cout << showmost << C << std::endl;
    }

    {
        // rate-2/3 systematic encoder with memory 1 over F4
        const auto zero = Polynomial<F4>(0);
        const auto one = Polynomial<F4>(1);
        const auto G_cc =
            Matrix<Polynomial<F4>>({{one, zero, Polynomial<F4>({1, 2})}, {zero, one, Polynomial<F4>({2, 1})}});

        auto Czt = ConvolutionalCode<F4>(G_cc, 4, termination_t::zero_terminated);
        std::cout << showall << Czt << std::endl;

        auto Ctb = ConvolutionalCode<F4>(G_cc, 4, termination_t::tailbitten);
        std::cout << showall << Ctb << std::endl;
    }

    const auto roundtrip_test = [](const ConvolutionalCode<F2>& CC) {
        const size_t k = CC.get_k();
        auto S = ZeroMatrix<F2>(k, k);
        do {
            S.randomize();
        } while (S.rank() < k);
        auto C_obscured = LinearCode(CC.get_n(), k, S * CC.get_G());  // structure hidden by change of basis

        auto C_recognized = ConvolutionalCode<F2>(C_obscured);  // ... and recovered by recognition
        assert(C_recognized == CC);
        std::cout << showmost << C_recognized << std::endl;
    };
    roundtrip_test(ConvolutionalCode<F2>({05, 07}, 8, termination_t::zero_terminated));
    roundtrip_test(ConvolutionalCode<F2>({05, 07}, 8, termination_t::tailbitten));

    // every polynomial code (cyclic binary [7, 4] Hamming here) is zero-terminated with k_cc = n_cc = 1
    {
        auto HamC = HammingCode<F2>(LinearCode<F2>(4, Polynomial<F2>({F2(1), F2(1), F2(0), F2(1)})));
        std::cout << showall << HamC << std::endl;
        auto CC = ConvolutionalCode<F2>(HamC);
        assert(CC == HamC && CC.is_zero_terminated());
        std::cout << showall << CC << std::endl;
    }

    {
        const size_t L = 12;
        auto CC = ConvolutionalCode<F2>({05, 07}, L, termination_t::tailbitten);
        std::cout << "Tailbitten mother code: " << showbasic << CC << ", dmin = " << CC.get_dmin() << std::endl;

        // messages divisible by p form a polynomial code with generator polynomial p, so the
        // expurgated tailbitten code is generated by the product of the two generator matrices
        const auto expurgated = [&CC, L](const Polynomial<F2>& p) {
            const size_t kp = L - p.degree();
            return LinearCode(CC.get_n(), kp, LinearCode<F2>(kp, p).get_G() * CC.get_G());
        };

        // all degree-3 ELFs (expurgating linear functions) with nonzero constant term, again in
        // octal notation
        auto p = Polynomial<F2>();
        size_t dmin_best = 0;
        for (unsigned g : {011u, 013u, 015u, 017u}) {
            const auto cand = from_octal(g);
            const auto CE_cand = expurgated(cand);
            if (CE_cand.get_dmin() > dmin_best) {
                dmin_best = CE_cand.get_dmin();
                p = cand;
            }
        }
        auto CELF = LinearCode<F2>(L - p.degree(), p);
        std::cout << "ELF code:               " << showbasic << CELF << ", dmin = " << CELF.get_dmin()
                  << ", gamma = " << CELF.get_gamma() << std::endl;
        auto CE = expurgated(p);
        std::cout << "Expurgated code CE:     " << showbasic << CE << ", dmin = " << CE.get_dmin() << std::endl;

        // TX
        auto u = Vector<F2>(CE.get_k()).randomize();
        std::cout << "Random message:                   " << u << std::endl;
        auto c = CE.enc(u);

        const double pe = 0.05;
        BSC channel(pe);

        // RX: serial list Viterbi decoding on the convolutional code, first ELF-passing candidate wins
        auto r = channel(c);
        bool success = false;
        for (const auto& c_est : CE.dec_Viterbi_list(r, 6)) {
            std::cout << "Viterbi list candidate:  " << CC.encinv(c_est) << std::endl;
            if ((Polynomial<F2>(CE.encinv(c_est)) % CELF.get_gamma()).is_zero()) {
                std::cout << "ELF-aided list message estimate:  " << CE.encinv(c_est) << std::endl;
                success = true;
                break;
            }
        }
        if (!success) std::cout << "ELF-aided list decoding failure!" << std::endl;
    }

    return 0;
}
