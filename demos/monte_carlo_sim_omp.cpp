#include <fstream>
#include <iostream>

// For clang/MacOS: add -lomp -Xpreprocessor -fopenmp
#include "omp.h"

// Use CECCO library, compile with -std=c++20 -I../
// Adapt -Ipath_to_hpp_files if necessary, append -O3 for performance
#include "cecco.hpp"

int main(void) {
    using namespace CECCO;
#define F2 Fp<2>        // define macro F2 (for convenience)
#define MC_TARGET 1000  // Monte Carlo simulation target
#define NUM_THREADS 10  // choose appropriately

    /* line code */
    BPSKEncoder bpsk;
    BPSKDecoder demap;

    /* error control code */
    auto C = extend(GolayCode<F2>());
    std::cout << showall << C << std::endl;
    Enc enc(C);
    Dec_ML dec(C);
    Encinv encinv(C);

    const int n = C.get_n();
    const int k = C.get_k();

    std::ofstream file;
    file.open(details::basename(__FILE__) + std::string(".txt"), std::fstream::out);

    for (double EbN0dB = -3.0; EbN0dB <= 13.0; EbN0dB += 0.5) {
        const double EsN0dB = EbN0dB - (10 * log10(n) - 10 * log10(k));

        long transmissions = 0;
        long bit_errors = 0;
        long block_errors = 0;

        const long target_per_thread = (MC_TARGET + NUM_THREADS - 1) / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS) reduction(+ : transmissions, bit_errors, block_errors)
        {
            AWGN awgn(EsN0dB, bpsk.get_a(), bpsk.get_b());

            do {
                /* count number of transmitted blocks/codewords */
                ++transmissions;

                /* random message */
                Vector<F2> u(k);
                u.randomize();

                /* buffers */
                Vector<F2> u_hat;
                Vector<F2> c;
                Vector<F2> c_hat;

                /* simulation chain */
                u >> enc >> c >> bpsk >> awgn >> demap >> dec >> c_hat >> encinv >> u_hat;

                /* count errors */
                bit_errors += dH(u, u_hat);
                if (c != c_hat) ++block_errors;

            } while (bit_errors < target_per_thread);
        }

        const double pe = bit_errors / (double)(transmissions * k);
        const double Pword = block_errors / (double) transmissions;
        std::cout << EbN0dB << " " << pe << " " << Pword << std::endl;
        file << EbN0dB << " " << pe << " " << Pword << std::endl;
    }

    return 0;
}
