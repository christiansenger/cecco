/**
 * @file blocks.hpp
 * @brief Communication system blocks library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.2
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026, Christian Senger <senger@inue.uni-stuttgart.de>
 *
 * Licensed for noncommercial use only, including academic teaching, research, and personal non-profit purposes.
 * Commercial use is prohibited without a separate commercial license. See the [LICENSE](../../LICENSE) file in the
 * repository root for full terms and how to request a commercial license.
 *
 * @section Description
 *
 * This header file provides communication system blocks for error control coding experiments/simulations. It supports:
 *
 * - **Channel models**: Symmetric Discrete Memoryless Erasure Channel (SDMEC), Symmetric Discrete Memoryless Channel
 * (SDMC), Binary Symmetric Channel (BSC), Binary Erasure Channel (BEC), and Additive White Gaussian Noise (AWGN)
 * channel with accurate error probability calculations
 * - **Modulation schemes**: Non-Return-to-Zero (NRZ) and Binary Phase Shift Keying (BPSK)
 *   with configurable constellation parameters
 * - **Demodulation**: Hard-decision and soft-decision demodulation with Log-Likelihood Ratio (LLR) computation
 * - **Field multiplexing**: DEMUX/MUX for expansion between extension fields and their subfields
 * - **Universal chaining**: Operator>> elegant communication chain construction
 * - **Performance optimized**: CRTP-based design eliminates virtual dispatch overhead
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * // Basic communication chain simulation
 * using F2 = Fp<2>;
 * Vector<F2> message = {1, 0, 1, 1};
 *
 * // Create communication chain components
 * BPSKEncoder enc;                           // BPSK modulation (a=0, b=2)
 * AWGN awgn(6.0, enc.get_a(), enc.get_b());  // 6 dB SNR
 * BPSKDecoder dec;                           // Hard-decision demodulation
 *
 * // Process through communication chain using operator>>
 * Vector<std::complex<double>> y;
 * Vector<F2> r;
 * message >> enc >> awgn >> y >> dec >> r;
 *
 * // Soft-decision processing with LLR
 * LLRCalculator calc(enc, awgn);
 * Vector<double> llrs;
 * y >> calc >> llrs;
 *
 * // Field tower multiplexing
 * using F4 = Ext<F2, {1, 1, 1}>;
 * Vector<F4> v = {F4(0), F4(1), F4(2), F4(3)};
 * Matrix<F2> M;
 * v >> DEMUX<F4, F2>() >> M;  // Expand to F2 matrix...
 * Vector<F4> w;
 * M >> MUX<F2, F4>() >> w;    // ... and compress back to F4 vector
 * @endcode
 *
 * @note All blocks are designed for finite field elements, complex numbers, and floating-point types.
 *       Channel models provide both element-wise and vector/matrix processing capabilities.
 *
 * @see Finite field concepts in @ref CECCO::FiniteFieldType for field element processing
 * @see Vector/Matrix classes in @ref CECCO::Vector, @ref CECCO::Matrix for container operations
 * @see Field relationships in @ref CECCO::SubfieldOf, @ref CECCO::ExtensionOf for DEMUX/MUX usage
 */

#ifndef BLOCKS_HPP
#define BLOCKS_HPP

// #include <cmath> // transitive through fields.hpp
// #include <complex> // transitive through fields.hpp
// #include <random> // transitive through fields.hpp
#include <numbers>

#include "fields.hpp"
#include "vectors.hpp"
// #include "field_concepts_traits.hpp" // transitive through vectors.hpp
// #include "matrices.hpp" // transitive through vectors.hpp

namespace CECCO {

namespace details {

/**
 * @brief CRTP base template for vector/matrix operator() implementations
 * @tparam T The derived class (CRTP pattern)
 * @tparam InputType The input element type for the vector/matrix operator() implementations
 * @tparam OutputType The output element type for the vector/matrix operator() implementations
 *
 * Provides automatic Vector and Matrix operator() overloads based on the element-wise
 * operator() implementation in the derived class.
 */
template <typename T, typename InputType, typename OutputType>
class BlockProcessor : private NonCopyable {
   protected:
    const T& derived() const noexcept { return static_cast<const T&>(*this); }

    T& derived() noexcept { return static_cast<T&>(*this); }

   public:
    /**
     * @brief Process vector of elements (const reference)
     * @param in Input vector to process
     * @return New vector with each element processed through derived class operator()
     */
    Vector<OutputType> operator()(const Vector<InputType>& in) noexcept {
        Vector<OutputType> res(in.get_n());
        for (size_t i = 0; i < in.get_n(); ++i) res.set_component(i, derived()(in[i]));
        return res;
    }

    /**
     * @brief Process vector of elements (rvalue reference)
     * @param in Input vector to process (moved)
     * @return New vector with each element processed
     */
    Vector<OutputType> operator()(Vector<InputType>&& in) noexcept {
        const size_t n = in.get_n();  // Get size before processing
        Vector<OutputType> res(n);
        for (size_t i = 0; i < n; ++i) res.set_component(i, derived()(std::move(in[i])));
        return res;
    }

    /**
     * @brief Process matrix of elements (const reference)
     * @param in Input matrix to process
     * @return New matrix with each element processed through derived class operator()
     */
    Matrix<OutputType> operator()(const Matrix<InputType>& in) noexcept {
        Matrix<OutputType> res(in.get_m(), in.get_n());
        for (size_t i = 0; i < res.get_m(); ++i) {
            for (size_t j = 0; j < res.get_n(); ++j) res.set_component(i, j, derived()(in(i, j)));
        }
        return res;
    }

    /**
     * @brief Process matrix of elements (rvalue reference)
     * @param in Input matrix to process (moved)
     * @return Processed matrix (reuses input storage when possible)
     */
    Matrix<OutputType> operator()(Matrix<InputType>&& in) noexcept {
        if constexpr (std::is_same_v<InputType, OutputType>) {
            // Same type: process in-place and reuse storage
            for (size_t i = 0; i < in.get_m(); ++i) {
                for (size_t j = 0; j < in.get_n(); ++j) in.set_component(i, j, derived()(std::move(in(i, j))));
            }
            return std::move(in);
        } else {
            // Different types: create new matrix
            Matrix<OutputType> res(in.get_m(), in.get_n());
            for (size_t i = 0; i < res.get_m(); ++i) {
                for (size_t j = 0; j < res.get_n(); ++j) res.set_component(i, j, derived()(std::move(in(i, j))));
            }
            return res;
        }
    }
};

// Type aliases for common usage patterns
template <typename T, typename ElementType>
using SameTypeProcessor = BlockProcessor<T, ElementType, ElementType>;

template <typename T>
using EncoderProcessor = BlockProcessor<T, Fp<2>, std::complex<double>>;

template <typename T>
using DecoderProcessor = BlockProcessor<T, std::complex<double>, Fp<2>>;

template <typename T>
using LLRProcessor = BlockProcessor<T, std::complex<double>, double>;

}  // namespace details

/**
 * @brief Symmetric Discrete Memoryless Erasure Channel (SDMEC)
 * @tparam T Finite field type for channel input/output symbols
 *
 * Simulates a symmetric discrete memoryless channel that introduces random errors and erasures
 * in finite field symbols with specified probabilities. The channel uses geometric distributions
 * to efficiently model the Bernoulli error and erasure processes.
 *
 * **Channel Model**:
 *
 * The SDMEC implements an errors-and-erasures channel where each transmitted symbol undergoes
 * independent error and erasure checks. The error probability is adjusted (scaled by 1/(1-px))
 * to achieve the desired overall error rate despite potential erasures.
 *
 * **Processing Steps**:
 * 1. **Error check**: With probability pe/(1-px), the symbol is changed to a random incorrect value
 * 2. **Erasure check**: With probability px, the symbol is marked as erased (overwrites any prior error)
 * 3. **Result**: The final symbol state is returned to the receiver
 *
 * **Observed Outcomes** (from receiver's perspective):
 * - **Overall error probability**: Pr(observe error) = pe/(1-px) × (1-px) = pe
 * - **Overall erasure probability**: Pr(observe erasure) = px
 * - **Success probability**: Pr(observe correct) = (1 - pe/(1-px)) × (1-px) = 1 - pe - px
 * - **Memoryless property**: Each symbol is processed independently with i.i.d. error and erasure events
 *
 * **Implementation Details**:
 *
 * The channel uses two independent geometric distributions to model Bernoulli processes:
 * - Error distribution with parameter pe/(1-px) (scaled to account for subsequent erasures)
 * - Erasure distribution with parameter px
 *
 * **Usage Example**:
 * @code{.cpp}
 * #define CECCO_ERASURE_SUPPORT
 * using F4 = Ext<Fp<2>, {1, 1, 1}>;
 * SDMEC<F4> channel(0.05, 0.1);        // 5% errors, 10% erasures
 * auto c = Vector<F4>(20).randomize();
 * Vector<F4> r;
 * c >> channel >> r;
 * @endcode
 *
 * @note For binary fields (F₂), this implements the Binary Symmetric Channel (BSC) when px=0,
 *       or the Binary Erasure Channel (BEC) when pe=0.
 * @note Erasure functionality requires compilation with CECCO_ERASURE_SUPPORT defined.
 *       Without this macro, px must be 0.0 (errors-only mode).
 * @note For larger fields (F_q with q>2), this implements the q-ary Symmetric Channel when px=0.
 * @note The error probability is internally scaled to pe/(1-px) to compensate for symbols that
 *       are subsequently erased, ensuring the observed error rate matches the specified pe parameter.
 *
 * @see @ref CECCO::SDMC for errors-only convenience wrapper
 * @see @ref CECCO::BSC for binary symmetric channel
 * @see @ref CECCO::BEC for binary erasure channel
 */
template <FiniteFieldType T>
class SDMEC : public details::SameTypeProcessor<SDMEC<T>, T> {
   public:
    // Bring base class operator() overloads into scope
    using details::SameTypeProcessor<SDMEC<T>, T>::operator();

    /**
     * @brief Construct SDMEC with specified error and erasure probabilities
     * @param pe Symbol error probability - probability of observing an error at receiver (must be in [0, 1])
     * @param px Symbol erasure probability - probability of observing an erasure at receiver (must be in [0, 1-pe],
     * default: 0.0)
     * @throws std::out_of_range if pe is outside [0, 1]
     * @throws std::out_of_range if pe is too small for numerical precision (0 < pe < 1e-9)
     * @throws std::out_of_range if px is outside [0, 1-pe] when CECCO_ERASURE_SUPPORT is defined
     * @throws std::out_of_range if px is too small for numerical precision (0 < px < 1e-9)
     * @throws std::invalid_argument if px != 0 when CECCO_ERASURE_SUPPORT is not defined
     *
     * Creates a symmetric discrete memoryless channel that produces errors with observed probability pe
     * and erasures with observed probability px.
     *
     * **Constraint**: pe + px ≤ 1.0 (the remaining probability 1-pe-px represents correctly received symbols)
     *
     * **Special Cases**:
     * - pe = 0, px = 0: Perfect channel (no errors or erasures)
     * - pe > 0, px = 0: Error-only channel (standard SDMC/BSC)
     * - pe = 0, px > 0: Erasure-only channel (requires CECCO_ERASURE_SUPPORT)
     * - pe > 0, px > 0: Errors-and-erasures channel (requires CECCO_ERASURE_SUPPORT)
     *
     * @note Edge case probabilities pe=0.0 and px=0.0 are valid. The channel uses early
     *       return optimization when both probabilities are zero.
     * @note When CECCO_ERASURE_SUPPORT is not defined, px must be 0.0. Any non-zero px
     *       will throw std::invalid_argument.
     */
    SDMEC(double pe, double px = 0.0);

    /**
     * @brief Process single symbol through SDMEC
     * @param in Input symbol
     * @return Output symbol (possibly erased when CECCO_ERASURE_SUPPORT is defined)
     *
     * Applies the channel error and erasure model to a single symbol through two independent checks:
     *
     * @note This method uses noexcept specification and relies on geometric distributions
     *       for efficient Bernoulli process simulation via independent trial counting.
     */
    T operator()(const T& in) noexcept;

    /**
     * @brief Get symbol error probability
     * @return Error probability pe
     */
    double get_pe() const noexcept { return error_dist.p(); }

    /**
     * @brief Get symbol erasure probability
     * @return Erasure probability px
     */
    double get_px() const noexcept { return erasure_dist.p(); }

    /**
     * @brief Calculate channel capacity in bits per symbol
     * @return Channel capacity C in bits per symbol
     *
     * Computes the Shannon capacity of the SDMEC channel using the formula for q-ary
     * symmetric channels with optional erasures. The capacity represents the maximum
     * achievable rate of reliable communication through this channel.
     *
     * **Mathematical Formula**:
     *
     * For errors-only channel (px = 0):
     * - C = log₂(q) + (1-pe)·log₂(1-pe) + pe·log₂(pe) - pe·log₂(q-1)
     *
     * For errors-and-erasures channel:
     * - C = (1-px) · C_error
     * - where C_error uses the conditional error probability pe/(1-px)
     *
     * For erasures-only channel:
     * - C = (1-px)·log₂(q)
     *
     * **Implementation Notes**:
     * - Uses internal error probability pe/(1-px) from error_dist.p()
     * - Handles edge cases pe=0 and pe=1 using limit: lim_{x→0} x·log₂(x) = 0
     * - For CECCO_ERASURE_SUPPORT, scales result by (1-px) to account for erasures
     * - Returns capacity in bits per channel use (symbol)
     *
     * **Usage Example**:
     * @code{.cpp}
     * SDMEC<F4> sdmec(0.05, 0.1);  // 5% errors, 10% erasures over F₄
     * double capacity = sdmec.get_capacity();
     * std::cout << "Channel capacity: " << capacity << " bits/symbol" << std::endl;
     * @endcode
     *
     * @note Returned value is in bits (log base 2).
     */
    double get_capacity() const noexcept;

   private:
    std::geometric_distribution<unsigned int> error_dist;
    unsigned int error_trials{0};
    unsigned int error_failures_before_hit;
    std::geometric_distribution<unsigned int> erasure_dist;
    unsigned int erasure_trials{0};
    unsigned int erasure_failures_before_hit;
};

template <FiniteFieldType T>
SDMEC<T>::SDMEC(double pe, double px) : error_dist(pe / (1 - px)), erasure_dist(px) {
#ifndef CECCO_ERASURE_SUPPORT
    if (px != 0) throw std::invalid_argument("px!=0 requires CECCO_ERASURE_SUPPORT");
#endif
    if (pe < 0.0 || pe > 1.0)
        throw std::out_of_range("SDMEC error probability must be in [0,1], got: " + std::to_string(pe));
    if (pe != 0.0 && pe < 0.000000001) throw std::out_of_range("pe too small");
    error_failures_before_hit = error_dist(gen());

    if (px < 0.0 || px > 1.0 - pe)
        throw std::out_of_range("SDMEC erasure probability must be in [0," + std::to_string(1 - pe) +
                                "], got: " + std::to_string(px));
    if (px != 0.0 && px < 0.000000001) throw std::out_of_range("px too small");
    erasure_failures_before_hit = erasure_dist(gen());
}

template <FiniteFieldType T>
T SDMEC<T>::operator()(const T& in) noexcept {
    if (error_dist.p() == 0.0 && erasure_dist.p() == 0.0) return in;
    T res(in);
    if (error_trials == error_failures_before_hit) {
        res.randomize_force_change();
        error_trials = 0;
        error_failures_before_hit = error_dist(gen());
    } else {
        ++error_trials;
    }
#ifdef CECCO_ERASURE_SUPPORT
    if (erasure_dist.p() == 0.0) return res;
    if (erasure_trials == erasure_failures_before_hit) {
        res.erase();
        erasure_trials = 0;
        erasure_failures_before_hit = erasure_dist(gen());
    } else {
        ++erasure_trials;
    }
#endif
    return res;
}

template <FiniteFieldType T>
double SDMEC<T>::get_capacity() const noexcept {
    const double pe = error_dist.p();
    const double q = static_cast<double>(T::get_size());

    const double term1 = (pe > 0.0 && pe < 1.0) ? pe * std::log2(pe) : 0.0;
    const double term2 = (pe > 0.0 && pe < 1.0) ? (1 - pe) * std::log2(1 - pe) : 0.0;

    double res = std::log2(q) + term2 + term1 - pe * std::log2(q - 1);

    const double px = erasure_dist.p();
    res *= (1 - px);

    return res;
}

/**
 * @brief Symmetric Discrete Memoryless Channel (SDMC)
 * @tparam T Finite field type for channel input/output symbols
 *
 * Specialized SDMEC for error-only channels (no erasures). Provides convenient
 * single-parameter constructor for the traditional symmetric channel model.
 *
 * **Usage Example**:
 * @code{.cpp}
 * using F4 = Ext<Fp<2>, {1, 1, 1}>;
 * SDMC<F4> sdmc(0.05);  // 5% symbol errors, 0% erasures
 * auto c = Vector<F4>(20).randomize();
 * Vector<F4> r;
 * c >> sdmc >> r;
 * @endcode
 *
 * @note Equivalent to SDMEC<T>(pe, 0.0)
 * @see @ref CECCO::SDMEC for the full errors-and-erasures implementation
 * @see @ref CECCO::BSC for binary symmetric channel
 */
template <FiniteFieldType T>
class SDMC : public SDMEC<T> {
   public:
    /**
     * @brief Construct SDMC with specified error probability
     * @param pe Symbol error probability (must be in [0, 1])
     */
    SDMC(double pe) : SDMEC<T>(pe, 0.0) {}
};

/**
 * @brief Binary Symmetric Channel (BSC)
 *
 * Binary SDMC with convenient single-parameter constructor. Implements the classical
 * binary symmetric channel model where bits are flipped with probability pe.
 *
 * **Usage Example**:
 * @code{.cpp}
 * BSC bsc(0.1);                         // 10% bit error probability
 * Vector<Fp<2>> c = {1, 0, 1, 0};
 * Vector<Fp<2>> r;
 * c >> bsc >> r;
 * @endcode
 *
 * @note Equivalent to SDMC<Fp<2>>(pe) or SDMEC<Fp<2>>(pe, 0.0)
 * @see @ref CECCO::SDMEC for the full errors-and-erasures implementation
 * @see @ref CECCO::SDMC for the general errors-only channel
 * @see @ref CECCO::BEC for binary erasure channel
 */
using BSC = SDMC<Fp<2>>;

/**
 * @brief Binary Erasure Channel (BEC)
 *
 * Specialized SDMEC for binary erasure channels where symbols are either received correctly
 * or marked as erased (no errors). Provides convenient single-parameter constructor that
 * takes only the erasure probability.
 *
 * **Usage Example**:
 * @code{.cpp}
 * BEC bec(0.1);                         // 10% erasure probability
 * Vector<Fp<2>> c = {1, 0, 1, 1};
 * Vector<Fp<2>> r;
 * c >> bec >> r;
 * @endcode
 *
 * @note Equivalent to SDMEC<Fp<2>>(0.0, px)
 * @note Requires CECCO_ERASURE_SUPPORT to be defined at compile time
 * @see @ref CECCO::SDMEC for the full errors-and-erasures implementation
 * @see @ref CECCO::BSC for binary symmetric channel (errors only)
 */
class BEC : public SDMEC<Fp<2>> {
   public:
    /**
     * @brief Construct BEC with specified erasure probability
     * @param px Symbol erasure probability (must be in [0, 1])
     */
    BEC(double px) : SDMEC<Fp<2>>(0.0, px) {}
};

/**
 * @brief Binary Asymmetric Channel (BAC) - Z-Channel
 *
 * Implements the z-channel model where input bit 0 is never corrupted, but input bit 1
 * can be flipped to 0 with probability p.
 *
 * **Usage Example**:
 * @code{.cpp}
 * BAC bac(0.1);                   // 10% probability that 1 flips to 0
 * Vector<Fp<2>> c = {0, 1, 1, 0};
 * Vector<Fp<2>> r;
 * c >> bac >> r;
 *
 * // Check capacity
 * double capacity = bac.get_capacity();
 * std::cout << "Z-channel capacity: " << capacity << " bits/symbol\n";
 * @endcode
 *
 * @see @ref CECCO::BSC for symmetric binary channel
 * @see @ref CECCO::BEC for binary erasure channel
 */
class BAC : public details::SameTypeProcessor<BAC, Fp<2>> {
   public:
    using details::SameTypeProcessor<BAC, Fp<2>>::operator();

    /**
     * @brief Construct BAC with specified flip probability
     * @param p Probability that 1 flips to 0 (must be in [0, 1])
     * @throws std::out_of_range if p is outside [0, 1]
     * @throws std::out_of_range if p is too small for numerical precision (0 < p < 1e-9)
     *
     * Creates a binary asymmetric channel where input 1 is flipped to 0 with probability p,
     * while input 0 is always preserved.
     *
     * **Special Cases**:
     * - p = 0: Perfect channel (no errors, capacity = 1 bit)
     * - p = 1: All 1s become 0s (only 0 can be reliably transmitted, capacity = 0)
     */
    BAC(double p) : bsc(p) {}

    /**
     * @brief Process single bit through BAC
     * @param in Input bit
     * @return Output bit
     *
     * Applies the Z-channel model: input 0 is always preserved, input 1 is
     * passed through internal BSC which flips to 0 with probability p.
     */
    Fp<2> operator()(const Fp<2>& in) noexcept {
        if (in == Fp<2>(0)) return in;
        return bsc(in);
    }

    /**
     * @brief Get flip probability
     * @return Flip probability p (probability that 1 → 0)
     */
    double get_pe() const noexcept { return bsc.get_pe(); }

    /**
     * @brief Calculate channel capacity in bits per symbol
     * @return Channel capacity C in bits per symbol
     *
     * Computes the Shannon capacity of the z-channel (binary asymmetric channel).
     * The capacity represents the maximum achievable rate of reliable communication
     * through this asymmetric channel.
     *
     * **Mathematical Formula**:
     * - C = log₂(1 + (1-p)·p^(p/(1-p)))
     *
     * **Implementation Notes**:
     * - Handles edge cases p=0 and p=1 using explicit checks
     * - Uses internal flip probability from bsc.get_pe()
     * - Returns capacity in bits per channel use (symbol)
     *
     * **Usage Example**:
     * @code{.cpp}
     * BAC bac(0.1);  // 10% flip probability (1 → 0)
     * double capacity = bac.get_capacity();
     * std::cout << "Z-channel capacity: " << capacity << " bits/symbol" << std::endl;
     * @endcode
     *
     * @note Returned value is in bits (log base 2).
     */
    double get_capacity() const noexcept {
        const double pe = bsc.get_pe();

        if (pe == 0.0) return 1.0;  // Perfect channel
        if (pe == 1.0) return 0.0;  // Useless channel

        return std::log2(1 + (1 - pe) * pow(pe, pe / (1 - pe)));
    }

   private:
    BSC bsc;
};

/**
 * @brief Non-Return-to-Zero (NRZ) encoder for binary modulation
 *
 * Implements NRZ line coding that maps binary symbols to complex constellation points.
 * The encoder supports configurable constellation parameters.
 *
 * **Mathematical Model**:
 * - **Constellation mapping**: 0 → (a - b/2, 0), 1 → (a + b/2, 0)
 * - **Energy per bit**: Eb = a² + b²/4 (assuming unit symbol duration)
 * - **Constellation distance**: b (minimum euclidean distance between symbols)
 * - **DC offset**: a (shifts entire constellation along real axis)
 *
 * **Common Configurations**:
 * - **BPSK**: a = 0, b = 2 → constellation {-1, +1} with Eb = 1
 * - **OOK**: a = 1, b = 2 → constellation {0, +2} with Eb = 2
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // BPSK modulation (optimal for AWGN)
 * NRZEncoder bpsk(0.0, 2.0);
 * Vector<std::complex<double>> signal;
 * Fp<2>(1) >> bpsk >> signal;          // Result: ( (1.0, 0.0) )
 * double energy = bpsk.get_Eb();       // Result: 1.0
 *
 * // On-Off Keying (OOK)
 * NRZEncoder ook(1.0, 2.0);
 * Vector<Fp<2>> bits = {0, 1, 0, 1};
 * Vector<std::complex<double>> signal;
 * bits >> ook >> signal;  // ( (0, 0), (2, 0), (0, 0), (2, 0) )
 * @endcode
 *
 * @note This encoder assumes symbol duration Δ = 1 for energy calculations.
 *       For different symbol durations, scale energy results accordingly.
 *
 * @see @ref CECCO::BPSKEncoder for convenient BPSK configuration
 * @see @ref CECCO::AWGN for compatible channel model with accurate Pe calculations
 */
class NRZEncoder : public details::EncoderProcessor<NRZEncoder> {
   public:
    // Bring base class operator() overloads into scope
    using details::EncoderProcessor<NRZEncoder>::operator();

    /**
     * @brief Construct NRZ encoder with constellation parameters
     * @param a DC offset parameter (real part offset)
     * @param b Constellation distance parameter (separation between symbols)
     *
     * Creates NRZ encoder with constellation points at (a ± b/2, 0).
     * Common values: BPSK (0, 2), OOK (1, 2).
     */
    constexpr NRZEncoder(double a, double b) noexcept : a(a), b(b) {}

    /** @name Constellation Parameters
     * @{
     */

    /**
     * @brief Get energy per bit (Eb)
     * @return Energy per bit assuming unit symbol duration
     *
     * Calculates Eb = a² + b²/4 based on constellation geometry.
     */
    constexpr double get_Eb() const noexcept { return (a * a) + (b * b) / 4.0; }

    /**
     * @brief Get DC offset parameter
     * @return DC offset value (a parameter)
     */
    constexpr double get_a() const noexcept { return a; }

    /**
     * @brief Get distance parameter
     * @return Distance parameter (b parameter)
     */
    constexpr double get_b() const noexcept { return b; }

    /** @} */

    /**
     * @brief Encode binary symbol to complex constellation point
     * @param in Binary input symbol (0 or 1)
     * @return Complex constellation point corresponding to input symbol
     *
     * Maps input according to: 0 → (a - b/2, 0), 1 → (a + b/2, 0).
     */
    constexpr std::complex<double> operator()(const Fp<2>& in) noexcept {
        if (in == Fp<2>(0))
            return {a - b / 2.0, 0};
        else
            return {a + b / 2.0, 0};
    }

   private:
    const double a{};
    const double b{};
};

/**
 * @brief Binary Phase Shift Keying (BPSK) encoder
 *
 * Specialized NRZ encoder configured for  BPSK modulation with constellation
 * points at {-1, +1}. This configuration provides optimal performance in AWGN channels
 * by maximizing the Euclidean distance while maintaining unit energy per bit.
 *
 * **Mathematical Properties**:
 * - **Constellation**: {-1, +1} (real-valued antipodal signaling)
 * - **Energy per bit**: Eb = 1 (normalized)
 * - **Minimum distance**: d = 2 (optimal for binary signaling)
 * - **DC component**: 0 (balanced signaling)
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // Standard BPSK encoding
 * BPSKEncoder enc;
 * Vector<Fp<2>> message = {0, 1, 1, 0};
 * Vector<std::complex<double>> signal;
 * message >> enc >> signal;              // Result: ( (-1, 0), (1, 0), (1, 0), (-1, 0) )
 *
 * double energy = enc.get_Eb();                        // Result: 1.0
 * double distance = enc.get_b();  // Result: 2.0
 *
 * // Use with AWGN channel for theoretical Pe calculations
 * AWGN awgn(6.0, enc.get_a(), enc.get_b());  // 6 dB SNR
 * double theoretical_pe = awgn.get_pe();
 * @endcode
 *
 * @see @ref CECCO::NRZEncoder for general constellation configuration
 * @see @ref CECCO::BPSKDecoder for corresponding demodulator
 */
class BPSKEncoder : public NRZEncoder {
   public:
    /**
     * @brief Construct BPSK encoder with optimal parameters
     *
     * Creates BPSK encoder with a=0, b=2 for constellation {-1, +1}.
     */
    constexpr BPSKEncoder() noexcept : NRZEncoder(0.0, 2.0) {}
};

/**
 * @brief Additive White Gaussian Noise (AWGN) channel simulation
 *
 * Implements a continuous-valued AWGN channel that adds independent Gaussian noise
 * to both real and imaginary components of complex-valued symbols. The channel
 * provides mathematically accurate modeling of thermal noise in communication systems
 * with precise error probability calculations for various modulation schemes.
 *
 * **Mathematical Model**:
 * - **Noise model**: N ~ 𝒩(0, σ²) independently for real and imaginary parts
 * - **Noise variance**: σ² = N₀/2 where N₀ is the two-sided noise spectral density
 * - **Signal-to-noise ratio**: Eb/N₀ (energy per bit to noise spectral density ratio)
 * - **Error probability**: Pe = ½ erfc(√(d²·Eb/N₀)/(4Eb)) for NRZ/BPSK modulation
 *
 * **Channel Parameters**:
 * - **Eb**: Energy per bit from the modulation scheme
 * - **d**: Constellation distance (minimum Euclidean distance between symbols)
 * - **Eb/N₀**: Signal-to-noise ratio in dB, determines noise power
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // BPSK over AWGN at 6 dB SNR
 * BPSKEncoder enc;  // Eb = 1, d = 2
 * AWGN awgn(6.0, enc.get_a(), enc.get_b());
 * Vector<Fp<2>> message = {0, 1, 1, 0};
 * Vector<std::complex<double>> y;
 * message >> enc >> awgn >> y;
 *
 * // Channel noise parameters
 * double sigma = awgn.get_standard_deviation();  // Noise std deviation
 * double variance = awgn.get_variance();         // Noise variance
 * @endcode
 *
 * **Error Probability Calculation**:
 * For binary modulation schemes (NRZ/BPSK), the bit error probability is:
 * Pe = ½ erfc(√(SNR_constellation)) where SNR_constellation = d²·(Eb/N₀)/(4·Eb)
 *
 * @note The channel assumes complex-valued input symbols and adds independent
 *       Gaussian noise to both real and imaginary components.
 *
 * @see @ref CECCO::NRZEncoder, @ref CECCO::BPSKEncoder for compatible modulation schemes
 * @see @ref CECCO::LLRCalculator for soft-decision demodulation using this channel
 */
class AWGN : public details::SameTypeProcessor<AWGN, std::complex<double>> {
   public:
    // Bring base class operator() overloads into scope
    using details::SameTypeProcessor<AWGN, std::complex<double>>::operator();

    /**
     * @brief Construct AWGN channel with specified parameters
     * @param EbNodB Signal-to-noise ratio (Eb/N₀) in decibels
     * @param a DC offset parameter (real part offset)
     * @param b Constellation distance parameter (separation between symbols)
     *
     * Creates an AWGN channel that adds Gaussian noise with variance σ² = Eb/(2·10^(EbNodB/10)).
     * The error probability is calculated based on the DC offset and constellation distance.
     */
    AWGN(double EbNodB, double a, double b)
        : Eb(NRZEncoder(a, b).get_Eb()),
          dist(0, std::sqrt(0.5 * Eb / pow(10.0, EbNodB / 10.0))),
          pe(calculate_pe(EbNodB, b)) {}

    /** @name Noise Parameters
     * @{
     */

    /**
     * @brief Get noise variance
     * @return Variance of the Gaussian noise (σ²)
     *
     * Returns the variance of the additive noise for both real and imaginary components.
     */
    double get_variance() const noexcept { return pow(dist.stddev(), 2.0); }

    /**
     * @brief Get noise standard deviation
     * @return Standard deviation of the Gaussian noise (σ)
     *
     * Returns the standard deviation of the additive noise for both components.
     */
    double get_standard_deviation() const noexcept { return dist.stddev(); }

    /**
     * @brief Get theoretical bit error probability
     * @return Theoretical Pe for the configured modulation and SNR
     *
     * Returns the mathematically calculated bit error probability after hard decisions based on
     * the constellation parameters and signal-to-noise ratio.
     */
    constexpr double get_pe() const noexcept { return pe; }

    /** @} */

    /**
     * @brief Add AWGN to complex symbol
     * @param in Input complex symbol
     * @return Noisy output symbol with independent Gaussian noise added to both components
     *
     * Adds independent Gaussian noise samples to both the real and imaginary parts
     * of the input symbol, simulating thermal noise in a communication channel.
     */
    std::complex<double> operator()(const std::complex<double>& in) noexcept {
        std::complex<double> res(in.real() + dist(gen()), in.imag() + dist(gen()));
        return res;
    }

    /**
     * @brief Calculate channel capacity in bits per symbol
     * @return Channel capacity C in bits per symbol
     *
     * Computes the Shannon capacity of the real-valued AWGN channel with binary input:
     * C = ½·log₂(1 + Eb/σ²)
     *
     * This represents the maximum achievable rate of reliable communication through
     * this continuous-valued real AWGN channel with binary signaling (NRZ/BPSK).
     *
     * **Implementation Notes**:
     * - Formula applies to real-valued signaling (one dimension)
     * - For complex AWGN, capacity would be C = log₂(1 + Eb/(2σ²)) due to noise in both I and Q
     *
     * @note Returned value is in bits per channel use (symbol).
     */
    double get_capacity() const noexcept { return 1 / 2.0 * std::log2(1 + Eb / get_variance()); }

   private:
    const double Eb{};
    std::normal_distribution<double> dist;
    const double pe{};

    /**
     * @brief Calculate theoretical bit error probability
     * @param EbNodB Signal-to-noise ratio in dB
     * @param b Constellation distance parameter (separation between symbols)
     * @return Theoretical Pe for NRZ/BPSK modulation
     */
    double calculate_pe(double EbNodB, double b) const noexcept;
};

double AWGN::calculate_pe(double EbNodB, double b) const noexcept {
    // For NRZ/BPSK: Pe = 0.5 * erfc(b/(2*sigma))
    // where b is constellation distance and sigma is noise std dev
    const double EbN0_linear = pow(10.0, EbNodB / 10.0);

    // Signal-to-noise ratio in terms of constellation distance
    // SNR = (b/2)² / σ² = (b/2)² / (No/2) = (b/2)² * 2/No = (b²/2) / No
    // where No = Eb/EbN0_linear
    const double constellation_snr = (b * b * EbN0_linear) / (4.0 * Eb);

    return 0.5 * std::erfc(std::sqrt(constellation_snr));
}

/**
 * @brief Binary Input - Additive White Gaussian Noise (BI-AWGN) channel
 *
 * Combines NRZ modulation and AWGN transmission into a single block that maps binary inputs
 * to noisy complex-valued channel outputs. Compatible with NRZDecoder for hard decisions
 * or LLRCalculator for soft decisions.
 *
 * **Usage Example**:
 * @code{.cpp}
 * // BPSK-AWGN at 6 dB (default parameters)
 * BI_AWGN channel(6.0); // BPSK: a=0, b=2
 * Vector<Fp<2>> c = {1, 0, 1, 0};
 * Vector<std::complex<double>> y;
 * Vector<Fp<2>> r;
 * c >> channel >> y >> NRZDecoder(channel.get_encoder()) >> r;
 *
 * // Custom NRZ constellation
 * BI_AWGN ook(8.0, 1.0, 2.0); // OOK: a=1, b=2
 * @endcode
 *
 * @note Default parameters implement BPSK (a=0, b=2)
 * @see @ref CECCO::NRZEncoder for constellation parameters
 * @see @ref CECCO::AWGN for transmission/noise model
 * @see @ref CECCO::NRZDecoder for hard-decision
 * @see @ref CECCO::LLRCalculator for soft-decision
 */
class BI_AWGN : public details::BlockProcessor<BI_AWGN, Fp<2>, std::complex<double>> {
   public:
    using details::BlockProcessor<BI_AWGN, Fp<2>, std::complex<double>>::operator();

    /**
     * @brief Construct BI-AWGN channel with SNR and optional NRZ parameters
     * @param EbN0dB Signal-to-noise ratio (Eb/N0) in decibels
     * @param a DC offset parameter for NRZ encoder (default: 0.0 for BPSK)
     * @param b Constellation distance parameter for NRZ encoder (default: 2.0 for BPSK)
     *
     * Default parameters (a=0, b=2) implement BPSK modulation with constellation {-1, +1}.
     * For other modulation schemes (e.g., OOK with a=1, b=2), specify custom parameters.
     */
    BI_AWGN(double EbN0dB, double a = 0.0, double b = 2.0) : encoder(a, b), transmission(EbN0dB, a, b) {}

    /**
     * @brief Process single bit through BI-AWGN channel
     * @param in Input bit
     * @return Noisy complex-valued channel output
     */
    std::complex<double> operator()(const Fp<2>& in) noexcept { return transmission(encoder(in)); }

    /**
     * @brief Get reference to internal NRZ encoder
     * @return Const reference to NRZ encoder (needed for NRZDecoder construction)
     */
    const NRZEncoder& get_encoder() const noexcept { return encoder; }

    /**
     * @brief Calculate channel capacity in bits per symbol
     * @return Channel capacity C in bits per symbol
     *
     * Computes the Shannon capacity of the binary-input, real-valued AWGN channel
     * using numerical integration with composite Simpson's rule.
     *
     * **Mathematical Model**:
     * - Binary input X ∈ {0, 1} mapped to real constellation points via NRZ
     * - Real-valued output Y ∈ ℝ (one-dimensional continuous observation)
     * - Capacity computed by numerical integration (no closed-form solution)
     *
     * **Edge Cases**:
     * - b = 0: C = 0 bits (no separation between symbols)
     * - σ = 0: C = 1 bit (noiseless channel)
     *
     * @note Result is clamped to [0, 1] for numerical stability (binary input limits capacity to 1 bit).
     * @note Integration performed over real line ℝ, reflecting real-valued signaling.
     */
    double get_capacity() const noexcept;

    /**
     * @brief Get theoretical bit error probability
     * @return Theoretical Pe for the configured modulation and SNR
     *
     * Returns the mathematically calculated bit error probability after hard decisions based on
     * the constellation parameters and signal-to-noise ratio.
     */
    constexpr double get_pe() const noexcept { return transmission.get_pe(); }

   private:
    NRZEncoder encoder;
    AWGN transmission;
};

double BI_AWGN::get_capacity() const noexcept {
    const double a = encoder.get_a();
    const double b = encoder.get_b();
    const double sigma = transmission.get_standard_deviation();

    if (b == 0.0) return 0.0;      // no separation -> capacity zero
    if (sigma == 0.0) return 1.0;  // noiseless -> capacity one

    auto f_Y_star = [a, b, sigma](double x) -> double {
        double m = 1.0 / (2.0 * sigma * std::sqrt(2.0 * std::numbers::pi));
        double s0 = std::exp(-0.5 * std::pow((x - (a - b / 2.0)) / sigma, 2.0));
        double s1 = std::exp(-0.5 * std::pow((x - (a + b / 2.0)) / sigma, 2.0));
        return m * (s0 + s1);
    };

    auto g = [&f_Y_star](double x) -> double {
        double fy = f_Y_star(x);
        if (fy <= 0.0) return 0.0;  // avoid log2(0)
        return fy * std::log2(fy);
    };

    const double K = 8.0;
    const double mu0 = a - b / 2.0;
    const double mu1 = a + b / 2.0;
    const double lb = std::min(mu0, mu1) - K * sigma;
    const double ub = std::max(mu0, mu1) + K * sigma;

    // ---- composite Simpson's rule on [lb, ub] ----
    const int N = 16000;  // must be even, tune for accuracy/speed
    const double h = (ub - lb) / N;

    double sum = g(lb) + g(ub);
    for (int i = 1; i < N; ++i) {
        double x = lb + i * h;
        sum += (i % 2 == 0 ? 2.0 : 4.0) * g(x);
    }

    const double integral = h * sum / 3.0;  // ≈ ∫ g(x) dx

    return std::clamp(-integral - std::log2(sigma * std::sqrt(2 * std::numbers::pi * std::numbers::e)), 0.0, 1.0);
}

/**
 * @brief Non-Return-to-Zero (NRZ) hard-decision decoder
 *
 * Implements optimal hard-decision demodulation for NRZ-encoded signals by applying
 * maximum likelihood (ML) decision rules based on constellation geometry. The decoder
 * uses the DC offset parameter from the corresponding encoder to determine the
 * optimal decision threshold for binary symbol detection.
 *
 * **Mathematical Model**:
 * - **Decision rule**: Decide 1 if Re(r) ≥ a, decide 0 if Re(r) < a
 * - **Decision threshold**: a (DC offset from the NRZ encoder)
 * - **Optimality**: Minimizes bit error probability for AWGN channels
 * - **Processing**: Uses only the real part of received complex symbols
 *
 * **Decision Regions**:
 * For NRZ constellation (a - b/2, 0) and (a + b/2, 0):
 * - **Region 0**: Re(r) ∈ (-∞, a) → output 0
 * - **Region 1**: Re(r) ∈ [a, +∞) → output 1
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // Standard NRZ decoding
 * NRZEncoder enc(0.5, 1.0);  // Constellation: {0, 1}
 * NRZDecoder dec(enc);       // Threshold at a = 0.5
 *
 * Vector<std::complex<double>> received = {(0.3, 0.1), (0.8, -0.2)};
 * Vector<Fp<2>> c_est;
 * received >> dec >> c_est;  // Result: {0, 1}
 *
 * // BPSK decoding (threshold at zero)
 * BPSKEncoder enc;      // a = 0, constellation: {-1, +1}
 * NRZDecoder dec(enc);  // Threshold at zero
 * @endcode
 *
 * @note The decoder ignores the imaginary part of received symbols, making it
 *       suitable for real-valued modulation schemes like NRZ and BPSK.
 *
 * @see @ref CECCO::NRZEncoder for the corresponding modulator
 * @see @ref CECCO::BPSKDecoder for optimized BPSK demodulation
 * @see @ref CECCO::LLRCalculator for soft-decision demodulation
 */
class NRZDecoder : public details::DecoderProcessor<NRZDecoder> {
   public:
    // Bring base class operator() overloads into scope
    using details::DecoderProcessor<NRZDecoder>::operator();

    /**
     * @brief Construct NRZ decoder from corresponding encoder
     * @param nrz The NRZ encoder used for modulation
     *
     * Creates a hard-decision decoder with decision threshold set to the
     * DC offset parameter (a) of the NRZ encoder for optimal ML decoding.
     */
    constexpr NRZDecoder(const NRZEncoder& nrz) noexcept : a(nrz.get_a()) {}

    /**
     * @brief Decode complex symbol to binary (hard) output
     * @param in Received complex symbol (possibly noisy)
     * @return Decoded binary symbol (0 or 1)
     *
     * Applies hard-decision rule: output 1 if Re(in) ≥ a, otherwise output 0.
     * This minimizes bit error probability for the NRZ constellation under AWGN.
     */
    constexpr Fp<2> operator()(const std::complex<double>& in) noexcept {
        if (in.real() >= a)
            return {1};
        else
            return {0};
    }

   private:
    const double a{};  ///< Decision threshold (DC offset from encoder)
};

/**
 * @brief Binary Phase Shift Keying (BPSK) hard-decision decoder
 *
 * Specialized NRZ decoder optimized for BPSK demodulation with zero-threshold
 * decision rule. This decoder implements the optimal ML detector for BPSK
 * modulation over AWGN channels, providing maximum performance for antipodal
 * signaling with constellation {-1, +1}.
 *
 * **Mathematical Properties**:
 * - **Decision threshold**: 0 (optimal for BPSK constellation {-1, +1})
 * - **Decision rule**: Decide 1 if Re(r) ≥ 0, decide 0 if Re(r) < 0
 * - **Optimality**: Minimizes bit error probability for BPSK over AWGN
 * - **Symmetry**: Exploits antipodal symmetry of BPSK constellation
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // Standard BPSK demodulation
 * BPSKDecoder dec;
 * Vector<std::complex<double>> y = {(-0.8, 0.3), (1.2, -0.1), (-0.1, 0.5)};
 * Vector<Fp<2>> r;
 * y >> dec >> r;  // Result: (0, 1, 0)
 *
 * // Complete BPSK communication chain
 * BPSKEncoder enc;
 * AWGN awgn(6.0, enc.get_a(), enc.get_b());
 * BPSKDecoder dec;
 *
 * Vector<Fp<2>> c = {1, 0, 1, 1, 0};
 * Vector<Fp<2>> c_est;
 * c >> enc >> awgn >> dec >> c_est;
 * @endcode
 *
 * @see @ref CECCO::BPSKEncoder for the corresponding BPSK modulator
 * @see @ref CECCO::NRZDecoder for general NRZ demodulation
 */
class BPSKDecoder : public NRZDecoder {
   public:
    /**
     * @brief Construct BPSK decoder with zero threshold
     *
     * Creates a BPSK decoder with optimal zero-threshold decision rule
     * for antipodal constellation {-1, +1}.
     */
    constexpr BPSKDecoder() noexcept : NRZDecoder(BPSKEncoder()) {}
};

/**
 * @brief Log-Likelihood Ratio (LLR) calculator for soft-decision demodulation
 *
 * Computes Log-Likelihood Ratios for binary symbols received through AWGN channels,
 * enabling soft-decision decoding algorithms such as belief propagation, Viterbi
 * decoding, and iterative decoding schemes.
 *
 * **Mathematical Model**:
 * - **LLR definition**: LLR(r) = ln(P(x=1|r)/P(x=0|r)) where r is received symbol
 * - **For NRZ over AWGN**: LLR(r) = (2·Re(r)·b - 2·a·b) / σ² = b·(2·Re(r) - 2·a) / σ²
 * - **Simplified form**: LLR(r) = b·(a - Re(r)) / σ² (with sign convention)
 * - **Interpretation**: LLR > 0 suggests bit = 0, LLR < 0 suggests bit = 1
 *
 * **LLR Properties**:
 * - **Magnitude**: |LLR| indicates reliability (larger magnitude = higher confidence)
 * - **Sign**: Sign indicates most likely bit value under the adopted convention
 * - **Units**: Natural logarithm scale (nats)
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // BPSK soft demodulation
 * BPSKEncoder enc;  // a = 0, b = 2
 * AWGN awgn(4.0, enc.get_a(), enc.get_b());
 * LLRCalculator llr_calc(enc, awgn);
 *
 * Vector<std::complex<double>> y = {(-0.8, 0.1), (1.2, -0.3), (-0.1, 0.2)};
 * Vector<double> llrs;
 * received >> llr_calc >> llrs;
 * // Positive LLR → likely bit 0, Negative LLR → likely bit 1
 *
 * // Integration with soft-decision decoder
 * Vector<Fp<2>> c = {0, 1, 0, 1};
 * Vector<std::complex<double>> x;
 * Vector<std::complex<double>> y;
 * Vector<double> llrs;
 * c >> enc >> x >> awgn >> y >> llr_calc >> llrs;
 * // Use soft_info with iterative decoder, LDPC, turbo codes, etc.
 * @endcode
 *
 * **Sign Convention**:
 * This implementation uses the convention where:
 * - **Positive LLR**: Suggests transmitted bit was 0
 * - **Negative LLR**: Suggests transmitted bit was 1
 *
 * @note LLR calculation assumes perfect knowledge of channel parameters (noise variance)
 *       and constellation parameters from the encoder (constructor takes both as parameters).
 *
 * @see @ref CECCO::NRZEncoder, @ref CECCO::BPSKEncoder for compatible modulation schemes
 * @see @ref CECCO::AWGN for noise variance estimation used in LLR computation
 */
class LLRCalculator : public details::LLRProcessor<LLRCalculator> {
   public:
    // Bring base class operator() overloads into scope
    using details::LLRProcessor<LLRCalculator>::operator();

    /**
     * @brief Construct LLR calculator from encoder and channel
     * @param nrz The NRZ encoder that defines the constellation
     * @param transmission The AWGN channel that provides noise statistics
     *
     * Creates an LLR calculator configured for the specific constellation
     * and noise parameters of the communication system.
     */
    LLRCalculator(const NRZEncoder& nrz, const AWGN& transmission) noexcept
        : a(nrz.get_a()), b(nrz.get_b()), sigmasq(pow(transmission.get_standard_deviation(), 2.0)) {}

    /**
     * @brief Calculate Log-Likelihood Ratio for received symbol
     * @param in Received complex symbol
     * @return LLR value in nats (positive suggests bit 0, negative suggests bit 1)
     *
     * Computes LLR = b·(a - Re(in)) / σ² based on the NRZ constellation
     * parameters and AWGN noise variance.
     */
    double operator()(const std::complex<double>& in) noexcept { return b * (a - in.real()) / sigmasq; }

   private:
    const double a{};        ///< DC offset from encoder
    const double b{};        ///< Constellation distance from encoder
    const double sigmasq{};  ///< Noise variance from channel
};

/**
 * @brief Field demultiplexer for expanding extension field elements to subfield representations
 * @tparam E Extension field type (e.g., F₄, F₈, F₁₆)
 * @tparam S Subfield type (must be a subfield of E)
 *
 * Converts extension field elements into subfield vectors and extension field vectors into subfield matrices.
 *
 * **Mathematical Foundation**:
 * - **Field relationship**: S ⊆ E (S is a subfield of E)
 * - **Vector space**: E viewed as vector space over S with dimension [E:S]
 * - **Basis representation**: Each E element expressed as S-linear combination of E basis elements
 * - **Matrix expansion**: Vector<E> → Matrix&lt;S&gt; where columns represent expanded elements
 *
 * **Expansion Rules**:
 * - **Element expansion**: E → Vector&lt;S&gt; using field's polynomial basis representation
 * - **Vector expansion**: Vector<E> → Matrix&lt;S&gt; where Matrix&lt;S&gt;(i,j) corresponds to
 *   the i-th basis coefficient of the j-th vector element
 * - **Dimension preservation**: n elements of E expand to n columns of [E:S] rows in S
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // F₄ to F₂ expansion
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;  // F₄ = F₂[x]/(x² + x + 1)
 *
 * DEMUX<F4, F2> demux;
 *
 * // Single element expansion
 * F4 e(3);  // Element 3 in F₄
 * Vector<F2> v
 * e >> demux >> v;  // Result: basis coefficients in F₂
 *
 * // Vector expansion
 * Vector<F4> v = {F4(0), F4(1), F4(2), F4(3)};
 * Matrix<F2> M;
 * v >> demux >> M;  // 2 × 4 matrix over F₂
 *
 * // Complex field hierarchy: F₁₆ → F₄
 * using F16 = Ext<F4, {2, 1, 1}>;  // F₁₆ = F₄[y]/(y² + y + F₄(2))
 * DEMUX<F16, F4> f16_to_f4;
 * DEMUX<F4, F2> f4_to_f2;
 * F16 e(11);
 * Vector<F4> v;
 * Matrix<F2> M;
 * e >> f16_to_f4 >> v >> f4_to_f2 >> M;
 * @endcode
 *
 * **Memory Layout**:
 * For Vector<E> → Matrix&lt;S&gt; conversion:
 * - **Rows**: [E:S] (extension degree)
 * - **Columns**: Vector<E>.size() (number of elements)
 * - **Column j**: Subfield coefficients of Vector<E>[j]
 *
 * @note The DEMUX operation is the inverse of MUX for compatible field relationships.
 *       Use with MUX to perform round-trip conversions: E → Vector&lt;S&gt; → E.
 *
 * @see @ref CECCO::MUX for the inverse operation (subfield to extension field)
 * @see Field relationships in @ref CECCO::SubfieldOf concept
 */
template <FiniteFieldType E, FiniteFieldType S>
    requires(SubfieldOf<E, S>)
class DEMUX : private details::NonCopyable {
   public:
    /**
     * @brief Default construct DEMUX converter
     */
    constexpr DEMUX() = default;

    /**
     * @brief Expand single extension field element to subfield vector
     * @param in Extension field element to expand
     * @return Vector of subfield coefficients representing the element
     *
     * Converts extension field element to its vector representation over the subfield
     * using the polynomial basis of the field extension.
     */
    constexpr Vector<S> operator()(const E& in) noexcept { return in.template as_vector<S>(); }

    /**
     * @brief Expand vector of extension field elements to subfield matrix
     * @param in Vector of extension field elements
     * @return Matrix where each column contains subfield coefficients of corresponding input element
     *
     * Creates matrix representation where Matrix(i,j) is the i-th subfield coefficient
     * of the j-th input vector element.
     */
    Matrix<S> operator()(const Vector<E>& in) noexcept;
};

template <FiniteFieldType E, FiniteFieldType S>
    requires(SubfieldOf<E, S>)
Matrix<S> DEMUX<E, S>::operator()(const Vector<E>& in) noexcept {
    if (in.get_n() == 0) return Matrix<S>(0, 0);

    auto temp = in[0].template as_vector<S>();
    Matrix<S> M(temp.get_n(), in.get_n());

    // First column: reuse the temp vector we already computed
    M.set_submatrix(0, 0, transpose(Matrix<S>(std::move(temp))));

    // Remaining columns: use move semantics throughout the chain
    for (size_t i = 1; i < in.get_n(); ++i) {
        auto col_vector = in[i].template as_vector<S>();
        M.set_submatrix(0, i, transpose(Matrix<S>(std::move(col_vector))));
    }

    return M;
}

/**
 * @brief Field multiplexer for converting subfield representations to extension field elements
 * @tparam S Subfield type (e.g., F₂, F₄)
 * @tparam E Extension field type (must be an extension of S)
 *
 * Converts subfield vector/matrix representations back into extension field elements and vectors,
 * providing the inverse operation to DEMUX.
 *
 * **Mathematical Foundation**:
 * - **Field relationship**: S ⊆ E (E is an extension field of S)
 * - **Vector space**: E viewed as vector space over S with dimension [E:S]
 * - **Basis reconstruction**: Vector&lt;S&gt; coefficients combined using E basis elements
 * - **Matrix compression**: Matrix&lt;S&gt; → Vector<E> where columns become individual E elements
 *
 * **Compression Rules**:
 * - **Vector compression**: Vector&lt;S&gt; → E using polynomial basis reconstruction
 * - **Matrix compression**: Matrix&lt;S&gt; → Vector<E> where Matrix&lt;S&gt; column j becomes Vector<E>[j]
 * - **Dimension validation**: Input dimensions must match [E:S] extension degree
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // F₂ to F₄ compression
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;  // F₄ = F₂[x]/(x² + x + 1)
 *
 * MUX<F2, F4> mux;
 *
 * // Single element reconstruction
 * Vector<F2> v = {1, 0};  // Coefficients for F₄ element
 * F4 e;
 * v >> mux >> e;          // Reconstruct F₄ element
 *
 * // Matrix compression to vector
 * Matrix<F2> M(2, 4);              // 2×4 matrix (2 = [F₄:F₂])
 * M.set_component(0, 0, F2(1)); M.set_component(1, 0, F2(0));  // First column: F₄(1)
 * M.set_component(0, 1, F2(0)); M.set_component(1, 1, F2(1));  // Second column: F₄(α)
 * M.set_component(0, 2, F2(1)); M.set_component(1, 2, F2(1));  // Third column: F₄(α+1)
 * M.set_component(0, 3, F2(0)); M.set_component(1, 3, F2(0));  // Fourth column: F₄(0)
 * Vector<F4> v;
 * M >> mux >> v;                   // Result: {F₄(1), F₄(α), F₄(α+1), F₄(0)}
 *
 * // Round-trip conversion (DEMUX → MUX)
 * DEMUX<F4, F2> demux;
 * Vector<F4> original = {F4(1), F4(2), F4(3)};
 * Matrix<F2> M;
 * Vector<F4> restored;
 * original >> demux >> M >> mux >> restored;
 * assert (original == restored);
 * @endcode
 *
 * **Memory Layout**:
 * For Matrix&lt;S&gt; → Vector<E> conversion:
 * - **Input rows**: [E:S] (extension degree)
 * - **Input columns**: n (desired Vector<E>.size())
 * - **Column j**: Subfield coefficients that become Vector<E>[j]
 *
 * @note MUX is the exact inverse of DEMUX for compatible field relationships.
 *       Round-trip conversions preserve all field element values exactly.
 *
 * @see @ref CECCO::DEMUX for the inverse operation (extension field to subfield)
 * @see Field relationships in @ref CECCO::ExtensionOf concept
 */
template <FiniteFieldType S, FiniteFieldType E>
    requires(ExtensionOf<S, E>)
class MUX : private details::NonCopyable {
   public:
    /**
     * @brief Default construct MUX converter
     */
    constexpr MUX() = default;

    /**
     * @brief Construct extension field element from subfield coefficient vector
     * @param in Vector of subfield coefficients (length must equal [E:S])
     * @return Extension field element reconstructed from coefficients
     *
     * Combines subfield coefficients using the polynomial basis of the extension field
     * to reconstruct the original extension field element.
     */
    constexpr E operator()(const Vector<S>& in) noexcept { return E(in); }

    /**
     * @brief Convert subfield matrix to extension field vector
     * @param in Matrix of subfield elements (rows = [E:S], columns = desired vector size)
     * @return Vector where each element is reconstructed from corresponding matrix column
     *
     * Creates extension field vector where Vector<E>[j] is reconstructed from Matrix&lt;S&gt; column j.
     */
    Vector<E> operator()(const Matrix<S>& in) noexcept;
};

template <FiniteFieldType S, FiniteFieldType E>
    requires(ExtensionOf<S, E>)
Vector<E> MUX<S, E>::operator()(const Matrix<S>& in) noexcept {
    Vector<E> v(in.get_n());
    for (size_t i = 0; i < in.get_n(); ++i) v.set_component(i, E(in.get_col(i)));

    return v;
}

/**
 * @brief Universal operator>> for callable objects (function call chaining)
 * @tparam LHS Input type (left-hand side of >>)
 * @tparam RHS Callable type (right-hand side of >>, must be callable with LHS)
 * @param lhs Input value to be processed
 * @param rhs Callable object (function, lambda, class with operator())
 * @return Result of calling rhs(lhs) with perfect forwarding
 *
 * Enables elegant chaining syntax where `x >> f` becomes `f(x)`, supporting
 * perfect forwarding for both move and copy semantics. This overload applies
 * when RHS is callable with LHS as argument.
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // Communication chain with processing blocks
 * Vector<Fp<2>> c = {1, 0, 1, 0};
 * BPSKEncoder enc;
 * AWGN awgn(4.0, enc.get_a(), enc.get_b());
 * BPSKDecoder dec;
 *
 * // Chain operations using operator>>
 * Vector<Fp<2>> r;
 * c >> enc >> awgn >> dec >> r;
 *
 * // Equivalent to nested function calls:
 * // r = deco(awgn(enc(c)));
 * @endcode
 *
 * @note This operator is disabled for iostream types to avoid conflicts with
 *       standard stream operators.
 */
template <class LHS, class RHS>
    requires(!std::is_base_of_v<std::ios_base, std::remove_cv_t<std::remove_reference_t<LHS>>>) &&
            requires(RHS&& r, LHS&& x) { std::forward<RHS>(r)(std::forward<LHS>(x)); }
decltype(auto) operator>>(LHS&& lhs, RHS&& rhs) {
    return std::forward<RHS>(rhs)(std::forward<LHS>(lhs));
}

/**
 * @brief Universal operator>> for assignment targets (assignment chaining)
 * @tparam LHS Input type (left-hand side of >>)
 * @tparam RHS Assignment target type (right-hand side, must be assignable from LHS)
 * @param lhs Input value to be assigned
 * @param dst Destination variable for assignment
 * @return Reference to dst for continued chaining
 *
 * Enables assignment chaining syntax where `x >> dst` becomes `dst = x` and returns
 * dst for continued chaining. This overload applies when RHS is not callable but
 * is assignable from LHS.
 *
 * **Usage Examples**:
 * @code{.cpp}
 * // Assignment with continued processing
 * Vector<Fp<2>> c = {1, 0, 1, 1};
 * Vector<std::complex<double>> x;
 * Vector<std::complex<double>> y;
 * Vector<Fp<2>> r;
 *
 * BPSKEncoder enc;
 * AWGN awgn(4.0, enc.get_a(), enc.get_b());
 * BPSKDecoder dec;
 *
 * // Chain with intermediate storage
 * c >> enc >> x >> awgn >> y >> dec >> r;
 * // x and y variables now contain intermediate results
 *
 * // In case intermediate results are not needed:
 * c >> enc >> awgn >> dec >> r;
 * @endcode
 *
 * @note This overload has lower priority than the callable overload, ensuring
 *       callable objects are preferred when both assignment and function call are possible.
 */
template <class LHS, class RHS>
    requires(!std::is_base_of_v<std::ios_base, std::remove_cv_t<std::remove_reference_t<LHS>>>) &&
            (!requires(RHS&& r, LHS&& x) { std::forward<RHS>(r)(std::forward<LHS>(x)); }) &&
            requires(RHS& dst, LHS&& x) { dst = std::forward<LHS>(x); }
RHS& operator>>(LHS&& lhs, RHS& dst) {
    dst = std::forward<LHS>(lhs);
    return dst;
}

}  // namespace CECCO

#endif