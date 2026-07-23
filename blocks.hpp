/**
 * @file blocks.hpp
 * @brief Communication system blocks library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.9.1
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
 * Channels, modulation and demodulation, and extension-/subfield multiplexing blocks
 * for coding experiments. Blocks compose left-to-right with `operator>>` and generally
 * accept scalars, vectors, and matrices.
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;
 * Vector<F4> c = {F4(0), F4(1), F4(2), F4(3)};
 * BI_AWGN channel(6.0);
 * LLRCalculator<F4> llr(channel);
 * Matrix<double> L = c >> DEMUX<F4, F2>{} >> channel >> llr;
 * @endcode
 */

#ifndef BLOCKS_HPP
#define BLOCKS_HPP

#include <numbers>

#include "fields.hpp"
#include "vectors.hpp"

/*
// transitive
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <ios>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "field_concepts_traits.hpp"
#include "matrices.hpp"
*/

namespace CECCO {

namespace details {

/**
 * @brief CRTP base providing scalar, vector, and matrix block operations
 * @tparam T Derived block type
 * @tparam InputType Scalar input type
 * @tparam OutputType Scalar output type
 *
 * Derived classes implement the scalar call operator; this base supplies vector and matrix
 * overloads, reusing rvalue storage when input and output types match.
 *
 * Calls with another finite-field type are accepted syntactically and throw at runtime,
 * allowing otherwise discarded non-template branches to remain well-formed.
 *
 * @note Blocks are movable but non-copyable. Channel copies would duplicate distribution
 * caches and stochastic countdown state, producing statistically coupled block instances.
 */
template <typename T, typename InputType, typename OutputType>
class BlockProcessor : private NonCopyable {
   protected:
    const T& derived() const noexcept { return static_cast<const T&>(*this); }

    T& derived() noexcept { return static_cast<T&>(*this); }

   public:
    /**
     * @brief Apply the block element-wise to each vector entry
     * @param in Input vector
     * @return Vector of processed outputs
     */
    Vector<OutputType> operator()(const Vector<InputType>& in) {
        Vector<OutputType> res(in.get_n());
        for (size_t i = 0; i < in.get_n(); ++i) res.set_component(i, derived()(in[i]));
        return res;
    }

    /**
     * @brief Apply the block element-wise (rvalue input)
     * @param in Input vector (moved)
     * @return Vector of processed outputs (reuses input storage when InputType == OutputType)
     */
    Vector<OutputType> operator()(Vector<InputType>&& in) {
        if constexpr (std::is_same_v<InputType, OutputType>) {
            for (size_t i = 0; i < in.get_n(); ++i) in.set_component(i, derived()(std::move(in[i])));
            return std::move(in);
        } else {
            const size_t n = in.get_n();
            Vector<OutputType> res(n);
            for (size_t i = 0; i < n; ++i) res.set_component(i, derived()(std::move(in[i])));
            return res;
        }
    }

    /**
     * @brief Apply the block element-wise to each matrix entry
     * @param in Input matrix
     * @return Matrix of processed outputs
     */
    Matrix<OutputType> operator()(const Matrix<InputType>& in) {
        Matrix<OutputType> res(in.get_m(), in.get_n());
        for (size_t i = 0; i < res.get_m(); ++i) {
            for (size_t j = 0; j < res.get_n(); ++j) res.set_component(i, j, derived()(in(i, j)));
        }
        return res;
    }

    /**
     * @brief Apply the block element-wise (rvalue input)
     * @param in Input matrix (moved)
     * @return Matrix of processed outputs (reuses input storage when InputType == OutputType)
     */
    Matrix<OutputType> operator()(Matrix<InputType>&& in) {
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

    /**
     * @brief Catch-all for inputs whose element type differs from `InputType`
     * @tparam U Any @ref CECCO::FiniteFieldType other than `InputType`
     *
     * Always throws `std::logic_error` at runtime. See the class doc for the `if constexpr`
     * use case that motivates these overloads.
     */
    template <FiniteFieldType U>
        requires(!std::is_same_v<U, InputType>)
    OutputType operator()(const U&) {
        throw std::logic_error("Instance of BlockProcessor can only accept inputs from " + InputType::get_info() +
                               ", got " + U::get_info());
    }

    template <FiniteFieldType U>
        requires(!std::is_same_v<U, InputType>)
    Vector<OutputType> operator()(const Vector<U>&) {
        throw std::logic_error("Instance of BlockProcessor can only accept inputs from " + InputType::get_info() +
                               ", got " + U::get_info());
    }

    template <FiniteFieldType U>
        requires(!std::is_same_v<U, InputType>)
    Matrix<OutputType> operator()(const Matrix<U>&) {
        throw std::logic_error("Instance of BlockProcessor can only accept inputs from " + InputType::get_info() +
                               ", got " + U::get_info());
    }
};

/** @brief CRTP base for blocks where input and output element types coincide. */
template <typename T, typename ElementType>
using SameTypeProcessor = BlockProcessor<T, ElementType, ElementType>;

/** @brief CRTP base for blocks mapping bits to complex symbols (e.g. NRZ/BPSK modulation). */
template <typename T>
using EncoderProcessor = BlockProcessor<T, Fp<2>, std::complex<double>>;

/** @brief CRTP base for blocks mapping complex symbols back to bits (hard demod). */
template <typename T>
using DecoderProcessor = BlockProcessor<T, std::complex<double>, Fp<2>>;

}  // namespace details

/**
 * @brief Symmetric errors-and-erasures channel over a finite field
 * @tparam T Finite field of the transmitted symbols
 *
 * Independently replaces a symbol by a uniformly selected different symbol with probability
 * `pe`, then marks it erased with probability `px`; erasure takes precedence over error.
 *
 * For `px == 0` this is an @ref CECCO::SDMC; over 𝔽₂, `pe == 0` gives an
 * @ref CECCO::BEC. Erasures require @ref CECCO_ERASURE_SUPPORT.
 *
 * The capacity is
 * `C = (1 − px)[log₂(q) + (1 − p̃)log₂(1 − p̃) + p̃log₂(p̃) − p̃log₂(q − 1)]`,
 * where `p̃ = pe / (1 − px)`.
 */
template <FieldType T>
class SDMEC : public details::SameTypeProcessor<SDMEC<T>, T> {
   public:
    // Bring base class operator() overloads into scope
    using details::SameTypeProcessor<SDMEC<T>, T>::operator();

    /**
     * @brief Construct with error probability pe and erasure probability px
     * @param pe Symbol error probability ∈ [0, 1]
     * @param px Symbol erasure probability ∈ [0, 1 − pe]; default 0
     *
     * @throws std::out_of_range if pe ∉ [0, 1] or px ∉ [0, 1 − pe]
     * @throws std::invalid_argument if px ≠ 0 and @ref CECCO_ERASURE_SUPPORT is not defined
     */
    SDMEC(double pe, double px = 0.0);

    /**
     * @brief Process one symbol
     * @param in Input symbol
     * @return Output symbol (possibly erased when @ref CECCO_ERASURE_SUPPORT is defined)
     */
    T operator()(const T& in);

    /** @brief Symbol error probability pe (erroneous and not erased). */
    double get_pe() const noexcept { return pe; }

    /** @brief Symbol erasure probability px. */
    double get_px() const noexcept { return px; }

    /**
     * @brief Shannon capacity in bits per symbol
     * @return C = (1 − px) · [log₂(q) + (1 − p̃)·log₂(1 − p̃) + p̃·log₂(p̃) − p̃·log₂(q − 1)],
     *         where p̃ = pe / (1 − px). Edge cases pe ∈ {0, 1} use lim_{x→0} x·log₂(x) = 0.
     */
    double get_capacity() const noexcept
        requires(FiniteFieldType<T>);

    /**
     * @brief Pairwise Bhattacharyya parameter
     *
     * Uses γ = px + 2·√((1 − pe − px)·pe/(q − 1)) + (q − 2)·pe/(q − 1),
     * with q = |T| and pe the probability of an erroneous, non-erased output.
     *
     * @return Common pairwise Bhattacharyya parameter γ for distinct input symbols
     */
    long double get_Bhattacharyya_param() const noexcept
        requires(FiniteFieldType<T>)
    {
        const long double q = T::get_size();
        const long double pe = get_pe();
        const long double px = get_px();

        return px + 2.0L * std::sqrt((1.0L - pe - px) * pe / (q - 1.0L)) + (q - 2.0L) * pe / (q - 1.0L);
    }

   private:
    double pe;
    double px;
    double p_error_given_not_erased{0.0};
    std::geometric_distribution<unsigned long long> error_dist;
    unsigned long long error_trials{0};
    unsigned long long error_failures_before_hit{0};
#ifdef CECCO_ERASURE_SUPPORT
    std::geometric_distribution<unsigned long long> erasure_dist;
    unsigned long long erasure_trials{0};
    unsigned long long erasure_failures_before_hit{0};
#endif
};

template <FieldType T>
SDMEC<T>::SDMEC(double pe, double px) : pe(pe), px(px) {
    if (!std::isfinite(pe) || pe < 0.0 || pe > 1.0)
        throw std::out_of_range("SDMEC error probability must be in [0,1], got: " + std::to_string(pe));
    if (!std::isfinite(px) || px < 0.0 || px > 1.0 - pe)
        throw std::out_of_range("SDMEC erasure probability must be in [0," + std::to_string(1.0 - pe) +
                                "], got: " + std::to_string(px));
#ifndef CECCO_ERASURE_SUPPORT
    if (px != 0.0) throw std::invalid_argument("px != 0 requires CECCO_ERASURE_SUPPORT");
#endif

    p_error_given_not_erased = (px >= 1.0) ? 0.0 : pe / (1.0 - px);
    if (p_error_given_not_erased > 0.0 && p_error_given_not_erased < 1.0) {
        error_dist = std::geometric_distribution<unsigned long long>(p_error_given_not_erased);
        error_failures_before_hit = error_dist(gen());
    }
#ifdef CECCO_ERASURE_SUPPORT
    if (px > 0.0 && px < 1.0) {
        erasure_dist = std::geometric_distribution<unsigned long long>(px);
        erasure_failures_before_hit = erasure_dist(gen());
    }
#endif
}

template <FieldType T>
T SDMEC<T>::operator()(const T& in) {
    T res(in);
    if (p_error_given_not_erased >= 1.0) {
        res.randomize_force_change();
    } else if (p_error_given_not_erased > 0.0) {
        if (error_trials == error_failures_before_hit) {
            res.randomize_force_change();
            error_trials = 0;
            error_failures_before_hit = error_dist(gen());
        } else {
            ++error_trials;
        }
    }
#ifdef CECCO_ERASURE_SUPPORT
    if (px >= 1.0) {
        res.erase();
    } else if (px > 0.0) {
        if (erasure_trials == erasure_failures_before_hit) {
            res.erase();
            erasure_trials = 0;
            erasure_failures_before_hit = erasure_dist(gen());
        } else {
            ++erasure_trials;
        }
    }
#endif
    return res;
}

template <FieldType T>
double SDMEC<T>::get_capacity() const noexcept
    requires(FiniteFieldType<T>)
{
    const double ptilde = p_error_given_not_erased;
    const double q = static_cast<double>(T::get_size());

    const double term1 = (ptilde > 0.0 && ptilde < 1.0) ? ptilde * std::log2(ptilde) : 0.0;
    const double term2 = (ptilde > 0.0 && ptilde < 1.0) ? (1.0 - ptilde) * std::log2(1.0 - ptilde) : 0.0;

    return (std::log2(q) + term2 + term1 - ptilde * std::log2(q - 1.0)) * (1.0 - px);
}

/**
 * @brief Symmetric Discrete Memoryless Channel: errors only, no erasures
 * @tparam T Finite field type for channel input/output symbols
 *
 * Convenience wrapper for `SDMEC<T>(pe, 0.0)` with a single-parameter constructor.
 *
 * @see @ref CECCO::SDMEC for the full errors-and-erasures channel
 * @see @ref CECCO::BSC for the binary case
 */
template <FieldType T>
class SDMC : public SDMEC<T> {
   public:
    /**
     * @brief Construct with symbol error probability pe
     * @param pe Symbol error probability ∈ [0, 1]
     */
    SDMC(double pe) : SDMEC<T>(pe, 0.0) {}
};

/**
 * @brief Binary Symmetric Channel: type alias for `SDMC<Fp<2>>`
 *
 * Bits are flipped with probability pe.
 *
 * @see @ref CECCO::SDMC, @ref CECCO::BEC
 */
using BSC = SDMC<Fp<2>>;

/**
 * @brief Binary Erasure Channel: symbols are received correctly or marked erased
 *
 * Convenience wrapper for `SDMEC<Fp<2>>(0.0, px)`. Requires @ref CECCO_ERASURE_SUPPORT.
 *
 * @see @ref CECCO::SDMEC, @ref CECCO::BSC
 */
class BEC : public SDMEC<Fp<2>> {
   public:
    /**
     * @brief Construct with symbol erasure probability px
     * @param px Symbol erasure probability ∈ [0, 1]
     */
    BEC(double px) : SDMEC<Fp<2>>(0.0, px) {}
};

/**
 * @brief Binary Asymmetric Channel (Z-channel): 0 is preserved; 1 flips to 0 with probability p
 *
 * @code{.cpp}
 * BAC bac(0.1);                            // 10% probability that 1 flips to 0
 * Vector<Fp<2>> c = {0, 1, 1, 0};
 * Vector<Fp<2>> r = c >> bac;
 * double C = bac.get_capacity();
 * @endcode
 *
 * @see @ref CECCO::BSC, @ref CECCO::BEC
 */
class BAC : public details::SameTypeProcessor<BAC, Fp<2>> {
   public:
    using details::SameTypeProcessor<BAC, Fp<2>>::operator();

    /**
     * @brief Construct with flip probability p
     * @param p Probability that 1 → 0, p ∈ [0, 1]
     *
     * @throws std::out_of_range if p ∉ [0, 1]
     *
     * @note p = 0 is the perfect channel (capacity 1); p = 1 makes only 0 transmissible (capacity 0).
     */
    BAC(double p) : bsc(p) {}

    /**
     * @brief Process one bit
     * @param in Input bit
     * @return Output: 0 is preserved; 1 flips to 0 with probability p
     */
    Fp<2> operator()(const Fp<2>& in) {
        if (in == Fp<2>(0)) return in;
        return bsc(in);
    }

    /** @brief Flip probability p (probability that 1 → 0). */
    double get_pe() const noexcept { return bsc.get_pe(); }

    /**
     * @brief Shannon capacity in bits per symbol
     * @return C = log₂(1 + (1 − p)·p^{p/(1−p)}); edge cases p ∈ {0, 1} handled explicitly.
     */
    double get_capacity() const noexcept {
        const double pe = bsc.get_pe();

        if (pe == 0.0) return 1.0;
        if (pe == 1.0) return 0.0;

        return std::log2(1 + (1 - pe) * std::pow(pe, pe / (1 - pe)));
    }

   private:
    BSC bsc;
};

/**
 * @brief Non-Return-to-Zero (NRZ) mapper for binary modulation
 *
 * Maps 𝔽_2 symbols to real constellation points (a − b/2, 0) and (a + b/2, 0), where `a` is
 * the DC offset and `b` is the constellation distance. Energy per bit is Eb = a² + b²/4
 * (unit symbol duration assumed).
 *
 * Typical configurations: BPSK (a = 0, b = 2, constellation {−1, +1}, Eb = 1) and
 * OOK (a = 1, b = 2, constellation {0, +2}, Eb = 2).
 *
 * @code{.cpp}
 * NRZMapper ook(1.0, 2.0);
 * Vector<Fp<2>> bits = {0, 1, 0, 1};
 * Vector<std::complex<double>> signal = bits >> ook;   // (0, 2, 0, 2)
 * @endcode
 *
 * @see @ref CECCO::BPSKMapper, @ref CECCO::AWGN, @ref CECCO::NRZDemapper
 */
class NRZMapper : public details::EncoderProcessor<NRZMapper> {
   public:
    // Bring base class operator() overloads into scope
    using details::EncoderProcessor<NRZMapper>::operator();

    /**
     * @brief Construct with constellation parameters
     * @param a DC offset (real-axis shift)
     * @param b Constellation distance (separation between symbols)
     */
    constexpr NRZMapper(double a, double b) noexcept : a(a), b(b) {}

    /** @name Constellation Parameters
     * @{
     */

    /** @brief Energy per bit Eb = a² + b²/4. */
    constexpr double get_Eb() const noexcept { return (a * a) + (b * b) / 4.0; }

    /** @brief DC offset parameter `a`. */
    constexpr double get_a() const noexcept { return a; }

    /** @brief Constellation distance parameter `b`. */
    constexpr double get_b() const noexcept { return b; }

    /** @} */

    /**
     * @brief Map a bit to its constellation point
     * @param in Binary input
     * @return (a − b/2, 0) for input 0; (a + b/2, 0) for input 1
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
 * @brief Binary Phase Shift Keying mapper: `NRZMapper` with a = 0, b = 2
 *
 * Antipodal constellation {−1, +1}, Eb = 1.
 *
 * @see @ref CECCO::NRZMapper, @ref CECCO::BPSKDemapper
 */
class BPSKMapper : public NRZMapper {
   public:
    /** @brief Construct BPSK mapper (a = 0, b = 2). */
    constexpr BPSKMapper() noexcept : NRZMapper(0.0, 2.0) {}
};

/**
 * @brief Additive White Gaussian Noise channel for complex-valued symbols
 *
 * Adds independent 𝒩(0, σ²) noise to the real and imaginary parts of each input symbol.
 * The noise variance is derived from the SNR Eb/N₀ and the constellation parameters
 * supplied at construction: σ² = Eb / (2 · 10^(EbN0dB/10)).
 *
 * For NRZ/BPSK input the bit error probability after a hard decision is
 *   Pe = ½ · erfc(√( b²·Eb/N₀ / (4·Eb) )),
 * available via @ref get_pe.
 *
 * @code{.cpp}
 * BPSKMapper map;
 * AWGN awgn(6.0, map.get_a(), map.get_b());        // 6 dB Eb/N₀
 * Vector<Fp<2>> message = {0, 1, 1, 0};
 * Vector<std::complex<double>> y = message >> map >> awgn;
 * @endcode
 *
 * @see @ref CECCO::NRZMapper, @ref CECCO::BPSKMapper, @ref CECCO::BI_AWGN, @ref CECCO::LLRCalculator
 */
class AWGN : public details::SameTypeProcessor<AWGN, std::complex<double>> {
   public:
    // Bring base class operator() overloads into scope
    using details::SameTypeProcessor<AWGN, std::complex<double>>::operator();

    /**
     * @brief Construct with SNR and constellation parameters
     * @param EbNodB SNR Eb/N₀ in dB
     * @param a DC offset of the source constellation
     * @param b Constellation distance of the source constellation
     */
    AWGN(double EbNodB, double a, double b)
        : Eb(NRZMapper(a, b).get_Eb()),
          sigma(std::sqrt(0.5 * Eb / std::pow(10.0, EbNodB / 10.0))),
          dist(0, sigma > 0.0 ? sigma : 1.0),  // sigma = 0 violates normal_distribution's precondition
          pe(calculate_pe(EbNodB, Eb, b)) {
        if (!std::isfinite(EbNodB) || !std::isfinite(a) || !std::isfinite(b))
            throw std::invalid_argument("AWGN parameters must be finite!");
        if (!std::isfinite(sigma))
            throw std::invalid_argument("AWGN noise standard deviation must be finite, adjust Eb/N0!");
    }

    /** @name Noise Parameters
     * @{
     */

    /** @brief Noise variance σ² (per real component). */
    double get_variance() const noexcept { return sigma * sigma; }

    /** @brief Noise standard deviation σ (per real component). */
    double get_standard_deviation() const noexcept { return sigma; }

    /** @brief Theoretical hard-decision bit error probability for the configured constellation/SNR. */
    constexpr double get_pe() const noexcept { return pe; }

    /** @} */

    /**
     * @brief Add Gaussian noise to a complex symbol
     * @param in Input symbol
     * @return Symbol with independent Gaussian noise added to each real component
     */
    std::complex<double> operator()(const std::complex<double>& in) {
        if (sigma == 0.0) return in;
        // sequenced draws: I before Q, so seeded runs reproduce across compilers
        const double noise_i = dist(gen());
        const double noise_q = dist(gen());
        return {in.real() + noise_i, in.imag() + noise_q};
    }

    /**
     * @brief Shannon capacity in bits per symbol (real signaling)
     * @return C = ½·log₂(1 + Eb/σ²) for one real dimension (NRZ/BPSK).
     *
     * For complex signaling with noise on both I and Q the formula is C = log₂(1 + Eb/(2σ²)).
     */
    double get_capacity() const noexcept {
        // sigma = 0 is either the noiseless channel (Eb > 0) or the zero-energy constellation (a = b = 0)
        if (sigma == 0.0) return Eb > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
        return 0.5 * std::log2(1 + Eb / (sigma * sigma));
    }

   private:
    const double Eb{};
    const double sigma{};
    std::normal_distribution<double> dist;
    const double pe{};

    /**
     * @brief Closed-form NRZ/BPSK hard-decision Pe
     * @param EbNodB SNR Eb/N₀ in dB
     * @param Eb Energy per bit of the source constellation
     * @param b Constellation distance
     * @return Pe = ½·erfc(√( b²·10^(EbNodB/10) / (4·Eb) ))
     */
    static double calculate_pe(double EbNodB, double Eb, double b) noexcept;
};

double AWGN::calculate_pe(double EbNodB, double Eb, double b) noexcept {
    if (Eb == 0.0) return 0.5;  // degenerate zero-energy constellation (a = b = 0), avoid 0/0

    // For NRZ/BPSK: Pe = 0.5 * erfc(b/(2*sigma))
    // where b is constellation distance and sigma is noise std dev
    const double EbN0_linear = std::pow(10.0, EbNodB / 10.0);

    // Signal-to-noise ratio in terms of constellation distance
    // SNR = (b/2)² / σ² = (b/2)² / (No/2) = (b/2)² * 2/No = (b²/2) / No
    // where No = Eb/EbN0_linear
    const double constellation_snr = (b * b * EbN0_linear) / (4.0 * Eb);

    return 0.5 * std::erfc(std::sqrt(constellation_snr));
}

/**
 * @brief Binary-Input AWGN: fused @ref NRZMapper + @ref AWGN block
 *
 * Maps binary inputs through an internal NRZMapper and then through AWGN, yielding noisy
 * complex symbols ready for hard decision (@ref NRZDemapper) or soft decision
 * (@ref LLRCalculator). Default parameters give BPSK (a = 0, b = 2). For codes over 𝔽_{2^m}
 * each code symbol occupies m channel uses via its binary image (cf. @ref CECCO::DEMUX);
 * energy accounting remains per transmitted bit.
 *
 * @code{.cpp}
 * BI_AWGN channel(6.0);                                       // BPSK at 6 dB
 * Vector<Fp<2>> c = {1, 0, 1, 0};
 * Vector<Fp<2>> r = c >> channel >> NRZDemapper(channel);     // hard decisions
 * @endcode
 *
 * @see @ref CECCO::AWGN, @ref CECCO::NRZDemapper, @ref CECCO::LLRCalculator
 */
class BI_AWGN : public details::BlockProcessor<BI_AWGN, Fp<2>, std::complex<double>> {
   public:
    using details::BlockProcessor<BI_AWGN, Fp<2>, std::complex<double>>::operator();

    /**
     * @brief Construct with SNR and constellation parameters
     * @param EbN0dB SNR Eb/N₀ in dB
     * @param a NRZ DC offset (default 0 for BPSK)
     * @param b NRZ constellation distance (default 2 for BPSK)
     */
    BI_AWGN(double EbN0dB, double a = 0.0, double b = 2.0) : encoder(a, b), transmission(EbN0dB, a, b) {}

    /**
     * @brief Map a bit and add noise
     * @param in Input bit
     * @return NRZ symbol corrupted by AWGN
     */
    std::complex<double> operator()(const Fp<2>& in) { return transmission(encoder(in)); }

    /** @brief Internal NRZ mapper (for constructing matching demappers/LLR calculators). */
    const NRZMapper& get_encoder() const noexcept { return encoder; }

    /** @brief Internal AWGN block (for noise statistics). */
    const AWGN& get_transmission() const noexcept { return transmission; }

    /**
     * @brief Shannon capacity in bits per symbol
     * @return C ∈ [0, 1], computed by composite Simpson's rule (no closed form for BI-AWGN).
     *
     * Returns 0 if the constellation distance is 0 and 1 if the noise vanishes.
     */
    double get_capacity() const noexcept;

    /** @brief Theoretical hard-decision bit error probability for this constellation/SNR. */
    constexpr double get_pe() const noexcept { return transmission.get_pe(); }

    /** @brief Noise standard deviation σ. */
    double get_sigma() const noexcept { return transmission.get_standard_deviation(); }

    /**
     * @brief Bhattacharyya parameter γ = exp(−b²/(8σ²))
     */
    long double get_Bhattacharyya_param() const noexcept {
        const long double b = encoder.get_b();
        const long double sigma = transmission.get_standard_deviation();
        return std::exp(-(b * b) / (8.0L * sigma * sigma));
    }

   private:
    NRZMapper encoder;
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
 * @brief Non-Return-to-Zero (NRZ) hard-decision demapper
 *
 * Maximum-likelihood threshold detector for NRZ-modulated symbols received over AWGN:
 * outputs 1 if Re(r) ≥ a, otherwise 0, where `a` is the DC offset of the corresponding
 * @ref CECCO::NRZMapper. The imaginary part is ignored.
 *
 * @code{.cpp}
 * NRZMapper map(0.5, 1.0);                  // Constellation: {0, 1}
 * NRZDemapper demap(map);                   // Threshold at a = 0.5
 * Vector<std::complex<double>> y = {{0.3, 0.1}, {0.8, −0.2}};
 * Vector<Fp<2>> c_est = y >> demap;         // {0, 1}
 * @endcode
 *
 * @see @ref CECCO::NRZMapper, @ref CECCO::BPSKDemapper, @ref CECCO::LLRCalculator
 */
class NRZDemapper : public details::DecoderProcessor<NRZDemapper> {
   public:
    // Bring base class operator() overloads into scope
    using details::DecoderProcessor<NRZDemapper>::operator();

    /**
     * @brief Construct from the corresponding mapper
     * @param nrz Source mapper whose DC offset becomes the decision threshold
     */
    constexpr NRZDemapper(const NRZMapper& nrz) noexcept : a(nrz.get_a()) {}

    /**
     * @brief Construct from a BI-AWGN channel (convenience)
     * @param bi_awgn Channel whose internal mapper provides the decision threshold
     */
    NRZDemapper(const BI_AWGN& bi_awgn) noexcept : a(bi_awgn.get_encoder().get_a()) {}

    /**
     * @brief Hard-decide a received symbol
     * @param in Received complex value (imaginary part ignored)
     * @return 1 if Re(in) ≥ a, else 0
     */
    constexpr Fp<2> operator()(const std::complex<double>& in) noexcept {
        if (in.real() >= a)
            return {1};
        else
            return {0};
    }

   private:
    const double a{};  ///< Decision threshold (DC offset from mapper)
};

/**
 * @brief BPSK hard-decision demapper: `NRZDemapper` with threshold 0
 *
 * @see @ref CECCO::BPSKMapper, @ref CECCO::NRZDemapper
 */
class BPSKDemapper : public NRZDemapper {
   public:
    /** @brief Construct BPSK demapper (threshold 0). */
    constexpr BPSKDemapper() noexcept : NRZDemapper(BPSKMapper()) {}
};

/**
 * @brief Compute symbol-level log-likelihood ratios
 * @tparam T Code-symbol field
 *
 * Produces a `(q − 1) × n` matrix whose entry `(a − 1, j)` is
 * `ln(P(r_j | 0) / P(r_j | a))`. Positive values favor zero; magnitude is reliability.
 *
 * Supports prime-field SDMEC observations and binary-image BI-AWGN observations. BI-AWGN
 * requires characteristic two and combines the independent bit LLRs of each extension-field symbol.
 */
template <FiniteFieldType T = Fp<2>>
class LLRCalculator : private details::NonCopyable {
   public:
    /**
     * @brief Construct a BI-AWGN calculator from a matching mapper and channel
     * @param nrz Source mapper providing (a, b)
     * @param transmission Channel providing variance
     */
    LLRCalculator(const NRZMapper& nrz, const AWGN& transmission) noexcept
        requires(T::get_characteristic() == 2)
        : model(channel_model_t::bi_awgn),
          a(nrz.get_a()),
          b(nrz.get_b()),
          sigmasq(std::pow(transmission.get_standard_deviation(), 2.0)),
          pe(0.0),
          px(0.0) {}

    /**
     * @brief Construct a BI-AWGN calculator from a BI-AWGN channel (convenience)
     * @param bi_awgn Channel providing (a, b) and variance
     */
    LLRCalculator(const BI_AWGN& bi_awgn) noexcept
        requires(T::get_characteristic() == 2)
        : model(channel_model_t::bi_awgn),
          a(bi_awgn.get_encoder().get_a()),
          b(bi_awgn.get_encoder().get_b()),
          sigmasq(std::pow(bi_awgn.get_transmission().get_standard_deviation(), 2.0)),
          pe(0.0),
          px(0.0) {}

    /**
     * @brief Construct an SDMEC calculator from a discrete channel
     * @param channel Symmetric discrete memoryless erasure channel providing pe and px
     */
    LLRCalculator(const SDMEC<T>& channel) noexcept
        requires(T::get_size() == T::get_characteristic())
        : model(channel_model_t::sdmec), a(0.0), b(0.0), sigmasq(0.0), pe(channel.get_pe()), px(channel.get_px()) {}

    /**
     * @brief Compute the LLR of a received complex symbol (BI-AWGN model)
     * @param in Received complex symbol
     * @return LLR = b·(a − Re(in)) / σ² in nats; positive ⇒ bit 0, negative ⇒ bit 1
     * @throws std::logic_error if this calculator was constructed from an SDMEC
     */
    double operator()(const std::complex<double>& in) {
        if (model != channel_model_t::bi_awgn)
            throw std::logic_error("This LLRCalculator was not constructed for a soft input!");
        if (sigmasq == 0.0) return 0.0;  // degenerate zero-energy constellation (a = b = 0)
        return b * (a - in.real()) / sigmasq;
    }

    /**
     * @brief Compute LLRs for a received complex word (BI-AWGN model)
     * @param in Received complex word
     * @return Vector of binary LLRs
     * @throws std::logic_error if this calculator was constructed from an SDMEC
     */
    Vector<double> operator()(const Vector<std::complex<double>>& in) {
        Vector<double> llrs(in.get_n());
        for (size_t i = 0; i < in.get_n(); ++i) llrs.set_component(i, (*this)(in[i]));
        return llrs;
    }

    /**
     * @brief Compute the symbol-level LLR matrix of a received binary image (BI-AWGN model)
     * @param in Received complex matrix; the [T:𝔽_2]-row binary image of the code word, column j
     *           holding the bits of symbol j in `as_vector()`/@ref CECCO::DEMUX order
     * @return (q−1) × n matrix; entry (a−1, j) = ln(P(r_j|0)/P(r_j|a)) for symbol a = 1, …, q−1
     * @throws std::logic_error if this calculator was constructed from an SDMEC
     * @throws std::invalid_argument if the input does not have [T:𝔽_2] rows
     *
     * The bits of a symbol are transmitted over independent channel uses, so a symbol LLR is the
     * sum of the bit LLRs selected by its nonzero coordinates. For a binary code (T = 𝔽_2) this is
     * one LLR per symbol.
     */
    Matrix<double> operator()(const Matrix<std::complex<double>>& in) {
        if (model != channel_model_t::bi_awgn)
            throw std::logic_error("This LLRCalculator was not constructed for a soft input!");

        constexpr size_t q = T::get_size();
        constexpr size_t m = details::degree_over_prime_v<T>;
        if (in.get_m() != m)
            throw std::invalid_argument("BI-AWGN input must have [T:F2] rows (the binary image of the code symbols)!");

        const size_t n = in.get_n();
        Matrix<double> llrs(q - 1, n);
        std::array<double, m> bit;
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) bit[i] = (*this)(in(i, j));
            for (size_t lab = 1; lab < q; ++lab) {
                double L = 0.0;
                size_t bits = lab;
                for (size_t i = 0; i < m; ++i) {
                    if (bits & 1u) L += bit[m - 1 - i];
                    bits >>= 1u;
                }
                llrs.set_component(lab - 1, j, L);
            }
        }
        return llrs;
    }

    /**
     * @brief Compute symbol LLRs for one received field symbol (SDMEC model)
     * @param r Received 𝔽_p symbol, possibly erased
     * @return Vector with entry a−1 equal to ln(P(r|0)/P(r|a))
     * @throws std::logic_error if this calculator was constructed from a BI-AWGN channel
     */
    Vector<double> operator()(const T& r)
        requires(T::get_size() == T::get_characteristic())
    {
        if (model != channel_model_t::sdmec)
            throw std::logic_error("This LLRCalculator was not constructed for hard input!");

        constexpr size_t q = T::get_size();
        Vector<double> llrs(q - 1);

#ifdef CECCO_ERASURE_SUPPORT
        if (r.is_erased()) return llrs;
#endif

        const double L = sdmec_reliability();
        const size_t y = r.get_label();
        if (y == 0)
            for (size_t a = 1; a < q; ++a) llrs.set_component(a - 1, L);
        else
            llrs.set_component(y - 1, -L);

        return llrs;
    }

    /**
     * @brief Compute the symbol-LLR matrix of a received word (SDMEC model)
     * @param r Received word of 𝔽_p symbols, possibly erased
     * @return (p−1) × n matrix; entry (a−1, j) = ln(P(r_j|0)/P(r_j|a))
     * @throws std::logic_error if this calculator was constructed from a BI-AWGN channel
     */
    Matrix<double> operator()(const Vector<T>& r)
        requires(T::get_size() == T::get_characteristic())
    {
        if (model != channel_model_t::sdmec)
            throw std::logic_error("This LLRCalculator was not constructed for hard input!");

        constexpr size_t q = T::get_size();
        const size_t n = r.get_n();
        const double L = sdmec_reliability();

        Matrix<double> llrs(q - 1, n);
        for (size_t j = 0; j < n; ++j) {
#ifdef CECCO_ERASURE_SUPPORT
            if (r[j].is_erased()) continue;
#endif
            const size_t y = r[j].get_label();
            if (y == 0)
                for (size_t a = 1; a < q; ++a) llrs.set_component(a - 1, j, L);
            else
                llrs.set_component(y - 1, j, -L);
        }
        return llrs;
    }

   private:
    enum class channel_model_t { bi_awgn, sdmec };

    const channel_model_t model;
    const double a;        ///< DC offset from mapper (BI-AWGN)
    const double b;        ///< Constellation distance from mapper (BI-AWGN)
    const double sigmasq;  ///< Noise variance from channel (BI-AWGN)
    const double pe;       ///< Symbol error probability (SDMEC)
    const double px;       ///< Symbol erasure probability (SDMEC)

    double sdmec_reliability() const noexcept {
        if (pe == 0.0) {
            if (px == 1.0) return 0.0;
            return std::numeric_limits<double>::infinity();
        }
        if (1.0 - pe - px == 0.0) return -std::numeric_limits<double>::infinity();
        return std::log((1.0 - pe - px) * (static_cast<double>(T::get_size()) - 1.0) / pe);
    }
};

LLRCalculator(const NRZMapper&, const AWGN&) -> LLRCalculator<Fp<2>>;
LLRCalculator(const BI_AWGN&) -> LLRCalculator<Fp<2>>;
template <FiniteFieldType T>
    requires(T::get_size() == T::get_characteristic())
LLRCalculator(const SDMEC<T>&) -> LLRCalculator<T>;

/**
 * @brief Expand extension-field elements into subfield coordinates
 * @tparam E Extension field
 * @tparam S Subfield of E (S ⊆ E)
 *
 * A vector of `n` elements of E becomes an `[E:S] × n` matrix over S. Column `j`
 * contains the coefficients of `input[j]` in `as_vector()` construction-tower order.
 * For `S = 𝔽₂`, this is the binary image used by binary-input channels.
 *
 * @see @ref CECCO::MUX for the inverse
 */
template <FiniteFieldType E, FiniteFieldType S>
    requires(SubfieldOf<E, S>)
class DEMUX : private details::NonCopyable {
   public:
    constexpr DEMUX() = default;

    /**
     * @brief Expand a single element into its subfield-coefficient vector
     * @param in Extension field element
     * @return Length-[E:S] vector of S coefficients
     */
    constexpr Vector<S> operator()(const E& in) { return in.template as_vector<S>(); }

    /**
     * @brief Expand a vector of elements into a subfield-coefficient matrix
     * @param in Vector of n extension field elements
     * @return [E:S] × n matrix; column j holds the coefficients of `in[j]`
     */
    Matrix<S> operator()(const Vector<E>& in);
};

template <FiniteFieldType E, FiniteFieldType S>
    requires(SubfieldOf<E, S>)
Matrix<S> DEMUX<E, S>::operator()(const Vector<E>& in) {
    constexpr size_t m = details::degree_over_prime_v<E> / details::degree_over_prime_v<S>;
    if (in.get_n() == 0) return Matrix<S>(m, 0);

    Matrix<S> M(m, in.get_n());

    for (size_t j = 0; j < in.get_n(); ++j) {
        const auto column = in[j].template as_vector<S>();
        for (size_t i = 0; i < m; ++i) M.set_component(i, j, column[i]);
    }

    return M;
}

/**
 * @brief Field multiplexer: reconstruct 𝔽_E elements from 𝔽_S coefficients
 * @tparam S Subfield
 * @tparam E Extension of S
 *
 * Inverse of @ref DEMUX. The input matrix must have [E:S] rows; each column is interpreted
 * as the coefficient vector of one E element, and the output `Vector<E>` has one entry per
 * input column. `original >> DEMUX{} >> MUX{} == original` for compatible field pairs.
 *
 * @see @ref CECCO::DEMUX for the inverse direction
 * @see @ref CECCO::Ext, @ref CECCO::ExtensionOf
 */
template <FiniteFieldType S, FiniteFieldType E>
    requires(ExtensionOf<S, E>)
class MUX : private details::NonCopyable {
   public:
    constexpr MUX() = default;

    /**
     * @brief Reconstruct an element from its subfield-coefficient vector
     * @param in Length-[E:S] coefficient vector
     * @return Reconstructed extension field element
     */
    constexpr E operator()(const Vector<S>& in) { return E(in); }

    /**
     * @brief Reconstruct a vector of elements from a subfield-coefficient matrix
     * @param in [E:S] × n matrix of S coefficients
     * @return Length-n vector of E elements (column j of `in` becomes element j)
     */
    Vector<E> operator()(const Matrix<S>& in);
};

template <FiniteFieldType S, FiniteFieldType E>
    requires(ExtensionOf<S, E>)
Vector<E> MUX<S, E>::operator()(const Matrix<S>& in) {
    Vector<E> v(in.get_n());
    for (size_t i = 0; i < in.get_n(); ++i) v.set_component(i, E(in.get_col(i)));

    return v;
}

/**
 * @brief Function-call chaining: `x >> f` ≡ `f(x)`
 * @tparam LHS Input type
 * @tparam RHS Callable on `LHS`
 * @param lhs Input value (perfect-forwarded)
 * @param rhs Callable (perfect-forwarded)
 * @return `rhs(lhs)`
 *
 * Selected when `RHS` is invocable with `LHS`. Disabled for `std::ios_base`-derived types
 * to avoid clashing with stream operators. The companion overload below handles the case
 * where the right-hand side is an assignment target.
 */
template <class LHS, class RHS>
    requires(!std::is_base_of_v<std::ios_base, std::remove_cv_t<std::remove_reference_t<LHS>>>) &&
            requires(RHS&& r, LHS&& x) { std::forward<RHS>(r)(std::forward<LHS>(x)); }
decltype(auto) operator>>(LHS&& lhs, RHS&& rhs) {
    return std::forward<RHS>(rhs)(std::forward<LHS>(lhs));
}

/**
 * @brief Assignment chaining: `x >> dst` ≡ `dst = x; return dst`
 * @tparam LHS Input type
 * @tparam RHS Assignment target type (assignable from `LHS`)
 * @param lhs Input value
 * @param dst Destination variable
 * @return Reference to `dst` for continued chaining
 *
 * Selected only when the callable overload above is not viable, so block calls always win
 * over capture-into-variable when both are possible. Useful for capturing intermediate
 * results in a chain (`c >> map >> x >> awgn >> y >> demap >> r;`).
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
