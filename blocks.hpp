/**
 * @file blocks.hpp
 * @brief Communication system blocks library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.2.6
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
 * Channel models, modulation/demodulation, and field multiplexing blocks for error-control
 * coding experiments. Provided blocks:
 *
 * - **Channels**: SDMEC (errors-and-erasures over any 𝔽_q), SDMC, BSC, BEC, BAC, AWGN, BI-AWGN.
 * - **Modulation**: NRZ and BPSK with configurable constellation.
 * - **Demodulation**: hard-decision (NRZDemapper, BPSKDemapper) and soft-decision (LLRCalculator).
 * - **Field multiplexing**: DEMUX/MUX between an extension field and a subfield.
 * - **Chaining**: `operator>>` for left-to-right block composition.
 *
 * All blocks expose element-wise, vector, and matrix overloads via the @ref CECCO::details::BlockProcessor
 * CRTP base; see its documentation for the canonical chain example.
 *
 * @see @ref CECCO::details::BlockProcessor — CRTP foundation and chain example
 * @see @ref CECCO::SubfieldOf, @ref CECCO::ExtensionOf — concepts behind DEMUX/MUX
 */

#ifndef BLOCKS_HPP
#define BLOCKS_HPP

#include <numbers>

#include "fields.hpp"
#include "vectors.hpp"

/*
// transitive
#include <algorithm>
#include <cmath>
#include <complex>
#include <ios>
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
 * @brief CRTP base providing element-wise, vector, and matrix `operator()` overloads
 * @tparam T Derived class (CRTP)
 * @tparam InputType Element type accepted by the block
 * @tparam OutputType Element type produced by the block
 *
 * Derived blocks implement a single-element `operator()(const InputType&)`; this base then
 * generates `Vector<OutputType> operator()(const Vector<InputType>&)` and the matrix
 * counterpart (plus rvalue versions, with in-place reuse when InputType == OutputType).
 *
 * Catch-all overloads accept any other @ref CECCO::FiniteFieldType and throw at runtime —
 * useful inside `if constexpr (std::is_same_v<F, InputType>)` guards in non-template
 * contexts, where the discarded branch must still compile.
 *
 * @code{.cpp}
 * Vector<Fp<2>> message = {1, 0, 1, 1};
 * BPSKMapper map;
 * AWGN awgn(6.0, map.get_a(), map.get_b());
 * BPSKDemapper demap;
 *
 * Vector<Fp<2>> r;
 * message >> map >> awgn >> demap >> r;
 * @endcode
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
     * @return Vector of processed outputs
     */
    Vector<OutputType> operator()(Vector<InputType>&& in) {
        const size_t n = in.get_n();
        Vector<OutputType> res(n);
        for (size_t i = 0; i < n; ++i) res.set_component(i, derived()(std::move(in[i])));
        return res;
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
        throw std::logic_error("BlockProcessor can only accept inputs from " + InputType::get_info());
    }

    template <FiniteFieldType U>
        requires(!std::is_same_v<U, InputType>)
    Vector<OutputType> operator()(const Vector<U>&) {
        throw std::logic_error("BlockProcessor can only accept inputs from " + InputType::get_info());
    }

    template <FiniteFieldType U>
        requires(!std::is_same_v<U, InputType>)
    Matrix<OutputType> operator()(const Matrix<U>&) {
        throw std::logic_error("BlockProcessor can only accept inputs from " + InputType::get_info());
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

/** @brief CRTP base for blocks producing log-likelihood ratios from complex symbols. */
template <typename T>
using LLRProcessor = BlockProcessor<T, std::complex<double>, double>;

}  // namespace details

/**
 * @brief Symmetric Discrete Memoryless Erasure Channel (SDMEC) over any finite field 𝔽_q
 * @tparam T Finite field type for channel input/output symbols
 *
 * q-ary symmetric channel with independent error and erasure processes. Each transmitted
 * symbol is changed to a uniformly random different value with observed probability pe,
 * marked erased (overwriting any error) with observed probability px, and otherwise passed
 * through unchanged.
 *
 * For px = 0 this reduces to the traditional symmetric channel (use @ref CECCO::SDMC, or
 * @ref CECCO::BSC for q = 2). For pe = 0 it is the erasure channel (use @ref CECCO::BEC for
 * q = 2). Erasure support requires the @ref CECCO_ERASURE_SUPPORT macro at compile time.
 *
 * Capacity (bits/symbol):
 *   C = (1 − px) · [ log₂(q) + (1 − p̃)·log₂(1 − p̃) + p̃·log₂(p̃) − p̃·log₂(q − 1) ],
 * where p̃ = pe / (1 − px) is the conditional error probability.
 *
 * @code{.cpp}
 * using F4 = Ext<Fp<2>, {1, 1, 1}>;
 * SDMEC<F4> channel(0.05, 0.1);                    // 5% errors, 10% erasures
 * Vector<F4> r = Vector<F4>(20).randomize() >> channel;
 * double C = channel.get_capacity();
 * @endcode
 *
 * @see @ref CECCO::SDMC, @ref CECCO::BSC, @ref CECCO::BEC, @ref CECCO::BAC
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
    std::geometric_distribution<unsigned long long> error_dist;
    unsigned long long error_trials{0};
    unsigned long long error_failures_before_hit;
#ifdef CECCO_ERASURE_SUPPORT
    std::geometric_distribution<unsigned long long> erasure_dist;
    unsigned long long erasure_trials{0};
    unsigned long long erasure_failures_before_hit;
#endif
};

template <FieldType T>
SDMEC<T>::SDMEC(double pe, double px) : pe(pe), px(px) {
#ifndef CECCO_ERASURE_SUPPORT
    if (px != 0.0) throw std::invalid_argument("px != 0 requires CECCO_ERASURE_SUPPORT");
#endif
    if (pe < 0.0 || pe > 1.0)
        throw std::out_of_range("SDMEC error probability must be in [0,1], got: " + std::to_string(pe));
    if (px < 0.0 || px > 1.0 - pe)
        throw std::out_of_range("SDMEC erasure probability must be in [0," + std::to_string(1.0 - pe) +
                                "], got: " + std::to_string(px));

    const double p_error_given_not_erased = (px == 1.0) ? 0.0 : pe / (1.0 - px);
    error_dist = std::geometric_distribution<unsigned long long>(p_error_given_not_erased);
    error_failures_before_hit = error_dist(gen());
#ifdef CECCO_ERASURE_SUPPORT
    erasure_dist = std::geometric_distribution<unsigned long long>(px);
    if (px > 0.0) {
        erasure_failures_before_hit = erasure_dist(gen());
    }
#endif
}

template <FieldType T>
T SDMEC<T>::operator()(const T& in) {
    if (error_dist.p() == 0.0 && px == 0.0) return in;
    T res(in);
    if (error_trials == error_failures_before_hit) {
        res.randomize_force_change();
        error_trials = 0;
        error_failures_before_hit = error_dist(gen());
    } else {
        ++error_trials;
    }
#ifdef CECCO_ERASURE_SUPPORT
    if (px == 0.0) return res;
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

template <FieldType T>
double SDMEC<T>::get_capacity() const noexcept
    requires(FiniteFieldType<T>)
{
    const double ptilde = error_dist.p();
    const double q = static_cast<double>(T::get_size());

    const double term1 = (ptilde > 0.0 && ptilde < 1.0) ? ptilde * std::log2(ptilde) : 0.0;
    const double term2 = (ptilde > 0.0 && ptilde < 1.0) ? (1.0 - ptilde) * std::log2(1.0 - ptilde) : 0.0;

    return (std::log2(q) + term2 + term1 - ptilde * std::log2(q - 1.0)) * (1.0 - px);
}

/**
 * @brief Symmetric Discrete Memoryless Channel — errors only, no erasures
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
 * @brief Binary Symmetric Channel — type alias for `SDMC<Fp<2>>`
 *
 * Bits are flipped with probability pe.
 *
 * @see @ref CECCO::SDMC, @ref CECCO::BEC
 */
using BSC = SDMC<Fp<2>>;

/**
 * @brief Binary Erasure Channel — symbols are received correctly or marked erased
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
 * @brief Binary Asymmetric Channel (Z-channel) — 0 is preserved; 1 flips to 0 with probability p
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
     * @throws std::out_of_range if p ∉ [0, 1] or 0 < p < 1e−9
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
 * @brief Binary Phase Shift Keying mapper — `NRZMapper` with a = 0, b = 2
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
          dist(0, std::sqrt(0.5 * Eb / std::pow(10.0, EbNodB / 10.0))),
          pe(calculate_pe(EbNodB, Eb, b)) {}

    /** @name Noise Parameters
     * @{
     */

    /** @brief Noise variance σ² (per real component). */
    double get_variance() const noexcept {
        const double s = dist.stddev();
        return s * s;
    }

    /** @brief Noise standard deviation σ (per real component). */
    double get_standard_deviation() const noexcept { return dist.stddev(); }

    /** @brief Theoretical hard-decision bit error probability for the configured constellation/SNR. */
    constexpr double get_pe() const noexcept { return pe; }

    /** @} */

    /**
     * @brief Add Gaussian noise to a complex symbol
     * @param in Input symbol
     * @return Symbol with independent Gaussian noise added to each real component
     */
    std::complex<double> operator()(const std::complex<double>& in) {
        std::complex<double> res(in.real() + dist(gen()), in.imag() + dist(gen()));
        return res;
    }

    /**
     * @brief Shannon capacity in bits per symbol (real signaling)
     * @return C = ½·log₂(1 + Eb/σ²) for one real dimension (NRZ/BPSK).
     *
     * For complex signaling with noise on both I and Q the formula is C = log₂(1 + Eb/(2σ²)).
     */
    double get_capacity() const noexcept { return 1 / 2.0 * std::log2(1 + Eb / get_variance()); }

   private:
    const double Eb{};
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
 * @brief Binary-Input AWGN — fused @ref NRZMapper + @ref AWGN block
 *
 * Maps binary inputs through an internal NRZMapper and then through AWGN, yielding noisy
 * complex symbols ready for hard decision (@ref NRZDemapper) or soft decision
 * (@ref LLRCalculator). Default parameters give BPSK (a = 0, b = 2).
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
 * @brief BPSK hard-decision demapper — `NRZDemapper` with threshold 0
 *
 * @see @ref CECCO::BPSKMapper, @ref CECCO::NRZDemapper
 */
class BPSKDemapper : public NRZDemapper {
   public:
    /** @brief Construct BPSK demapper (threshold 0). */
    constexpr BPSKDemapper() noexcept : NRZDemapper(BPSKMapper()) {}
};

/**
 * @brief Log-Likelihood Ratio calculator for NRZ-over-AWGN soft demodulation
 *
 * Computes LLR(r) = b·(a − Re(r)) / σ² in nats, where (a, b) are the NRZ constellation
 * parameters and σ² is the AWGN noise variance. Sign convention: positive LLR ⇒ bit 0,
 * negative ⇒ bit 1; magnitude indicates reliability. Output suitable for belief
 * propagation, LDPC, turbo, and other soft-decision decoders.
 *
 * Construct from a matching @ref NRZMapper + @ref AWGN pair, or directly from a
 * @ref BI_AWGN. See the @ref BI_AWGN class doc for an end-to-end chain example.
 *
 * @see @ref CECCO::NRZDemapper for the hard-decision counterpart
 */
class LLRCalculator : public details::LLRProcessor<LLRCalculator> {
   public:
    // Bring base class operator() overloads into scope
    using details::LLRProcessor<LLRCalculator>::operator();

    /**
     * @brief Construct from a matching mapper and channel
     * @param nrz Source mapper providing (a, b)
     * @param transmission Channel providing σ
     */
    LLRCalculator(const NRZMapper& nrz, const AWGN& transmission) noexcept
        : a(nrz.get_a()), b(nrz.get_b()), sigmasq(std::pow(transmission.get_standard_deviation(), 2.0)) {}

    /**
     * @brief Construct from a BI-AWGN channel (convenience)
     * @param bi_awgn Channel providing (a, b) and σ
     */
    LLRCalculator(const BI_AWGN& bi_awgn) noexcept
        : a(bi_awgn.get_encoder().get_a()),
          b(bi_awgn.get_encoder().get_b()),
          sigmasq(std::pow(bi_awgn.get_transmission().get_standard_deviation(), 2.0)) {}

    /**
     * @brief Compute LLR of a received symbol
     * @param in Received complex symbol
     * @return LLR = b·(a − Re(in)) / σ² in nats; positive ⇒ bit 0, negative ⇒ bit 1
     */
    double operator()(const std::complex<double>& in) noexcept { return b * (a - in.real()) / sigmasq; }

   private:
    const double a{};        ///< DC offset from mapper
    const double b{};        ///< Constellation distance from mapper
    const double sigmasq{};  ///< Noise variance from channel
};

/**
 * @brief Field demultiplexer — expand 𝔽_E elements into 𝔽_S coefficient vectors/matrices
 * @tparam E Extension field
 * @tparam S Subfield of E (S ⊆ E)
 *
 * Each E element decomposes into [E:S] coefficients over S via the polynomial basis of the
 * extension. For a `Vector<E>` of length n, the resulting `Matrix<S>` has [E:S] rows and n
 * columns; column j holds the coefficients of element j.
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;          // 𝔽_4 = 𝔽_2[x]/(x² + x + 1)
 * DEMUX<F4, F2> demux;
 *
 * Vector<F4> v = {F4(0), F4(1), F4(2), F4(3)};
 * Matrix<F2> M = v >> demux;              // 2×4 matrix over 𝔽_2
 * @endcode
 *
 * @see @ref CECCO::MUX for the inverse
 * @see @ref CECCO::Ext, @ref CECCO::SubfieldOf
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
    if (in.get_n() == 0) return Matrix<S>(0, 0);

    auto temp = in[0].template as_vector<S>();
    Matrix<S> M(temp.get_n(), in.get_n());

    M.set_submatrix(0, 0, transpose(Matrix<S>(std::move(temp))));

    for (size_t i = 1; i < in.get_n(); ++i) {
        auto col_vector = in[i].template as_vector<S>();
        M.set_submatrix(0, i, transpose(Matrix<S>(std::move(col_vector))));
    }

    return M;
}

/**
 * @brief Field multiplexer — reconstruct 𝔽_E elements from 𝔽_S coefficients
 * @tparam S Subfield
 * @tparam E Extension of S
 *
 * Inverse of @ref DEMUX. The input matrix must have [E:S] rows; each column is interpreted
 * as the coefficient vector of one E element, and the output `Vector<E>` has one entry per
 * input column. `original >> DEMUX{} >> MUX{} == original` for compatible field pairs.
 *
 * @see @ref CECCO::DEMUX for the inverse direction and a usage example
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
