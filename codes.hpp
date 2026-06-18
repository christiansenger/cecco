/**
 * @file codes.hpp
 * @brief Error control codes library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.3.0
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026, Christian Senger <senger@inue.uni-stuttgart.de>
 *
 * Licensed for noncommercial use only, including academic teaching, research, and personal non-profit purposes.
 * Commercial use is prohibited without a separate commercial license. See the [LICENSE](../../LICENSE) file in the
 * repository root for full terms and how to request a commercial license.
 */

#ifndef CODES_HPP
#define CODES_HPP

#include <bit>

#include "code_bounds.hpp"
#include "graphs.hpp"
/*
// transitive
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fields.hpp"
#include "helpers.hpp"
#include "matrices.hpp"
#include "polynomials.hpp"
#include "vectors.hpp"
*/

#define BOLD(x) "\033[1m" x "\033[0m"

// Docu note: default cap on the number of belief-propagation iterations. It is the default
// value of the max_iterations parameter on the base-class dec_BP virtuals in Code, hence also the
// cap used when decoding through Dec with method_t::BP. Override with -DBP_MAX_ITERATIONS=<value>.
#ifndef BP_MAX_ITERATIONS
#define BP_MAX_ITERATIONS 100
#endif

namespace CECCO {

template <FiniteFieldType T>
Polynomial<InfInt> MacWilliamsIdentity(const Polynomial<InfInt>& A, size_t n, size_t k) {
    constexpr size_t q = T::get_size();

    auto a = Vector<InfInt>(n + 1);
    for (size_t i = 0; i < n + 1; ++i) a.set_component(i, A[i]);

    Matrix<InfInt> K(n + 1, n + 1);
    for (size_t i = 0; i < n + 1; ++i) {
        if (a[i] == 0) continue;
        for (size_t j = 0; j < n + 1; ++j) {
            InfInt sum = 0;
            // for (size_t h = 0; h <= std::min<size_t>(i, j); ++h) {
            for (size_t h = 0; h <= j; ++h) {
                /*
                 sum += sqm<InfInt>(-1, h) * bin<InfInt>(i, h) * bin<InfInt>(n - i, j - h) *
                        sqm<InfInt>(q - 1, j - h);
                // the following code calculates exactly this expression - with a couple of
                // optimizations in order to avoid unnecessary computations
                */
                InfInt a = bin<InfInt>(n - i, j - h);
                if (a != 0) {
                    const auto b = bin<InfInt>(i, h);
                    if (b != 0) {
                        const auto c = sqm<InfInt>(q - 1, j - h);
                        if (h % 2)
                            sum -= a * b * c;
                        else
                            sum += a * b * c;
                    }
                }
            }
            K.set_component(i, j, sum);
        }
    }
    return Polynomial<InfInt>(Vector<InfInt>(a) * Matrix<InfInt>(K) / sqm<InfInt>(q, k));
}

namespace details {

inline const int index = std::ios_base::xalloc();

}  // namespace details

inline std::ostream& showbasic(std::ostream& os) {
    os.iword(details::index) = 0;
    return os;
}

inline std::ostream& showmost(std::ostream& os) {
    os.iword(details::index) = 1;
    return os;
}

inline std::ostream& showall(std::ostream& os) {
    os.iword(details::index) = 2;
    return os;
}

inline std::ostream& showspecial(std::ostream& os) {
    os.iword(details::index) = 3;
    return os;
}

template <ComponentType T>
class Code {
   public:
    Code(size_t n) noexcept : n(n) {}

    Code(const Code& other) noexcept : n(other.n) {}

    Code(Code&&) noexcept = default;

    virtual ~Code() = default;

    Code& operator=(const Code& other) noexcept {
        if (this != &other) {
            n = other.n;
        }
        return *this;
    }

    Code& operator=(Code&&) noexcept = default;

    size_t get_n() const noexcept { return n; }

    virtual void get_info(std::ostream&) const {};
    virtual Vector<T> enc(const Vector<T>&) const { throw std::logic_error("Encoding not supported for this code!"); }
    virtual Vector<T> encinv(const Vector<T>&) const {
        throw std::logic_error("Inverse encoding not supported for this code!");
    }
    virtual Vector<T> dec_BD(const Vector<T>&) const {
        throw std::logic_error("Bounded distance decoding not supported for this code!");
    }
    virtual Vector<T> dec_boosted_BD(const Vector<T>&) const {
        throw std::logic_error("Boosted bounded distance decoding not supported for this code!");
    }
    virtual Vector<T> dec_ML(const Vector<T>&) const {
        throw std::logic_error("ML decoding not supported for this code!");
    }
    virtual Vector<T> dec_Viterbi(const Vector<T>&, const std::string& = "") const {
        throw std::logic_error("Viterbi decoding not supported for this code!");
    }
    virtual Vector<T> dec_recursive(const Vector<T>&) const {
        throw std::logic_error("Recursive decoding not supported for this code!");
    }
    virtual Vector<T> dec_Meggitt(const Vector<T>&) const {
        throw std::logic_error("Meggitt decoding not supported for this code!");
    }
    virtual Vector<T> dec_WBA(const Vector<T>&) const {
        throw std::logic_error("Welch-Berlekamp decoding not supported for this code!");
    }
    virtual Vector<T> dec_BMA(const Vector<T>&) const {
        throw std::logic_error("Berlekamp-Massey decoding not supported for this code!");
    }
    // Docu note: soft-input decoders take an LLR matrix with [T:F_p]·(p−1) rows and one
    // column per code symbol, coordinate rows in as_vector()/DEMUX order (highest level of the
    // construction tower first), cf. details::symbol_costs_from_llrs in graphs.hpp; for binary
    // codes a single row. The Vector<double> overloads are the binary special case.
    virtual Vector<T> dec_ML_soft(const Vector<double>&, size_t) const {
        throw std::logic_error("Soft-input ML decoding not supported for this code!");
    }
    virtual Vector<T> dec_ML_soft(const Matrix<double>&, size_t) const {
        throw std::logic_error("Soft-input ML decoding not supported for this code!");
    }
    virtual Vector<T> dec_Viterbi_soft(const Vector<double>&, const std::string& = "") const {
        throw std::logic_error("Soft-input Viterbi decoding not supported for this code!");
    }
    virtual Vector<T> dec_Viterbi_soft(const Matrix<double>&, const std::string& = "") const {
        throw std::logic_error("Soft-input Viterbi decoding not supported for this code!");
    }
    // Docu note: pass nullptr as Lambda when only the filename argument is needed. Lambda
    // receives the (q−1)×n a-posteriori LLR matrix; entry (a−1, i) holds L_i(a) =
    // ln(P(c_i=0|r)/P(c_i=a|r)) for a = 1, …, q−1 by label. For binary codes this is a single row.
    virtual Vector<T> dec_BCJR(const Vector<double>&, Matrix<double>* = nullptr, const std::string& = "") const {
        throw std::logic_error("BCJR decoding not supported for this code!");
    }
    virtual Vector<T> dec_BCJR(const Matrix<double>&, Matrix<double>* = nullptr, const std::string& = "") const {
        throw std::logic_error("BCJR decoding not supported for this code!");
    }
    // Docu note: belief propagation runs exact log-domain sum-product on the Tanner graph
    // (via details::max_star), not min-sum or any other approximation. With max_iterations = 0 it
    // returns the intrinsic (channel-only) hard decision and posterior LLRs.
    // Docu note: the max_iterations default BP_MAX_ITERATIONS (defined at the top of this
    // file) lives only on these base virtuals. Virtual default arguments are bound from the static
    // type of the call, so this is the value Dec uses, as it decodes through a Code reference.
    virtual Vector<T> dec_BP(const Vector<double>&, size_t = BP_MAX_ITERATIONS, Matrix<double>* = nullptr,
                             const std::string& = "", size_t* = nullptr) const {
        throw std::logic_error("Belief-propagation decoding not supported for this code!");
    }
    virtual Vector<T> dec_BP(const Matrix<double>&, size_t = BP_MAX_ITERATIONS, Matrix<double>* = nullptr,
                             const std::string& = "", size_t* = nullptr) const {
        throw std::logic_error("Belief-propagation decoding not supported for this code!");
    }
#ifdef CECCO_ERASURE_SUPPORT
    virtual Vector<T> dec_BD_EE(const Vector<T>&) const {
        throw std::logic_error("BD error/erasure decoding not supported for this code!");
    }
    virtual Vector<T> dec_ML_EE(const Vector<T>&) const {
        throw std::logic_error("ML error/erasure decoding not supported for this code!");
    }
    virtual Vector<T> dec_Viterbi_EE(const Vector<T>&, const std::string& = "") const {
        throw std::logic_error("Viterbi error/erasure decoding not supported for this code!");
    }
    virtual Vector<T> dec_recursive_EE(const Vector<T>&) const {
        throw std::logic_error("Recursive error/erasure decoding not supported for this code!");
    }
    virtual Vector<T> dec_WBA_EE(const Vector<T>&) const {
        throw std::logic_error("Welch-Berlekamp error/erasure decoding not supported for this code!");
    }
    virtual Vector<T> dec_BMA_EE(const Vector<T>&) const {
        throw std::logic_error("Berlekamp-Massey error/erasure decoding not supported for this code!");
    }
#endif

   protected:
    size_t n;
#ifdef CECCO_ERASURE_SUPPORT
    std::vector<size_t> erasure_positions(const Vector<T>& r) const {
        std::vector<size_t> X;
        for (size_t i = 0; i < n; ++i) {
            if (r[i].is_erased()) X.push_back(i);
        }
        return X;
    }
#endif
};

template <ComponentType T>
class EmptyCode : public Code<T> {
   public:
    EmptyCode(size_t n) noexcept : Code<T>(n) {}

    EmptyCode(const EmptyCode&) noexcept = default;
    EmptyCode(EmptyCode&&) noexcept = default;
    EmptyCode& operator=(const EmptyCode&) noexcept = default;
    EmptyCode& operator=(EmptyCode&&) noexcept = default;

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) > 0) {
            Code<T>::get_info(os);
            os << "Empty code";
        }
    }
};

template <FieldType T>
class LinearCode;

template <FiniteFieldType T>
class UniverseCode;

template <FiniteFieldType T>
class ZeroCode;

template <FiniteFieldType T>
class SimplexCode;

template <FiniteFieldType T>
class SingleParityCheckCode;

template <FieldType T, class B>
    requires std::derived_from<B, LinearCode<T>>
class ExtendedCode;

template <FiniteFieldType T>
class CodewordIterator {
    friend bool operator==(const CodewordIterator& a, const CodewordIterator& b) { return a.counter == b.counter; }
    friend bool operator!=(const CodewordIterator& a, const CodewordIterator& b) { return !(a == b); }

   public:
    // Required for STL compatibility
    using iterator_category = std::forward_iterator_tag;
    using value_type = Vector<T>;
    using difference_type = std::ptrdiff_t;
    using pointer = const Vector<T>*;
    using reference = const Vector<T>&;

    CodewordIterator(const LinearCode<T>& C, InfInt s) : C(C), counter(s), u(C.get_k()) {
        constexpr size_t q = T::get_size();
        if (s < C.get_size()) {  // for cend()
            const size_t k = C.get_k();
            for (size_t i = 0; i < k; ++i) {
                u.set_component(i, T((s % q).toInt()));
                s /= q;
            }
            c = u * C.get_G();
        }
    }

    const Vector<T>& operator*() const noexcept { return c; }

    CodewordIterator& operator++() {
        constexpr size_t q = T::get_size();
        const size_t k = C.get_k();
        ++counter;
        if (counter < C.get_size()) {
            auto j = counter;
            for (size_t i = 0; i < k; ++i) {
                const auto quot = j / q;
                const auto rem = (j % q).toInt();
                u.set_component(i, T(rem));
                j = quot;
            }
            c = u * C.get_G();
        }
        return *this;
    }

   private:
    const LinearCode<T>& C;
    InfInt counter;
    Vector<T> u;
    Vector<T> c;
};

class decoding_failure : public std::exception {
   public:
    decoding_failure(const std::string& message) : message(message) {}
    const char* what() const noexcept override { return message.c_str(); }

   private:
    const std::string message;
};

template <FieldType T>
class LinearCode : public Code<T> {
   public:
    // expose base field (used by the extend() factories)
    using FIELD = T;

    // Docu note: X is interpreted as generator matrix G if it has k rows, or as
    // parity-check matrix H if it has n-k rows. If k == n-k, the ambiguity is resolved in favor
    // of G; callers who mean H must pass the dual code's generator instead.
    LinearCode(size_t n, size_t k, const Matrix<T>& X) : Code<T>(n), k(k), MI(k, k) {
        if (X.get_n() != this->n) throw std::invalid_argument("G must have " + std::to_string(this->n) + " columns");
        if (k == 0) {
            if (!X.is_zero() && !X.is_invertible())
                throw std::invalid_argument("Cannot construct linear code: G must be a zero matrix");
            G = ZeroMatrix<T>(1, n);
            HT = IdentityMatrix<T>(n);
            return;
        }

        if (X.get_m() == k) {
            // X supposed to be generator matrix G
            if (X.rank() != k)
                throw std::invalid_argument("Cannot construct linear code: G must have full rank " + std::to_string(k));
            G = X;
            HT = G.basis_of_nullspace().rref().transpose();
            const auto Gp = rref(G);
            size_t i = 0;
            for (size_t j = 0; j < k; ++j) {
                const auto u = unit_vector<T>(k, j);
                while (Gp.get_col(i) != u) ++i;
                infoset.push_back(i);
            }
        } else if (X.get_m() == this->n - k) {
            // X supposed to be parity check matrix H
            if (X.rank() != this->n - k)
                throw std::invalid_argument("Cannot construct linear code: H must have full rank " +
                                            std::to_string(this->n - k));
            G = X.basis_of_nullspace();
            HT = transpose(X);
            G.rref();
            size_t i = 0;
            for (size_t j = 0; j < k; ++j) {
                const auto u = unit_vector<T>(k, j);
                while (G.get_col(i) != u) ++i;
                infoset.push_back(i);
            }
        } else {
            throw std::invalid_argument("Cannot construct linear code: matrix must have " + std::to_string(k) +
                                        " rows (as G) or " + std::to_string(this->n - k) + " rows (as H), got " +
                                        std::to_string(X.get_m()));
        }
        size_t j = 0;
        for (auto it = infoset.cbegin(); it != infoset.cend(); ++it) {
            MI.set_submatrix(0, j, G.get_submatrix(0, *it, k, 1));
            ++j;
        }
        MI.invert();
    }

    LinearCode(size_t k, Polynomial<T> gamma) try
        : LinearCode(
              k + gamma.degree(), k,
              ToeplitzMatrix(pad_back(pad_front(Vector<T>(gamma), k + gamma.degree()), 2 * k + gamma.degree() - 1), k,
                             k + gamma.degree())) {
        set_gamma(std::move(gamma));
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct linear code from polynomial: ") + e.what());
    }

    LinearCode(const std::vector<Vector<T>>& codewords) : LinearCode(G_from_codewords(codewords)) {}

    LinearCode(const LinearCode& other)
        : Code<T>(other),
          k(other.k),
          G(other.G),
          HT(other.HT),
          MI(other.MI),
          infoset(other.infoset),
          dmin(other.dmin),
          weight_enumerator(other.weight_enumerator),
          p_ary_image_weight_enumerator(other.p_ary_image_weight_enumerator),
          codewords(other.codewords),
          standard_array(other.standard_array),
          tainted(other.tainted),
          Meggitt_table(other.Meggitt_table),
#ifdef CECCO_ERASURE_SUPPORT
          punctured_codes_BD(other.punctured_codes_BD),
          punctured_codes_ML(other.punctured_codes_ML),
#endif
          polynomial(other.polynomial),
          gamma(other.gamma),
          minimal_trellis(other.minimal_trellis),
          tanner_graph(other.tanner_graph) {
    }

    LinearCode(LinearCode&& other)
        : Code<T>(std::move(other)),
          k(other.k),
          G(std::move(other.G)),
          HT(std::move(other.HT)),
          MI(std::move(other.MI)),
          infoset(std::move(other.infoset)),
          dmin(std::move(other.dmin)),
          weight_enumerator(std::move(other.weight_enumerator)),
          p_ary_image_weight_enumerator(std::move(other.p_ary_image_weight_enumerator)),
          codewords(std::move(other.codewords)),
          standard_array(std::move(other.standard_array)),
          tainted(std::move(other.tainted)),
          Meggitt_table(std::move(other.Meggitt_table)),
#ifdef CECCO_ERASURE_SUPPORT
          punctured_codes_BD(std::move(other.punctured_codes_BD)),
          punctured_codes_ML(std::move(other.punctured_codes_ML)),
#endif
          polynomial(std::move(other.polynomial)),
          gamma(std::move(other.gamma)),
          minimal_trellis(std::move(other.minimal_trellis)),
          tanner_graph(std::move(other.tanner_graph)) {
    }

    LinearCode& operator=(const LinearCode& other) {
        if (this != &other) {
            Code<T>::operator=(other);
            k = other.k;
            G = other.G;
            HT = other.HT;
            MI = other.MI;
            infoset = other.infoset;
            dmin = other.dmin;
            weight_enumerator = other.weight_enumerator;
            p_ary_image_weight_enumerator = other.p_ary_image_weight_enumerator;
            codewords = other.codewords;
            standard_array = other.standard_array;
            tainted = other.tainted;
            Meggitt_table = other.Meggitt_table;
#ifdef CECCO_ERASURE_SUPPORT
            punctured_codes_BD = other.punctured_codes_BD;
            punctured_codes_ML = other.punctured_codes_ML;
#endif
            polynomial = other.polynomial;
            gamma = other.gamma;
            minimal_trellis = other.minimal_trellis;
            tanner_graph = other.tanner_graph;
        }
        return *this;
    }

    LinearCode& operator=(LinearCode&& other) {
        if (this != &other) {
            Code<T>::operator=(std::move(other));
            k = other.k;
            G = std::move(other.G);
            HT = std::move(other.HT);
            MI = std::move(other.MI);
            infoset = std::move(other.infoset);
            dmin = std::move(other.dmin);
            weight_enumerator = std::move(other.weight_enumerator);
            p_ary_image_weight_enumerator = std::move(other.p_ary_image_weight_enumerator);
            codewords = std::move(other.codewords);
            standard_array = std::move(other.standard_array);
            tainted = std::move(other.tainted);
            Meggitt_table = std::move(other.Meggitt_table);
#ifdef CECCO_ERASURE_SUPPORT
            punctured_codes_BD = std::move(other.punctured_codes_BD);
            punctured_codes_ML = std::move(other.punctured_codes_ML);
#endif
            polynomial = std::move(other.polynomial);
            gamma = std::move(other.gamma);
            minimal_trellis = std::move(other.minimal_trellis);
            tanner_graph = std::move(other.tanner_graph);
        }
        return *this;
    }

    size_t get_k() const noexcept { return k; }
    double get_R() const noexcept { return static_cast<double>(k) / this->n; }

    InfInt get_size() const
        requires FiniteFieldType<T>
    {
        return sqm<InfInt>(T::get_size(), k);
    }

    const Matrix<T>& get_G() const noexcept { return G; }
    const Matrix<T>& get_HT() const noexcept { return HT; }
    Matrix<T> get_H() const { return transpose(HT); }

    virtual size_t get_dmin() const {
        dmin.call_once([this] {
            if (dmin.has_value()) return;
            if (k == 0) throw std::logic_error("Cannot calculate dmin of a dimension zero code!");

            // if weight enumerator is calculated, use it...
            if (weight_enumerator.has_value()) {
                if constexpr (FiniteFieldType<T>) {
                    for (size_t i = 1; i <= weight_enumerator.value().degree(); ++i) {
                        if (weight_enumerator.value()[i] != 0) {
                            dmin = i;
                            return;
                        }
                    }
                }
                // ... otherwise:
            } else {
                if (k == 1) {
                    dmin = G.get_row(0).wH();
                    return;
                }

                // if less than 100 codewords calculate weight enumerator (even when not required)
                if constexpr (FiniteFieldType<T>) {
                    if (get_size() < 100) {
                        get_weight_enumerator();
                        for (size_t i = 1; i <= weight_enumerator.value().degree(); ++i) {
                            if (weight_enumerator.value()[i] != 0) {
                                dmin = i;
                                return;
                            }
                        }
                    }
                }

                std::clog
                    << "--> Calculating dmin, this requires finding minimal number of linearly dependent columns in H"
                    << std::endl;
                // find min. number of linearly dependent rows of HT
                for (size_t d = 1; d <= this->n - k + 1; ++d) {
                    Matrix<T> M(d, this->n - k);
                    std::vector<bool> selection(this->n);  // zero-initialized
                    std::fill(selection.begin() + this->n - d, selection.end(), true);

                    do {
                        size_t i = 0;
                        for (size_t j = 0; j < this->n; ++j) {
                            if (selection[j]) {
                                M.set_submatrix(i, 0, HT.get_submatrix(j, 0, 1, this->n - k));
                                ++i;
                            }
                        }
                        if (M.rank() < d) {
                            dmin = d;
                            return;
                        }
                    } while (next_permutation(selection.begin(), selection.end()));
                }
            }
        });

        return dmin.value();
    }

    size_t get_tmax() const {
        if (k == 0) throw std::logic_error("Cannot calculate tmax of a dimension zero code!");
        return (get_dmin() - 1) / 2;
    }

    virtual const Polynomial<InfInt>& get_weight_enumerator() const {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Cannot calculate weight enumerator of code over infinite field!");
        } else {
            weight_enumerator.call_once([this] {
                constexpr size_t q = T::get_size();
                if (weight_enumerator.has_value()) return;
                if (k == 0) {
                    weight_enumerator.emplace(Polynomial<InfInt>(1));
                } else if (k <= this->n - k) {  // calculate directly
                    std::clog << "--> Calculating weight enumerator, this requires iterating through "
                              << sqm<InfInt>(q, k) << " codewords" << std::endl;
                    weight_enumerator.emplace(ZeroPolynomial<InfInt>());
                    for (auto it = cbegin(); it != cend(); ++it)
                        weight_enumerator.value().add_to_coefficient(wH(*it), 1);
                } else if (k == this->n) {
                    UniverseCode<T> Cp(this->n);
                    weight_enumerator.emplace(Cp.get_weight_enumerator());
                } else {  // calculate based on dual code and MacWilliams identity
                    std::clog << "--> Using MacWilliams' identity for: " << std::endl;
                    LinearCode<T> Cp(this->n, this->n - k, transpose(HT));
                    weight_enumerator.emplace(MacWilliamsIdentity<T>(Cp.get_weight_enumerator(), this->n, this->n - k));
                }
            });

            return weight_enumerator.value();
        }
    }

    // Docu note: weight enumerator of the p-ary image code (length n·[T:F_p], dimension
    // k·[T:F_p] over F_p, p the characteristic), i.e. codeword weights counted in prime-field
    // coordinates of the as_vector()/DEMUX expansion. For codes over prime fields image and code
    // coincide.
    const Polynomial<InfInt>& get_p_ary_image_weight_enumerator() const
        requires FiniteFieldType<T>
    {
        constexpr size_t m = details::degree_over_prime_v<T>;
        if constexpr (m == 1) {
            return get_weight_enumerator();
        } else {
            p_ary_image_weight_enumerator.call_once([this] {
                if (p_ary_image_weight_enumerator.has_value()) return;
                constexpr uint16_t p = T::get_characteristic();
                const size_t n = this->n;

                if (k == 0) {
                    p_ary_image_weight_enumerator.emplace(Polynomial<InfInt>(1));
                    return;
                }

                std::clog << "--> Calculating p-ary image weight enumerator for:" << std::endl;
                Matrix<Fp<p>> Gp(k * m, n * m);
                for (size_t i = 0; i < k; ++i) {
                    for (size_t j = 0; j < m; ++j) {
                        const T beta(static_cast<int>(sqm<size_t>(p, j)));
                        for (size_t pos = 0; pos < n; ++pos) {
                            const auto image = (beta * G(i, pos)).template as_vector<Fp<p>>();
                            Gp.set_submatrix(i * m + j, pos * m, Matrix<Fp<p>>(image));
                        }
                    }
                }
                p_ary_image_weight_enumerator.emplace(LinearCode<Fp<p>>(n * m, k * m, Gp).get_weight_enumerator());
            });

            return p_ary_image_weight_enumerator.value();
        }
    }

    long double P_word(double pe) const {
        const size_t tmax = get_tmax();
        long double res = 0.0;
        for (size_t i = tmax + 1; i <= this->n; ++i)
            res += bin<InfInt>(this->n, i).toUnsignedLongLong() * std::pow(static_cast<long double>(pe), i) *
                   std::pow(1.0L - pe, this->n - i);
        return res;
    }

    long double P_error(double pe) const
        requires FiniteFieldType<T>
    {
        const auto& A = get_weight_enumerator();
        const size_t tmax = get_tmax();
        long double res = 0.0;
        for (size_t h = 1; h <= this->n; ++h) {
            InfInt sum = 0;
            for (size_t s = 0; s <= tmax; ++s) {
                for (size_t ell = 1; ell <= this->n; ++ell) sum += A[ell] * N(ell, h, s);
            }
            res += std::pow(static_cast<long double>(pe) / (T::get_size() - 1), h) * std::pow(1.0L - pe, this->n - h) *
                   sum.toUnsignedLongLong();
        }
        return res;
    }

    long double P_failure(double pe) const
        requires FiniteFieldType<T>
    {
        long double res = P_word(pe) - P_error(pe);
        if (std::fabs(res) < 10 * std::numeric_limits<long double>::epsilon())
            return 0;
        else
            return res;
    }

    // Docu note: union-Bhattacharyya bound on ML decoding for transmission of the p-ary
    // image (cf. DEMUX/as_vector()) over a channel with prime-field input, gamma e.g. from
    // BI_AWGN::get_Bhattacharyya_param(). For codes over prime fields this is the classical
    // symbol-level bound.
    long double Bhattacharyya_bound(long double gamma) const
        requires FiniteFieldType<T>
    {
        const auto& A = get_p_ary_image_weight_enumerator();
        const size_t dmin = get_dmin();

        long double res = 0;
        for (size_t i = dmin; i <= A.degree(); ++i) res += A[i].toUnsignedLongLong() * std::pow(gamma, i);

        return res;
    }

    const Polynomial<T>& get_gamma() const {
        if (!is_polynomial())
            throw std::logic_error("Cannot calculate generator polynomial of a code that is not polynomial!");
        return gamma.value();
    }

    void set_dmin(size_t d) const { dmin.emplace(d); }

    void set_weight_enumerator(const Polynomial<InfInt>& p) const
        requires FiniteFieldType<T>
    {
        weight_enumerator.emplace(p);
    }

    void set_weight_enumerator(Polynomial<InfInt>&& p) const
        requires FiniteFieldType<T>
    {
        weight_enumerator.emplace(std::move(p));
    }

    void set_gamma(const Polynomial<T>& g) const {
        polynomial.emplace(true);
        gamma.emplace(g);
    }

    void set_gamma(Polynomial<T>&& g) const {
        polynomial.emplace(true);
        gamma.emplace(std::move(g));
    }

    const std::vector<Vector<T>>& get_standard_array() const
        requires FiniteFieldType<T>
    {
        standard_array.call_once([this] {
            if (standard_array.has_value()) return;

            std::clog << "--> Calculating standard array" << std::endl;

            constexpr size_t q = T::get_size();

            const size_t nof_cosets = sqm<size_t>(q, this->n - k);
            size_t count = 0;
            bool done = false;

            try {
                standard_array.emplace(std::vector<Vector<T>>(nof_cosets));
                tainted.emplace(std::vector<bool>(nof_cosets, false));
            } catch (const std::bad_alloc& e) {
                std::cerr << "Memory allocation failed, standard array would be too large!" << std::endl;
                throw e;
            }

            std::vector<size_t> leader_wH(standard_array.value().size(), std::numeric_limits<size_t>::max());
            std::vector<size_t> leader_tie_count(standard_array.value().size(), 0);

            for (size_t wt = 0; wt <= this->n; ++wt) {
                std::vector<bool> mask(this->n, false);
                std::fill(mask.begin(), mask.begin() + wt, true);

                do {
                    std::vector<size_t> pos;
                    pos.reserve(wt);
                    for (auto it = mask.cbegin(); it != mask.cend(); ++it) {
                        if (*it) pos.push_back(static_cast<size_t>(it - mask.cbegin()));
                    }

                    Vector<T> v(this->n, T(0));
                    for (size_t j = 0; j < wt; ++j) v.set_component(pos[j], T(1));

                    std::vector<size_t> digits(wt, 1);

                    for (;;) {
                        const auto s = v * HT;
                        const size_t i = s.as_integer();

                        if (standard_array.value()[i].is_empty()) {
                            standard_array.value()[i] = v;

                            leader_wH[i] = wt;
                            leader_tie_count[i] = 1;

                            ++count;
                            if (count == nof_cosets) done = true;
                        } else {
                            if (wt == leader_wH[i]) {
                                tainted.value()[i] = true;
                                if (details::reservoir_accept(++leader_tie_count[i])) {
                                    standard_array.value()[i] = v;
                                }
                            }
                        }

                        size_t j = 0;
                        while (j < wt) {
                            if (digits[j] < q - 1) {
                                ++digits[j];
                                v.set_component(pos[j], T(digits[j]));
                                break;
                            }
                            digits[j] = 1;
                            v.set_component(pos[j], T(1));
                            ++j;
                        }
                        if (j == wt) break;
                    }
                } while (std::prev_permutation(mask.begin(), mask.end()));

                if (done) return;
            }
        });

        return standard_array.value();
    }

    const std::vector<bool>& get_tainted() const
        requires FiniteFieldType<T>
    {
        return tainted.value();
    }

    const std::unordered_map<size_t, Vector<T>>& get_Meggitt_table() const
        requires FiniteFieldType<T>
    {
        Meggitt_table.call_once([this] {
            if (Meggitt_table.has_value()) return;

            if (k == 0) throw std::invalid_argument("Meggitt table only available for codes with k>0!");
            if (!is_cyclic()) throw std::logic_error("Meggitt table only available for cyclic codes!");

            std::clog << "--> Calculating Meggitt table" << std::endl;

            constexpr size_t q = T::get_size();
            const size_t n = this->n;
            const size_t nk = n - k;
            const size_t t = get_tmax();
            const auto& gamma = get_gamma();

            try {
                Meggitt_table.emplace();
                auto& table = Meggitt_table.value();

                for (size_t wt = 1; wt <= t; ++wt) {
                    std::vector<bool> mask(n, false);
                    // Docu note: we trap only error patterns with nonzero first component
                    std::fill(mask.begin(), mask.begin() + wt, true);

                    do {
                        std::vector<size_t> pos;
                        pos.reserve(wt);
                        for (size_t j = 0; j < n; ++j)
                            if (mask[j]) pos.push_back(j);

                        Vector<T> v(n, T(0));
                        for (size_t j = 0; j < wt; ++j) v.set_component(pos[j], T(1));

                        std::vector<size_t> digits(wt, 1);

                        for (;;) {
                            table.emplace(pad_back(Vector<T>(Polynomial<T>(v) % gamma), nk).as_integer(), v);

                            size_t j = 0;
                            while (j < wt) {
                                if (digits[j] < q - 1) {
                                    ++digits[j];
                                    v.set_component(pos[j], T(digits[j]));
                                    break;
                                }
                                digits[j] = 1;
                                v.set_component(pos[j], T(1));
                                ++j;
                            }
                            if (j == wt) break;
                        }

                    } while (std::prev_permutation(mask.begin() + 1, mask.end()));
                }
            } catch (const std::bad_alloc& e) {
                Meggitt_table.reset();
                std::cerr << "Memory allocation failed, Meggitt table too large!" << std::endl;
                throw;
            }
        });

        return Meggitt_table.value();
    }

    auto cbegin() const
        requires FiniteFieldType<T>
    {
        if (k == 0)
            return CodewordIterator<T>(*this, get_size());
        else
            return CodewordIterator<T>(*this, 0);
    }

    auto cend() const
        requires FiniteFieldType<T>
    {
        return CodewordIterator<T>(*this, get_size());
    }

    /**
     * @brief Test whether two linear codes are identical
     *
     * @tparam S Field type of the other code (must equal T for a meaningful comparison)
     * @param other Linear code to compare against
     * @param L_ptr If non-null, receives the invertible k × k matrix L such that G' = L · G
     * @return True if both codes span the same codebook, false otherwise
     *
     * Two [n, k] linear codes are identical if they have the same codebook, which
     * is the case if and only if their generator matrices have the same RREF.
     * When n − k < k, the comparison is performed on the parity-check matrices
     * for efficiency.
     */
    template <FieldType S>
    bool is_identical(const LinearCode<S>& other, Matrix<T>* L_ptr = nullptr) const {
        if constexpr (!std::is_same_v<T, S>) {
            return false;
        } else {
            if (this->n != other.n || k != other.k) return false;
            if (this == &other) {
                if (L_ptr != nullptr) *L_ptr = IdentityMatrix<T>(k);
                return true;
            }

            bool res;
            if (this->n - k < other.k)
                res = rref(transpose(HT)) == rref(transpose(other.HT));
            else
                res = rref(G) == rref(other.G);

            if (res && L_ptr != nullptr) {
                auto Bp = transpose(Matrix<T>(other.get_G().get_col(infoset[0])));
                for (size_t t = 1; t < k; ++t)
                    Bp.horizontal_join(transpose(Matrix<T>(other.get_G().get_col(infoset[t]))));
                *L_ptr = Bp * MI;
            }

            return res;
        }
    }

    /**
     * @brief Test whether two linear codes are equivalent
     *
     * @tparam S Field type of the other code (must equal T for a meaningful comparison)
     * @param other Linear code to compare against
     * @param L_ptr If non-null, receives the invertible k × k matrix L such that G' = L · G · P
     * @param P_ptr If non-null, receives the n × n permutation matrix P such that G' = L · G · P
     * @return True if the codes are equivalent, false otherwise
     *
     * Two [n, k] linear codes are equivalent if their generator matrices are related
     * by G' = L · G · P for some invertible matrix L and permutation matrix P.
     * When n − k < k, the search is performed on the dual codes for efficiency.
     *
     * @note The search has combinatorial complexity and may be slow for large codes
     */
    template <FieldType S>
    bool is_equivalent(const LinearCode<S>& other, Matrix<T>* L_ptr = nullptr, Matrix<T>* P_ptr = nullptr) const {
        if constexpr (!std::is_same_v<T, S>) {
            return false;
        } else {
            if (this->n != other.n || k != other.k) return false;

            if (this == &other) {
                if (L_ptr != nullptr) *L_ptr = IdentityMatrix<T>(k);
                if (P_ptr != nullptr) *P_ptr = IdentityMatrix<T>(this->n);
                return true;
            }

            if (this->n - k < k) {
                auto Cperp = this->get_dual();
                auto Cotherperp = other.get_dual();

                Matrix<T> Pperp;
                Matrix<T>* Pperp_ptr = (P_ptr != nullptr || L_ptr != nullptr) ? &Pperp : nullptr;

                // L on the dual side is not useful here, so don't request it.
                if (!Cperp.is_equivalent(Cotherperp, nullptr, Pperp_ptr)) return false;

                if (P_ptr != nullptr) *P_ptr = Pperp;

                if (L_ptr != nullptr) {
                    std::vector<size_t> p(this->n, this->n);
                    for (size_t i = 0; i < this->n; ++i) {
                        for (size_t j = 0; j < this->n; ++j) {
                            if (Pperp(i, j) == T(1)) {
                                p[i] = j;
                                break;
                            }
                        }
                    }

                    auto Bp = transpose(Matrix<T>(other.get_G().get_col(p[this->infoset[0]])));
                    for (size_t t = 1; t < this->k; ++t)
                        Bp.horizontal_join(transpose(Matrix<T>(other.get_G().get_col(p[this->infoset[t]]))));
                    *L_ptr = Bp * this->MI;
                }

                return true;
            } else {
                auto next_combination = [](std::vector<size_t>& comb, size_t n_total) -> bool {
                    const size_t kk = comb.size();
                    for (size_t i = kk; i-- > 0;) {
                        if (comb[i] < n_total - kk + i) {
                            ++comb[i];
                            for (size_t j = i + 1; j < kk; ++j) comb[j] = comb[j - 1] + 1;
                            return true;
                        }
                    }
                    return false;
                };

                std::vector<Vector<T>> Gp_cols;
                Gp_cols.reserve(this->n);
                for (size_t j = 0; j < this->n; ++j) Gp_cols.push_back(other.get_G().get_col(j));

                std::vector<size_t> target_col_keys;
                target_col_keys.reserve(this->n);
                for (size_t j = 0; j < this->n; ++j) target_col_keys.push_back(Gp_cols[j].as_integer());
                std::sort(target_col_keys.begin(), target_col_keys.end());

                std::vector<size_t> J(k);
                std::iota(J.begin(), J.end(), 0);

                do {
                    auto Bp_sorted = transpose(Matrix<T>(Gp_cols[J[0]]));
                    for (size_t t = 1; t < k; ++t) Bp_sorted.horizontal_join(transpose(Matrix<T>(Gp_cols[J[t]])));

                    if (Bp_sorted.rank() != k) continue;

                    std::vector<size_t> Jperm = J;
                    do {
                        auto Bp = transpose(Matrix<T>(Gp_cols[Jperm[0]]));
                        for (size_t t = 1; t < k; ++t) Bp.horizontal_join(transpose(Matrix<T>(Gp_cols[Jperm[t]])));

                        const auto L = Bp * MI;
                        const auto Lt = transpose(L);  // buffered once per candidate

                        std::vector<size_t> transformed_col_keys;
                        transformed_col_keys.reserve(this->n);

                        for (size_t col = 0; col < this->n; ++col) {
                            auto h = Vector<T>(G.get_col(col) * Lt);
                            transformed_col_keys.push_back(h.as_integer());
                        }
                        std::sort(transformed_col_keys.begin(), transformed_col_keys.end());

                        if (transformed_col_keys == target_col_keys) {
                            // Fast path: caller only wants yes/no
                            if (L_ptr == nullptr && P_ptr == nullptr) return true;

                            if (L_ptr != nullptr) *L_ptr = L;

                            if (P_ptr != nullptr) {
                                std::vector<Vector<T>> transformed_cols;
                                transformed_cols.reserve(this->n);
                                for (size_t col = 0; col < this->n; ++col)
                                    transformed_cols.push_back(Vector<T>(G.get_col(col) * Lt));

                                std::vector<size_t> perm(this->n, this->n);
                                std::vector<bool> used(this->n, false);

                                bool match_ok = true;
                                for (size_t i = 0; i < this->n && match_ok; ++i) {
                                    bool found = false;
                                    for (size_t j = 0; j < this->n; ++j) {
                                        if (used[j]) continue;
                                        if (transformed_cols[i] == Gp_cols[j]) {
                                            perm[i] = j;
                                            used[j] = true;
                                            found = true;
                                            break;
                                        }
                                    }
                                    if (!found) match_ok = false;
                                }

                                if (!match_ok) {
                                    // If matching fails, continue searching (e.g. due to key collisions)
                                    continue;
                                }

                                *P_ptr = PermutationMatrix<T>(perm);
                            }

                            return true;
                        }

                    } while (std::next_permutation(Jperm.begin(), Jperm.end()));

                } while (next_combination(J, this->n));

                return false;
            }
        }
    }

    bool is_perfect() const
        requires FiniteFieldType<T>
    {
        if (k == 0) return false;
        return std::fabs(HammingUpperBound<T>(this->n, get_dmin()) - k) <
               10 * std::numeric_limits<long double>::epsilon();
    }

    bool is_MDS() const {
        if (k == 0) return false;
        return SingletonUpperBound(this->n, get_dmin()) == k;
    }

    bool is_equidistant() const
        requires FiniteFieldType<T>
    {
        if (k == 0) return true;
        return std::fabs(PlotkinUpperBound<T>(this->n, get_dmin()) - k) <
               10 * std::numeric_limits<long double>::epsilon();
    }

    bool is_weakly_self_dual() const { return G * transpose(G) == ZeroMatrix<T>(k, k); }
    bool is_dual_containing() const { return transpose(HT) * HT == ZeroMatrix<T>(this->n - k, this->n - k); }
    bool is_self_dual() const { return 2 * k == this->n && is_weakly_self_dual() && is_dual_containing(); }

    bool is_polynomial() const {
        polynomial.call_once([this] {
            if (polynomial.has_value()) return;

            if (k == 0) {
                polynomial.emplace(true);
                auto g = ZeroPolynomial<T>();
                g.set_coefficient(0, -T(1));
                g.set_coefficient(this->n, T(1));
                gamma.emplace(g);
                return;
            }

            auto g = ZeroPolynomial<T>();
            for (size_t i = 0; i < k; ++i) {
                g = GCD(g, Polynomial<T>(G.get_row(i)));
                if (g.degree() < this->n - k && !g.is_zero()) {
                    polynomial.emplace(false);
                    return;
                }
            }
            if (g.trailing_degree() > 0) {
                polynomial.emplace(false);
                return;
            }
            g = normalize(g);  // degree at this point can only be n-k
            polynomial.emplace(true);
            gamma.emplace(g);
        });
        return polynomial.value();
    }

    bool is_cyclic() const {
        if (!is_polynomial()) return false;
        if (k == 0) return true;

        auto p = ZeroPolynomial<T>();
        p.set_coefficient(0, -T(1));
        p.set_coefficient(gamma.value().degree() + k, T(1));
        p %= gamma.value();
        if (p.is_zero())
            return true;
        else
            return false;
    }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) != 3) {
            if (os.iword(details::index) > 0) Code<T>::get_info(os);
            if constexpr (FiniteFieldType<T>) {
                constexpr size_t q = T::get_size();
                os << "[F_" << q << "; " << this->n << ", " << k << "]";
                if (os.iword(details::index) > 1) {
                    get_weight_enumerator();
                    os << ", dmin = ";
                    try {
                        os << get_dmin();
                    } catch (const std::logic_error& e) {
                        os << "undefined";
                    }
                }
            } else {
                os << "[Q; " << this->n << ", " << k << "]";
                if (os.iword(details::index) > 1) {
                    os << ", dmin = ";
                    try {
                        os << get_dmin();
                    } catch (const std::logic_error& e) {
                        os << "undefined";
                    }
                }
            }
        }
        if (os.iword(details::index) > 0) {
            if (os.iword(details::index) != 3) os << std::endl;
            os << BOLD("Linear code") " with properties: { ";
            if (os.iword(details::index) != 3) {
                os << std::endl;
                os << "G = " << std::endl;
                os << G << std::endl;
                os << "H = " << std::endl;
                os << get_H();
                os << std::endl;
            }
        }
        if (os.iword(details::index) > 1 && os.iword(details::index) < 3) {
            if constexpr (FiniteFieldType<T>) os << "A(x) = " << get_weight_enumerator() << std::setfill(' ') << " ";
            os << "tmax = ";
            try {
                os << get_tmax() << " ";
            } catch (const std::logic_error& e) {
                os << "undefined ";
            }
        }

        if (os.iword(details::index) > 1) {
            if (is_polynomial()) {
                os << "polynomial(";
                if (is_cyclic()) os << "cyclic, ";
                os << "gamma = " << get_gamma() << ") " << std::flush;
            }
            if constexpr (FiniteFieldType<T>)
                if (is_perfect()) os << "perfect " << std::flush;
            if (is_MDS()) os << "MDS " << std::flush;
            if constexpr (FiniteFieldType<T>)
                if (is_equidistant()) os << "equidistant " << std::flush;
        }
        if (os.iword(details::index) > 0) {
            if (!is_self_dual() && is_weakly_self_dual()) os << "weakly_self-dual " << std::flush;
            if (!is_self_dual() && is_dual_containing()) os << "dual-containing " << std::flush;
            if (is_self_dual()) os << "self-dual " << std::flush;
            if (os.iword(details::index) != 3) os << std::endl;
            os << "}" << std::flush;
        }
    }

    LinearCode<T> get_dual() const {
        auto dual_code = LinearCode<T>(this->n, this->n - k, get_H());
        if constexpr (FiniteFieldType<T>) {
            if (weight_enumerator.has_value())
                dual_code.set_weight_enumerator(MacWilliamsIdentity<T>(weight_enumerator.value(), this->n, k));
        }
        return dual_code;
    }

    LinearCode<T> get_equivalent_code_in_standard_form() const {
        auto Gp = G;
        Gp.rref();
        for (size_t i = 0; i < k; ++i) {
            const auto u = unit_vector<T>(k, i);
            for (size_t j = 0; j < this->n; ++j) {
                if (Gp.get_col(j) == u) {
                    if (j != i) Gp.swap_columns(i, j);
                    break;
                }
            }
        }
        LinearCode<T> res(this->n, k, Gp);
        if (dmin.has_value()) res.set_dmin(*dmin);
        if constexpr (FiniteFieldType<T>) {
            if (weight_enumerator.has_value()) res.set_weight_enumerator(*weight_enumerator);
            if (p_ary_image_weight_enumerator.has_value())
                res.p_ary_image_weight_enumerator = p_ary_image_weight_enumerator;
        }
        return res;
    }

    LinearCode<T> get_identical_code_in_polynomial_form() const {
        LinearCode<T> res(this->n, k, get_G_in_polynomial_form());
        if (dmin.has_value()) res.set_dmin(*dmin);
        if constexpr (FiniteFieldType<T>) {
            if (weight_enumerator.has_value()) res.set_weight_enumerator(*weight_enumerator);
            if (p_ary_image_weight_enumerator.has_value())
                res.p_ary_image_weight_enumerator = p_ary_image_weight_enumerator;
            if (minimal_trellis.has_value()) res.minimal_trellis = minimal_trellis;
            if (codewords.has_value()) res.codewords = codewords;
        }
        if (gamma.has_value()) res.set_gamma(*gamma);
        return res;
    }

    Matrix<T> get_G_in_polynomial_form() const {
        if (!is_polynomial()) throw std::invalid_argument("Code is not polynomial, cannot bring G in polynomial form!");

        const auto& g = get_gamma();
        auto Gp = ToeplitzMatrix(pad_back(pad_front(Vector<T>(g), this->n), this->n + k - 1), k, this->n);
        return Gp;
    }

    Matrix<T> get_G_in_trellis_oriented_form() const {
        auto Gp = G;
        for (;;) {
            // find start and end indices
            std::vector<size_t> starts(k);
            std::vector<size_t> ends(k);
            for (size_t i = 0; i < k; ++i) {
                for (size_t j = 0; j < this->n; ++j) {
                    if (Gp(i, j) != T(0)) {
                        starts[i] = j;
                        break;
                    }
                }
                for (size_t j = this->n; j-- > 0;) {
                    if (Gp(i, j) != T(0)) {
                        ends[i] = j;
                        break;
                    }
                }
            }

            // break if in trellis-oriented form
            std::set<size_t> start_set(starts.cbegin(), starts.cend());
            std::set<size_t> end_set(ends.cbegin(), ends.cend());

            if (start_set.size() == k && end_set.size() == k) break;

            bool found_start = false;
            for (size_t i = 0; i < k; ++i) {
                for (size_t j = i + 1; j < k; ++j) {
                    if (starts[i] == starts[j]) {
                        const size_t s = starts[i];
                        Gp.scale_row(T(1) / Gp(i, s), i);
                        Gp.scale_row(T(1) / Gp(j, s), j);
                        if (ends[i] < ends[j])
                            Gp.add_scaled_row(-T(1), i, j);
                        else
                            Gp.add_scaled_row(-T(1), j, i);
                        found_start = true;
                        break;
                    }
                }
                if (found_start) break;
            }

            if (!found_start) {
                bool found_end = false;
                for (size_t i = 0; i < k; ++i) {
                    for (size_t j = i + 1; j < k; ++j) {
                        if (ends[i] == ends[j]) {
                            const size_t e = ends[i];
                            Gp.scale_row(T(1) / Gp(i, e), i);
                            Gp.scale_row(T(1) / Gp(j, e), j);
                            if (starts[i] > starts[j])
                                Gp.add_scaled_row(-T(1), i, j);
                            else
                                Gp.add_scaled_row(-T(1), j, i);
                            found_end = true;
                            break;
                        }
                    }
                    if (found_end) break;
                }
            }
        }

        return Gp;
    }

    Trellis<T> get_trivial_trellis() const
        requires FiniteFieldType<T>
    {
        const size_t n = this->n;

        Trellis<T> res;
        size_t i = 0;
        for (auto it = cbegin(); it != cend(); ++it) {
            res.add_edge(0, 0, i, (*it)[0]);
            for (size_t j = 1; j < n - 1; ++j) {
                res.add_edge(j, i, i, (*it)[j]);
            }
            res.add_edge(n - 1, i, 0, (*it)[n - 1]);
            ++i;
        }

        return res;
    }

    const Trellis<T>& get_minimal_trellis() const
        requires FiniteFieldType<T>
    {
        minimal_trellis.call_once([this] {
            if (minimal_trellis.has_value()) return;
            constexpr size_t q = T::get_size();
            const size_t n = this->n;
            const auto Gp = get_G_in_trellis_oriented_form();

            std::clog << "--> Calculating minimal trellis for code with " << sqm<InfInt>(q, k) << " codewords"
                      << std::endl;

            auto row_trellis = [n, &Gp](size_t i) {
                size_t s = 0, e = 0;

                for (size_t j = 0; j < n; ++j) {
                    if (Gp(i, j) != T(0)) {
                        s = j;
                        break;
                    }
                }
                for (size_t j = n; j-- > 0;) {
                    if (Gp(i, j) != T(0)) {
                        e = j;
                        break;
                    }
                }

                const bool span_one = (s == e);
                const bool open_ended = span_one && (e == n - 1);

                Trellis<T> tr;

                for (size_t j = 0; j < n; ++j) {
                    if (j < s || (!open_ended && j > (span_one ? s + 1 : e)) || (open_ended && j > e)) {
                        tr.add_edge(j, 0, 0, T(0));
                    } else if (j == s) {
                        for (uint32_t a = 0; a < q; ++a) tr.add_edge(j, 0, a, T(a) * Gp(i, s));
                    } else if (!open_ended && j == (span_one ? s + 1 : e)) {
                        if (span_one) {
                            for (uint32_t a = 0; a < q; ++a) tr.add_edge(j, a, 0, T(0));
                        } else {
                            for (uint32_t a = 0; a < q; ++a) tr.add_edge(j, a, 0, T(a) * Gp(i, j));
                        }
                    } else {
                        for (uint32_t a = 0; a < q; ++a) tr.add_edge(j, a, a, T(a) * Gp(i, j));
                    }
                }

                return tr;
            };

            auto result = row_trellis(0);
            for (size_t i = 1; i < k; ++i) result = result * row_trellis(i);

            minimal_trellis.emplace(std::move(result));
        });
        return *minimal_trellis;
    }

    Vector<T> enc(const Vector<T>& u) const override { return u * G; }

    Vector<T> encinv(const Vector<T>& c) const override {
        if (k == 0) throw std::logic_error("Cannot invert encoding wrt. a dimension zero code!");

        Vector<T> c_sub(k);
        size_t i = 0;
        for (auto it = infoset.cbegin(); it != infoset.cend(); ++it) {
            c_sub.set_component(i, c[*it]);
            ++i;
        }
        return c_sub * MI;
    }

    virtual Vector<T> dec_BD(const Vector<T>& r) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("BD decoding only available for codes over finite fields!");
        } else {
#ifdef CECCO_ERASURE_SUPPORT
            if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
#endif
            validate_length(r);

            if (k == 0) return Vector<T>(this->n);

            const auto c_est = LinearCode<T>::dec_ML(r);
            if (dH(r, c_est) > this->get_tmax()) throw decoding_failure("Linear code BD decoder failed!");
            return c_est;
        }
    }

    virtual Vector<T> dec_boosted_BD(const Vector<T>& r) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("BD decoding only available for codes over finite fields!");
        } else {
#ifdef CECCO_ERASURE_SUPPORT
            if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
#endif
            validate_length(r);

            if (k == 0) return Vector<T>(this->n);

            get_standard_array();
            const auto s = r * HT;  // calculate syndrome...
            if (s.is_zero()) return r;
            const size_t i = s.as_integer();  // ... and interpret it as q-ary number
            if (tainted.value()[i]) throw decoding_failure("Linear code boosted BD decoder failed!");
            return r - standard_array.value()[i];
        }
    }

    virtual Vector<T> dec_ML(const Vector<T>& r) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("ML decoding only available for codes over finite fields!");
        } else {
#ifdef CECCO_ERASURE_SUPPORT
            if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
#endif
            validate_length(r);

            if (k == 0) return Vector<T>(this->n);

            get_standard_array();
            const auto s = r * HT;  // calculate syndrome...
            if (s.is_zero()) return r;
            const size_t i = s.as_integer();  // ... and interpret it as q-ary number
            return r - standard_array.value()[i];
        }
    }

    virtual Vector<T> dec_Meggitt(const Vector<T>& r) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Meggitt BD decoding only available for codes over finite fields!");
        } else {
#ifdef CECCO_ERASURE_SUPPORT
            if (LinearCode<T>::erasures_present(r))
                throw std::invalid_argument("Trying to correct erasures with a Meggitt BD decoder!");
#endif
            validate_length(r);

            if (k == 0) return Vector<T>(this->n);

            const size_t n = this->n;
            const size_t redundancy = n - k;
            const auto& gamma = get_gamma();
            const T lead_inv = T(1) / gamma[redundancy];
            const auto& table = get_Meggitt_table();

            // Initial polynomial syndrome: s = r(x) mod gamma(x)
            // x * r(x) mod (x^n - 1) = rotate_right(r, 1), so each LFSR step
            // advances the syndrome to match the next right-cyclic shift of r.
            Vector<T> s = pad_back(Vector<T>(Polynomial<T>(r) % gamma), redundancy);

            if (s.is_zero()) return r;

            for (size_t i = 0; i < n; ++i) {
                const auto it = table.find(s.as_integer());
                if (it != table.end()) return r - rotate_left(it->second, i);
                // LFSR: s <- x * s mod gamma
                const T feedback = s[redundancy - 1] * lead_inv;
                for (size_t j = redundancy - 1; j > 0; --j) s.set_component(j, s[j - 1] - feedback * gamma[j]);
                s.set_component(0, -feedback * gamma[0]);
            }

            throw decoding_failure("Meggitt BD decoder failed!");
        }
    }

    virtual Vector<T> dec_Viterbi(const Vector<T>& r, const std::string& filename = "") const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Viterbi decoding only available for codes over finite fields!");
        } else {
#ifdef CECCO_ERASURE_SUPPORT
            if (LinearCode<T>::erasures_present(r)) return dec_Viterbi_EE(r);
#endif
            validate_length(r);
            if (!filename.empty() && T::get_size() > 64)
                throw std::invalid_argument("Viterbi trellis TikZ export not supported for fields with size > 64!");
            if (k == 0) return Vector<T>(this->n);

            const auto& Tr = get_minimal_trellis();
            typename Trellis<T>::template Viterbi_Workspace<uint16_t> ws(Tr);
            ws.calculate_edge_costs(Tr, r);
            if (!filename.empty()) ws.v.emplace(r);
            auto c_est = viterbi_forward_pass_and_traceback<uint16_t>(Tr, ws, filename);
            return c_est;
        }
    }

    virtual Vector<T> dec_Viterbi_soft(const Vector<double>& llrs, const std::string& filename = "") const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Soft-input Viterbi decoding only available for codes over finite fields!");
        } else if constexpr (T::get_size() != 2) {
            throw std::logic_error(
                "Soft-input Viterbi vector decoding only available for binary codes; use the "
                "Matrix<double> overload!");
        } else {
            return dec_Viterbi_soft(Matrix<double>(llrs), filename);
        }
    }

    virtual Vector<T> dec_Viterbi_soft(const Matrix<double>& llrs, const std::string& filename = "") const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Soft-input Viterbi decoding only available for codes over finite fields!");
        } else {
            validate_length(llrs);
            if (!filename.empty() && T::get_size() > 2)
                throw std::invalid_argument("Soft-input Viterbi trellis TikZ export only supported for binary codes!");
            if (k == 0) return Vector<T>(this->n);

            const auto& Tr = get_minimal_trellis();
            typename Trellis<T>::template Viterbi_Workspace<double> ws(Tr);
            ws.calculate_edge_costs(Tr, llrs);
            if (!filename.empty()) ws.v.emplace(llrs);
            return viterbi_forward_pass_and_traceback<double>(Tr, ws, filename);
        }
    }

    // Docu note: pass nullptr as Lambda when only the filename argument is needed.
    virtual Vector<T> dec_BCJR(const Vector<double>& llrs, Matrix<double>* Lambda = nullptr,
                               const std::string& filename = "") const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("BCJR decoding only available for codes over finite fields!");
        } else if constexpr (T::get_size() != 2) {
            throw std::logic_error(
                "Soft-input BCJR vector decoding only available for binary codes; use the Matrix<double> overload!");
        } else {
            return bcjr_decode(Matrix<double>(llrs), Lambda, filename);
        }
    }

    virtual Vector<T> dec_BCJR(const Matrix<double>& llrs, Matrix<double>* Lambda = nullptr,
                               const std::string& filename = "") const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("BCJR decoding only available for codes over finite fields!");
        } else {
            return bcjr_decode(llrs, Lambda, filename);
        }
    }

    virtual Vector<T> dec_ML_soft(const Vector<double>& llrs, size_t cache_limit) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Soft-input ML decoding only available for codes over finite fields!");
        } else if constexpr (T::get_size() != 2) {
            throw std::logic_error(
                "Soft-input ML vector decoding only available for binary codes; use the Matrix<double> overload!");
        } else {
            return dec_ML_soft(Matrix<double>(llrs), cache_limit);
        }
    }

    virtual Vector<T> dec_ML_soft(const Matrix<double>& llrs, size_t cache_limit) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Soft-input ML decoding only available for codes over finite fields!");
        } else {
            validate_length(llrs);

            if (k == 0) return Vector<T>(this->n);

            std::vector<std::array<double, T::get_size()>> costs;
            costs.reserve(this->n);
            for (size_t i = 0; i < this->n; ++i) costs.push_back(details::symbol_costs_from_llrs<T>(llrs, i));

            Vector<T> c_est;
            double best = std::numeric_limits<double>::max();
            uint16_t ties = 1;

            if (this->get_size() <= cache_limit) {
                codewords.call_once([this] {
                    if (codewords.has_value()) return;
                    codewords.emplace(this->get_G().span());
                });

                for (auto it = codewords.value().cbegin(); it != codewords.value().cend(); ++it) {
                    double val = 0.0;
                    for (size_t i = 0; i < this->n; ++i) val += costs[i][(*it)[i].get_label()];
                    if (val < best) {
                        c_est = *it;
                        best = val;
                        ties = 1;
                    } else if (val == best) {
                        if (details::reservoir_accept(++ties)) c_est = *it;
                    }
                }
            } else {
                for (auto it = cbegin(); it != cend(); ++it) {
                    double val = 0.0;
                    for (size_t i = 0; i < this->n; ++i) val += costs[i][(*it)[i].get_label()];
                    if (val < best) {
                        c_est = *it;
                        best = val;
                        ties = 1;
                    } else if (val == best) {
                        if (details::reservoir_accept(++ties)) c_est = *it;
                    }
                }
            }

            return c_est;
        }
    }

    const TannerGraph<T>& get_tanner_graph() const
        requires FiniteFieldType<T>
    {
        tanner_graph.call_once([this] {
            if (tanner_graph.has_value()) return;
            const size_t n = this->n;
            const size_t redundancy = n - k;
            TannerGraph<T> g(n);
            for (size_t i = 0; i < redundancy; ++i)
                for (size_t j = 0; j < n; ++j)
                    if (!HT(j, i).is_zero()) g.add_edge(i, static_cast<uint32_t>(j), HT(j, i));
            tanner_graph.emplace(std::move(g));
        });
        return tanner_graph.value();
    }

    virtual Vector<T> dec_BP(const Vector<double>& llrs, size_t max_iterations = BP_MAX_ITERATIONS,
                             Matrix<double>* Lambda = nullptr, const std::string& filename = "",
                             size_t* nof_iterations = nullptr) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Belief-propagation decoding only available for codes over finite fields!");
        } else if constexpr (T::get_size() != 2) {
            throw std::logic_error(
                "Soft-input BP vector decoding only available for binary codes; use the Matrix<double> overload!");
        } else {
            return bp_decode(Matrix<double>(llrs), max_iterations, Lambda, filename, nof_iterations);
        }
    }

    virtual Vector<T> dec_BP(const Matrix<double>& llrs, size_t max_iterations = BP_MAX_ITERATIONS,
                             Matrix<double>* Lambda = nullptr, const std::string& filename = "",
                             size_t* nof_iterations = nullptr) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Belief-propagation decoding only available for codes over finite fields!");
        } else {
            return bp_decode(llrs, max_iterations, Lambda, filename, nof_iterations);
        }
    }

#ifdef CECCO_ERASURE_SUPPORT
    virtual Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("BD error/erasure decoding only available for codes over finite fields!");
        } else {
            validate_length(r);

            if (k == 0) return Vector<T>(this->n);

            std::vector<size_t> X;
            std::vector<size_t> E;
            for (size_t i = 0; i < this->n; ++i) {
                if (r[i].is_erased())
                    X.push_back(i);
                else
                    E.push_back(i);
            }
            const size_t tau = X.size();

            if (tau == 0) return dec_BD(r);
            if (tau > get_dmin() - 1) {
                throw decoding_failure("Linear code BD error/erasure decoder failed!");
            }

            init_punctured_codes_BD();

            const auto& PC = punctured_codes_BD.value()[pos_to_index(X)].value();

            const auto r_E = delete_components(r, X);
            const auto c_E = PC.dec_ML(r_E);
            const auto HT_E = delete_rows(this->HT, X);
            const auto b = -c_E * HT_E;
            const auto HT_X = delete_rows(this->HT, E);
            const auto B = transpose(vertical_join(HT_X, Matrix(b))).basis_of_nullspace();

            Vector<T> sol(tau);
            bool found = false;
            for (size_t i = 0; i < B.get_m(); ++i) {
                const T a = B(i, B.get_n() - 1);
                if (!a.is_zero()) {
                    sol = B.get_row(i).delete_component(B.get_n() - 1);
                    sol /= -a;
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw decoding_failure("Linear code BD error/erasure decoder failed!");
            }

            Vector<T> c_est(this->n);
            for (size_t i = 0; i < E.size(); ++i) c_est.set_component(E[i], c_E[i]);
            for (size_t j = 0; j < tau; ++j) c_est.set_component(X[j], sol[j]);

            size_t t = 0;
            for (size_t i = 0; i < this->n; ++i) {
                if (!r[i].is_erased() && r[i] != c_est[i]) ++t;
            }

            if (2 * t + tau > get_dmin() - 1) {
                throw decoding_failure("Linear code BD error/erasure decoder failed!");
            }

            return c_est;
        }
    }

    virtual Vector<T> dec_Viterbi_EE(const Vector<T>& r, const std::string& filename = "") const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Viterbi error/erasure decoding only available for codes over finite fields!");
        } else {
            validate_length(r);
            if (!filename.empty() && T::get_size() > 64)
                throw std::invalid_argument("Viterbi trellis TikZ export not supported for fields with size > 64!");
            if (k == 0) return Vector<T>(this->n);

            const auto& Tr = get_minimal_trellis();
            typename Trellis<T>::template Viterbi_Workspace<uint16_t> ws(Tr);
            ws.calculate_edge_costs(Tr, r);
            if (!filename.empty()) ws.v.emplace(r);
            auto c_est = viterbi_forward_pass_and_traceback<uint16_t>(Tr, ws, filename);
            return c_est;
        }
    }

    virtual Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("ML error/erasure decoding only available for codes over finite fields!");
        } else {
            validate_length(r);

            if (k == 0) return Vector<T>(this->n);

            std::vector<size_t> X;
            std::vector<size_t> E;
            for (size_t i = 0; i < this->n; ++i) {
                if (r[i].is_erased())
                    X.push_back(i);
                else
                    E.push_back(i);
            }
            const size_t tau = X.size();

            if (tau == 0) return dec_ML(r);
            if (tau == this->n) return Vector<T>(this->n);

            const LinearCode<T>* pc_ptr;

            if (tau <= get_dmin() - 1) {
                init_punctured_codes_BD();
                pc_ptr = &punctured_codes_BD.value()[pos_to_index(X)].value();
            } else {
                init_punctured_codes_ML();
                size_t bd_count = 0;
                for (size_t t = 1; t <= get_dmin() - 1; ++t) bd_count += bin<size_t>(this->n, t);
                pc_ptr = &punctured_codes_ML.value()[pos_to_index(X) - bd_count].value();
            }

            const auto& PC = *pc_ptr;

            const auto r_E = delete_components(r, X);
            const auto c_E = PC.dec_ML(r_E);
            const auto HT_E = delete_rows(this->HT, X);
            const auto b = -c_E * HT_E;
            const auto HT_X = delete_rows(this->HT, E);
            const auto B = transpose(vertical_join(HT_X, Matrix(b))).basis_of_nullspace();

            Vector<T> sol(tau);
            bool found = false;
            for (size_t i = 0; i < B.get_m(); ++i) {
                const T a = B(i, B.get_n() - 1);
                if (!a.is_zero()) {
                    sol = B.get_row(i).delete_component(B.get_n() - 1);
                    sol /= -a;
                    found = true;
                    break;
                }
            }
            if (!found) throw decoding_failure("Linear code ML error/erasure decoder failed!");

            Vector<T> c_est(this->n);
            for (size_t i = 0; i < E.size(); ++i) c_est.set_component(E[i], c_E[i]);
            for (size_t j = 0; j < tau; ++j) c_est.set_component(X[j], sol[j]);

            return c_est;
        }
    }

    Vector<T> dec_GMD(const Vector<T>& r, const std::vector<double>& reliability) const
        requires FiniteFieldType<T>
    {
        const size_t n = this->n;

        if (reliability.size() != n)
            throw std::invalid_argument("GMD decoder: length of reliability vector must match code length");

        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), gen());
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return reliability[a] < reliability[b]; });

        std::optional<Vector<T>> best;
        double best_score = std::numeric_limits<double>::infinity();

        const size_t d = get_dmin();

        auto rp = r;
        const auto trial = [&]() {
            try {
                const auto c_est = dec_BD_EE(rp);
                double score = 0.0;
                for (size_t i = 0; i < n; ++i)
                    if (c_est[i] != r[i]) score += reliability[i];
                if (score < best_score) {
                    best_score = score;
                    best = c_est;
                }
            } catch (const decoding_failure&) {
            }
        };

        trial();
        for (size_t tau = 2; tau < d; tau += 2) {
            rp.erase_components({order[tau - 2], order[tau - 1]});
            trial();
            if (best_score == 0.0) break;
        }

        if (!best.has_value()) throw decoding_failure("GMD decoder failed (all trials failed)!");
        return std::move(*best);
    }
#endif

   protected:
#ifdef CECCO_ERASURE_SUPPORT
    static bool erasures_present(const Vector<T>& r) {
        for (size_t i = 0; i < r.get_n(); ++i) {
            if (r[i].is_erased()) return true;
        }
        return false;
    }
#endif

    template <ComponentType S>
    void validate_length(const Vector<S>& r) const {
        if (r.get_n() != this->n)
            throw std::invalid_argument(std::string("Received vector length must be ") + std::to_string(this->n));
    }

    void validate_length(const Matrix<double>& llrs) const
        requires FiniteFieldType<T>
    {
        constexpr size_t rows = details::degree_over_prime_v<T> * (T::get_characteristic() - 1);
        if (llrs.get_m() != rows || llrs.get_n() != this->n)
            throw std::invalid_argument("LLR matrix must have " + std::to_string(rows) + " rows and " +
                                        std::to_string(this->n) + " columns");
    }

    size_t k;
    Matrix<T> G;
    Matrix<T> HT;
    Matrix<T> MI;
    std::vector<size_t> infoset{};
    mutable details::OnceCache<size_t> dmin;
    mutable details::OnceCache<Polynomial<InfInt>> weight_enumerator;
    mutable details::OnceCache<Polynomial<InfInt>> p_ary_image_weight_enumerator;
    mutable details::OnceCache<std::vector<Vector<T>>> codewords;
    mutable details::OnceCache<std::vector<Vector<T>>> standard_array;
    mutable details::OnceCache<std::vector<bool>> tainted;
    mutable details::OnceCache<std::unordered_map<size_t, Vector<T>>> Meggitt_table;
#ifdef CECCO_ERASURE_SUPPORT
    mutable details::OnceCache<std::vector<std::optional<LinearCode<T>>>> punctured_codes_BD;
    mutable details::OnceCache<std::vector<std::optional<LinearCode<T>>>> punctured_codes_ML;
#endif
    mutable details::OnceCache<bool> polynomial;
    mutable details::OnceCache<Polynomial<T>> gamma;
    mutable details::OnceCache<Trellis<T>> minimal_trellis;
    mutable details::OnceCache<TannerGraph<T>> tanner_graph;

   private:
    explicit LinearCode(Matrix<T> G) : LinearCode(G.get_n(), G.get_m(), G) {}

    static Matrix<T> G_from_codewords(const std::vector<Vector<T>>& codewords) {
        if (codewords.empty())
            throw std::invalid_argument("Cannot construct linear code from codewords: no codewords provided");

        const size_t n = codewords.front().get_n();
        Matrix<T> Gp(codewords.size(), n);

        for (size_t i = 0; i < codewords.size(); ++i) {
            if (codewords[i].get_n() != n)
                throw std::invalid_argument(
                    "Cannot construct linear code from codewords: codewords must have the same length");

#ifdef CECCO_ERASURE_SUPPORT
            for (size_t j = 0; j < n; ++j) {
                if (codewords[i][j].is_erased())
                    throw std::invalid_argument(
                        "Cannot construct linear code from codewords: codewords must not contain erasures");
            }
#endif

            Gp.set_submatrix(i, 0, Matrix<T>(codewords[i]));
        }

        size_t k;
        Gp.rref(&k);
        if (k == 0) return Matrix<T>(0, n);
        return Gp.get_submatrix(0, 0, k, n);
    }

    template <typename cost_t>
    Vector<T> viterbi_forward_pass_and_traceback(const Trellis<T>& Tr,
                                                 typename Trellis<T>::template Viterbi_Workspace<cost_t>& ws,
                                                 const std::string& filename = "") const {
        const size_t n = this->n;
        using ws_t = typename Trellis<T>::template Viterbi_Workspace<cost_t>;

        std::ofstream file;
        if constexpr (T::get_size() <= 64) {
            if (!filename.empty()) {
                file.open(filename);
                Tr.template tikz_header<ws_t>(file);
            }
        }

        std::fill(ws.path_costs_prev.begin(), ws.path_costs_prev.end(), ws_t::init);
        ws.path_costs_prev[0] = cost_t{0};

        for (size_t s = 0; s < n; ++s) {
            std::fill(ws.path_costs_curr.begin(), ws.path_costs_curr.end(), ws_t::init);
            std::fill(ws.tie_counts.begin(), ws.tie_counts.end(), 0);
            for (size_t j = 0; j < Tr.E[s].size(); ++j) {
                const auto& e = Tr.E[s][j];
                const cost_t cost = ws.path_costs_prev[e.from_id] + ws.edge_costs[s][j];
                if (cost < ws.path_costs_curr[e.to_id]) {
                    ws.path_costs_curr[e.to_id] = cost;
                    ws.backptrs[s + 1][e.to_id] = &e;
                    ws.tie_counts[e.to_id] = 1;
                } else if (cost == ws.path_costs_curr[e.to_id]) {
                    if (details::reservoir_accept(++ws.tie_counts[e.to_id])) ws.backptrs[s + 1][e.to_id] = &e;
                }
            }
            std::swap(ws.path_costs_prev, ws.path_costs_curr);
            if constexpr (T::get_size() <= 64) {
                if (!filename.empty()) Tr.tikz_picture(file, &ws, s + 1);
            }
        }

        Vector<T> c_est(n);

        size_t v = 0;
        cost_t best = ws.path_costs_prev[0];
        uint16_t sink_ties = 1;
        for (size_t u = 1; u < Tr.V[n].size(); ++u) {
            const cost_t c = ws.path_costs_prev[u];
            if (c < best) {
                best = c;
                v = u;
                sink_ties = 1;
            } else if (c == best) {
                if (details::reservoir_accept(++sink_ties)) v = u;
            }
        }

        for (size_t s = n; s > 0; --s) {
            const auto* e = ws.backptrs[s][v];
            c_est.set_component(s - 1, e->value);
            v = e->from_id;
        }

        return c_est;
    }

    Matrix<double> bcjr_forward_backward(const Trellis<T>& Tr, typename Trellis<T>::BCJR_Workspace& ws) const
        requires FiniteFieldType<T>
    {
        constexpr double neg_inf = -std::numeric_limits<double>::infinity();
        constexpr size_t q = T::get_size();
        const size_t n = this->n;

        ws.alpha[0][0] = 0.0;
        for (size_t s = 0; s < n; ++s) {
            for (size_t j = 0; j < Tr.E[s].size(); ++j) {
                const auto& e = Tr.E[s][j];
                const double val = ws.alpha[s][e.from_id] - ws.edge_costs[s][j];
                ws.alpha[s + 1][e.to_id] = details::max_star(ws.alpha[s + 1][e.to_id], val);
            }
        }

        for (size_t u = 0; u < Tr.V[n].size(); ++u) ws.beta[n][u] = 0.0;
        for (size_t s = n; s > 0; --s) {
            for (size_t j = 0; j < Tr.E[s - 1].size(); ++j) {
                const auto& e = Tr.E[s - 1][j];
                const double val = ws.beta[s][e.to_id] - ws.edge_costs[s - 1][j];
                ws.beta[s - 1][e.from_id] = details::max_star(ws.beta[s - 1][e.from_id], val);
            }
        }

        Matrix<double> res(q - 1, n);
        std::array<double, q> Lambda;
        for (size_t s = 0; s < n; ++s) {
            Lambda.fill(neg_inf);
            for (size_t j = 0; j < Tr.E[s].size(); ++j) {
                const auto& e = Tr.E[s][j];
                const double val = ws.alpha[s][e.from_id] - ws.edge_costs[s][j] + ws.beta[s + 1][e.to_id];
                const size_t a = e.value.get_label();
                Lambda[a] = details::max_star(Lambda[a], val);
            }
            for (size_t a = 1; a < q; ++a) res.set_component(a - 1, s, Lambda[0] - Lambda[a]);
        }

        return res;
    }

    Vector<T> bcjr_decode(const Matrix<double>& llrs, Matrix<double>* Lambda, const std::string& filename) const
        requires FiniteFieldType<T>
    {
        validate_length(llrs);
        if (!filename.empty() && T::get_size() > 2)
            throw std::invalid_argument("BCJR trellis TikZ export only supported for binary codes!");

        constexpr size_t q = T::get_size();
        const size_t n = this->n;

        if (k == 0) {
            if (Lambda != nullptr) *Lambda = Matrix<double>(q - 1, n, std::numeric_limits<double>::infinity());
            return Vector<T>(n);
        }

        const auto& Tr = get_minimal_trellis();
        typename Trellis<T>::BCJR_Workspace ws(Tr);
        ws.calculate_edge_costs(Tr, llrs);
        const Matrix<double> Lambda_local = bcjr_forward_backward(Tr, ws);
        if constexpr (T::get_size() == 2) {
            if (!filename.empty()) {
                ws.v.emplace(llrs);
                Tr.export_as_tikz(filename, &ws);
            }
        }
        if (Lambda != nullptr) *Lambda = Lambda_local;

        Vector<T> c_est(n);
        for (size_t i = 0; i < n; ++i) {
            size_t label = 0;
            double min_LLR = 0.0;  // the LLR of label 0 is 0 by definition
            uint16_t ties = 1;
            for (size_t a = 1; a < q; ++a) {
                const double L = Lambda_local(a - 1, i);
                if (L < min_LLR) {
                    min_LLR = L;
                    label = a;
                    ties = 1;
                } else if (L == min_LLR) {
                    if (details::reservoir_accept(++ties)) label = a;
                }
            }
            c_est.set_component(i, T(label));
        }
        return c_est;
    }

    Vector<T> bp_decode(const Matrix<double>& llrs, size_t max_iterations, Matrix<double>* Lambda,
                        const std::string& filename, size_t* nof_iterations) const
        requires FiniteFieldType<T>
    {
        validate_length(llrs);
        if (!filename.empty() && T::get_size() > 2)
            throw std::invalid_argument("BP Tanner graph TikZ export only supported for binary codes!");

        constexpr double neg_inf = -std::numeric_limits<double>::infinity();
        constexpr size_t q = T::get_size();
        const size_t n = this->n;

        if (k == 0) {
            if (Lambda != nullptr) *Lambda = Matrix<double>(q - 1, n, std::numeric_limits<double>::infinity());
            if (nof_iterations != nullptr) *nof_iterations = 0;
            return Vector<T>(n);
        }

        const auto& g = get_tanner_graph();
        typename TannerGraph<T>::BP_Workspace ws(g);
        ws.calculate_intrinsic(g, llrs);

        std::ofstream file;
        if constexpr (T::get_size() == 2) {
            if (!filename.empty()) {
                ws.v.emplace(llrs);
                file.open(filename);
                g.tikz_header(file);
            }
        }

        auto conv = [neg_inf](const std::array<double, q>& A, const std::array<double, q>& B) {
            std::array<double, q> R;
            R.fill(neg_inf);
            for (size_t x = 0; x < q; ++x) {
                if (A[x] == neg_inf) continue;
                for (size_t y = 0; y < q; ++y) {
                    if (B[y] == neg_inf) continue;
                    const size_t t = (T(x) + T(y)).get_label();
                    R[t] = details::max_star(R[t], A[x] + B[y]);
                }
            }
            return R;
        };

        auto normalize = [](std::array<double, q>& m) {
            const double ref = m[0];
            for (auto& x : m) x -= ref;
        };

        Vector<T> c_est(n);
        auto decide = [&](size_t j) {
            size_t best = 0;
            double best_val = ws.posterior[j][0];
            uint16_t ties = 1;
            for (size_t a = 1; a < q; ++a) {
                const double val = ws.posterior[j][a];
                if (val > best_val) {
                    best_val = val;
                    best = a;
                    ties = 1;
                } else if (val == best_val) {
                    if (details::reservoir_accept(++ties)) best = a;
                }
            }
            c_est.set_component(j, T(best));
        };

        for (size_t i = 0; i < g.checks.size(); ++i)
            for (size_t e = 0; e < g.checks[i].size(); ++e) ws.m_vc[i][e] = ws.intrinsic[g.checks[i][e].var_id];

        // Intrinsic (channel-only) posterior and hard decision; this is the result for max_iterations == 0.
        for (size_t j = 0; j < n; ++j) {
            ws.posterior[j] = ws.intrinsic[j];
            decide(j);
        }

        if constexpr (T::get_size() == 2) {
            if (!filename.empty()) g.tikz_picture(file, &ws);
        }

        std::vector<std::array<double, q>> W, F, B;
        size_t iter = 0;
        for (; !(c_est * HT).is_zero() && iter < max_iterations; ++iter) {
            for (size_t i = 0; i < g.checks.size(); ++i) {
                const size_t d = g.checks[i].size();
                W.resize(d);
                F.resize(d + 1);
                B.resize(d + 1);
                for (size_t e = 0; e < d; ++e) {
                    const T h = g.checks[i][e].value;
                    W[e].fill(neg_inf);
                    for (size_t a = 0; a < q; ++a) W[e][(h * T(a)).get_label()] = ws.m_vc[i][e][a];
                }
                F[0].fill(neg_inf);
                F[0][0] = 0.0;
                for (size_t e = 0; e < d; ++e) F[e + 1] = conv(F[e], W[e]);
                B[d].fill(neg_inf);
                B[d][0] = 0.0;
                for (size_t e = d; e-- > 0;) B[e] = conv(W[e], B[e + 1]);
                for (size_t e = 0; e < d; ++e) {
                    const std::array<double, q> excl = conv(F[e], B[e + 1]);
                    const T h = g.checks[i][e].value;
                    for (size_t a = 0; a < q; ++a) ws.m_cv[i][e][a] = excl[(-(h * T(a))).get_label()];
                    normalize(ws.m_cv[i][e]);
                }
            }

            for (size_t j = 0; j < n; ++j) {
                ws.posterior[j] = ws.intrinsic[j];
                for (const auto& [i, e] : ws.var_edges[j])
                    for (size_t a = 0; a < q; ++a) ws.posterior[j][a] += ws.m_cv[i][e][a];
                decide(j);
                for (const auto& [i, e] : ws.var_edges[j]) {
                    for (size_t a = 0; a < q; ++a) ws.m_vc[i][e][a] = ws.posterior[j][a] - ws.m_cv[i][e][a];
                    normalize(ws.m_vc[i][e]);
                }
            }

            if constexpr (T::get_size() == 2) {
                if (!filename.empty()) g.tikz_picture(file, &ws);
            }
        }

        if (nof_iterations != nullptr) *nof_iterations = iter;

        if (Lambda != nullptr) {
            Matrix<double> res(q - 1, n);
            for (size_t j = 0; j < n; ++j)
                for (size_t a = 1; a < q; ++a) res.set_component(a - 1, j, ws.posterior[j][0] - ws.posterior[j][a]);
            *Lambda = std::move(res);
        }

        return c_est;
    }

#ifdef CECCO_ERASURE_SUPPORT
    void init_punctured_codes_BD() const {
        punctured_codes_BD.call_once([this] {
            if (punctured_codes_BD.has_value()) return;
            std::clog << "--> Preparing punctured codes for BD error/erasure decoding" << std::endl;
            size_t count = 0;
            for (size_t tau = 1; tau <= get_dmin() - 1; ++tau) count += bin<size_t>(this->n, tau);
            punctured_codes_BD.emplace(count);
            for (size_t tau = 1; tau <= get_dmin() - 1; ++tau) {
                std::vector<bool> mask(this->n, false);
                std::fill(mask.begin(), mask.begin() + tau, true);
                do {
                    std::vector<size_t> X;
                    X.reserve(tau);
                    for (auto it = mask.cbegin(); it != mask.cend(); ++it)
                        if (*it) X.push_back(static_cast<size_t>(it - mask.cbegin()));
                    punctured_codes_BD.value()[pos_to_index(X)].emplace(puncture(*this, X));
                } while (std::prev_permutation(mask.begin(), mask.end()));
            }
        });
    }

    void init_punctured_codes_ML() const {
        punctured_codes_ML.call_once([this] {
            if (punctured_codes_ML.has_value()) return;
            std::clog << "--> Preparing punctured codes for ML error/erasure decoding" << std::endl;
            size_t bd_count = 0;
            for (size_t t = 1; t <= get_dmin() - 1; ++t) bd_count += bin<size_t>(this->n, t);
            size_t count = 0;
            for (size_t tau = get_dmin(); tau <= this->n; ++tau) count += bin<size_t>(this->n, tau);
            punctured_codes_ML.emplace(count);
            for (size_t tau = get_dmin(); tau <= this->n; ++tau) {
                std::vector<bool> mask(this->n, false);
                std::fill(mask.begin(), mask.begin() + tau, true);
                do {
                    std::vector<size_t> X;
                    X.reserve(tau);
                    for (auto it = mask.cbegin(); it != mask.cend(); ++it)
                        if (*it) X.push_back(static_cast<size_t>(it - mask.cbegin()));
                    punctured_codes_ML.value()[pos_to_index(X) - bd_count].emplace(puncture(*this, X));
                } while (std::prev_permutation(mask.begin(), mask.end()));
            }
        });
    }

    size_t pos_to_index(const std::vector<size_t>& pos) const {
        const size_t tau = pos.size();

        if (tau == 0) throw std::invalid_argument("Cannot calculate punctured code index from erasure positions!");

        size_t offset = 0;
        for (size_t t = 1; t < tau; ++t) offset += bin<size_t>(this->n, t);

        size_t rank = 0;
        for (size_t i = 0; i < tau; ++i) {
            const size_t start = (i == 0) ? 0 : (pos[i - 1] + 1);
            for (size_t x = start; x < pos[i]; ++x) rank += bin<size_t>(this->n - 1 - x, tau - 1 - i);
        }
        return offset + rank;
    }
#endif

    InfInt N(size_t ell, size_t h, size_t s) const {
        const size_t n = this->n;

        if constexpr (std::is_same_v<T, Fp<2>>) {
            InfInt res = 0;
            for (size_t u = 0; u <= n; ++u) {
                for (size_t w = 0; w <= n; ++w) {
                    if (u + w == s && ell + u - w == h) {
                        res += bin<InfInt>(n - ell, u) * bin<InfInt>(ell, w);
                    }
                }
            }
            return res;
        } else {
            InfInt res = 0;

            for (size_t u = 0; u <= n; ++u) {
                for (size_t v = 0; v <= n; ++v) {
                    for (size_t w = 0; w <= n; ++w) {
                        if (u + v + w == s && ell + u - w == h) {
                            res += bin<InfInt>(n - ell, u) * bin<InfInt>(ell, v) * bin<InfInt>(ell - v, w) *
                                   sqm<InfInt>(T::get_size() - 1, u) * sqm<InfInt>(T::get_size() - 2, v);
                            break;  // no need to check for further matching w
                        }
                    }
                }
            }

            /*
            // Blahut alternative
            for (size_t i=0; i<=n; ++i) {
                for (size_t j=0; j<=n; ++j) {
                    if (i+2*j+h==s+ell) {
                        if (ell>n || j+h-ell>n-ell || ell>j+h || i>ell || i>ell || j>ell-i) continue;
                        sum+=bin<InfInt>(n-ell, j+h-ell)*bin<InfInt>(ell, i)*bin<InfInt>(ell-i, j)*powl(F::get_size(),
            j+h-ell)*powl(F::get_size()-2, i);
                    }
                }
            }
            */

            return res;
        }
    }
};

template <FiniteFieldType T>
class UniverseCode : public LinearCode<T> {
   public:
    UniverseCode(size_t n) : LinearCode<T>(n, n, IdentityMatrix<T>(n)) {
        auto weight_enumerator = Polynomial<InfInt>();
        for (size_t i = 0; i <= n; ++i)
            weight_enumerator.set_coefficient(i, bin<InfInt>(n, i) * sqm<InfInt>(T::get_size() - 1, i));
        this->set_weight_enumerator(std::move(weight_enumerator));
    }

    UniverseCode(const LinearCode<T>& C) : LinearCode<T>(C) {
        if (this->n != this->k) throw std::invalid_argument("Linear code cannot be converted into universe code!");
    }

    UniverseCode(const UniverseCode&) = default;
    UniverseCode(UniverseCode&&) = default;
    UniverseCode& operator=(const UniverseCode&) = default;
    UniverseCode& operator=(UniverseCode&&) = default;

    UniverseCode<T> get_equivalent_code_in_standard_form() const { return UniverseCode<T>(this->n); }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Universe code";
    }

    ZeroCode<T> get_dual() const { return ZeroCode<T>(this->n); }
    Vector<T> enc(const Vector<T>& u) const override { return u; }
    Vector<T> encinv(const Vector<T>& c) const override { return c; }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        this->validate_length(r);
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r))
            throw decoding_failure("Universe code BD decoder failed, received vector contains erasures!");
#endif
        return r;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
        this->validate_length(r);
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
#endif
        return r;
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        this->validate_length(r);
        if (LinearCode<T>::erasures_present(r))
            throw decoding_failure("Universe code BD error/erasure decoder failed, received vector contains erasures!");
        return r;
    }
    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        this->validate_length(r);
        auto c_est = r;
        for (size_t i = 0; i < this->n; ++i) {
            if (c_est[i].is_erased()) c_est.set_component(i, T(0));
        }
        return c_est;
    }
#endif
};

template <FiniteFieldType T>
class ZeroCode : public LinearCode<T> {
   public:
    ZeroCode(size_t n) : LinearCode<T>(n, 0, ZeroMatrix<T>(1, n)) {}

    ZeroCode(const LinearCode<T>& C) : LinearCode<T>(C) {
        if (this->k != 0) throw std::invalid_argument("Linear code cannot be converted into zero code!");
    }

    ZeroCode(const ZeroCode&) = default;
    ZeroCode(ZeroCode&&) = default;
    ZeroCode& operator=(const ZeroCode&) = default;
    ZeroCode& operator=(ZeroCode&&) = default;

    ZeroCode<T> get_equivalent_code_in_standard_form() const { return ZeroCode<T>(this->n); }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Zero code";
    }
};

template <FiniteFieldType T>
class HammingCode : public LinearCode<T> {
    friend class SimplexCode<T>;

   public:
    HammingCode(size_t s) : LinearCode<T>(Hamming_n(s), Hamming_k(s), Hamming_H(s)), s(s) {
        constexpr size_t q = T::get_size();
        Polynomial<InfInt> weight_enumerator_dual;     // weight enumerator of the dual...
        weight_enumerator_dual.set_coefficient(0, 1);  // ... code, a q-ary simplex code...
        weight_enumerator_dual.set_coefficient(sqm<size_t>(q, s - 1), sqm<InfInt>(q, s) - 1);  // ... easy to calculate
        this->set_weight_enumerator(MacWilliamsIdentity<T>(weight_enumerator_dual, this->n, this->n - this->k));
    }

    HammingCode(const LinearCode<T>& C) : LinearCode<T>(C), s(0) {
        for (size_t s_cand = 2; s_cand < std::numeric_limits<size_t>::max(); ++s_cand) {
            const size_t n = Hamming_n(s_cand);
            if (n > this->n) break;
            if (n != this->n || Hamming_k(s_cand) != this->k) continue;

            std::vector<size_t> seen;
            seen.reserve(this->n);
            bool valid = true;
            for (size_t i = 0; i < this->n && valid; ++i) {
                const auto h = this->HT.get_row(i);
                T inv;
                for (size_t j = 0; j < h.get_n(); ++j)
                    if (!h[j].is_zero()) {
                        inv = T(1) / h[j];
                        break;
                    }
                if (inv.is_zero()) {
                    valid = false;
                    break;
                }
                const size_t key = (inv * h).as_integer();
                if (std::ranges::find(seen, key) != seen.end()) {
                    valid = false;
                    break;
                }
                seen.push_back(key);
            }
            if (valid) {
                this->s = s_cand;
                this->set_dmin(3);
                return;
            }
        }
        throw std::invalid_argument("Linear code cannot be converted into Hamming code!");
    }

    HammingCode(const HammingCode&) = default;
    HammingCode(HammingCode&&) = default;
    HammingCode& operator=(const HammingCode&) = default;
    HammingCode& operator=(HammingCode&&) = default;

    HammingCode<T> get_equivalent_code_in_standard_form() const {
        return HammingCode<T>(LinearCode<T>::get_equivalent_code_in_standard_form());
    }

    size_t get_s() const noexcept { return s; }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) os << BOLD("Hamming code") " with properties: { s = " << s << " }";
    }

    SimplexCode<T> get_dual() const { return SimplexCode<T>(s); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
#endif
        this->validate_length(r);

        constexpr size_t q = T::get_size();
        const auto s = r * this->HT;
        if (s.is_zero()) return r;
        auto c_est = r;
        for (size_t i = 0; i < this->n; ++i) {
            for (size_t j = 1; j < q; ++j) {
                T a(j);
                if (s == a * this->HT.get_row(i)) {
                    c_est.set_component(i, c_est[i] - a);
                    return c_est;
                }
            }
        }

        return c_est;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
#endif

        return dec_BD(r);
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        const auto X = this->erasure_positions(r);
        const size_t tau = X.size();

        if (tau == 0) return dec_BD(r);
        if (tau > 2) throw decoding_failure("Hamming code BD error/erasure decoder failed!");

        auto s = Vector<T>(this->n - this->k);
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased()) s += r[i] * this->HT.get_row(i);
        }

        auto c_est = r;

        if (tau == 1) {
            if (s.is_zero()) {
                c_est.set_component(X[0], 0);
            } else {
                for (size_t i = 0; i < this->n - this->k; ++i) {
                    if (!this->HT(X[0], i).is_zero()) {
                        c_est.set_component(X[0], -s[i] / this->HT(X[0], i));
                        break;
                    }
                }
            }
        } else if (tau == 2) {
            const auto c0 = transpose(Matrix<T>(this->HT.get_row(X[0])));
            const auto c1 = transpose(Matrix<T>(this->HT.get_row(X[1])));
            const auto c2 = -transpose(Matrix<T>(s));
            const auto M = horizontal_join(horizontal_join(c0, c1), c2);
            const auto B = M.basis_of_nullspace();
            c_est.set_component(X[0], B(0, 0));
            c_est.set_component(X[1], B(0, 1));
        }

        if (!(c_est * this->HT).is_zero()) throw decoding_failure("Hamming code BD error/erasure decoder failed!");

        return c_est;
    }

    Vector<T> dec_ML_EE(const Vector<T>& r) const override { return dec_BD_EE(r); }
#endif

   private:
    size_t s;

    static size_t Hamming_n(size_t s) {
        constexpr size_t q = T::get_size();
        return (sqm<size_t>(q, s) - 1) / (q - 1);
    }

    static size_t Hamming_k(size_t s) { return Hamming_n(s) - s; }

    static Matrix<T> Hamming_H(size_t s) {
        const size_t n = Hamming_n(s);
        auto H = Matrix<T>(n, s);
        size_t i = 0;
        // topmost element loop
        for (size_t top = 0; top < s; ++top) {
            const auto v = IdentityMatrix<T>(s - top - 1).rowspace();
            for (size_t j = 0; j < v.size(); ++j) {
                H.set_component(n - i - 1, top, T(1));
                H.set_submatrix(n - i - 1, top + 1, Matrix(v[v.size() - j - 1]));
                ++i;
            }
        }
        H.transpose();
        return H;
    }
};

template <FiniteFieldType T>
class SimplexCode : public LinearCode<T> {
   public:
    SimplexCode(size_t s)
        : LinearCode<T>(HammingCode<T>::Hamming_n(s), s, HammingCode<T>::Hamming_H(s)),
          s(s),
          cols_as_integers(this->n) {
        constexpr size_t q = T::get_size();
        Polynomial<InfInt> weight_enumerator;
        weight_enumerator.set_coefficient(0, 1);
        weight_enumerator.set_coefficient(sqm<size_t>(q, s - 1), sqm<InfInt>(q, s) - 1);
        this->set_weight_enumerator(std::move(weight_enumerator));
        for (size_t i = 0; i < this->n; ++i) cols_as_integers[i] = this->G.get_col(i).as_integer();
    }

    SimplexCode(const LinearCode<T>& C) : LinearCode<T>(C), s(0), cols_as_integers(this->n) {
        constexpr size_t q = T::get_size();
        for (size_t s_cand = 2; s_cand < std::numeric_limits<size_t>::max(); ++s_cand) {
            const size_t n = HammingCode<T>::Hamming_n(s_cand);
            if (n > this->n) break;
            if (n == this->n) {
                if (this->k == s_cand && this->get_dmin() == sqm<size_t>(q, s_cand - 1)) {
                    this->s = s_cand;
                    for (size_t i = 0; i < this->n; ++i) cols_as_integers[i] = this->G.get_col(i).as_integer();
                    return;
                }
            }
        }
        throw std::invalid_argument("Linear code cannot be converted into Simplex code!");
    }

    SimplexCode(const SimplexCode&) = default;
    SimplexCode(SimplexCode&&) = default;
    SimplexCode& operator=(const SimplexCode&) = default;
    SimplexCode& operator=(SimplexCode&&) = default;

    SimplexCode<T> get_equivalent_code_in_standard_form() const {
        return SimplexCode<T>(LinearCode<T>::get_equivalent_code_in_standard_form());
    }

    size_t get_s() const noexcept { return s; }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) os << BOLD("Simplex code") " with properties: { s = " << s << " }";
    }

    HammingCode<T> get_dual() const { return HammingCode<T>(s); }
    Vector<T> dec_BD(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
#endif
        if constexpr (T::get_size() != 2) return LinearCode<T>::dec_BD(r);

        const auto c_est = dec_ML(r);
        if (dH(r, c_est) > this->get_tmax())
            throw decoding_failure("Simplex code BD decoder failed!");
        else
            return c_est;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
#endif
        if constexpr (T::get_size() != 2) return LinearCode<T>::dec_ML(r);
        this->validate_length(r);

        Vector<double> y(this->n + 1, 0.0);
        y.set_component(0, 0.0);
        for (size_t i = 0; i < this->n; ++i) {
            const size_t j = cols_as_integers[i];
            y.set_component(j, r[i].is_zero() ? 1.0 : -1.0);
        }

        FWHT(y);

        size_t hit = 0;
        double best = -std::numeric_limits<double>::infinity();
        uint16_t ties = 1;
        for (size_t i = 0; i < y.get_n(); ++i) {
            if (y[i] > best) {
                best = y[i];
                hit = i;
                ties = 1;
            } else if (y[i] == best) {
                if (details::reservoir_accept(++ties)) hit = i;
            }
        }

        Vector<T> u_est;
        u_est.from_integer(hit, this->k);

        return u_est * this->G;
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        if constexpr (T::get_size() != 2) return LinearCode<T>::dec_BD_EE(r);
        this->validate_length(r);

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }

        if (tau > this->get_dmin() - 1) throw decoding_failure("Simplex code BD error/erasure decoder failed!");

        const Vector<T> c_est = dec_ML_EE(r);

        size_t t = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased() && c_est[i] != r[i]) ++t;
        }

        if (2 * t + tau > this->get_dmin() - 1) throw decoding_failure("Simplex code BD error/erasure decoder failed!");

        return c_est;
    }

    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        if constexpr (T::get_size() != 2) return LinearCode<T>::dec_ML_EE(r);
        this->validate_length(r);

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }
        if (tau == this->n) return Vector<T>(this->n, T(0));

        Vector<double> y(this->n + 1, 0.0);
        y.set_component(0, 0.0);
        for (size_t i = 0; i < this->n; ++i) {
            const size_t j = cols_as_integers[i];
            if (!r[i].is_erased()) y.set_component(j, r[i].is_zero() ? 1.0 : -1.0);
        }

        FWHT(y);

        size_t hit = 0;
        double best = -std::numeric_limits<double>::infinity();
        uint16_t ties = 1;
        for (size_t u = 0; u < y.get_n(); ++u) {
            if (y[u] > best) {
                best = y[u];
                hit = u;
                ties = 1;
            } else if (y[u] == best) {
                if (details::reservoir_accept(++ties)) hit = u;
            }
        }

        Vector<T> u_est;
        u_est.from_integer(hit, this->k);

        return u_est * this->G;
    }
#endif

   private:
    static void FWHT(Vector<double>& a) {
        const size_t n = a.get_n();
        if (!std::has_single_bit(n)) throw std::invalid_argument("FWHT requires length 2^m!");
        for (size_t len = 1; 2 * len <= n; len *= 2) {
            for (size_t i = 0; i < n; i += 2 * len) {
                for (size_t j = 0; j < len; ++j) {
                    const double u = a[i + j];
                    const double v = a[i + j + len];
                    a.set_component(i + j, u + v);
                    a.set_component(i + j + len, u - v);
                }
            }
        }
    }

    size_t s;
    std::vector<size_t> cols_as_integers;
};

template <FiniteFieldType T>
class RepetitionCode : public LinearCode<T> {
    friend class SingleParityCheckCode<T>;

   public:
    RepetitionCode(size_t n) : LinearCode<T>(n, 1, Repetition_G(n)) {
        Polynomial<InfInt> weight_enumerator;
        weight_enumerator.set_coefficient(0, 1);
        weight_enumerator.set_coefficient(n, T::get_size() - 1);
        this->set_weight_enumerator(std::move(weight_enumerator));
        auto p = ZeroPolynomial<T>();
        p.set_coefficient(0, -T(1));
        p.set_coefficient(n, T(1));
        auto q = ZeroPolynomial<T>();
        q.set_coefficient(0, -T(1));
        q.set_coefficient(1, T(1));
        this->set_gamma(p / q);
    }

    RepetitionCode(const LinearCode<T>& C) : LinearCode<T>(C) {
        if (this->k != 1 || this->get_dmin() != this->n)
            throw std::invalid_argument("Linear code cannot be converted into repetition code!");
    }

    RepetitionCode(const RepetitionCode&) = default;
    RepetitionCode(RepetitionCode&&) = default;
    RepetitionCode& operator=(const RepetitionCode&) = default;
    RepetitionCode& operator=(RepetitionCode&&) = default;

    RepetitionCode<T> get_equivalent_code_in_standard_form() const { return RepetitionCode<T>(this->n); }
    RepetitionCode<T> get_identical_code_in_polynomial_form() const {
        return RepetitionCode<T>(LinearCode<T>::get_identical_code_in_polynomial_form());
    }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Repetition code";
    }

    SingleParityCheckCode<T> get_dual() const { return SingleParityCheckCode<T>(this->n); }
    Vector<T> enc(const Vector<T>& u) const override { return Vector<T>(this->n, u[0]); }
    Vector<T> encinv(const Vector<T>& c) const override { return Vector<T>(1, c[0]); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
#endif

        const auto c_est = dec_ML(r);
        constexpr size_t q = T::get_size();
        if (q != 2 || this->n % 2 == 0) {
            if (2 * dH(r, c_est) > this->get_dmin() - 1) throw decoding_failure("Repetition code BD decoder failed!");
        }

        return c_est;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
#endif
        this->validate_length(r);

        constexpr size_t q = T::get_size();
        std::array<size_t, q> counters{};
        for (size_t i = 0; i < this->n; ++i) ++counters[r[i].get_label()];
        const auto it = std::max_element(counters.cbegin(), counters.cend());
        std::size_t label = static_cast<std::size_t>(std::distance(counters.cbegin(), it));

        return Vector<T>(this->n, T(label));
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }

        if (tau == 0) return dec_BD(r);
        if (tau > this->get_dmin() - 1) throw decoding_failure("Repetition code BD error/erasure decoder failed!");

        constexpr size_t q = T::get_size();
        std::array<size_t, q> counters{};
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased()) ++counters[r[i].get_label()];
        }
        const auto it = std::max_element(counters.cbegin(), counters.cend());
        std::size_t label = static_cast<std::size_t>(std::distance(counters.cbegin(), it));

        const auto c_est = Vector<T>(this->n, T(label));
        size_t t = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased() && r[i] != c_est[i]) ++t;
        }

        if (q != 2 || this->n % 2 == 0) {
            if (2 * t + tau > this->get_dmin() - 1)
                throw decoding_failure("Repetition code BD error/erasure decoder failed!");
        }

        return c_est;
    }

    virtual Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }

        if (tau == 0) return dec_ML(r);
        if (tau == this->n) return Vector<T>(this->n, T(0));

        constexpr size_t q = T::get_size();
        std::array<size_t, q> counters{};
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased()) ++counters[r[i].get_label()];
        }
        const auto it = std::max_element(counters.cbegin(), counters.cend());
        std::size_t label = static_cast<std::size_t>(std::distance(counters.cbegin(), it));
        return Vector<T>(this->n, T(label));
    }
#endif

   private:
    static Matrix<T> Repetition_G(size_t n) { return Matrix<T>(1, n, T(1)); }
};

template <FiniteFieldType T>
class SingleParityCheckCode : public LinearCode<T> {
   public:
    SingleParityCheckCode(size_t n) : LinearCode<T>(n, n - 1, RepetitionCode<T>::Repetition_G(n)) {
        Polynomial<InfInt> weight_enumerator_rep;
        weight_enumerator_rep.set_coefficient(0, 1);
        weight_enumerator_rep.set_coefficient(n, T::get_size() - 1);
        this->set_weight_enumerator(MacWilliamsIdentity<T>(weight_enumerator_rep, n, 1));
        auto q = ZeroPolynomial<T>();
        q.set_coefficient(0, -T(1));
        q.set_coefficient(1, T(1));
        this->set_gamma(q);
    }

    SingleParityCheckCode(const LinearCode<T>& C) : LinearCode<T>(C) {
        if (this->k != this->n - 1)
            throw std::invalid_argument("Linear code cannot be converted into single parity check code!");
        const auto& HTp = this->get_HT();
        for (size_t i = 0; i < this->n; ++i)
            if (HTp(i, 0).is_zero())
                throw std::invalid_argument("Linear code cannot be converted into single parity check code!");
    }

    SingleParityCheckCode(const SingleParityCheckCode&) = default;
    SingleParityCheckCode(SingleParityCheckCode&&) = default;
    SingleParityCheckCode& operator=(const SingleParityCheckCode&) = default;
    SingleParityCheckCode& operator=(SingleParityCheckCode&&) = default;

    SingleParityCheckCode<T> get_equivalent_code_in_standard_form() const {
        auto Gp = this->G;
        Gp.rref();
        return SingleParityCheckCode<T>(LinearCode<T>(this->n, this->k, Gp));
    }
    SingleParityCheckCode<T> get_identical_code_in_polynomial_form() const {
        return SingleParityCheckCode<T>(LinearCode<T>::get_identical_code_in_polynomial_form());
    }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Single parity check code";
    }

    RepetitionCode<T> get_dual() const { return RepetitionCode<T>(this->n); }
    Vector<T> dec_BD(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
#endif
        this->validate_length(r);

        if (!(r * this->HT)[0].is_zero()) throw decoding_failure("Single parity check code BD decoder failed!");

        return r;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
#endif
        this->validate_length(r);

        const T s = (r * this->HT)[0];
        if (s.is_zero()) return r;

        Vector<T> c_est = r;
        c_est.set_component(0, r[0] - s / (this->HT)(0, 0));
        return c_est;
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        const auto X = this->erasure_positions(r);
        const size_t tau = X.size();

        if (tau == 0) return dec_BD(r);
        if (tau > 1) throw decoding_failure("Single parity check code BD error/erasure decoder failed!");

        T s = T(0);
        for (size_t i = 0; i < this->n; ++i)
            if (!r[i].is_erased()) s += (this->HT)(i, 0) * r[i];

        Vector<T> c_est = r;
        c_est.set_component(X[0], -s / (this->HT)(X[0], 0));
        return c_est;
    }

    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        const auto X = this->erasure_positions(r);
        const size_t tau = X.size();

        if (tau == 0) return dec_ML(r);
        if (tau == this->n) return Vector<T>(this->n, T(0));

        T s = T(0);
        for (size_t i = 0; i < this->n; ++i)
            if (!r[i].is_erased()) s += (this->HT)(i, 0) * r[i];

        Vector<T> c_est = r;
        c_est.set_component(X[0], -s / (this->HT)(X[0], 0));
        for (size_t i = 1; i < tau; ++i) c_est.set_component(X[i], T(0));
        return c_est;
    }
#endif
};

template <FiniteFieldType T>
    requires(std::is_same_v<T, Fp<2>> || std::is_same_v<T, Fp<3>>)
class GolayCode : public LinearCode<T> {
   public:
    GolayCode() : LinearCode<T>(Golay_n(), Golay_k(), Golay_G()) {
        if constexpr (T::get_size() == 2) {
            this->set_weight_enumerator(Polynomial<InfInt>(
                {1, 0, 0, 0, 0, 0, 0, 253, 506, 0, 0, 1288, 1288, 0, 0, 506, 253, 0, 0, 0, 0, 0, 0, 1}));
            this->set_gamma(Polynomial<T>({1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1}));
        } else {
            this->set_weight_enumerator(Polynomial<InfInt>({1, 0, 0, 0, 0, 132, 132, 0, 330, 110, 0, 24}));
            this->set_gamma(Polynomial<T>({2, 0, 1, 2, 1, 1}));
        }
    }

    GolayCode(const LinearCode<T>& C) : LinearCode<T>(C) {
        // this is a non-trivial result, uses uniqueness of Steiner systems
        if (this->n == Golay_n() && this->k == Golay_k()) {
            if (std::is_same_v<T, Fp<2>> && this->get_dmin() == 7)
                return;
            else if (std::is_same_v<T, Fp<3>> && this->get_dmin() == 5)
                return;
        }
        throw std::invalid_argument("Linear code cannot be converted into Golay code!");
    }

    GolayCode(const GolayCode&) = default;
    GolayCode(GolayCode&&) = default;
    GolayCode& operator=(const GolayCode&) = default;
    GolayCode& operator=(GolayCode&&) = default;

    GolayCode<T> get_equivalent_code_in_standard_form() const {
        return GolayCode<T>(LinearCode<T>::get_equivalent_code_in_standard_form());
    }
    GolayCode<T> get_identical_code_in_polynomial_form() const {
        return GolayCode<T>(LinearCode<T>::get_identical_code_in_polynomial_form());
    }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            if constexpr (std::is_same_v<T, Fp<2>>) {
                os << "Binary Golay code";
            } else {
                os << "Ternary Golay code";
            }
        }
    }

   private:
    static constexpr size_t Golay_n() noexcept {
        if constexpr (std::is_same_v<T, Fp<2>>)
            return 23;
        else
            return 11;
    }

    static constexpr size_t Golay_k() noexcept {
        if constexpr (std::is_same_v<T, Fp<2>>)
            return 12;
        else
            return 6;
    }

    static Matrix<T> Golay_G() {
        Polynomial<T> gamma;
        if constexpr (std::is_same_v<T, Fp<2>>)
            gamma = Polynomial<T>({1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1});
        else
            gamma = Polynomial<T>({2, 0, 1, 2, 1, 1});

        const size_t n = Golay_n();
        const size_t k = Golay_k();

        return ToeplitzMatrix(pad_back(pad_front(Vector<T>(gamma), n), 2 * k + gamma.degree() - 1), k, n);
    }
};

template <FieldType T>
class GRSCode : public LinearCode<T> {
   public:
    GRSCode(const Vector<T>& a, const Vector<T>& d, size_t k) try : LinearCode
        <T>(a.get_n(), k, GRS_G(a, d, k)), a(a), d(d) {
            const size_t n = this->n;

            for (size_t i = 0; i < n; ++i) {
                if (d[i] == T(0)) throw std::invalid_argument("GRS codes must have nonzero column multipliers!");
            }

            if constexpr (FiniteFieldType<T>) {
                constexpr size_t q = T::get_size();

                Polynomial<InfInt> weight_enumerator;
                weight_enumerator.set_coefficient(0, 1);
                for (size_t i = n - k + 1; i <= n; ++i) {
                    InfInt sum = 0;
                    for (size_t j = 0; j <= i - (n - k + 1); ++j) {
                        InfInt s = j % 2 ? -1 : 1;
                        sum += s * bin<InfInt>(i - 1, j) * sqm<InfInt>(q, i - j - (n - k + 1));
                    }
                    weight_enumerator.set_coefficient(i, bin<InfInt>(n, i) * (q - 1) * sum);
                }

                this->set_weight_enumerator(std::move(weight_enumerator));
            } else {
                this->set_dmin(n - k + 1);
            }
        }
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct GRS code: ") + e.what());
    }

    GRSCode(const GRSCode& other)
        : LinearCode<T>(other),
          a(other.a),
          d(other.d),
          G_canonical(other.G_canonical),
          HT_canonical(other.HT_canonical) {}

    GRSCode(GRSCode&& other)
        : LinearCode<T>(std::move(other)),
          a(std::move(other.a)),
          d(std::move(other.d)),
          G_canonical(std::move(other.G_canonical)),
          HT_canonical(std::move(other.HT_canonical)) {}

    GRSCode& operator=(const GRSCode& other) {
        if (this != &other) {
            LinearCode<T>::operator=(other);
            a = other.a;
            d = other.d;
            G_canonical = other.G_canonical;
            HT_canonical = other.HT_canonical;
        }
        return *this;
    }

    GRSCode& operator=(GRSCode&& other) {
        if (this != &other) {
            LinearCode<T>::operator=(std::move(other));
            a = std::move(other.a);
            d = std::move(other.d);
            G_canonical = std::move(other.G_canonical);
            HT_canonical = std::move(other.HT_canonical);
        }
        return *this;
    }

    const Vector<T>& get_a() const noexcept { return a; }
    const Vector<T>& get_d() const noexcept { return d; }

    GRSCode<T> get_equivalent_code_in_standard_form() const {
        auto Gp = this->G;
        Gp.rref();
        GRSCode res(a, d, this->k, std::move(Gp));
        if (this->dmin.has_value()) res.set_dmin(*(this->dmin));
        if constexpr (FiniteFieldType<T>) {
            if (this->weight_enumerator.has_value()) res.set_weight_enumerator(*(this->weight_enumerator));
            if (this->p_ary_image_weight_enumerator.has_value())
                res.p_ary_image_weight_enumerator = this->p_ary_image_weight_enumerator;
        }
        return res;
    }

    bool is_singly_extended() const {
        for (size_t i = 0; i < this->n; ++i) {
            if (a[i] == T(0)) return true;
        }
        return false;
    }

    bool is_primitive() const {
        if constexpr (!FiniteFieldType<T>) {
            return false;
        } else {
            const size_t n = this->n;
            if (n != T::get_size() - 1) return false;
            for (size_t i = 0; i < n; ++i) {
                if (a[i] == T(0)) return false;
            }
            return true;
        }
    }

    bool is_narrow_sense() const {
        for (size_t i = 0; i < this->n; ++i) {
            if (d[i] != T(1)) return false;
        }
        return true;
    }

    bool is_normalized() const { return a == d; }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            os << BOLD("GRS code") " with properties: { a = " << a << ", d = " << d;
            if (is_singly_extended()) {
                os << " singly-extended";
            }
            if (is_primitive()) {
                os << " primitive";
            }
            if (is_narrow_sense()) {
                os << " narrow-sense";
            }
            if (is_normalized()) {
                os << " normalized";
            }
            os << " }";
        }
    }

    Vector<T> dec_BD(const Vector<T>& r) const override { return dec_WBA(r); }
    Vector<T> dec_WBA(const Vector<T>& r) const override { return dec_WBA_impl(r, {}); }
    Vector<T> dec_BMA(const Vector<T>& r) const override { return dec_BMA_impl(r, {}); }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override { return dec_WBA_EE(r); }

    Vector<T> dec_WBA_EE(const Vector<T>& r) const override { return dec_WBA_impl(r, this->erasure_positions(r)); }

    Vector<T> dec_BMA_EE(const Vector<T>& r) const override { return dec_BMA_impl(r, this->erasure_positions(r)); }
#endif

   protected:
    GRSCode(const Vector<T>& a, const Vector<T>& d, size_t k, Matrix<T> G)
        : LinearCode<T>(a.get_n(), k, std::move(G)), a(a), d(d) {}

   private:
    static Matrix<T> GRS_G(const Vector<T>& a, const Vector<T>& d, size_t k) {
        if (a.get_n() != d.get_n())
            throw std::invalid_argument("code locators a and column multipliers d must have the same length");
        return VandermondeMatrix<T>(a, k) * DiagonalMatrix<T>(d);
    }

    Vector<T> dec_WBA_impl(const Vector<T>& r, const std::vector<size_t>& X) const {
        this->validate_length(r);

        const size_t tau = X.size();
        if (tau > this->get_dmin() - 1)
            throw decoding_failure("GRS code WBA error/erasure decoder failed (too many erasures)!");

        const size_t n = this->n;
        const size_t k = this->k;

        const auto ap = delete_components(a, X);
        const auto dp = delete_components(d, X);
        const size_t np = n - tau;
        const size_t tmaxp = (np - k) / 2;
        const auto rp = delete_components(r, X);

        Vector<T> rp_norm(np);
        for (size_t i = 0; i < np; ++i) rp_norm.set_component(i, rp[i] / dp[i]);

        const auto M0 = transpose(VandermondeMatrix<T>(ap, np - tmaxp));
        const auto M1 = DiagonalMatrix(rp_norm) * transpose(VandermondeMatrix<T>(ap, np - tmaxp - k + 1));
        const auto M = horizontal_join(M0, M1);

        const auto B = M.basis_of_kernel();

        size_t i = 0;
        Polynomial<T> Q0;
        for (size_t j = 0; j <= np - tmaxp - 1; ++j, ++i) Q0.set_coefficient(j, B(0, i));
        Polynomial<T> Q1;
        for (size_t j = 0; j <= np - tmaxp - k; ++j, ++i) Q1.set_coefficient(j, B(0, i));

        const auto temp = poly_long_div(-Q0, Q1);
        const auto quotient = temp.first;
        const auto remainder = temp.second;
        if (!remainder.is_zero() || quotient.degree() >= k)
            throw decoding_failure("GRS code WBA error/erasure decoder failed (true error beyond BD radius)!");

        const auto u_est = pad_back(Vector<T>(quotient), k);
        G_canonical.call_once([this] {
            if (G_canonical.has_value()) return;
            G_canonical.emplace(VandermondeMatrix<T>(a, this->k) * DiagonalMatrix<T>(d));
        });
        const auto c_est = u_est * G_canonical.value();
        if (dH(r, c_est) > tmaxp)
            throw decoding_failure("GRS code WBA error/erasure decoder failed (identified error beyond BD radius)!");

        return c_est;
    }

    Vector<T> dec_BMA_impl(const Vector<T>& r, const std::vector<size_t>& X) const {
        this->validate_length(r);

        const size_t n = this->n;
        const size_t redundancy = n - this->k;
        const size_t tau = X.size();
        if (tau > redundancy) throw decoding_failure("GRS code BMA error/erasure decoder failed (too many erasures)!");

        Vector<T> rp = r;
        for (size_t i = 0; i < tau; ++i) rp.set_component(X[i], T(0));

        HT_canonical.call_once([this, n, redundancy] {
            if (HT_canonical.has_value()) return;

            auto dells = VandermondeMatrix<T>(a, n).invert().get_col(n - 1);
            for (size_t i = 0; i < n; ++i) dells.set_component(i, dells[i] / d[i]);
            const auto D = DiagonalMatrix<T>(dells);
            const auto V = VandermondeMatrix<T>(a, redundancy);
            HT_canonical.emplace(transpose(V * D));
        });
        const auto& HT = HT_canonical.value();

        // syndrome
        const auto s = rp * HT;
        if (s.is_zero() && tau == 0) return r;
        const auto sx = Polynomial<T>(s);

        // erasure locator
        auto Psi = Monomial<T>(0, T(1));
        for (size_t i = 0; i < tau; ++i) Psi *= Polynomial<T>({-a[X[i]], T(1)});

        // updated syndrome
        auto sxp = Polynomial<T>(T(0));
        for (size_t i = 0; i < redundancy - tau; ++i) {
            T coeff(0);
            for (size_t j = 0; j <= tau; ++j) coeff += Psi[j] * s[i + j];
            sxp.set_coefficient(i, coeff);
        }

        // error locator (Berlekamp-Massey)
        auto Lambda_e = Monomial<T>(0, T(1));
        size_t t = 0;
        auto B_ref = Monomial<T>(0, T(1));
        size_t t_ref = 0;
        T discrepancy_ref(1);
        size_t last_update = 0;

        for (size_t i = 0; i < redundancy - tau; ++i) {
            T discrepancy(0);
            const size_t j0 = (t > i ? t - i : 0);
            for (size_t j = j0; j <= t; ++j) discrepancy += Lambda_e[j] * sxp[i + j - t];
            const auto ratio = discrepancy / discrepancy_ref;

            if (!discrepancy.is_zero()) {
                const auto B = Lambda_e;
                const size_t shift = i - last_update + 1;

                if (shift + t_ref <= t) {
                    Lambda_e -= ratio * Monomial<T>(t - shift - t_ref, T(1)) * B_ref;
                } else {
                    const auto temp = t;
                    Lambda_e = Monomial<T>(shift + t_ref - t, T(1)) * Lambda_e - ratio * B_ref;
                    t = shift + t_ref;
                    B_ref = B;
                    t_ref = temp;
                    discrepancy_ref = discrepancy;
                    last_update = i + 1;
                }
            }
        }

        if (2 * t + tau > redundancy)
            throw decoding_failure("GRS code BMA error/erasure decoder failed (identified error beyond BD radius)!");

        // error/erasure locator
        auto Lambda = Psi * Lambda_e;

        // Chien search
        std::vector<size_t> E;
        E.reserve(tau + t);
        for (size_t i = 0; i < n; ++i)
            if (Lambda(a[i]).is_zero()) E.push_back(i);

        if (E.size() != tau + t)
            throw decoding_failure("GRS code BMA error/erasure decoder failed (true error beyond BD radius)!");

        // Forney
        auto Omega = Polynomial<T>(T(0));
        for (size_t j = 0; j < tau + t; ++j) {
            T coeff(0);
            for (size_t m = j + 1; m <= tau + t; ++m) coeff += Lambda[m] * sx[m - j - 1];
            Omega.set_coefficient(j, coeff);
        }

        Lambda.differentiate(1);

        Vector<T> c_est = rp;
        for (size_t i = 0; i < E.size(); ++i) {
            const auto locator = a[E[i]];
            c_est.set_component(E[i], c_est[E[i]] - Omega(locator) / (HT(E[i], 0) * Lambda(locator)));
        }

        return c_est;
    }

    Vector<T> a;
    Vector<T> d;
    mutable details::OnceCache<Matrix<T>> G_canonical;
    mutable details::OnceCache<Matrix<T>> HT_canonical;
};

template <FiniteFieldType T>
class RSCode : public GRSCode<T> {
   public:
    RSCode(const T& alpha, size_t b, size_t k) try : RSCode(RS_a_and_D(alpha, b), k, alpha, b) {
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct RS code: ") + e.what());
    }

    RSCode(const RSCode&) = default;
    RSCode(RSCode&&) = default;
    RSCode& operator=(const RSCode&) = default;
    RSCode& operator=(RSCode&&) = default;

    T get_alpha() const { return alpha; }
    size_t get_b() const noexcept { return b; }

    RSCode<T> get_equivalent_code_in_standard_form() const {
        auto Gp = this->G;
        Gp.rref();
        RSCode<T> res(std::make_pair(this->get_a(), this->get_d()), this->k, alpha, b, std::move(Gp));
        if (this->dmin.has_value()) res.set_dmin(*(this->dmin));
        if (this->weight_enumerator.has_value()) res.set_weight_enumerator(*(this->weight_enumerator));
        if (this->p_ary_image_weight_enumerator.has_value())
            res.p_ary_image_weight_enumerator = this->p_ary_image_weight_enumerator;
        return res;
    }

    RSCode<T> get_identical_code_in_polynomial_form() const {
        RSCode<T> res(std::make_pair(this->get_a(), this->get_d()), this->k, alpha, b,
                      this->get_G_in_polynomial_form());
        if (this->dmin.has_value()) res.set_dmin(*(this->dmin));
        if (this->weight_enumerator.has_value()) res.set_weight_enumerator(*(this->weight_enumerator));
        if (this->p_ary_image_weight_enumerator.has_value())
            res.p_ary_image_weight_enumerator = this->p_ary_image_weight_enumerator;
        if (this->minimal_trellis.has_value()) res.minimal_trellis = this->minimal_trellis;
        if (this->codewords.has_value()) res.codewords = this->codewords;
        return res;
    }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            GRSCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            os << "RS code: { alpha = " << alpha << ", b = " << b;
            if (this->is_primitive()) os << ", primitive";
            if (this->is_narrow_sense()) os << ", narrow-sense";
            if (this->is_normalized()) os << ", normalized";
            os << " }";
        }
    }

   private:
    static std::pair<Vector<T>, Vector<T>> RS_a_and_D(const T& alpha, size_t b) {
        if (alpha == T(0)) throw std::invalid_argument("alpha must not be zero");
        const size_t n = alpha.get_multiplicative_order();
        const T alpha_1mb = sqm<T>(alpha, n + 1 - b % n);
        Vector<T> a(n), d(n);
        T a_pow = T(1), d_pow = T(1);
        for (size_t i = 0; i < n; ++i) {
            a.set_component(i, a_pow);
            d.set_component(i, d_pow);
            a_pow = a_pow * alpha;
            d_pow = d_pow * alpha_1mb;
        }
        return {std::move(a), std::move(d)};
    }

    static Polynomial<T> RS_gamma(const Vector<T>& a, size_t b, size_t k) {
        const size_t n = a.get_n();
        const size_t start = b % n;  // index of alpha^b in a
        auto gamma = ZeroPolynomial<T>();
        gamma.set_coefficient(0, T(1));
        for (size_t j = 0; j < n - k; ++j) {
            auto factor = ZeroPolynomial<T>();
            factor.set_coefficient(0, -a[(start + j) % n]);
            factor.set_coefficient(1, T(1));
            gamma = gamma * factor;
        }
        return gamma;
    }

    RSCode(std::pair<Vector<T>, Vector<T>> params, size_t k, const T& alpha, size_t b)
        : GRSCode<T>(params.first, params.second, k), alpha(alpha), b(b) {
        this->set_gamma(RS_gamma(params.first, b, k));
    }

    RSCode(std::pair<Vector<T>, Vector<T>> params, size_t k, const T& alpha, size_t b, Matrix<T> G)
        : GRSCode<T>(params.first, params.second, k, std::move(G)), alpha(alpha), b(b) {
        this->set_gamma(RS_gamma(params.first, b, k));
    }

    T alpha;
    size_t b;
};

class CordaroWagnerCode : public LinearCode<Fp<2>> {
   public:
    CordaroWagnerCode(size_t r, int8_t m) : LinearCode<Fp<2>>(3 * r + m, 2, CordaroWagner_G(r, m)), r(r), m(m) {
        if (r < 1) throw std::invalid_argument("Cordaro-Wagner codes must have r > 0!");
        if (m != -1 && m != 0 && m != 1)
            throw std::invalid_argument("Cordaro-Wagner codes must have m either -1, 0, or 1!");

        auto weight_enumerator = ZeroPolynomial<InfInt>();
        weight_enumerator.add_to_coefficient(0, 1);
        weight_enumerator.add_to_coefficient(2 * r, 1);
        weight_enumerator.add_to_coefficient(2 * r + m, 2);
        this->set_weight_enumerator(std::move(weight_enumerator));
    }

    CordaroWagnerCode(const LinearCode<Fp<2>>& C) : LinearCode<Fp<2>>(C), r(0), m(0) {
        if (this->k != 2) throw std::invalid_argument("Linear code cannot be converted into Cordaro-Wagner code!");

        r = std::floor(this->n / 3.0 + 1.0 / 2.0);
        m = this->n - 3 * r;

        const auto c1 = this->G.get_row(0);
        const auto c2 = this->G.get_row(1);
        std::array<size_t, 3> actual = {c1.wH(), c2.wH(), (c1 + c2).wH()};
        std::array<size_t, 3> expected = {2 * r, this->n - r, this->n - r};  // n - r = 2r + m
        std::ranges::sort(actual);
        std::ranges::sort(expected);
        if (actual != expected)
            throw std::invalid_argument("Linear code cannot be converted into Cordaro-Wagner code!");
    }

    CordaroWagnerCode(const CordaroWagnerCode&) = default;
    CordaroWagnerCode(CordaroWagnerCode&&) = default;
    CordaroWagnerCode& operator=(const CordaroWagnerCode&) = default;
    CordaroWagnerCode& operator=(CordaroWagnerCode&&) = default;

    CordaroWagnerCode get_equivalent_code_in_standard_form() const {
        return CordaroWagnerCode(LinearCode<Fp<2>>::get_equivalent_code_in_standard_form());
    }

    size_t get_r() const noexcept { return r; }
    int8_t get_m() const noexcept { return m; }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<Fp<2>>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0)
            os << BOLD("Cordaro-Wagner code") " with properties: { r = " << r << ", m = " << static_cast<int>(m)
               << " }";
    }

    Vector<Fp<2>> dec_BD(const Vector<Fp<2>>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<Fp<2>>::erasures_present(r)) return dec_BD_EE(r);
#endif
        this->validate_length(r);

        const size_t t = this->get_tmax();
        for (auto it = this->cbegin(); it != this->cend(); ++it) {
            if (dH(*it, r) <= t) return *it;
        }
        throw decoding_failure("Cordaro-Wagner code BD decoder failed!");
    }

    Vector<Fp<2>> dec_ML(const Vector<Fp<2>>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<Fp<2>>::erasures_present(r)) return dec_ML_EE(r);
#endif
        this->validate_length(r);

        Vector<Fp<2>> best;
        size_t best_t = std::numeric_limits<size_t>::max();
        uint16_t ties = 1;

        for (auto it = this->cbegin(); it != this->cend(); ++it) {
            const size_t t = dH(*it, r);
            if (t < best_t) {
                best = *it;
                best_t = t;
                ties = 1;
            } else if (t == best_t) {
                if (details::reservoir_accept(++ties)) best = *it;
            }
        }
        return best;
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<Fp<2>> dec_BD_EE(const Vector<Fp<2>>& r) const override {
        this->validate_length(r);

        const size_t dmin = this->get_dmin();

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }

        if (tau > dmin - 1) throw decoding_failure("Cordaro-Wagner code BD error/erasure decoder failed!");

        for (auto it = this->cbegin(); it != this->cend(); ++it) {
            size_t t = 0;
            for (size_t i = 0; i < this->n; ++i) {
                if (!r[i].is_erased() && r[i] != (*it)[i]) ++t;
            }

            if (2 * t + tau <= dmin - 1) return *it;
        }

        throw decoding_failure("Cordaro-Wagner code BD error/erasure decoder failed!");
    }

    Vector<Fp<2>> dec_ML_EE(const Vector<Fp<2>>& r) const override {
        this->validate_length(r);

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }
        if (tau == this->n) return Vector<Fp<2>>(this->n, Fp<2>(0));

        Vector<Fp<2>> best;
        size_t best_t = std::numeric_limits<size_t>::max();
        uint16_t ties = 1;

        for (auto it = this->cbegin(); it != this->cend(); ++it) {
            size_t t = 0;
            for (size_t i = 0; i < this->n; ++i) {
                if (!r[i].is_erased() && r[i] != (*it)[i]) ++t;
            }

            if (t < best_t) {
                best_t = t;
                best = *it;
                ties = 1;
            } else if (t == best_t) {
                if (details::reservoir_accept(++ties)) best = *it;
            }
        }

        return best;
    }
#endif

   private:
    static Matrix<Fp<2>> CordaroWagner_G(size_t r, int8_t m) {
        const size_t n = 3 * r + m;
        Matrix<Fp<2>> G(2, n);

        const Matrix<Fp<2>> type1_column(2, 1, {Fp<2>(1), Fp<2>(0)});
        for (size_t s = 0; s < r; ++s) G.set_submatrix(0, s, type1_column);  // r of them

        const Matrix<Fp<2>> type2_column(2, 1, {Fp<2>(0), Fp<2>(1)});
        for (size_t s = r; s < 2 * r + m; ++s) G.set_submatrix(0, s, type2_column);  // r+m of them

        const Matrix<Fp<2>> type3_column(2, 1, {Fp<2>(1), Fp<2>(1)});
        for (size_t s = 2 * r + m; s < 3 * r + m; ++s) G.set_submatrix(0, s, type3_column);  // r of them

        // => weight distribution is 1 + 2*x^(2r) + x^(2r+m)
        return G;
    }

    size_t r;
    int8_t m;
};

namespace details {

template <FieldType T>
Matrix<T> LDC_G(const Matrix<T>& G_U, const Matrix<T>& G_V) {
    const size_t n = G_U.get_n();
    if (G_V.get_n() != n) throw std::invalid_argument("Codes must have same length for LDC construction!");

    const size_t kU = G_U.get_m();
    const size_t kV = G_V.get_m();

    Matrix<T> G(kU + kV, 2 * n);
    G.set_submatrix(0, 0, G_U);
    G.set_submatrix(0, n, G_U);
    G.set_submatrix(kU, n, G_V);
    return G;
}

}  // namespace details

template <class BU, class BV>
    requires(std::derived_from<BU, LinearCode<typename BU::FIELD>> &&
             std::derived_from<BV, LinearCode<typename BV::FIELD>> &&
             std::same_as<typename BU::FIELD, typename BV::FIELD>)
class LDCCode : public LinearCode<typename BU::FIELD> {
   public:
    using T = typename BU::FIELD;

    LDCCode(const BU& U, const BV& V) try : LinearCode
        <T>(2 * U.get_n(), U.get_k() + V.get_k(), details::LDC_G(U.get_G(), V.get_G())), U(U), V(V) {}
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Trying to perform invalid length-doubling construction: ") + e.what());
    }

    LDCCode(const LDCCode&) = default;
    LDCCode(LDCCode&&) = default;
    LDCCode& operator=(const LDCCode&) = default;
    LDCCode& operator=(LDCCode&&) = default;

    const BU& get_U() const noexcept { return U; }
    const BV& get_V() const noexcept { return V; }

    virtual size_t get_dmin() const override {
        if constexpr (std::is_same_v<T, Fp<2>>) this->set_dmin(std::min({2 * U.get_dmin(), V.get_dmin()}));
        return this->LinearCode<T>::get_dmin();
    }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            const auto old = os.iword(details::index);
            os << BOLD("LDC code") " with properties: { ";
            os << "U = " << showbasic;
            U.LinearCode<T>::get_info(os);
            os << ", V = ";
            V.LinearCode<T>::get_info(os);
            os << " }";
            os.iword(details::index) = old;
        }
    }

    Vector<T> dec_recursive(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_recursive_EE(r);
#endif
        this->validate_length(r);

        auto rl = r.get_subvector(0, U.get_n());
        auto rr = r.get_subvector(U.get_n(), U.get_n());

        // U code
        const Vector<T> cl_hat_1 = dec_wrapper(U, rl);

        // V code
        const Vector<T> cr_hat = dec_wrapper(V, rr - rl);
        // ... then U code
        const Vector<T> cl_hat_2 = dec_wrapper(U, rr - cr_hat);

        const auto c_est_1 = concatenate(cl_hat_1, cl_hat_1 + cr_hat);
        const auto c_est_2 = concatenate(cl_hat_2, cl_hat_2 + cr_hat);
        if (dH(r, c_est_1) < dH(r, c_est_2))
            return c_est_1;
        else
            return c_est_2;
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_recursive_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }

        if (tau == 0) return dec_recursive(r);
        if (tau == this->n) return Vector<T>(this->n, T(0));

        auto rl = r.get_subvector(0, U.get_n());
        auto rr = r.get_subvector(U.get_n(), U.get_n());

        // U code
        const Vector<T> cl_hat_1 = dec_wrapper_EE(U, rl);

        // V code
        const Vector<T> cr_hat = dec_wrapper_EE(V, rr - rl);
        // ... then U code
        const Vector<T> cl_hat_2 = dec_wrapper_EE(U, rr - cr_hat);

        const auto c_est_1 = concatenate(cl_hat_1, cl_hat_1 + cr_hat);
        const auto c_est_2 = concatenate(cl_hat_2, cl_hat_2 + cr_hat);

        size_t t_1 = 0;
        size_t t_2 = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased()) {
                if (r[i] != c_est_1[i]) ++t_1;
                if (r[i] != c_est_2[i]) ++t_2;
            }
        }

        if (t_1 < t_2)
            return c_est_1;
        else
            return c_est_2;
    }
#endif

   private:
    BU U;
    BV V;

    Vector<T> dec_wrapper(const LinearCode<T>& C, const Vector<T>& r) const { return C.dec_ML(r); }

    Vector<T> dec_wrapper(const LDCCode& C, const Vector<T>& r) const { return C.dec_recursive(r); }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_wrapper_EE(const LinearCode<T>& C, const Vector<T>& r) const { return C.dec_ML_EE(r); }

    Vector<T> dec_wrapper_EE(const LDCCode& C, const Vector<T>& r) const { return C.dec_recursive_EE(r); }
#endif
};

class RMCode : public LinearCode<Fp<2>> {
   public:
    using T = Fp<2>;

    RMCode(size_t r, size_t m) try : LinearCode
        <T>(sqm(2, m), RM_k(r, m), RM_G(r, m)), r(r), m(m) { this->set_dmin(sqm(2, m - r)); }
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Trying to construct invalid RM code: ") + e.what());
    }

    RMCode(const RMCode&) = default;
    RMCode(RMCode&&) = default;
    RMCode& operator=(const RMCode&) = default;
    RMCode& operator=(RMCode&&) = default;

    size_t get_r() const noexcept { return r; }
    size_t get_m() const noexcept { return m; }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            os << BOLD("RM code") " with properties: { r = " << r << ", m = " << m << " }";
        }
    }

    // Docu note: there is no dec_ML, falls back to LinearCode since dec_recursive is only approximative
    // ML!

    Vector<T> dec_recursive(const Vector<T>& r) const override {
        this->validate_length(r);
        return dec_wrapper(this->r, m, r);
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_recursive_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }

        if (tau == 0) return dec_recursive(r);
        if (tau == this->n) return Vector<T>(this->n, T(0));

        return dec_wrapper_EE(this->r, m, r);
    }
#endif

   private:
    size_t r;
    size_t m;

    static size_t RM_k(size_t r, size_t m) {
        size_t k = 0;
        for (size_t i = 0; i <= r; ++i) k += bin<size_t>(m, i);
        return k;
    }

    static Matrix<T> RM_G(size_t r, size_t m) {
        if (r > m) throw std::invalid_argument("RM codes require 0 <= r <= m");
        const size_t n = sqm(2, m);

        if (r == 0) return Matrix<T>(1, n, T(1));
        if (r == m) return IdentityMatrix<T>(n);

        const auto GU = RM_G(r, m - 1);
        const auto GV = RM_G(r - 1, m - 1);
        return details::LDC_G(GU, GV);
    }

    static Vector<T> dec_wrapper(size_t r, size_t m, const Vector<T>& v) {
        const size_t n = sqm(2, m);

        // UC, decoding means returning v
        if (r == m) return v;

        // RepC, majority decision
        if (r == 0) {
            constexpr size_t q = T::get_size();
            std::array<size_t, q> counters{};
            for (size_t i = 0; i < n; ++i) ++counters[v[i].get_label()];
            const auto it = std::max_element(counters.cbegin(), counters.cend());
            return Vector<T>(n, T(static_cast<size_t>(std::distance(counters.cbegin(), it))));
        }

        const size_t np = n / 2;
        auto vl = v.get_subvector(0, np);
        auto vr = v.get_subvector(np, np);

        // U code
        const Vector<T> cl_hat_1 = dec_wrapper(r, m - 1, vl);

        // V code
        const Vector<T> cr_hat = dec_wrapper(r - 1, m - 1, vr - vl);
        // ... then U code
        const Vector<T> cl_hat_2 = dec_wrapper(r, m - 1, vr - cr_hat);

        if (dH(vl, cl_hat_1) + dH(vr, cl_hat_1 + cr_hat) < dH(vl, cl_hat_2) + dH(vr, cl_hat_2 + cr_hat))
            return concatenate(cl_hat_1, cl_hat_1 + cr_hat);
        else
            return concatenate(cl_hat_2, cl_hat_2 + cr_hat);
    }

#ifdef CECCO_ERASURE_SUPPORT
    static Vector<T> dec_wrapper_EE(size_t r, size_t m, const Vector<T>& v) {
        const size_t n = sqm(2, m);

        if (r == m) {
            auto c = v;
            for (size_t i = 0; i < n; ++i)
                if (c[i].is_erased()) c.set_component(i, T(0));
            return c;
        }

        if (r == 0) {
            constexpr size_t q = T::get_size();
            std::array<size_t, q> counters{};
            for (size_t i = 0; i < n; ++i)
                if (!v[i].is_erased()) ++counters[v[i].get_label()];
            const auto it = std::max_element(counters.cbegin(), counters.cend());
            return Vector<T>(n, T(static_cast<size_t>(std::distance(counters.cbegin(), it))));
        }

        const size_t np = n / 2;
        auto vl = v.get_subvector(0, np);
        auto vr = v.get_subvector(np, np);

        // U code
        const Vector<T> cl_hat_1 = dec_wrapper_EE(r, m - 1, vl);

        // V code...
        const Vector<T> cr_hat = dec_wrapper_EE(r - 1, m - 1, vr - vl);
        // ... then U code
        const Vector<T> cl_hat_2 = dec_wrapper_EE(r, m - 1, vr - cr_hat);

        const auto c_est_1 = concatenate(cl_hat_1, cl_hat_1 + cr_hat);
        const auto c_est_2 = concatenate(cl_hat_2, cl_hat_2 + cr_hat);

        size_t t_1 = 0;
        size_t t_2 = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!v[i].is_erased()) {
                if (v[i] != c_est_1[i]) ++t_1;
                if (v[i] != c_est_2[i]) ++t_2;
            }
        }

        if (t_1 < t_2)
            return c_est_1;
        else
            return c_est_2;
    }
#endif
};

template <class B>
    requires std::derived_from<B, LinearCode<typename B::FIELD>>
class SubfieldSubcode : public LinearCode<typename B::FIELD::BASE_FIELD> {
    using SUPER = typename B::FIELD;
    using SUB = typename SUPER::BASE_FIELD;

   public:
    SubfieldSubcode(const B& SuperCode) try : SubfieldSubcode(SuperCode, SSC_parameters(SuperCode)) {
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(
            std::string("Trying to construct a subfield subcode from a super code that is not over a superfield: ") +
            e.what());
    }

    SubfieldSubcode(const SubfieldSubcode&) = default;
    SubfieldSubcode(SubfieldSubcode&&) = default;
    SubfieldSubcode& operator=(const SubfieldSubcode&) = default;
    SubfieldSubcode& operator=(SubfieldSubcode&&) = default;

    const B& get_SuperCode() const noexcept { return SuperCode; }

    Vector<SUB> dec_supercode_BD(const Vector<SUB>& r) const {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<SUB>::erasures_present(r)) return dec_supercode_BD_EE(r);
#endif
        return project(SuperCode.dec_BD(Vector<SUPER>(r)));
    }

    Vector<SUB> dec_supercode_ML(const Vector<SUB>& r) const {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<SUB>::erasures_present(r)) return dec_supercode_ML_EE(r);
#endif
        return project(SuperCode.dec_ML(Vector<SUPER>(r)));
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<SUB> dec_supercode_BD_EE(const Vector<SUB>& r) const {
        return project(SuperCode.dec_BD_EE(Vector<SUPER>(r)));
    }

    Vector<SUB> dec_supercode_ML_EE(const Vector<SUB>& r) const {
        return project(SuperCode.dec_ML_EE(Vector<SUPER>(r)));
    }
#endif

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<SUB>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            const auto old = os.iword(details::index);
            os << BOLD("Subfield-subcode") " with properties: {" << std::endl;
            os << "Gamma = " << std::endl << Gamma << ", " << std::endl;
            os << "Supercode = " << showbasic;
            SuperCode.get_info(os);
            os << " " << showspecial << SuperCode << std::endl;
            os << "}";
            os.iword(details::index) = old;
        }
    }

   private:
    static Vector<SUB> project(const Vector<SUPER>& c) {
        try {
            return Vector<SUB>(c);
        } catch (const std::invalid_argument& e) {
            throw decoding_failure(std::string("Subfield subcode supercode decoder failed: ") + e.what());
        }
    }

    SubfieldSubcode(const B& SuperCode, std::pair<size_t, Matrix<SUPER>> parameters)
        : LinearCode<SUB>(SuperCode.get_n(), parameters.first, parameters.second * SuperCode.get_G()),
          SuperCode(SuperCode),
          Gamma(parameters.second) {}

    static std::pair<size_t, Matrix<SUPER>> SSC_parameters(const B& SuperCode) {
        const size_t n = SuperCode.get_n();
        const size_t kp = SuperCode.get_k();
        const size_t m = SUPER(0).template as_vector<SUB>().get_n();
        const auto Gp = SuperCode.get_G();

        auto T = [m](const SUPER& Gij) -> Matrix<SUB> {
            const auto v = pad_back(pad_front(Gij.template as_vector<SUB>().reverse(), 2 * m - 1), 3 * m - 2);
            return ToeplitzMatrix<SUB>(v, m, 2 * m - 1);
        };

        auto R = IdentityMatrix<SUB>(2 * m - 1);
        for (size_t j = 2 * m - 1; j-- > m;) {
            const auto I = IdentityMatrix<SUB>(j - m);
            const auto E = Matrix<SUB>(unit_vector<SUB>(m, 0));
            const auto C = CompanionMatrix<SUB>(Vector(SUPER::get_modulus())).transpose();
            const auto Rj = diagonal_join(I, vertical_join(E, C));
            R *= Rj;
        }

        Matrix<SUB> Gt(kp * m, n * (m - 1));
        for (size_t i = 0; i < kp; ++i) {
            for (size_t j = 0; j < n; ++j) Gt.set_submatrix(i * m, j * (m - 1), (T(Gp(i, j)) * R).delete_column(0));
        }
        const auto Gammat = Gt.transpose().basis_of_kernel();
        const size_t k = Gammat.rank();
        if (k == 0) return std::make_pair(0, Matrix<SUPER>(k, kp));

        Matrix<SUPER> Gamma(k, kp);
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < kp; ++j)
                Gamma.set_component(i, j, SUPER(Gammat.get_submatrix(i, j * m, 1, m).to_vector().reverse()));
        }

        return std::make_pair(k, Gamma);
    }

    B SuperCode;
    Matrix<SUPER> Gamma;
};

template <class B>
SubfieldSubcode(const B&) -> SubfieldSubcode<B>;

template <class B>
SubfieldSubcode(B&&) -> SubfieldSubcode<B>;

template <class B>
    requires std::derived_from<B, GRSCode<typename B::FIELD>>
class AlternantCode : public SubfieldSubcode<B> {
    using Base = SubfieldSubcode<B>;
    using SUPER = typename B::FIELD;
    using SUB = typename SUPER::BASE_FIELD;

   public:
    AlternantCode(const B& supercode) try : Base(supercode), delta(supercode.get_n() - supercode.get_k() + 1) {
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct alternant code: ") + e.what());
    }

    AlternantCode(const Vector<SUPER>& a, const Vector<SUPER>& d, size_t delta)
        requires std::is_same_v<B, GRSCode<SUPER>>
    try : Base(GRSCode<SUPER>(a, d, k_from_delta(a.get_n(), delta))), delta(delta) {
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct alternant code: ") + e.what());
    }

    AlternantCode(const AlternantCode&) = default;
    AlternantCode(AlternantCode&&) = default;
    AlternantCode& operator=(const AlternantCode&) = default;
    AlternantCode& operator=(AlternantCode&&) = default;

    const Vector<SUPER>& get_a() const noexcept { return this->get_SuperCode().get_a(); }
    const Vector<SUPER>& get_d() const noexcept { return this->get_SuperCode().get_d(); }
    virtual size_t get_delta() const noexcept { return delta; }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            Base::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            os << BOLD("Alternant code") " with properties: { delta = " << get_delta();
            if (this->get_SuperCode().is_primitive()) os << ", primitive";
            if (this->get_SuperCode().is_narrow_sense()) os << ", narrow-sense";
            if (this->get_SuperCode().is_normalized()) os << ", normalized";
            os << " }";
        }
    }

   protected:
    static size_t k_from_delta(size_t n, size_t delta) {
        if (delta == 0 || delta > n) throw std::invalid_argument("delta must satisfy 1 <= delta <= n");
        return n - delta + 1;
    }

   private:
    size_t delta;
};

template <class B>
    requires std::derived_from<B, RSCode<typename B::FIELD>>
class BCHCode : public AlternantCode<B> {
    using Base = AlternantCode<B>;
    using SUPER = typename B::FIELD;
    using SUB = typename SUPER::BASE_FIELD;

   public:
    BCHCode(const B& rscode) try : Base(rscode) {
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct BCH code: ") + e.what());
    }

    BCHCode(const SUPER& alpha, size_t b, size_t delta)
        requires std::is_same_v<B, RSCode<SUPER>>
    try : Base(RSCode<SUPER>(alpha, b, Base::k_from_delta(alpha.get_multiplicative_order(), delta))) {
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct BCH code: ") + e.what());
    }

    BCHCode(const BCHCode&) = default;
    BCHCode(BCHCode&&) = default;
    BCHCode& operator=(const BCHCode&) = default;
    BCHCode& operator=(BCHCode&&) = default;

    SUPER get_alpha() const { return this->get_SuperCode().get_alpha(); }
    size_t get_b() const noexcept { return this->get_SuperCode().get_b(); }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            Base::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) os << BOLD("BCH code");
    }
};

template <FiniteFieldType SUPER>
class GoppaCode : public AlternantCode<GRSCode<SUPER>> {
    using Base = AlternantCode<GRSCode<SUPER>>;
    using SUB = typename SUPER::BASE_FIELD;

   public:
    GoppaCode(const Vector<SUPER>& a, size_t delta) try : GoppaCode(a, Goppa_polynomial(delta)) {
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct Goppa code: ") + e.what());
    }

    GoppaCode(const Vector<SUPER>& a, Polynomial<SUPER> g) try
        : Base(a, Goppa_multipliers(a, g), g.degree() + 1),
          g(std::move(g)),
          squarefree(GCD(this->g, derivative(this->g, 1)).degree() == 0) {
    } catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Cannot construct Goppa code: ") + e.what());
    }

    GoppaCode(const GoppaCode& other)
        : Base(other), g(other.g), squarefree(other.squarefree), Patterson_inv_cache(other.Patterson_inv_cache) {}

    GoppaCode(GoppaCode&& other)
        : Base(std::move(other)),
          g(std::move(other.g)),
          squarefree(other.squarefree),
          Patterson_inv_cache(std::move(other.Patterson_inv_cache)) {}

    GoppaCode& operator=(const GoppaCode& other) {
        if (this != &other) {
            Base::operator=(other);
            g = other.g;
            squarefree = other.squarefree;
            Patterson_inv_cache = other.Patterson_inv_cache;
        }
        return *this;
    }

    GoppaCode& operator=(GoppaCode&& other) {
        if (this != &other) {
            Base::operator=(std::move(other));
            g = std::move(other.g);
            squarefree = other.squarefree;
            Patterson_inv_cache = std::move(other.Patterson_inv_cache);
        }
        return *this;
    }

    const Polynomial<SUPER>& get_g() const noexcept { return g; }
    bool is_squarefree() const noexcept { return squarefree; }

    size_t get_delta() const noexcept override {
        if constexpr (std::is_same_v<SUB, Fp<2>>)
            if (squarefree) return 2 * g.degree() + 1;
        return Base::get_delta();
    }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            Base::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            os << BOLD("Goppa code") " with properties: { g = " << g;
            if (squarefree) os << ", square-free";
            os << " }";
        }
    }

    Vector<SUB> dec_Patterson(const Vector<SUB>& r) const
        requires std::is_same_v<SUB, Fp<2>>
    {
        this->validate_length(r);

        if (!squarefree) throw std::invalid_argument("Patterson decoder requires a square-free Goppa polynomial");

        const size_t n = this->n;
        const auto& a = this->get_a();
        const size_t t = g.degree();

        if (t == 0) throw std::invalid_argument("Patterson decoder requires a nonconstant Goppa polynomial");

        calculate_Patterson_inv_cache();
        const auto& inv = *Patterson_inv_cache;

        const auto one = Monomial<SUPER>(0, SUPER(1));
        const auto x = Monomial<SUPER>(1, SUPER(1));
        const auto zero = ZeroPolynomial<SUPER>();

        auto S = zero;
        for (size_t i = 0; i < n; ++i)
            if (r[i] == SUB(1)) S = (S + inv[i]) % g;
        if (S.is_zero()) return r;

        Polynomial<SUPER> v, u;
        const auto d = GCD(g, S, &v, &u);
        if (d.is_empty() || d.is_zero() || d.degree() != 0)
            throw decoding_failure("Patterson decoder failed (syndrome not invertible modulo g)");
        const auto h = (((SUPER(1) / d[0]) * u) + x) % g;

        Polynomial<SUPER> sigma;
        if (h.is_zero()) {
            sigma = x;
        } else {
            auto R = h;
            const size_t squarings = std::bit_width(SUPER::get_size() - 1) * t - 1;
            for (size_t i = 0; i < squarings; ++i) R = (R * R) % g;

            auto r0 = g, r1 = R;
            auto t0 = zero, t1 = one;
            while (!r1.is_zero() && r1.degree() > t / 2) {
                const auto [q, rem] = poly_long_div(r0, r1);
                const auto next_t = t0 - q * t1;
                r0 = std::move(r1);
                r1 = rem;
                t0 = std::move(t1);
                t1 = next_t;
            }
            if (r1.is_zero()) throw decoding_failure("Patterson decoder failed (half-GCD)");
            sigma = r1 * r1 + x * (t1 * t1);
        }

        // Chien search
        std::vector<size_t> E;
        E.reserve(t);
        for (size_t i = 0; i < n; ++i)
            if (sigma(a[i]).is_zero()) E.push_back(i);

        if (E.size() > t) throw decoding_failure("Patterson decoder failed (too many errors)");

        Vector<SUB> c_est = r;
        for (size_t i = 0; i < E.size(); ++i) c_est.set_component(E[i], r[E[i]] + SUB(1));

        return c_est;
    }

   private:
    Polynomial<SUPER> g;
    bool squarefree;
    mutable details::OnceCache<std::vector<Polynomial<SUPER>>> Patterson_inv_cache;

    void calculate_Patterson_inv_cache() const {
        Patterson_inv_cache.call_once([this] {
            if (Patterson_inv_cache.has_value()) return;
            const size_t n = this->get_n();
            const auto& a = this->get_a();
            std::vector<Polynomial<SUPER>> inv(n);
            for (size_t i = 0; i < n; ++i) {
                Polynomial<SUPER> v, u;
                const auto d = GCD(g, Polynomial<SUPER>({a[i], SUPER(1)}), &v, &u);
                if (d.is_empty() || d.is_zero() || d.degree() != 0)
                    throw std::invalid_argument("Goppa locator a[" + std::to_string(i) + "] is a root of g");
                inv[i] = ((SUPER(1) / d[0]) * u) % g;
            }
            Patterson_inv_cache.emplace(std::move(inv));
        });
    }

    // Docu note: these are G multipliers (in literature typically: H multipliers)
    static Vector<SUPER> Goppa_multipliers(const Vector<SUPER>& a, const Polynomial<SUPER>& h) {
        const size_t n = a.get_n();
        Vector<SUPER> res(n);

        for (size_t i = 0; i < n; ++i) {
            const SUPER ha = h(a[i]);
            if (ha.is_zero()) throw std::invalid_argument("Goppa polynomial vanishes at a code locator");

            SUPER denominator(1);
            for (size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                const SUPER diff = a[i] - a[j];  // are guaranteed to be pairwise distinct
                denominator *= diff;
            }
            res.set_component(i, ha / denominator);
        }

        return res;
    }

    static Polynomial<SUPER> Goppa_polynomial(size_t delta) {
        if (delta < 2) throw std::invalid_argument("delta must be at least 2 for a nonconstant Goppa polynomial");
        return find_irreducible<SUPER>(delta - 1);
    }
};

template <FieldType T, class B>
    requires std::derived_from<B, LinearCode<T>>
class ExtendedCode : public LinearCode<T> {
   public:
    ExtendedCode(const B& BaseCode, size_t i, const Vector<T>& v) try : LinearCode
        <T>(BaseCode.get_n() + 1, BaseCode.get_k() == 0 ? v.wH() : BaseCode.get_k(),
            Extended_G(BaseCode.get_G(), i, v)),
            BaseCode(BaseCode), i(i), v(v), parity(false) {
            if (v == Extended_v(this->BaseCode.get_G())) parity = true;
        }
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Trying to extend a code with invalid extension parameters: ") +
                                    e.what());
    }

    ExtendedCode(B&& BaseCode, size_t i, const Vector<T>& v) try : LinearCode
        <T>(BaseCode.get_n() + 1, BaseCode.get_k() == 0 ? v.wH() : BaseCode.get_k(),
            Extended_G(BaseCode.get_G(), i, v)),
            BaseCode(std::move(BaseCode)), i(i), v(v), parity(v == Extended_v(this->BaseCode.get_G())) {}
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Trying to extend a code with invalid extension parameters: ") +
                                    e.what());
    }

    explicit ExtendedCode(const B& BaseCode)
        : LinearCode<T>(BaseCode.get_n() + 1, BaseCode.get_k(),
                        Extended_G(BaseCode.get_G(), 0, Extended_v(BaseCode.get_G()))),
          BaseCode(BaseCode),
          i(0),
          v(this->G.get_col(i)),
          parity(true) {}

    ExtendedCode(const ExtendedCode&) = default;
    ExtendedCode(ExtendedCode&&) = default;
    ExtendedCode& operator=(const ExtendedCode&) = default;
    ExtendedCode& operator=(ExtendedCode&&) = default;

    size_t get_i() const noexcept { return i; }
    const B& get_BaseCode() const noexcept { return BaseCode; }

    const Polynomial<InfInt>& get_weight_enumerator() const override {
        if constexpr (!FiniteFieldType<T>) {
            throw std::logic_error("Cannot calculate weight enumerator of code over infinite field!");
        } else {
            // In non-binary case, zero row sum is not the same as even Hamming distance so this really only
            // works for the binary Fp<2> case! Otherwise we fall back to linear code weight weight enumerator
            // calculation.
            if constexpr (std::is_same_v<T, Fp<2>>) {
                if (parity) {
                    auto weight_enumerator = BaseCode.get_weight_enumerator();
                    for (size_t w = 0; w <= weight_enumerator.degree(); ++w) {
                        if (w % 2 && weight_enumerator[w] != 0) {
                            weight_enumerator.add_to_coefficient(w + 1, weight_enumerator[w]);
                            weight_enumerator.set_coefficient(w, 0);
                        }
                    }
                    this->set_weight_enumerator(std::move(weight_enumerator));
                }
            }
            return LinearCode<T>::get_weight_enumerator();
        }
    }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            const auto old = os.iword(details::index);
            os << BOLD("Extended code") " with properties: { i = " << i;
            os << ", v = " << v;
            if (parity) os << ", even parity";
            os << ", BaseCode = " << showbasic;
            BaseCode.LinearCode<T>::get_info(os);
            os << " " << showspecial << BaseCode;
            os << " }";
            os.iword(details::index) = old;
        }
    }

    virtual Vector<T> dec_BD(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
#endif
        return LinearCode<T>::dec_BD(r);
    }

    virtual Vector<T> dec_ML(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
#endif
        return LinearCode<T>::dec_ML(r);
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        if (!parity || !r[i].is_erased()) return LinearCode<T>::dec_BD_EE(r);
        this->validate_length(r);

        const auto r_base = concatenate(r.get_subvector(0, i), r.get_subvector(i + 1, this->n - i - 1));

        const auto cp_est = BaseCode.dec_ML_EE(r_base);  // sic!

        T p(0);
        for (size_t j = 0; j < cp_est.get_n(); ++j) p += cp_est[j];

        Vector<T> c_est(this->n);
        c_est.set_subvector(cp_est.get_subvector(0, i), 0);
        c_est.set_component(i, -p);
        c_est.set_subvector(cp_est.get_subvector(i, cp_est.get_n() - i), i + 1);

        size_t tau = 0;
        size_t t = 0;
        for (size_t j = 0; j < this->n; ++j) {
            if (r[j].is_erased())
                ++tau;
            else if (r[j] != c_est[j])
                ++t;
        }

        if (2 * t + tau > this->get_dmin() - 1)
            throw decoding_failure("Extended code BD error/erasure decoder failed!");

        return c_est;
    }

    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        if (!parity || !r[i].is_erased()) return LinearCode<T>::dec_ML_EE(r);
        this->validate_length(r);

        const auto r_base = concatenate(r.get_subvector(0, i), r.get_subvector(i + 1, this->n - i - 1));

        const auto cp_est = BaseCode.dec_ML_EE(r_base);

        T p(0);
        for (size_t j = 0; j < cp_est.get_n(); ++j) p += cp_est[j];

        Vector<T> c_est(this->n);
        c_est.set_subvector(cp_est.get_subvector(0, i), 0);
        c_est.set_component(i, -p);
        c_est.set_subvector(cp_est.get_subvector(i, cp_est.get_n() - i), i + 1);
        return c_est;
    }
#endif

   private:
    static Matrix<T> Extended_G(const Matrix<T>& Gp, size_t i, const Vector<T>& v) {
        const size_t n = Gp.get_n();
        const size_t k = Gp.get_m();

        if (i > n) throw std::invalid_argument("Extension index invalid");
        if (v.get_n() != k) throw std::invalid_argument(std::string("Length of v must be ") + std::to_string(k));

        Matrix<T> G(k, n + 1);
        for (size_t j = 0; j < i; ++j) G.set_submatrix(0, j, Gp.get_submatrix(0, j, k, 1));
        G.set_submatrix(0, i, transpose(Matrix<T>(v)));
        for (size_t j = i; j < n; ++j) G.set_submatrix(0, j + 1, Gp.get_submatrix(0, j, k, 1));
        return G;
    }

    static Vector<T> Extended_v(const Matrix<T>& Gp) {
        const size_t n = Gp.get_n();
        const size_t k = Gp.get_m();

        Vector<T> v(k);
        for (size_t j = 0; j < n; ++j) v += Gp.get_col(j);
        v *= -T(1);
        return v;
    }

    B BaseCode;
    size_t i;
    Vector<T> v;
    bool parity;
};

template <FieldType T, class B>
    requires std::derived_from<B, LinearCode<T>>
class AugmentedCode : public LinearCode<T> {
   public:
    AugmentedCode(const B& BaseCode, size_t j, const Vector<T>& w) try : LinearCode
        <T>(BaseCode.get_n(), BaseCode.get_k() + 1, Augmented_G(BaseCode.get_G(), j, w)), BaseCode(BaseCode), j(j),
            w(w) {}
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Trying to augment a code by an invalid w: ") + e.what());
    }

    AugmentedCode(B&& BaseCode, size_t j, const Vector<T>& w) try : LinearCode
        <T>(BaseCode.get_n(), BaseCode.get_k() + 1, Augmented_G(BaseCode.get_G(), j, w)), BaseCode(std::move(BaseCode)),
            j(j), w(w) {}
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Trying to augment a code by an invalid w: ") + e.what());
    }

    AugmentedCode(const AugmentedCode&) = default;
    AugmentedCode(AugmentedCode&&) = default;
    AugmentedCode& operator=(const AugmentedCode&) = default;
    AugmentedCode& operator=(AugmentedCode&&) = default;

    const B& get_BaseCode() const noexcept { return BaseCode; }
    size_t get_j() const noexcept { return j; }
    const Vector<T>& get_w() const noexcept { return w; }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            if (os.iword(details::index) > 0) os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            const auto old = os.iword(details::index);
            os << BOLD("Augmented code") " with properties: { w = " << w;
            os << ", BaseCode = " << showbasic;
            BaseCode.LinearCode<T>::get_info(os);
            os << " " << showspecial << BaseCode;
            os << " }";
            os.iword(details::index) = old;
        }
    }

    Vector<T> dec_BD(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
#endif
        if constexpr (!FiniteFieldType<T>) {
            return LinearCode<T>::dec_BD(r);
        } else {
            this->validate_length(r);

            const size_t t = this->get_tmax();

            for (size_t i = 0; i < T::get_size(); ++i) {
                const T alpha = T(i);

                try {
                    Vector<T> cp_est = BaseCode.dec_BD(r - alpha * w);
                    Vector<T> c_est = cp_est + alpha * w;
                    if (dH(r, c_est) <= t) return c_est;
                } catch (const decoding_failure&) {
                    continue;
                }
            }

            throw decoding_failure("Augmented code BD decoder failed!");
        }
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
#ifdef CECCO_ERASURE_SUPPORT
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
#endif
        if constexpr (!FiniteFieldType<T>) {
            return LinearCode<T>::dec_ML(r);
        } else {
            this->validate_length(r);

            const size_t tmax = this->get_tmax();

            Vector<T> best = BaseCode.dec_ML(r);  // r - 0*w
            size_t best_t = dH(r, best);

            if (best_t <= tmax) return best;

            uint16_t ties = 1;
            for (size_t i = 1; i < T::get_size(); ++i) {
                const T alpha = T(i);

                Vector<T> cp_est = BaseCode.dec_ML(r - alpha * w);
                Vector<T> c_est = cp_est + alpha * w;

                const size_t t = dH(r, c_est);

                if (t <= tmax) return c_est;

                if (t < best_t) {
                    best_t = t;
                    best = std::move(c_est);
                    ties = 1;
                } else if (t == best_t) {
                    if (details::reservoir_accept(++ties)) best = std::move(c_est);
                }
            }

            return best;
        }
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        if constexpr (!FiniteFieldType<T>) {
            return LinearCode<T>::dec_BD_EE(r);
        } else {
            this->validate_length(r);

            const size_t dmin = this->get_dmin();

            size_t tau = 0;
            for (size_t j = 0; j < this->n; ++j)
                if (r[j].is_erased()) ++tau;

            if (tau > dmin - 1) throw decoding_failure("Augmented code BD error/erasure decoder failed!");

            for (size_t i = 0; i < T::get_size(); ++i) {
                const T alpha = T(i);

                try {
                    const auto cp_est = BaseCode.dec_BD_EE(r - alpha * w);
                    const auto c_est = cp_est + alpha * w;

                    size_t t = 0;
                    for (size_t j = 0; j < this->n; ++j) {
                        if (!r[j].is_erased() && r[j] != c_est[j]) ++t;
                    }

                    if (2 * t + tau <= dmin - 1) return c_est;

                } catch (const decoding_failure&) {
                    continue;
                }
            }

            throw decoding_failure("Augmented code BD error/erasure decoder failed!");
        }
    }

    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        if constexpr (!FiniteFieldType<T>) {
            return LinearCode<T>::dec_ML_EE(r);
        } else {
            this->validate_length(r);

            Vector<T> best = BaseCode.dec_ML_EE(r);  // r - 0*w
            size_t best_t = 0;
            for (size_t j = 0; j < this->n; ++j) {
                if (!r[j].is_erased() && r[j] != best[j]) ++best_t;
            }

            uint16_t ties = 1;
            for (size_t i = 1; i < T::get_size(); ++i) {
                const T alpha = T(i);

                const auto cp_est = BaseCode.dec_ML_EE(r - alpha * w);
                const auto c_est = cp_est + alpha * w;

                size_t t = 0;
                for (size_t j = 0; j < this->n; ++j) {
                    if (!r[j].is_erased() && r[j] != c_est[j]) ++t;
                }

                if (t < best_t) {
                    best_t = t;
                    best = std::move(c_est);
                    ties = 1;
                } else if (t == best_t) {
                    if (details::reservoir_accept(++ties)) best = std::move(c_est);
                }
            }

            return best;
        }
    }
#endif

   private:
    static Matrix<T> Augmented_G(const Matrix<T>& Gp, size_t j, const Vector<T>& w) {
        const size_t n = Gp.get_n();
        const size_t k = Gp.get_m();

        if (w.get_n() != n) throw std::invalid_argument(std::string("Length of w must be ") + std::to_string(n));

        if (Gp.get_m() == 1 && Gp.rank() == 0) return Matrix<T>(w);

        if (j == 0) {
            const auto G = vertical_join(Matrix(w), Gp);
            return G;
        } else if (j == k) {
            const auto G = vertical_join(Gp, Matrix(w));
            return G;
        } else {
            const auto G =
                vertical_join(vertical_join(Gp.get_submatrix(0, 0, j, n), Matrix(w)), Gp.get_submatrix(j, 0, k - j, n));
            return G;
        }
    }

    B BaseCode;
    size_t j;
    Vector<T> w;
};

template <FieldType T>
auto dual(const T& C) {
    return C.get_dual();
}

namespace details {
bool validate(const std::vector<size_t>& v, std::size_t n) {
    std::vector<bool> seen(n, false);
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (*it >= n || seen[*it]) return false;
        seen[*it] = true;
    }
    return true;
}
}  // namespace details

template <class B>
using base_t = std::remove_cvref_t<B>;

template <class C>
using field_t = typename base_t<C>::FIELD;

template <class B>
auto extend(B&& base, size_t i, const Vector<field_t<B>>& v) {
    using D = base_t<B>;
    using T = field_t<B>;
    return ExtendedCode<T, D>(std::forward<B>(base), i, v);
}

template <class B>
auto extend(B&& base) {
    using D = base_t<B>;
    using T = field_t<B>;
    return ExtendedCode<T, D>(std::forward<B>(base));
}

// Docu note: unextend maintains special structure, puncture does not
template <FieldType T, class B>
B unextend(const ExtendedCode<T, B>& C) {
    return C.get_BaseCode();
}

template <class B>
auto augment(B&& base, size_t j, const Vector<field_t<B>>& w) {
    using D = base_t<B>;
    using T = field_t<B>;
    return AugmentedCode<T, D>(std::forward<B>(base), j, w);
}

// Docu note: unaugment maintains special structure, expurgate does not
template <FieldType T, class B>
B unaugment(const AugmentedCode<T, B>& C) {
    return C.get_BaseCode();
}

template <class B>
auto lengthen(B&& base, size_t j, const Vector<field_t<B>>& w, size_t i, const Vector<field_t<B>>& v) {
    using D = base_t<B>;
    using T = field_t<B>;
    return ExtendedCode<T, AugmentedCode<T, D>>(AugmentedCode<T, D>(std::forward<B>(base), j, w), i, v);
}

template <class B>
auto lengthen(B&& base, size_t j, const Vector<field_t<B>>& w) {
    using D = base_t<B>;
    using T = field_t<B>;
    return ExtendedCode<T, AugmentedCode<T, D>>(AugmentedCode<T, D>(std::forward<B>(base), j, w));
}

// Docu note: unlengthen maintains special structure, shorten does not
template <FieldType T, class D>
auto unlengthen(const ExtendedCode<T, AugmentedCode<T, D>>& C) {
    return C.get_BaseCode().get_BaseCode();
}

template <FieldType T>
LinearCode<T> puncture(const LinearCode<T>& C, const std::vector<size_t>& v) {
    if (!details::validate(v, C.get_n())) throw std::invalid_argument("Invalid pattern for puncturing linear code!");

    auto G = C.get_G();
    G.delete_columns(v);
    size_t rank;
    G.rref(&rank);
    G = G.get_submatrix(0, 0, rank, G.get_n());
    return LinearCode<T>(C.get_n() - v.size(), rank, std::move(G));
}

template <FieldType T>
auto puncture(const EmptyCode<T>& C, const std::vector<size_t>& v) {
    if (v.empty()) return C;
    throw std::invalid_argument("Cannot puncture an empty code!");
}

template <FieldType T>
auto puncture(const ZeroCode<T>& C, const std::vector<size_t>& v) {
    if (!details::validate(v, C.get_n())) throw std::invalid_argument("Invalid pattern for puncturing zero code!");
    return ZeroCode<T>(C.get_n() - v.size());
}

template <FieldType T>
auto puncture(const UniverseCode<T>& C, const std::vector<size_t>& v) {
    if (!details::validate(v, C.get_n())) throw std::invalid_argument("Invalid pattern for puncturing universe code!");
    return UniverseCode<T>(C.get_n() - v.size());
}

template <FieldType T>
auto puncture(const RepetitionCode<T>& C, const std::vector<size_t>& v) {
    if (!details::validate(v, C.get_n()))
        throw std::invalid_argument("Invalid pattern for puncturing repetition code!");
    return RepetitionCode<T>(C.get_n() - v.size());
}

template <class C>
    requires std::derived_from<base_t<C>, LinearCode<field_t<C>>>
auto puncture(C&& code, size_t i) {
    return puncture(std::forward<C>(code), std::vector<size_t>{i});
}

template <FieldType T>
LinearCode<T> expurgate(const LinearCode<T>& C, const std::vector<size_t>& v) {
    if (!details::validate(v, C.get_k())) throw std::invalid_argument("Invalid pattern for expurgating linear code!");
    if (C.get_k() == 0) return C;

    auto G = C.get_G();
    G.delete_rows(v);
    return LinearCode<T>(C.get_n(), G.get_m(), std::move(G));
}

template <FieldType T>
auto expurgate(const EmptyCode<T>& C, const std::vector<size_t>& v) {
    if (v.empty()) return C;
    throw std::invalid_argument("Cannot expurgate empty code!");
}

template <FieldType T>
auto expurgate(const ZeroCode<T>& C, const std::vector<size_t>& v) {
    if (!details::validate(v, C.get_k()) || v.size() > 1 || v[0] != 0)
        throw std::invalid_argument("Invalid pattern for expurgating zero code!");
    return EmptyCode<T>(C.get_n());
}

template <FieldType T>
auto expurgate(const RepetitionCode<T>& C, const std::vector<size_t>& v) {
    if (!details::validate(v, C.get_k()) || v.size() > 1 || v[0] != 0)
        throw std::invalid_argument("Invalid pattern for expurgating repetition code!");
    return EmptyCode<T>(C.get_n());
}

template <class C>
    requires std::derived_from<base_t<C>, LinearCode<field_t<C>>>
auto expurgate(C&& code, size_t j) {
    return expurgate(std::forward<C>(code), std::vector<size_t>{j});
}

template <class B>
auto shorten(B&& base, size_t j, size_t i) {
    return puncture(expurgate(std::forward<B>(base), j), i);
}

template <FieldType L, FieldType R>
bool operator==(const LinearCode<L>& lhs, const LinearCode<R>& rhs) {
    return lhs.is_identical(rhs);
}

template <FieldType L, FieldType R>
bool operator!=(const LinearCode<L>& lhs, const LinearCode<R>& rhs) {
    return !(lhs == rhs);
}

template <FieldType L, FieldType R>
bool identical(const LinearCode<L>& lhs, const LinearCode<R>& rhs) {
    return lhs.is_identical(rhs);
}

template <FieldType L, FieldType R>
bool equivalent(const LinearCode<L>& lhs, const LinearCode<R>& rhs) {
    return lhs.is_equivalent(rhs);
}

template <FieldType T>
std::ostream& operator<<(std::ostream& os, const Code<T>& rhs) {
    rhs.get_info(os);
    return os;
}

template <FieldType T>
class Enc {
   public:
    Enc(const Code<T>& C) noexcept : C(C) {}

    Vector<T> operator()(const Vector<T>& in) const { return C.enc(in); }

   private:
    const Code<T>& C;
};

enum class method_t {
    BD,
    boosted_BD,
    ML,
    ML_soft,
    Viterbi,
    Viterbi_soft,
    BCJR,
    BP,
    recursive,
    Meggitt,
    WBA,
    BMA,
#ifdef CECCO_ERASURE_SUPPORT
    WBA_EE,
    BMA_EE,
    BD_EE,
    ML_EE,
    Viterbi_EE,
    recursive_EE
#endif
};

template <FieldType T>
class Dec {
   public:
    explicit Dec(const Code<T>& C, method_t method = method_t::ML) : C(C), method(method) {
        switch (method) {
            case method_t::BD:
                dec = [this](const Vector<T>& r) { return this->C.dec_BD(r); };
                break;
            case method_t::boosted_BD:
                dec = [this](const Vector<T>& r) { return this->C.dec_boosted_BD(r); };
                break;
            case method_t::ML:
                dec = [this](const Vector<T>& r) { return this->C.dec_ML(r); };
                break;
            case method_t::Viterbi:
                dec = [this](const Vector<T>& r) { return this->C.dec_Viterbi(r); };
                break;
            case method_t::recursive:
                dec = [this](const Vector<T>& r) { return this->C.dec_recursive(r); };
                break;
            case method_t::Meggitt:
                dec = [this](const Vector<T>& r) { return this->C.dec_Meggitt(r); };
                break;
            case method_t::WBA:
                dec = [this](const Vector<T>& r) { return this->C.dec_WBA(r); };
                break;
            case method_t::BMA:
                dec = [this](const Vector<T>& r) { return this->C.dec_BMA(r); };
                break;
            case method_t::ML_soft:
            case method_t::Viterbi_soft:
            case method_t::BCJR:
            case method_t::BP:
                // Soft-input only; decoded through operator()(Vector<double>) / operator()(Matrix<double>).
                break;
#ifdef CECCO_ERASURE_SUPPORT
            case method_t::WBA_EE:
                dec = [this](const Vector<T>& r) { return this->C.dec_WBA_EE(r); };
                break;
            case method_t::BMA_EE:
                dec = [this](const Vector<T>& r) { return this->C.dec_BMA_EE(r); };
                break;
            case method_t::BD_EE:
                dec = [this](const Vector<T>& r) { return this->C.dec_BD_EE(r); };
                break;
            case method_t::ML_EE:
                dec = [this](const Vector<T>& r) { return this->C.dec_ML_EE(r); };
                break;
            case method_t::Viterbi_EE:
                dec = [this](const Vector<T>& r) { return this->C.dec_Viterbi_EE(r); };
                break;
            case method_t::recursive_EE:
                dec = [this](const Vector<T>& r) { return this->C.dec_recursive_EE(r); };
                break;
#endif
            default:
                break;
        }
    }

    Vector<T> operator()(const Vector<T>& in) const {
        if (!dec) throw std::logic_error("Selected decoding method does not support hard-decision input!");
        return dec(in);
    }

    Vector<T> operator()(const Vector<double>& in) const {
        if (method == method_t::Viterbi || method == method_t::Viterbi_soft)
            return C.dec_Viterbi_soft(in);
        else if (method == method_t::BCJR)
            return C.dec_BCJR(in);
        else if (method == method_t::BP)
            return C.dec_BP(in, bp_max_iterations);
        else if (method == method_t::ML_soft)
            return C.dec_ML_soft(in, cache_limit);
        throw std::logic_error("Vector soft input requires a soft method (Viterbi_soft, BCJR, BP, or ML_soft)!");
    }

    Vector<T> operator()(const Matrix<double>& in) const {
        if (method == method_t::Viterbi || method == method_t::Viterbi_soft)
            return C.dec_Viterbi_soft(in);
        else if (method == method_t::BCJR)
            return C.dec_BCJR(in);
        else if (method == method_t::BP)
            return C.dec_BP(in, bp_max_iterations);
        else if (method == method_t::ML_soft)
            return C.dec_ML_soft(in, cache_limit);
        throw std::logic_error("Matrix soft input requires a soft method (Viterbi_soft, BCJR, BP, or ML_soft)!");
    }

    // Docu note: cache_limit (ML_soft) and bp_max_iterations (BP) are method-specific knobs;
    // both default sensibly and can be overridden after construction via these setters.
    void set_cache_limit(size_t l) { cache_limit = l; }
    void set_BP_max_iterations(size_t l) { bp_max_iterations = l; }

   private:
    const Code<T>& C;
    std::function<Vector<T>(const Vector<T>&)> dec;
    method_t method;
    size_t cache_limit = 10000;
    size_t bp_max_iterations = BP_MAX_ITERATIONS;
};

template <FiniteFieldType T>
class Encinv {
   public:
    Encinv(const Code<T>& C) noexcept : C(C) {}

    Vector<T> operator()(const Vector<T>& in) const { return C.encinv(in); }

   private:
    const Code<T>& C;
};

}  // namespace CECCO

#endif
