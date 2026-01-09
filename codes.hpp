/**
 * @file codes.hpp
 * @brief Error control codes library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.0.8
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026, Christian Senger <senger@inue.uni-stuttgart.de>
 *
 * Licensed for noncommercial use only, including academic teaching, research, and personal non-profit purposes.
 * Commercial use is prohibited without a separate commercial license. See the [LICENSE](../../LICENSE) file in the
 * repository root for full terms and how to request a commercial license.
 */

/*
 * +==============================================================+
 * |  !!! WARNING !!!                                             |
 * |  this file is work in progress                               |
 * |  and not production-ready                                    |
 * +==============================================================+
 */

#ifndef CODES_HPP
#define CODES_HPP

#include "blocks.hpp"
// #include <ranges> // transitive through blocks.hpp
// #include <algorithm> // transitive through blocks.hpp
// #include <vector> // transitive through blocks.hpp
// #include <iostream> // transitive through blocks.hpp
// #include "InfInt.hpp" // transitive through blocks.hpp
// #include "field_concepts_traits.hpp" // transitive through blocks.hpp
// #include "helpers.hpp" // transitive through blocks.hpp
// #include "matrices.hpp" // transitive through blocks.hpp
// #include "vectors.hpp" // transitive through blocks.hpp
// #include "fields.hpp" // transitive through blocks.hpp
// #include "polynomials.hpp" // transitive through blocks.hpp
#include <bit>

namespace CECCO {

template <FiniteFieldType T>
long double HammingUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    try {
        const size_t tmax = (dmin - 1.0) / 2;
        InfInt h = 0;
        for (size_t i = 0; i <= tmax; ++i) h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
        return n - std::log2l(h.toUnsignedLongLong()) / std::log2l(q);
    } catch (const InfIntException& e) {
        std::cerr << " [Hamming bound overflow]";
        return 0;
    }
}

namespace details {

template <FiniteFieldType T>
static size_t A(size_t n, size_t d, size_t w) {
    if (2 * w < d) return 1;
    constexpr size_t q = T::get_size();
    const size_t e = std::ceill(d / 2.0);
    size_t res = 1;
    for (size_t i = e; i <= w; ++i) res *= (n - w + i) * (q - 1) / (long double)i;
    return res;
}

}  // namespace details

template <FiniteFieldType T>
long double JohnsonUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    try {
        const size_t tmax = (dmin - 1) / 2;
        const size_t s = dmin % 2;
        InfInt h = 0;
        for (size_t i = 0; i <= tmax; ++i) h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
        return n - std::log2l(h.toUnsignedLongLong() + (bin<InfInt>(n, tmax + 1) * sqm<InfInt>(q - 1, tmax + 1) -
                                                        InfInt(s) * bin<InfInt>(dmin, tmax) * A<T>(n, dmin, dmin))
                                                               .toUnsignedLongLong() /
                                                           (long double)A<T>(n, dmin, tmax + 1)) /
                       std::log2l(q);
    } catch (const InfIntException& e) {
        std::cerr << " [Johnson bound overflow]";
        return 0;
    }
}

template <FiniteFieldType T>
long double PlotkinUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    try {
        if ((long double)dmin / n > (long double)(q - 1) / q) {  // conventional
            return 1 - std::log2l(q - (long double)n * (q - 1) / dmin) / std::log2l(q);
        } else {  // improved
            const size_t Delta = n - std::floorl((long double)dmin * q / (q - 1)) + 1;
            const size_t M = sqm(q, Delta + 1) * (long double)dmin / ((q * dmin - (n - Delta) * (q - 1)));
            return std::log2l(M) / std::log2l(q);
        }
    } catch (const InfIntException& e) {
        std::cerr << " [Plotkin bound overflow]";
        return 0;
    }
}

template <FiniteFieldType T>
long double EliasUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    try {
        const long double r = 1 - (long double)1 / q;
        long double min = std::numeric_limits<long double>::max();
        for (size_t w = 0; w <= r * n; ++w) {
            if (w * w - 2 * r * n * w + r * n * dmin > 0) {
                InfInt h = 0;
                for (size_t i = 0; i <= w; ++i) h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
                long double temp = (long double)(r * n * dmin) / (w * w - 2 * r * n * w + r * n * dmin);
                temp /= h.toUnsignedLongLong();
                min = std::min<long double>(temp, min);
            }
        }
        return n + std::log2l(min) / std::log2l(q);
    } catch (const InfIntException& e) {
        std::cerr << " [Elias bound overflow]";
        return 0;
    }
}

size_t SingletonUpperBound(size_t n, size_t dmin) { return n - dmin + 1; }

template <FiniteFieldType T>
size_t GriesmerUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    size_t k = 0;
    for (size_t kp = 1; kp <= n; ++kp) {
        size_t sum = 0;
        for (size_t i = 0; i < kp; ++i) sum += std::ceill(dmin / sqm<long double>(q, i));
        if (sum <= n)
            k = kp;
        else
            break;
    }
    return k;
}

template <FiniteFieldType T>
double UpperBound(size_t n, size_t dmin) {
    long double min = std::numeric_limits<long double>::max();
    for (size_t delta = 0; delta < dmin; ++delta) {
        const long double hamming = HammingUpperBound<T>(n - delta, dmin - delta);
        const long double johnson = JohnsonUpperBound<T>(n - delta, dmin - delta);
        const long double plotkin = PlotkinUpperBound<T>(n - delta, dmin - delta);
        const long double elias = EliasUpperBound<T>(n - delta, dmin - delta);
        const long double singleton = SingletonUpperBound(n - delta, dmin - delta);
        const long double griesmer = GriesmerUpperBound<T>(n - delta, dmin - delta);
        for (long double bound : {hamming, johnson, plotkin, elias, singleton, griesmer}) min = std::min(bound, min);
    }
    return min;
}

template <FiniteFieldType T>
size_t GilbertVarshamovLowerBound(size_t n, size_t dmin) {
    if (dmin == 1) return n;
    if (dmin == n) return 1;
    constexpr size_t q = T::get_size();
    try {
        InfInt sum = 0;
        for (size_t i = 0; i < dmin; ++i) sum += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
        return std::ceill(n - std::log2l(sum.toUnsignedLongLong()) / std::log2l(q));
    } catch (const InfIntException& e) {
        std::cerr << " [Gilbert-Varshamov bound overflow]";
        return 0;
    }
}

template <FiniteFieldType T>
size_t BurstUpperBound(size_t n, size_t ell) {
    constexpr size_t q = T::get_size();
    return std::floorl(n - ell - std::log2l(1 + (q - 1) * (n - ell) / q) / std::log2l(q));
}

size_t ReigerBurstUpperBound(size_t n, size_t ell) {
    if (2 * ell > n) return 0;
    return n - 2 * ell;
}

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
            K(i, j) = sum;
        }
    }
    return Polynomial<InfInt>(Vector<InfInt>(a) * Matrix<InfInt>(K) / sqm<InfInt>(q, k));
}

namespace details {

static const int index = std::ios_base::xalloc();

}  // namespace details

std::ostream& showbasic(std::ostream& os) {
    os.iword(details::index) = 0;
    return os;
}

std::ostream& showmost(std::ostream& os) {
    os.iword(details::index) = 1;
    return os;
}

std::ostream& showall(std::ostream& os) {
    os.iword(details::index) = 2;
    return os;
}

std::ostream& showspecial(std::ostream& os) {
    os.iword(details::index) = 3;
    return os;
}

template <FieldType T>
class Code {
   public:
    Code(size_t n) : n(n) {}

    Code(const Code& other) : n(other.n) {}

    Code(Code&&) = default;

    virtual ~Code() = default;

    Code& operator=(const Code& other) {
        if (this != &other) {
            n = other.n;
        }
        return *this;
    }

    Code& operator=(Code&&) = default;

    size_t get_n() const noexcept { return n; }

    virtual void get_info(std::ostream& os) const {};
    virtual Vector<T> enc(const Vector<T>& u) const = 0;
    virtual Vector<T> encinv(const Vector<T>& c) const = 0;
    virtual Vector<T> dec_BD(const Vector<T>& r) const = 0;
    virtual Vector<T> dec_ML(const Vector<T>& r) const = 0;
    virtual Vector<T> dec_burst(const Vector<T>& r) const = 0;
    virtual Vector<T> dec_BD_EE(const Vector<T>& r) const = 0;
    virtual Vector<T> dec_ML_EE(const Vector<T>& r) const = 0;

   protected:
    size_t n;
};

template <FieldType T>
class EmptyCode : public Code<T> {
   public:
    EmptyCode(size_t n) : Code<T>(n) {}

    EmptyCode(const EmptyCode&) = default;
    EmptyCode(EmptyCode&&) = default;
    EmptyCode& operator=(const EmptyCode&) = default;
    EmptyCode& operator=(EmptyCode&&) = default;

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) > 0) {
            Code<T>::get_info(os);
            os << "Empty code";
        }
    }

    Vector<T> enc(const Vector<T>& u) const override {
        throw std::invalid_argument("Cannot encode wrt. an empty code!");
    }

    Vector<T> encinv(const Vector<T>& c) const override {
        throw std::invalid_argument("Cannot invert encoding wrt. an empty code!");
    }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        throw std::invalid_argument("Cannot BD decode an empty code!");
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
        throw std::invalid_argument("Cannot ML decode an empty code!");
    }

    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        throw std::invalid_argument("Cannot BD error/erasure decode an empty code!");
    }

    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        throw std::invalid_argument("Cannot ML error/erasure decode an empty code!");
    }

    Vector<T> dec_burst(const Vector<T>& r) const override {
        throw std::invalid_argument("Cannot burst decode an empty code!");
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
                const auto rem = (j - quot).toInt();
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
    decoding_failure(const char* message) : message(message) {}
    const char* what() const noexcept override { return message.c_str(); }

   private:
    const std::string message;
};

template <FieldType T>
class LinearCode : public Code<T> {
   public:
    // expose base field (used by the extend() factories)
    using field = T;

    LinearCode(size_t n, size_t k, const Matrix<T>& X) : Code<T>(n), k(k), MI(k, k) {
        if (X.get_n() != this->n) throw std::invalid_argument("G must have " + std::to_string(this->n) + " columns");
        if (k == 0) {
            if (!X.is_zero()) throw std::invalid_argument("G must be a zero matrix");
            G = ZeroMatrix<T>(1, n);
            HT = IdentityMatrix<T>(n);
            return;
        }

        if (X.get_m() == k) {
            // X supposed to be generator matrix G
            if (X.rank() != k) throw std::invalid_argument("G must have full rank " + std::to_string(k));
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
                throw std::invalid_argument("H must have full rank " + std::to_string(this->n - k));
            G = X.basis_of_nullspace();
            HT = transpose(X);
            G.rref();
            size_t i = 0;
            for (size_t j = 0; j < k; ++j) {
                const auto u = unit_vector<T>(k, j);
                while (G.get_col(i) != u) ++i;
                infoset.push_back(i);
            }
        }
        size_t j = 0;
        for (auto it = infoset.cbegin(); it != infoset.cend(); ++it) {
            MI.set_submatrix(0, j, G.get_submatrix(0, *it, k, 1));
            ++j;
        }
        MI.invert();
    }

    LinearCode(const LinearCode& other)
        : Code<T>(other),
          k(other.k),
          G(other.G),
          HT(other.HT),
          MI(other.MI),
          infoset(other.infoset),
          dmin(other.dmin),  // copy computed value
          dmin_flag(),       // fresh flag
          weight_enumerator(other.weight_enumerator),
          weight_enumerator_flag(),
          standard_array(other.standard_array),
          standard_array_flag(),
          tainted(other.tainted),
          tainted_flag(),
          tainted_burst(other.tainted_burst),
          tainted_burst_flag(),
          punctured_codes(other.punctured_codes),
          punctured_codes_flag(),
          polynomial(other.polynomial),
          polynomial_flag(),
          gamma(other.gamma),
          gamma_flag() {}

    LinearCode(LinearCode&& other)
        : Code<T>(std::move(other)),
          k(other.k),
          G(std::move(other.G)),
          HT(std::move(other.HT)),
          MI(std::move(other.MI)),
          infoset(std::move(other.infoset)),
          dmin(std::move(other.dmin)),  // move computed value
          dmin_flag(),                  // fresh flag
          weight_enumerator(std::move(other.weight_enumerator)),
          weight_enumerator_flag(),
          standard_array(std::move(other.standard_array)),
          standard_array_flag(),
          tainted(std::move(other.tainted)),
          tainted_flag(),
          tainted_burst(std::move(other.tainted_burst)),
          tainted_burst_flag(),
          punctured_codes(std::move(other.punctured_codes)),
          punctured_codes_flag(),
          polynomial(std::move(other.polynomial)),
          polynomial_flag(),
          gamma(std::move(other.gamma)),
          gamma_flag() {}

    LinearCode& operator=(const LinearCode& other) {
        if (this != &other) {
            k = other.k;
            G = other.G;
            HT = other.HT;
            MI = other.MI;
            infoset = other.infoset;
            dmin = other.dmin;
            weight_enumerator = other.weight_enumerator;
            standard_array = other.standard_array;
            tainted = other.tainted;
            tainted_burst = other.tainted_burst;
            punctured_codes = other.punctured_codes;
            polynomial = other.polynomial;
            gamma = other.gamma;
        }
        return *this;
    }

    LinearCode& operator=(LinearCode&& other) {
        if (this != &other) {
            k = other.k;
            G = std::move(other.G);
            HT = std::move(other.HT);
            MI = std::move(other.MI);
            infoset = std::move(other.infoset);
            dmin = std::move(other.dmin);
            weight_enumerator = std::move(other.weight_enumerator);
            standard_array = std::move(other.standard_array);
            tainted = std::move(other.tainted);
            tainted_burst = std::move(other.tainted_burst);
            punctured_codes = std::move(other.punctured_codes);
            polynomial = std::move(other.polynomial);
            gamma = std::move(other.gamma);
        }
        return *this;
    }

    size_t get_k() const noexcept { return k; }
    double get_R() const noexcept { return static_cast<double>(k) / this->n; }

    InfInt get_size() const {
        if constexpr (FiniteFieldType<T>) {
            return sqm<InfInt>(T::get_size(), k);
        } else {
            throw std::logic_error("get_size() only available for finite fields");
        }
    }
    const Matrix<T>& get_G() const noexcept { return G; }
    const Matrix<T>& get_HT() const noexcept { return HT; }
    Matrix<T> get_H() const { return transpose(HT); }

    size_t get_dmin() const {
        std::call_once(dmin_flag, [this] {
            if (dmin.has_value()) return;
            // if weight enumerator is calculated, use it...
            if (weight_enumerator.has_value()) {
                for (size_t i = 1; i <= weight_enumerator.value().degree(); ++i) {
                    if (weight_enumerator.value()[i] != 0) {
                        dmin = i;
                        return;
                    }
                }
                // ... otherwise:
            } else {
                if (k == 1) {
                    dmin = G.get_row(0).wH();
                    return;
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

    size_t get_tmax() const { return (get_dmin() - 1) / 2; }

    virtual const Polynomial<InfInt>& get_weight_enumerator() const {
        if constexpr (!FiniteFieldType<T>)
            throw std::logic_error("Weight enumerators make only sense for codes over finite fields!");

        std::call_once(weight_enumerator_flag, [this] {
            constexpr size_t q = T::get_size();
            if (weight_enumerator.has_value()) return;
            if (k <= this->n - k) {  // calculate directly
                std::clog << "--> Calculating weight enumerator, this requires iterating through " << sqm<InfInt>(q, k)
                          << " codewords" << std::endl;
                weight_enumerator.emplace(ZeroPolynomial<InfInt>());
                for (auto it = cbegin(); it != cend(); ++it) weight_enumerator.value().add_to_coefficient(wH(*it), 1);
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

    const Polynomial<T>& get_gamma() const {
        if (!is_polynomial()) throw std::logic_error("Trying to get gamma of a code that is not polynomial!");
        return gamma.value();
    }

    void set_dmin(size_t d) const noexcept { dmin.emplace(d); }

    void set_weight_enumerator(const Polynomial<InfInt>& p) const noexcept
        requires FiniteFieldType<T>
    {
        weight_enumerator.emplace(p);
    }

    void set_weight_enumerator(Polynomial<InfInt>&& p) const noexcept
        requires FiniteFieldType<T>
    {
        weight_enumerator.emplace(std::move(p));
    }

    void set_gamma(const Polynomial<T>& g) const noexcept {
        polynomial.emplace(true);
        gamma.emplace(g);
    }

    void set_gamma(Polynomial<T>&& g) const noexcept {
        polynomial.emplace(true);
        gamma.emplace(std::move(g));
    }

    const std::vector<Vector<T>>& get_standard_array() const
        requires FiniteFieldType<T>
    {
        std::call_once(standard_array_flag, [this] {
            if (standard_array.has_value()) return;

            std::clog << "--> Calculating standard array" << std::endl;

            constexpr size_t q = T::get_size();

            const size_t nof_cosets = sqm<size_t>(q, this->n - k);
            size_t count = 0;
            bool done = false;

            try {
                standard_array.emplace(std::vector<Vector<T>>(sqm<size_t>(q, this->n - k)));
                tainted.emplace(std::vector<bool>(standard_array.value().size(), false));
                tainted_burst.emplace(std::vector<bool>(standard_array.value().size(), false));
            } catch (const std::bad_alloc& e) {
                std::cerr << "Memory allocation failed, standard array would be too large!" << std::endl;
                throw e;
            }

            std::vector<size_t> leader_wH(standard_array.value().size(), std::numeric_limits<size_t>::max());
            std::vector<size_t> leader_cyclic_burst_length(standard_array.value().size(),
                                                           std::numeric_limits<size_t>::max());

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
                            leader_cyclic_burst_length[i] = cyclic_burst_length(v);

                            ++count;
                            if (count == nof_cosets) done = true;
                        } else {
                            if (wt == leader_wH[i]) {
                                tainted.value()[i] = true;
                                if (!tainted_burst.value()[i]) {
                                    const size_t cbl = cyclic_burst_length(v);
                                    if (cbl == leader_cyclic_burst_length[i]) tainted_burst.value()[i] = true;
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

    CodewordIterator<T> cbegin() const noexcept
        requires FiniteFieldType<T>
    {
        return CodewordIterator<T>(*this, 0);
    }

    CodewordIterator<T> cend() const noexcept
        requires FiniteFieldType<T>
    {
        return CodewordIterator<T>(*this, get_size());
    }

    bool is_perfect() const {
        if (k == 0) return true;
        return std::fabsl(HammingUpperBound<T>(this->n, get_dmin()) - k) <
               10 * std::numeric_limits<long double>::epsilon();
    }

    bool is_MDS() const { return SingletonUpperBound(this->n, get_dmin()) == k; }

    bool is_equidistant() const {
        return std::fabsl(PlotkinUpperBound<T>(this->n, get_dmin()) - k) <
               10 * std::numeric_limits<long double>::epsilon();
    }

    bool is_weakly_self_dual() const { return G * transpose(G) == ZeroMatrix<T>(k, k); }
    bool is_dual_containing() const { return transpose(HT) * HT == ZeroMatrix<T>(this->n - k, this->n - k); }
    bool is_self_dual() const { return 2 * k == this->n && is_weakly_self_dual() && is_dual_containing(); }

    bool is_polynomial() const {
        if (k == 0) return true;
        std::call_once(polynomial_flag, [this] {
            if (polynomial.has_value()) return;

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
        if (os.iword(details::index) > 0) Code<T>::get_info(os);
        if constexpr (FiniteFieldType<T>) {
            constexpr size_t q = T::get_size();
            os << "[F_" << q << "; " << this->n << ", " << k << "]";
        } else {
            os << "[Q; " << this->n << ", " << k << "]";
        }
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "G = " << std::endl;
            os << G << std::endl;
            os << "H = " << std::endl;
            os << get_H();
        }
        if (os.iword(details::index) > 1) {
            os << std::endl;
            const auto A = get_weight_enumerator();
            os << "A(x) = " << A << std::setfill(' ') << std::endl;
            size_t dmin = get_dmin();
            if (dmin == std::numeric_limits<size_t>::max())
                os << "dmin = undefined";
            else
                os << "dmin = " << dmin;
        }
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Linear code with properties: {" << std::flush;
        }
        if (os.iword(details::index) > 1) {
            if (is_polynomial()) {
                os << " polynomial(";
                if (is_cyclic()) os << "cyclic, ";
                os << "gamma = " << get_gamma() << ")" << std::flush;
            }
            if (is_perfect()) os << " perfect" << std::flush;
            if (is_MDS()) os << " MDS" << std::flush;
            if (is_equidistant()) os << " equidistant" << std::flush;
        }
        if (os.iword(details::index) > 0) {
            if (!is_self_dual() && is_weakly_self_dual()) os << " weakly_self-dual" << std::flush;
            if (!is_self_dual() && is_dual_containing()) os << " dual-containing" << std::flush;
            if (is_self_dual()) os << " self-dual" << std::flush;
            os << " }";
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

    Vector<T> enc(const Vector<T>& u) const override { return u * G; }

    Vector<T> encinv(const Vector<T>& c) const override {
        Vector<T> c_sub(k);
        size_t i = 0;
        for (auto it = infoset.cbegin(); it != infoset.cend(); ++it) {
            c_sub.set_component(i, c[*it]);
            ++i;
        }
        return c_sub * MI;
    }

    virtual Vector<T> dec_BD(const Vector<T>& r) const override {
        static_assert(FiniteFieldType<T>, "Bounded distance decoding only available for finite fields");
        validate_length(r);
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
        const auto c_est = dec_ML(r);
        if (dH(r, c_est) > this->get_tmax()) throw decoding_failure("Linear code BD decoder failed!");
        return c_est;
    }

    virtual Vector<T> dec_ML(const Vector<T>& r) const override {
        static_assert(FiniteFieldType<T>, "Maximum likelihood decoding only available for finite fields");
        validate_length(r);
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
        get_standard_array();
        const auto s = r * HT;  // calculate syndrome...
        if (s.is_zero()) return r;
        const size_t i = s.as_integer();  // ... and interpret it as binary number
        return r - standard_array.value()[i];
    }

    virtual Vector<T> dec_burst(const Vector<T>& r) const override {
        static_assert(FiniteFieldType<T>, "Burst decoding only available for finite fields");
        validate_length(r);
        if (LinearCode<T>::erasures_present(r))
            throw std::invalid_argument("Trying to correct erasures with a burst decoder!");
        get_standard_array();
        const auto s = r * HT;  // calculate syndrome...
        if (s.is_zero()) return r;
        const size_t i = s.as_integer();  // ... and interpret it as binary number
        if (tainted_burst.value()[i]) {   // decoding failure
            throw decoding_failure(
                "Linear code burst error decoder failed, coset of syndrome empty or tainted (ambiguous leader)!");
        } else {  // correct decoding or decoding error
            return r - standard_array.value()[i];
        }
    }

#ifdef CECCO_ERASURE_SUPPORT
    virtual Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        static_assert(FiniteFieldType<T>, "Bounded distance error/erasure decoding only available for finite fields");
        validate_length(r);

        size_t tau = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) ++tau;
        }

        if (tau == 0) return dec_BD(r);
        if (tau > get_dmin() - 1) throw decoding_failure("Linear code BD error/erasure decoder failed!");

        const auto c_est = dec_ML_EE(r);

        size_t t = 0;
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased() && r[i] != c_est[i]) ++t;
        }

        if (2 * t + tau > get_dmin() - 1)
            throw decoding_failure("Linear code BD error/erasure decoder failed!");
        else
            return c_est;
    }

    virtual Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        static_assert(FiniteFieldType<T>,
                      "Maximum likelihood error/erasure decoding only available for finite fields!");
        validate_length(r);

        std::call_once(punctured_codes_flag, [this] {
            if (punctured_codes.has_value()) return;

            std::clog << "--> Preparing punctured codes for error/erasure decoding" << std::endl;

            size_t count = 0;
            for (size_t tau = 1; tau <= get_dmin() - 1; ++tau) count += bin<size_t>(this->n, tau);
            punctured_codes.emplace(count);

            for (size_t tau = 1; tau <= get_dmin() - 1; ++tau) {
                std::vector<bool> mask(this->n, false);
                std::fill(mask.begin(), mask.begin() + tau, true);

                do {
                    std::vector<size_t> X;
                    X.reserve(tau);
                    for (auto it = mask.cbegin(); it != mask.cend(); ++it) {
                        if (*it) X.push_back(static_cast<size_t>(it - mask.cbegin()));
                    }

                    const size_t i = pos_to_index(X);
                    punctured_codes.value()[i].emplace(puncture(*this, X));

                } while (std::prev_permutation(mask.begin(), mask.end()));
            }
        });

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
        if (tau > get_dmin() - 1) throw decoding_failure("Linear code ML error/erasure decoder failed!");

        const auto& PC = punctured_codes.value()[pos_to_index(X)].value();

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
#endif

   protected:
    static bool erasures_present(const Vector<T>& r) {
#ifdef CECCO_ERASURE_SUPPORT
        for (size_t i = 0; i < r.get_n(); ++i) {
            if (r[i].is_erased()) return true;
        }
        return false;
#else
        return false;
#endif
    }

    void validate_length(const Vector<T>& r) const {
        if (r.get_n() != this->n)
            throw std::invalid_argument(std::string("Received vector length must be ") + std::to_string(this->n));
    }

    size_t k;
    Matrix<T> G;
    Matrix<T> HT;
    Matrix<T> MI;
    std::vector<size_t> infoset{};
    mutable std::optional<size_t> dmin;
    mutable std::once_flag dmin_flag;
    mutable std::optional<Polynomial<InfInt>> weight_enumerator;
    mutable std::once_flag weight_enumerator_flag;
    mutable std::optional<std::vector<Vector<T>>> standard_array;
    mutable std::once_flag standard_array_flag;
    mutable std::optional<std::vector<bool>> tainted;
    mutable std::once_flag tainted_flag;
    mutable std::optional<std::vector<bool>> tainted_burst;
    mutable std::once_flag tainted_burst_flag;
#ifdef CECCO_ERASURE_SUPPORT
    mutable std::optional<std::vector<std::optional<LinearCode<T>>>> punctured_codes;
    mutable std::once_flag punctured_codes_flag;
#endif
    mutable std::optional<bool> polynomial;
    mutable std::once_flag polynomial_flag;
    mutable std::optional<Polynomial<T>> gamma;
    mutable std::once_flag gamma_flag;

   private:
#ifdef CECCO_ERASURE_SUPPORT
    size_t pos_to_index(const std::vector<size_t>& pos) const {
        const size_t tau = pos.size();
        const size_t tau_max = get_dmin() - 1;

        if (tau == 0 || tau > tau_max)
            throw std::invalid_argument("Cannot calculate punctured code index from erasure positions!");

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
};

template <FiniteFieldType T>
class UniverseCode : public LinearCode<T> {
   public:
    UniverseCode(size_t n) : LinearCode<T>(n, n, IdentityMatrix<T>(n)) {
        auto weight_enumerator = Polynomial<InfInt>();
        for (size_t i = 0; i <= n; ++i) weight_enumerator.set_coefficient(i, bin<InfInt>(n, i));
        this->set_weight_enumerator(std::move(weight_enumerator));
    }

    UniverseCode(const LinearCode<T>& C) : LinearCode<T>(C) {
        if (this->n != this->k) throw std::invalid_argument("Linear code cannot be converted into a universe code!");
    }

    UniverseCode(const UniverseCode&) = default;
    UniverseCode(UniverseCode&&) = default;
    UniverseCode& operator=(const UniverseCode&) = default;
    UniverseCode& operator=(UniverseCode&&) = default;

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Universe code";
    }

    ZeroCode<T> get_dual() const noexcept { return ZeroCode<T>(this->n); }
    Vector<T> enc(const Vector<T>& u) const override { return u; }
    Vector<T> encinv(const Vector<T>& c) const override { return c; }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        this->validate_length(r);
        if (LinearCode<T>::erasures_present(r))
            throw decoding_failure("Universe code BD decoder failed, received vector contains erasures!");
        return r;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
        this->validate_length(r);
        if (LinearCode<T>::erasures_present(r))
            throw decoding_failure("Universe code ML decoder failed, received vector contains erasures!");
        return r;
    }

    Vector<T> dec_burst(const Vector<T>& r) const override {
        this->validate_length(r);
        if (LinearCode<T>::erasures_present(r))
            throw decoding_failure("Universe code burst decoder failed, received vector contains erasures!");
        return r;
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        this->validate_length(r);
        if (LinearCode<T>::erasures_present(r))
            throw decoding_failure("Universe code BD error/erasure decoder failed, contains erasures!");
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
    ZeroCode(size_t n) : LinearCode<T>(n, 0, ZeroMatrix<T>(1, n)) {
        this->set_dmin(std::numeric_limits<size_t>::max());
        const auto weight_enumerator = Polynomial<InfInt>(1);
        this->set_weight_enumerator(std::move(weight_enumerator));
        this->set_gamma(ZeroPolynomial<T>());
    }

    ZeroCode(const LinearCode<T>& C) : LinearCode<T>(C) {
        if (this->k != 0) throw std::invalid_argument("Linear code cannot be converted into a zero code!");
    }

    ZeroCode(const ZeroCode&) = default;
    ZeroCode(ZeroCode&&) = default;
    ZeroCode& operator=(const ZeroCode&) = default;
    ZeroCode& operator=(ZeroCode&&) = default;

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Zero code";
    }

    UniverseCode<T> get_dual() const noexcept { return UniverseCode<T>(this->n); }
    Vector<T> enc(const Vector<T>& u) const override { return Vector<T>(this->n, T(0)); }
    Vector<T> encinv(const Vector<T>& c) const override { throw std::logic_error("Zero code has no encoder inverse!"); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
        this->validate_length(r);
        return Vector<T>(this->n);
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
        this->validate_length(r);
        return Vector<T>(this->n);
    }

    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        this->validate_length(r);
        return Vector<T>(this->n);
    }

    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        this->validate_length(r);
        return Vector<T>(this->n);
    }
};

template <FiniteFieldType T>
class HammingCode : public LinearCode<T> {
    friend class SimplexCode<T>;

   public:
    HammingCode(size_t s) : LinearCode<T>(Hamming_n(s), Hamming_k(s), Hamming_H(s)), s(s) {
        constexpr size_t q = T::get_size();
        Polynomial<InfInt> weight_enumerator_dual;     // weight enumerator of the dual...
        weight_enumerator_dual.set_coefficient(0, 1);  // ... code, a binary simplex code...
        weight_enumerator_dual.set_coefficient(sqm<size_t>(q, s - 1), sqm<InfInt>(q, s) - 1);  // ... easy to calculate
        this->set_weight_enumerator(MacWilliamsIdentity<T>(weight_enumerator_dual, this->n, this->n - this->k));
    }

    HammingCode(const LinearCode<T>& C) : LinearCode<T>(C), s(0) {
        for (size_t s_cand = 2; s_cand < std::numeric_limits<size_t>::max(); ++s_cand) {
            const size_t n = Hamming_n(s_cand);
            if (n > this->n) break;
            if (n == this->n) {
                if (Hamming_k(s_cand) == this->k && this->get_dmin() == 3) {
                    this->s = s_cand;
                    return;
                }
            }
        }
        throw std::invalid_argument("Linear code cannot be converted into a Hamming code!");
    }

    HammingCode(const HammingCode&) = default;
    HammingCode(HammingCode&&) = default;
    HammingCode& operator=(const HammingCode&) = default;
    HammingCode& operator=(HammingCode&&) = default;

    size_t get_s() const noexcept { return s; }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Hamming code with properties: { s = " << s << " }";
    }

    SimplexCode<T> get_dual() const noexcept { return SimplexCode<T>(s); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
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
        if (!LinearCode<T>::erasures_present(r))
            return dec_BD(r);
        else
            return LinearCode<T>::dec_ML_EE(r);
    }

#ifdef CECCO_ERASURE_SUPPORT
    virtual Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        std::vector<size_t> X;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) X.push_back(i);
        }
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
#endif

   private:
    size_t s;

    static size_t Hamming_n(size_t s) noexcept {
        constexpr size_t q = T::get_size();
        return (sqm<size_t>(q, s) - 1) / (q - 1);
    }

    static size_t Hamming_k(size_t s) noexcept { return Hamming_n(s) - s; }

    static Matrix<T> Hamming_H(size_t s) {
        const size_t n = Hamming_n(s);
        auto H = Matrix<T>(n, s);
        size_t i = 0;
        // topmost element loop
        for (size_t top = 0; top < s; ++top) {
            const auto v = IdentityMatrix<T>(s - top - 1).rowspace();
            for (size_t j = 0; j < v.size(); ++j) {
                H(n - i - 1, top) = T(1);
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
        throw std::invalid_argument("Linear code cannot be converted into a Simplex code!");
    }

    SimplexCode(const SimplexCode&) = default;
    SimplexCode(SimplexCode&&) = default;
    SimplexCode& operator=(const SimplexCode&) = default;
    SimplexCode& operator=(SimplexCode&&) = default;

    size_t get_s() const noexcept { return s; }

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Simplex code with properties: { s = " << s << " }";
    }

    HammingCode<T> get_dual() const noexcept { return HammingCode<T>(s); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
        if constexpr (T::get_size() != 2) return LinearCode<T>::dec_BD(r);
        this->validate_length(r);

        const auto c_est = dec_ML(r);
        if (dH(r, c_est) > this->get_tmax())
            throw decoding_failure("Simplex code BD decoder failed!");
        else
            return c_est;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
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
        for (size_t i = 0; i < y.get_n(); ++i) {
            if (y[i] > best) {
                best = y[i];
                hit = i;
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
        for (size_t u = 0; u < y.get_n(); ++u) {
            if (y[u] > best) {
                best = y[u];
                hit = u;
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
            throw std::invalid_argument("Linear code cannot be converted into a repetition code!");
    }

    RepetitionCode(const RepetitionCode&) = default;
    RepetitionCode(RepetitionCode&&) = default;
    RepetitionCode& operator=(const RepetitionCode&) = default;
    RepetitionCode& operator=(RepetitionCode&&) = default;

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Repetition code";
    }

    SingleParityCheckCode<T> get_dual() const noexcept { return SingleParityCheckCode<T>(this->n); }
    Vector<T> enc(const Vector<T>& u) const override { return Vector<T>(this->n, u[0]); }
    Vector<T> encinv(const Vector<T>& c) const override { return Vector<T>(1, c[0]); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return LinearCode<T>::dec_BD_EE(r);

        const auto c_est = dec_ML(r);
        constexpr size_t q = T::get_size();
        if (q != 2 || this->n % 2 == 0) {
            if (2 * dH(r, c_est) > this->get_dmin() - 1) throw decoding_failure("Repetition code BD decoder failed!");
        }

        return c_est;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
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

        if (q > 2 || q % 2 == 0) {
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
        if (this->k != this->n - 1 || this->get_dmin() != 2)
            throw std::invalid_argument("Linear code cannot be converted into a single parity check code!");
    }

    SingleParityCheckCode(const SingleParityCheckCode&) = default;
    SingleParityCheckCode(SingleParityCheckCode&&) = default;
    SingleParityCheckCode& operator=(const SingleParityCheckCode&) = default;
    SingleParityCheckCode& operator=(SingleParityCheckCode&&) = default;

    void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0) os << "Single parity check code";
    }

    RepetitionCode<T> get_dual() const noexcept { return RepetitionCode<T>(this->n); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
        this->validate_length(r);

        T s = T(0);
        for (size_t i = 0; i < this->n; ++i) s += r[i];

        if (!s.is_zero()) throw decoding_failure("Single parity check code BD decoder failed!");

        return r;
    }

    Vector<T> dec_ML(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
        this->validate_length(r);

        T s = T(0);
        for (size_t i = 0; i < this->n; ++i) s += r[i];
        if (s.is_zero()) return r;

        Vector<T> c_est = r;
        c_est.set_component(0, r[0] - s);
        return c_est;
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        std::vector<size_t> X;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) X.push_back(i);
        }
        const size_t tau = X.size();

        if (tau == 0) return dec_BD(r);
        if (tau > 1) throw decoding_failure("Single parity check code BD error/erasure decoder failed!");

        T s = T(0);
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased()) s += r[i];
        }

        Vector<T> c_est = r;
        c_est.set_component(X[0], -s);
        return c_est;
    }

    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        std::vector<size_t> X;
        for (size_t i = 0; i < this->n; ++i) {
            if (r[i].is_erased()) X.push_back(i);
        }
        const size_t tau = X.size();

        if (tau == 0) return dec_ML(r);
        if (tau == this->n) return Vector<T>(this->n, T(0));

        T s = T(0);
        for (size_t i = 0; i < this->n; ++i) {
            if (!r[i].is_erased()) s += r[i];
        }

        Vector<T> c_est = r;
        c_est.set_component(X[0], -s);
        for (size_t i = 1; i < tau; ++i) c_est.set_component(X[i], T(0));
        return c_est;
    }
#endif
};

template <FiniteFieldType T>
    requires(T::get_size() == 2 || T::get_size() == 3)
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
            if (T::get_size() == 2 && this->get_dmin() == 7)
                return;
            else if (T::get_size() == 3 && this->get_dmin() == 5)
                return;
        }
        throw std::invalid_argument("Linear code cannot be converted into a Golay code!");
    }

    GolayCode(const GolayCode&) = default;
    GolayCode(GolayCode&&) = default;
    GolayCode& operator=(const GolayCode&) = default;
    GolayCode& operator=(GolayCode&&) = default;

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            if constexpr (T::get_size() == 2) {
                os << "Binary Golay code";
            } else {
                os << "Ternary Golay code";
            }
        }
    }

   private:
    static constexpr size_t Golay_n() {
        if constexpr (T::get_size() == 2)
            return 23;
        else
            return 11;
    }

    static constexpr size_t Golay_k() {
        if constexpr (T::get_size() == 2)
            return 12;
        else
            return 6;
    }

    static Matrix<T> Golay_G() {
        Polynomial<T> gamma;
        if constexpr (T::get_size() == 2)
            gamma = Polynomial<T>({1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1});
        else
            gamma = Polynomial<T>({2, 0, 1, 2, 1, 1});

        const size_t n = Golay_n();
        const size_t k = Golay_k();

        return ToeplitzMatrix(pad_back(pad_front(Vector<T>(gamma), n), 2 * k + gamma.degree() - 1), k, n);
    }
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
        if (this->k != 2) throw std::invalid_argument("Linear code cannot be converted into a Cordaro-Wagner code!");

        r = std::floor(n / 3.0 + 1.0 / 2.0);
        m = this->n - 3 * r;

        if (this->get_dmin() != std::min(2 * r, 2 * r + m))
            throw std::invalid_argument("Linear code cannot be converted into a Cordaro-Wagner code!");
    }

    CordaroWagnerCode(const CordaroWagnerCode&) = default;
    CordaroWagnerCode(CordaroWagnerCode&&) = default;
    CordaroWagnerCode& operator=(const CordaroWagnerCode&) = default;
    CordaroWagnerCode& operator=(CordaroWagnerCode&&) = default;

    size_t get_r() const noexcept { return r; }
    int8_t get_m() const noexcept { return m; }

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<Fp<2>>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0)
            os << "Cordaro-Wagner code with properties: { r = " << r << ", m = " << (int)m << " }";
    }

    Vector<Fp<2>> dec_BD(const Vector<Fp<2>>& r) const override {
        if (LinearCode<Fp<2>>::erasures_present(r)) return dec_BD_EE(r);
        this->validate_length(r);

        const size_t t = this->get_tmax();
        for (auto it = this->cbegin(); it != this->cend(); ++it) {
            if (dH(*it, r) <= t) return *it;
        }
        throw decoding_failure("Cordaro-Wagner code BD failed!");
    }

    Vector<Fp<2>> dec_ML(const Vector<Fp<2>>& r) const override {
        if (LinearCode<Fp<2>>::erasures_present(r)) return dec_ML_EE(r);
        this->validate_length(r);

        Vector<Fp<2>> best;
        size_t best_t = std::numeric_limits<size_t>::max();

        for (auto it = this->cbegin(); it != this->cend(); ++it) {
            const size_t t = dH(*it, r);
            if (t < best_t) {
                best = *it;
                best_t = t;
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

        for (auto it = this->cbegin(); it != this->cend(); ++it) {
            size_t t = 0;
            for (size_t i = 0; i < this->n; ++i) {
                if (!r[i].is_erased() && r[i] != (*it)[i]) ++t;
            }

            if (t < best_t) {
                best_t = t;
                best = *it;
                if (best_t == 0) return best;
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

template <FieldType T, class B>
    requires std::derived_from<B, LinearCode<T>>
class ExtendedCode : public LinearCode<T> {
   public:
    ExtendedCode(const B& BaseCode, size_t i, const Vector<T>& v) try : LinearCode
        <T>(BaseCode.get_n() + 1, BaseCode.get_k(), Extended_G(BaseCode.get_G(), i, v)), BaseCode(BaseCode), i(i), v(v),
            parity(false) {
            if (v == Extended_v(this->BaseCode.get_G())) parity = true;
        }
    catch (const std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Trying to extend a code with invalid extension parameters: ") +
                                    e.what());
    }

    ExtendedCode(B&& BaseCode, size_t i, const Vector<T>& v) try : LinearCode
        <T>(BaseCode.get_n() + 1, BaseCode.get_k(), Extended_G(BaseCode.get_G(), i, v)), BaseCode(std::move(BaseCode)),
            i(i), v(v), parity(v == Extended_v(this->BaseCode.get_G())) {}
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
        if constexpr (T::get_size() == 2) {
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

    virtual void get_info(std::ostream& os) const override {
        if (os.iword(details::index) < 3) {
            LinearCode<T>::get_info(os);
            os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            const auto old = os.iword(details::index);
            os << "Extended code with properties: { i = " << i;
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
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
        return LinearCode<T>::dec_BD(r);
    }

    virtual Vector<T> dec_ML(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
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
            throw decoding_failure("Extended code BD error/erasure decoding failed!");

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
            os << std::endl;
        }
        if (os.iword(details::index) > 0) {
            const auto old = os.iword(details::index);
            os << "Augmented code with properties: { w = " << w;
            os << ", BaseCode = " << showbasic;
            BaseCode.LinearCode<T>::get_info(os);
            os << " " << showspecial << BaseCode;
            os << " }";
            os.iword(details::index) = old;
        }
    }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_BD_EE(r);
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

    Vector<T> dec_ML(const Vector<T>& r) const override {
        if (LinearCode<T>::erasures_present(r)) return dec_ML_EE(r);
        this->validate_length(r);

        const size_t tmax = this->get_tmax();

        Vector<T> best = BaseCode.dec_ML(r);  // r - 0*w
        size_t best_t = dH(r, best);

        if (best_t <= tmax) return best;

        for (size_t i = 1; i < T::get_size(); ++i) {
            const T alpha = T(i);

            Vector<T> cp_est = BaseCode.dec_ML(r - alpha * w);
            Vector<T> c_est = cp_est + alpha * w;

            const size_t t = dH(r, c_est);

            if (t <= tmax) return c_est;

            if (t < best_t) {
                best_t = t;
                best = std::move(c_est);
            }
        }

        return best;
    }

#ifdef CECCO_ERASURE_SUPPORT
    Vector<T> dec_BD_EE(const Vector<T>& r) const override {
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

    Vector<T> dec_ML_EE(const Vector<T>& r) const override {
        this->validate_length(r);

        Vector<T> best = BaseCode.dec_ML_EE(r);  // r - 0*w
        size_t best_t = 0;
        for (size_t j = 0; j < this->n; ++j) {
            if (!r[j].is_erased() && r[j] != best[j]) ++best_t;
        }
        if (best_t == 0) return best;

        for (size_t i = 1; i < T::get_size(); ++i) {
            const T alpha = T(i);

            const auto cp_est = BaseCode.dec_ML_EE(r - alpha * w);
            const auto c_est = cp_est + alpha * w;

            size_t t = 0;
            for (size_t j = 0; j < this->n; ++j) {
                if (!r[j].is_erased() && r[j] != c_est[j]) ++t;
            }

            if (t == 0) return c_est;

            if (t < best_t) {
                best_t = t;
                best = std::move(c_est);
            }
        }

        return best;
    }
#endif

   private:
    static Matrix<T> Augmented_G(const Matrix<T>& Gp, size_t j, const Vector<T>& w) {
        const size_t n = Gp.get_n();
        const size_t k = Gp.get_m();

        if (w.get_n() != n) throw std::invalid_argument(std::string("Length of w must be ") + std::to_string(n));

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

template <typename T>
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
using field_t = typename base_t<C>::field;

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

// docu note: unextend maintains special structure, puncture does not
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

// docu note: unaugment maintains special structure, expurgate does not
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

// docu note: unlengthen maintains special structure, shorten does not
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
    if constexpr (!std::is_same_v<L, R>) {
        return false;
    } else {
        if (lhs.get_n() != rhs.get_n() || lhs.get_k() != rhs.get_k()) return false;
        if (lhs.get_G() == rhs.get_G() || lhs.get_H() == rhs.get_H()) return true;
        if (lhs.get_n() - lhs.get_k() < lhs.get_k())
            return rref(lhs.get_H()) == rref(rhs.get_H());
        else
            return rref(lhs.get_G()) == rref(rhs.get_G());
    }
}

template <FieldType L, FieldType R>
bool operator!=(const LinearCode<L>& lhs, const LinearCode<R>& rhs) {
    return !(lhs == rhs);
}

template <FieldType T>
std::ostream& operator<<(std::ostream& os, const Code<T>& rhs) {
    rhs.get_info(os);
    return os;
}

}  // namespace CECCO

#endif