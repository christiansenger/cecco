/**
 * @file codes.hpp
 * @brief Error control codes library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.0.0
 * @date 2025
 *
 * @copyright
 * Copyright (c) 2025, Christian Senger <senger@inue.uni-stuttgart.de>
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
    const size_t tmax = std::floorl((long double)(dmin - 1) / 2);
    InfInt h = 0;
    for (size_t i = 0; i <= tmax; ++i) h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
    return n - std::log2l(h.toUnsignedLongLong()) / std::log2l(q);
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
    const size_t tmax = std::floorl((dmin - 1) / 2.0);
    const size_t s = dmin % 2;
    InfInt h = 0;
    for (size_t i = 0; i <= tmax; ++i) h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
    return n - std::log2l(h.toUnsignedLongLong() + (bin<InfInt>(n, tmax + 1) * sqm<InfInt>(q - 1, tmax + 1) -
                                                    InfInt(s) * bin<InfInt>(dmin, tmax) * A<T>(n, dmin, dmin))
                                                           .toUnsignedLongLong() /
                                                       (long double)A<T>(n, dmin, tmax + 1)) /
                   std::log2l(q);
}

template <FiniteFieldType T>
long double PlotkinUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    if ((long double)dmin / n > (long double)(q - 1) / q) {  // conventional
        return 1 - std::log2l(q - (long double)n * (q - 1) / dmin) / std::log2l(q);
    } else {  // improved
        const size_t Delta = n - std::floorl((long double)dmin * q / (q - 1)) + 1;
        const size_t M = sqm(q, Delta + 1) * (long double)dmin / ((q * dmin - (n - Delta) * (q - 1)));
        return std::log2l(M) / std::log2l(q);
    }
}

template <FiniteFieldType T>
long double EliasUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    const long double r = 1 - (long double)1 / q;
    double min = std::numeric_limits<long double>::max();
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
    InfInt sum = 0;
    for (size_t i = 0; i < dmin; ++i) sum += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
    return std::ceill(n - std::log2l(sum.toUnsignedLongLong()) / std::log2l(q));
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
                    auto b = bin<InfInt>(i, h);
                    if (b != 0) {
                        auto c = sqm<InfInt>(q - 1, j - h);
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

template <ComponentType T>
class Code {
   public:
    Code(size_t n) : n(n) {}

    Code(const Code& other) : n(other.n) {}

    Code(Code&&) = default;

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

   protected:
    size_t n;
};

template <ComponentType T>
class EmptyCode : public Code<T> {
   public:
    EmptyCode(size_t n) : Code<T>(n) {}

    EmptyCode(const EmptyCode&) = default;
    EmptyCode(EmptyCode&&) = default;
    EmptyCode& operator=(const EmptyCode&) = default;
    EmptyCode& operator=(EmptyCode&&) = default;

    void get_info(std::ostream& os) const override {
        Code<T>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
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
                auto quot = j / q;
                auto rem = (j - quot).toInt();
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

namespace details {

class decoding_failure : public std::exception {
   public:
    decoding_failure(const char* message) : message(message) {}
    const char* what() const noexcept override { return message.c_str(); }

   private:
    const std::string message;
};

}  // end namespace details

template <FieldType T>
class LinearCode : public Code<T> {
   public:
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
            auto Gp = rref(G);
            size_t i = 0;
            for (size_t j = 0; j < k; ++j) {
                auto u = unit_vector<T>(k, j);
                while (Gp.get_col(i) != u) ++i;
                infoset.push_back(i);
            }
        } else if (X.get_m() == this->n - k) {
            // X suppoed to be parity check matrix H
            if (X.rank() != this->n - k)
                throw std::invalid_argument("H must have full rank " + std::to_string(this->n - k));
            G = X.basis_of_nullspace();
            std::cout << G << std::endl;
            HT = transpose(X);
            G.rref();
            size_t i = 0;
            for (size_t j = 0; j < k; ++j) {
                auto u = unit_vector<T>(k, j);
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
          polynomial(other.polynomial),
          polynomial_flag(),
          gamma(other.gamma),
          gamma_flag() {}

    LinearCode(LinearCode&&) = default;

    LinearCode& operator=(const LinearCode& other) {
        if (this != &other) {
            k = other.k;
            G = other.G;
            HT = other.HT;
            MI = other.MI;
            infoset = other.infoset;
            dmin = other.dmin;
            dmin_flag = {};
            weight_enumerator = other.weight_enumerator;
            weight_enumerator_flag = {};
            standard_array = other.standard_array;
            standard_array_flag = {};
            tainted = other.tainted;
            tainted_flag = {};
            tainted_burst = other.tainted_burst;
            tainted_burst_flag = {};
            polynomial = other.polynomial;
            polynomial_flag = {};
            gamma = other.gamma;
            gamma_flag = {};
        }
        return *this;
    }

    LinearCode& operator=(LinearCode&&) = default;
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
    Matrix<T> get_H() const noexcept { return transpose(HT); }

    size_t get_dmin() const noexcept {
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
                            if (selection[j] == true) {
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

    size_t get_tmax() const noexcept { return std::floor((get_dmin() - 1) / 2.0); }

    const Polynomial<InfInt>& get_weight_enumerator() const noexcept
        requires FiniteFieldType<T>
    {
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
                std::clog << "--> Using MacWilliam's identity for: " << std::endl;
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

    void set_dmin(size_t d) noexcept { dmin.emplace(d); }

    void set_weight_enumerator(const Polynomial<InfInt>& p) noexcept
        requires FiniteFieldType<T>
    {
        weight_enumerator.emplace(p);
    }

    void set_gamma(const Polynomial<T>& g) const noexcept {
        polynomial.emplace(true);
        gamma.emplace(g);
    }

    const std::vector<Vector<T>>& get_standard_array() const noexcept
        requires FiniteFieldType<T>
    {
        std::call_once(standard_array_flag, [this] {
            constexpr size_t q = T::get_size();
            if (standard_array.has_value()) return;
            std::clog << "--> Calculating standard array, this requires iterating through all vectors of the universe"
                      << std::endl;
            standard_array.emplace(std::vector<Vector<T>>(sqm<size_t>(q, this->n - k)));
            tainted.emplace(std::vector<bool>(standard_array.value().size(), false));
            tainted_burst.emplace(std::vector<bool>(standard_array.value().size(), false));
            UniverseCode<T> U(this->n);
            for (auto it = U.cbegin(); it != U.cend(); ++it) {
                const auto s = *it * HT;
                const size_t i = s.as_integer();
                if (standard_array.value()[i].is_empty()) {
                    standard_array.value()[i] = *it;
                } else {
                    if (wH(standard_array.value()[i]) > wH(*it)) {
                        standard_array.value()[i] = *it;
                        tainted.value()[i] = false;
                        tainted_burst.value()[i] = false;
                    } else {
                        if (wH(standard_array.value()[i]) == wH(*it)) tainted.value()[i] = true;
                        const size_t cb_0 = cyclic_burst_length(standard_array.value()[i]);
                        const size_t cb_1 = cyclic_burst_length(*it);
                        if (cb_0 == cb_1) tainted_burst.value()[i] = true;
                    }
                }
            }
        });

        /*
        for (size_t i=0; i<sqm<size_t>(T::get_size(), this->n - k); ++i) {
            std::cout << standard_array.value()[i] << ", " << (int) tainted.value()[i] << ", " << (int)
        tainted_burst.value()[i] << std::endl;
        }
        */

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

    bool is_perfect() const noexcept {
        if (k == 0) return true;
        return std::fabsl(HammingUpperBound<T>(this->n, get_dmin()) - k) <
               10 * std::numeric_limits<long double>::epsilon();
    }

    bool is_MDS() const noexcept { return SingletonUpperBound(this->n, get_dmin()) == k; }

    bool is_equidistant() const noexcept {
        return std::fabsl(PlotkinUpperBound<T>(this->n, get_dmin()) - k) <
               10 * std::numeric_limits<long double>::epsilon();
    }

    bool is_weakly_self_dual() const noexcept { return G * transpose(G) == ZeroMatrix<T>(k, k); }
    bool is_dual_containing() const noexcept { return transpose(HT) * HT == ZeroMatrix<T>(this->n - k, this->n - k); }
    bool is_self_dual() const noexcept { return 2 * k == this->n && is_weakly_self_dual() && is_dual_containing(); }

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
        if (os.iword(details::index) > 0) {
            Code<T>::get_info(os);
            os << std::endl;
        }
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
            auto A = get_weight_enumerator();
            os << "A(x) = " << A << std::setfill(' ') << std::endl;
            size_t dmin = get_dmin();
            if (dmin == std::numeric_limits<size_t>::max())
                os << "dmin = undefined";
            else
                os << "dmin = " << dmin;
        }
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Linear code properties: {" << std::flush;
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

    LinearCode<T> dual() const noexcept {
        auto dual_code = LinearCode<T>(this->n, this->n - k, get_H());
        if constexpr (FiniteFieldType<T>) {
            if (weight_enumerator.has_value())
                dual_code.set_weight_enumerator(MacWilliamsIdentity<T>(weight_enumerator.value(), this->n, k));
        }
        return dual_code;
    }

    Vector<T> enc(const Vector<T>& u) const noexcept override { return u * G; }

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
        auto c_hat = dec_ML(r);
        if (dH(r, c_hat) > this->get_tmax())
            throw details::decoding_failure("Linear code BD decoder failed!");
        else
            return c_hat;

        /*
        // possible implemenation of boosted BD
        get_standard_array();
        auto s = r * HT;  // calculate syndrome...
        if (s.is_zero()) return r;
        const size_t i = s.as_integer();  // ... and interpret it as binary number
        if (tainted.value()[i]) {         // decoding failure
            throw details::decoding_failure(
                "Linear code BD decoder failed, coset of syndrome empty or tainted (ambiguous leader)!");
        } else {  // correct decoding or decoding error
            return r - standard_array.value()[i];
        }
        */
    }

    virtual Vector<T> dec_ML(const Vector<T>& r) const noexcept override {
        static_assert(FiniteFieldType<T>, "Maximum likelihood decoding only available for finite fields");
        get_standard_array();
        auto s = r * HT;  // calculate syndrome...
        if (s.is_zero()) return r;
        const size_t i = s.as_integer();  // ... and interpret it as binary number
        return r - standard_array.value()[i];
    }

    virtual Vector<T> dec_burst(const Vector<T>& r) const override {
        static_assert(FiniteFieldType<T>, "Burst decoding only available for finite fields");
        get_standard_array();
        auto s = r * HT;  // calculate syndrome...
        if (s.is_zero()) return r;
        const size_t i = s.as_integer();  // ... and interpret it as binary number
        if (tainted_burst.value()[i]) {   // decoding failure
            throw details::decoding_failure(
                "Linear code burst error decoder failed, coset of syndrome empty or tainted (ambiguous leader)!");
        } else {  // correct decoding or decoding error
            return r - standard_array.value()[i];
        }
    }

   protected:
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
    mutable std::optional<bool> polynomial;
    mutable std::once_flag polynomial_flag;
    mutable std::optional<Polynomial<T>> gamma;
    mutable std::once_flag gamma_flag;

    LinearCode() {};
};

template <FiniteFieldType T>
class UniverseCode : public LinearCode<T> {
   public:
    UniverseCode(size_t n) : LinearCode<T>(n, n, IdentityMatrix<T>(n)) {
        auto weight_enumerator = Polynomial<InfInt>();
        for (size_t i = 0; i <= n; ++i) weight_enumerator.set_coefficient(i, bin<InfInt>(n, i));
        this->set_weight_enumerator(weight_enumerator);
    }

    UniverseCode(const LinearCode<T>& C) : LinearCode<T>(C) {
        if (this->n != this->k) throw std::invalid_argument("Linear code cannot be converted into a universe code!");
    }

    UniverseCode(const UniverseCode&) = default;
    UniverseCode(UniverseCode&&) = default;
    UniverseCode& operator=(const UniverseCode&) = default;
    UniverseCode& operator=(UniverseCode&&) = default;

    void get_info(std::ostream& os) const override {
        LinearCode<T>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Universe code";
        }
    }

    ZeroCode<T> dual() const noexcept { return ZeroCode<T>(this->n); }
    Vector<T> enc(const Vector<T>& u) const noexcept override { return u; }
    Vector<T> encinv(const Vector<T>& c) const override { return c; }
    Vector<T> dec_BD(const Vector<T>& r) const override { return r; }
    Vector<T> dec_ML(const Vector<T>& r) const noexcept override { return r; }
    Vector<T> dec_burst(const Vector<T>& r) const override { return r; }
};

template <FiniteFieldType T>
class ZeroCode : public LinearCode<T> {
   public:
    ZeroCode(size_t n) : LinearCode<T>(n, 0, Matrix<T>(1, n, T(0))) {
        this->set_dmin(std::numeric_limits<size_t>::max());
        auto weight_enumerator = Polynomial<InfInt>(1);
        this->set_weight_enumerator(weight_enumerator);
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
        LinearCode<T>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Zero code";
        }
    }

    UniverseCode<T> dual() const noexcept { return UniverseCode<T>(this->n); }
    Vector<T> enc(const Vector<T>& u) const noexcept override { return Vector<T>(this->n, T(0)); }
    Vector<T> encinv(const Vector<T>& c) const override { throw std::logic_error("Zero code has no encoder inverse!"); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        if (!r.is_zero())
            throw details::decoding_failure("Zero code BD decoder failed, received vector is not the zero codeword!");
        else
            return r;
    }

    Vector<T> dec_ML(const Vector<T>& r) const noexcept override { return ZeroVector<T>(this->n); }

    Vector<T> dec_burst(const Vector<T>& r) const override {
        if (!r.is_zero())
            throw details::decoding_failure(
                "Zero code burst error decoder failed, received vector is not the zero codeword!");
        else
            return r;
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
        LinearCode<T>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Hamming code properties: { s = " << s << " }";
        }
    }

    SimplexCode<T> dual() const noexcept { return SimplexCode<T>(s); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        constexpr size_t q = T::get_size();
        auto s = r * this->HT;
        if (s.is_zero()) return r;
        auto c_hat = r;
        for (size_t i = 0; i < this->n; ++i) {
            for (size_t j = 1; j < q; ++j) {
                T a(j);
                if (s == a * this->HT.get_row(i)) {
                    c_hat.set_component(i, c_hat[i] - a);
                    return c_hat;
                }
            }
        }

        return c_hat;
    }

    Vector<T> dec_ML(const Vector<T>& r) const noexcept override { return dec_BD(r); }

   private:
    size_t s;

    static size_t Hamming_n(size_t s) noexcept {
        constexpr size_t q = T::get_size();
        return (sqm<size_t>(q, s) - 1) / (q - 1);
    }

    static size_t Hamming_k(size_t s) noexcept { return Hamming_n(s) - s; }

    static Matrix<T> Hamming_H(size_t s) noexcept {
        const size_t n = Hamming_n(s);
        const size_t k = Hamming_k(s);
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
        this->set_weight_enumerator(weight_enumerator);

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
                    return;
                }
            }
        }
        for (size_t i = 0; i < this->n; ++i) cols_as_integers[i] = this->G.get_col(i).as_integer();
        throw std::invalid_argument("Linear code cannot be converted into a Simplex code!");
    }

    SimplexCode(const SimplexCode&) = default;
    SimplexCode(SimplexCode&&) = default;
    SimplexCode& operator=(const SimplexCode&) = default;
    SimplexCode& operator=(SimplexCode&&) = default;

    size_t get_s() const noexcept { return s; }

    void get_info(std::ostream& os) const override {
        LinearCode<T>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Simplex code properties: { s = " << s << " }";
        }
    }

    HammingCode<T> dual() const noexcept { return HammingCode<T>(s); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        if constexpr (T::get_size() != 2) return LinearCode<T>::dec_BD(r);

        auto c_hat = dec_ML(r);
        if (dH(r, c_hat) > this->get_tmax())
            throw details::decoding_failure("Simplex code BD decoder failed!");
        else
            return c_hat;
    }

    Vector<T> dec_ML(const Vector<T>& r) const noexcept override {
        if constexpr (T::get_size() != 2) return LinearCode<T>::dec_ML(r);

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

        Vector<T> u_hat;
        u_hat.from_integer(hit, this->k);

        Vector<T> c_hat(this->n, T(0));
        for (size_t i = 0; i < this->n; ++i) c_hat.set_component(i, inner_product(u_hat, this->G.get_col(i)));

        return c_hat;
    }

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
        this->set_weight_enumerator(weight_enumerator);
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
        LinearCode<T>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Repetition code";
        }
    }

    SingleParityCheckCode<T> dual() const noexcept { return SingleParityCheckCode<T>(this->n); }
    Vector<T> enc(const Vector<T>& u) const noexcept override { return Vector<T>(this->n, u[0]); }
    Vector<T> encinv(const Vector<T>& c) const override { return Vector<T>(1, c[0]); }

    Vector<T> dec_BD(const Vector<T>& r) const override {
        constexpr size_t q = T::get_size();
        std::array<size_t, q> counters{};
        for (size_t i = 0; i < r.get_n(); ++i) ++counters[r[i].get_label()];
        const auto it = std::max_element(counters.cbegin(), counters.cend());
        std::size_t label = static_cast<std::size_t>(std::distance(counters.cbegin(), it));
        return Vector<T>(this->n, label);
    }

    Vector<T> dec_ML(const Vector<T>& r) const noexcept override { return dec_BD(r); }
    Vector<T> dec_burst(const Vector<T>& r) const noexcept override { return dec_BD(r); }

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
        LinearCode<T>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Single parity check code";
        }
    }

    RepetitionCode<T> dual() const noexcept { return RepetitionCode<T>(this->n); }
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
            if constexpr (T::get_size() == 2) {
                if (this->get_dmin() != 7) return;
            } else if (T::get_size() == 3) {
                if (this->get_dmin() != 5) return;
            }
        }
        throw std::invalid_argument("Linear code cannot be converted into a Golay code!");
    }

    GolayCode(const GolayCode&) = default;
    GolayCode(GolayCode&&) = default;
    GolayCode& operator=(const GolayCode&) = default;
    GolayCode& operator=(GolayCode&&) = default;

    virtual void get_info(std::ostream& os) const override {
        LinearCode<T>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
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
        this->set_weight_enumerator(weight_enumerator);
    }

    CordaroWagnerCode(const LinearCode<Fp<2>>& C) : LinearCode<Fp<2>>(C), r(0), m(0) {
        if (this->k != 2) throw std::invalid_argument("Linear code cannot be converted into a Cordaro-Wagner code!");

        r = std::floor(n / 3 + 1 / 2);
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
        LinearCode<Fp<2>>::get_info(os);
        if (os.iword(details::index) > 0) {
            os << std::endl;
            os << "Cordaro-Wagner code { r = " << r << ", m = " << (int)m << " }";
        }
    }

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

template <typename CodeType>
auto dual(const CodeType& C) {
    return C.dual();
}

template <ComponentType L, ComponentType R>
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

template <ComponentType L, ComponentType R>
bool operator!=(const LinearCode<L>& lhs, const LinearCode<R>& rhs) {
    return !(lhs == rhs);
}

template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Code<T>& rhs) {
    rhs.get_info(os);
    return os;
}

}  // namespace CECCO

#endif