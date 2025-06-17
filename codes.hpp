/*
   Copyright 2025 Christian Senger <senger@inue.uni-stuttgart.de>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   v1.0
*/

/* ToDo:
    - Goppa codes (https://en.wikipedia.org/wiki/Binary_Goppa_code), not only binary
*/

#ifndef CODES_HPP
#define CODES_HPP

#include <algorithm>
#include <exception>
#include <iostream>
#include <ranges>
#include <vector>

#include "InfInt.hpp"
#include "blocks.hpp"
#include "fields.hpp"
#include "helpers.hpp"
#include "matrices.hpp"
#include "polynomials.hpp"
#include "vectors.hpp"

namespace ECC {

namespace details {}
using namespace details;

template <class T>
long double HammingUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    const size_t tmax = floorl((long double)(dmin - 1) / 2);
    InfInt h = 0;
    for (size_t i = 0; i <= tmax; ++i) {
        h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
    }

    return n - log2l(h.toUnsignedLongLong()) / log2l(q);
}

namespace details {

template <class T>
static size_t A(size_t n, size_t d, size_t w) {
    if (2 * w < d) return 1;
    constexpr size_t q = T::get_size();
    const size_t e = ceill(d / 2.0);
    size_t res = 1;
    for (size_t i = e; i <= w; ++i) {
        res *= (n - w + i) * (q - 1) / (long double)i;
    }
    return res;
}

}  // namespace details

template <class T>
long double JohnsonUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    const size_t tmax = floorl((dmin - 1) / 2.0);
    const size_t s = dmin % 2;
    InfInt h = 0;
    for (size_t i = 0; i <= tmax; ++i) {
        h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
    }

    return n - log2l(h.toUnsignedLongLong() + (bin<InfInt>(n, tmax + 1) * sqm<InfInt>(q - 1, tmax + 1) -
                                               InfInt(s) * bin<InfInt>(dmin, tmax) * A<T>(n, dmin, dmin))
                                                      .toUnsignedLongLong() /
                                                  (long double)A<T>(n, dmin, tmax + 1)) /
                   log2l(q);
}

template <class T>
long double PlotkinUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    if ((long double)dmin / n > (long double)(q - 1) / q) {  // conventional
        return 1 - log2l(q - (long double)n * (q - 1) / dmin) / log2l(q);
    } else {  // improved
        const size_t Delta = n - floor((long double)dmin * q / (q - 1)) + 1;
        const size_t M = sqm(q, Delta + 1) * (long double)dmin / ((q * dmin - (n - Delta) * (q - 1)));
        return log2l(M) / log2l(q);
    }
}

template <class T>
long double EliasUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();
    const long double r = 1 - (long double)1 / q;

    double min = std::numeric_limits<long double>::max();

    for (size_t w = 0; w <= r * n; ++w) {
        if (w * w - 2 * r * n * w + r * n * dmin > 0) {
            InfInt h = 0;
            for (size_t i = 0; i <= w; ++i) {
                h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
            }

            long double temp = (long double)(r * n * dmin) / (w * w - 2 * r * n * w + r * n * dmin);
            temp /= h.toUnsignedLongLong();
            min = std::min<long double>(temp, min);
        }
    }

    return n + log2l(min) / log2l(q);
}

size_t SingletonUpperBound(size_t n, size_t dmin) { return n - dmin + 1; }

template <class T>
size_t GriesmerUpperBound(size_t n, size_t dmin) {
    constexpr size_t q = T::get_size();

    size_t k = 0;
    ;

    for (size_t kp = 1; kp <= n; ++kp) {
        size_t sum = 0;
        for (size_t i = 0; i < kp; ++i) {
            sum += ceill(dmin / sqm<long double>(q, i));
        }
        if (sum <= n)
            k = kp;
        else
            break;
    }

    return k;
}

template <class T>
double UpperBound(size_t n, size_t dmin) {
    long double min = std::numeric_limits<long double>::max();

    for (size_t delta = 0; delta < dmin; ++delta) {
        const long double hamming = HammingUpperBound<T>(n - delta, dmin - delta);
        const long double johnson = JohnsonUpperBound<T>(n - delta, dmin - delta);
        const long double plotkin = PlotkinUpperBound<T>(n - delta, dmin - delta);
        const long double elias = EliasUpperBound<T>(n - delta, dmin - delta);
        const long double singleton = SingletonUpperBound(n - delta, dmin - delta);
        const long double griesmer = GriesmerUpperBound<T>(n - delta, dmin - delta);
        for (long double bound : {hamming, johnson, plotkin, elias, singleton, griesmer}) {
            min = std::min(bound, min);
        }
    }
    return min;
}

template <class T>
size_t GilbertVarshamovLowerBound(size_t n, size_t dmin) {
    if (dmin == 1) return n;
    if (dmin == n) return 1;
    constexpr size_t q = T::get_size();
    InfInt sum = 0;
    for (size_t i = 0; i < dmin; ++i) {
        sum += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);
    }
    return ceill(n - log2l(sum.toUnsignedLongLong()) / log2l(q));
}

template <class T>
static Polynomial<InfInt> MacWilliamsIdentity(const Polynomial<InfInt>& A, size_t n, size_t k) {
    auto a = Vector<InfInt>(n + 1);
    for (size_t i = 0; i < n + 1; ++i) {
        a.set_component(i, A[i]);
    }

    Matrix<InfInt> K(n + 1, n + 1);
    for (size_t i = 0; i < n + 1; ++i) {
        if (a[i] == 0) continue;
        for (size_t j = 0; j < n + 1; ++j) {
            InfInt sum = 0;
            // for (size_t h = 0; h <= std::min<size_t>(i, j); ++h) {
            for (size_t h = 0; h <= j; ++h) {
                /*
                 sum += sqm<InfInt>(-1, h) * bin<InfInt>(i, h) * bin<InfInt>(n - i, j - h) *
                        sqm<InfInt>(T::get_size() - 1, j - h);
                // the following code calculates exactly this expression - with a couple of
                // optimizations in order to avoid unnecessary computations
                */
                InfInt a = bin<InfInt>(n - i, j - h);
                if (a != 0) {
                    InfInt b = bin<InfInt>(i, h);
                    if (b != 0) {
                        auto c = sqm<InfInt>(T::get_size() - 1, j - h);
                        if (h % 2) {
                            sum -= a * b * c;
                        } else {
                            sum += a * b * c;
                        }
                    }
                }
            }
            K(i, j) = sum;
        }
    }

    return Polynomial<InfInt>(a * K / sqm<InfInt>(T::get_size(), k));
}

namespace details {

static const int index = std::ios_base::xalloc();

}  // namespace details

std::ostream& showbasic(std::ostream& os) {
    os.iword(index) = 0;
    return os;
}

std::ostream& showmost(std::ostream& os) {
    os.iword(index) = 1;
    return os;
}

std::ostream& showall(std::ostream& os) {
    os.iword(index) = 2;
    return os;
}

template <class T, class SUPER = T>
class Code;

template <class T, class SUPER = T>
Code<T, SUPER> UniverseCode(size_t n);

template <class T>
Code<T> SimplexCode(size_t s);

namespace details {

class decoding_failure : public std::exception {
   public:
    decoding_failure(const char* message) : message(message) {}
    const char* what() const noexcept override { return message.c_str(); }

   private:
    const std::string message;
};

enum BD_t : uint8_t { BD, boosted_BD };

template <class T>
class LinearCodeBackend;

template <class T>
class CodewordIterator {
    friend bool operator==(const CodewordIterator& a, const CodewordIterator& b) { return a.counter == b.counter; }
    friend bool operator!=(const CodewordIterator& a, const CodewordIterator& b) { return !(a == b); }

   public:
    CodewordIterator(LinearCodeBackend<T>* backend, InfInt s) : backend(backend), counter(s) {
        if (s < backend->get_size()) {  // for cend()

            if (backend->is_codewords_calculated()) {
                c = backend->get_codewords()[counter.toUnsignedInt()];
            } else {
                const size_t k = backend->get_k();
                Vector<T> u(k);
                for (size_t i = 0; i < k; ++i) {
                    u.set_component(i, T((s % T::get_size()).toInt()));
                    s /= T::get_size();
                }
                c = u * backend->get_G();
            }
        }
    }

    Vector<T> operator*() const { return c; }

    CodewordIterator& operator++() {
        const size_t k = backend->get_k();

        ++counter;

        if (counter < backend->get_size()) {
            if (backend->is_codewords_calculated()) {
                c = backend->get_codewords()[counter.toUnsignedInt()];
            } else {
                Vector<T> u(k);
                auto j = counter;
                for (size_t i = 0; i < k; ++i) {
                    auto quot = j / T::get_size();
                    auto rem = (j - quot).toInt();
                    u.set_component(i, T(rem));
                    j = quot;
                }
                c = u * backend->get_G();
            }
        }
        return *this;
    }

   private:
    LinearCodeBackend<T>* backend;
    InfInt counter;
    Vector<T> c;
};

class Backend {
   public:
    virtual ~Backend() = default;
    Backend& operator=(const Backend& rhs) = delete;
    virtual void get_info(std::ostream& os) = 0;
    virtual Backend* clone() const = 0;
};

class EmptyCodeBackend : public Backend {
   public:
    EmptyCodeBackend() = default;
    //~EmptyCodeBackend() override = default;

    void get_info(std::ostream& os) override { os << std::endl << "Empty code"; }
    Backend* clone() const override { return new EmptyCodeBackend(); }
};

template <class T>
class PolynomialCodeBackend;

template <class T>
class LinearCodeBackend : public Backend {
    friend class PolynomialCodeBackend<T>;

   public:
    LinearCodeBackend(size_t n, size_t k, const Matrix<T>& X, size_t dmin = 0,
                      const Polynomial<InfInt>& weight_enumerator = {})
        : n(n), k(k), dmin(dmin), weight_enumerator(weight_enumerator), Mi(k, k), size(sqm<InfInt>(T::get_size(), k)) {
        if (X.get_n() != n) throw std::invalid_argument("G must have " + std::to_string(n) + " columns");
        if (X.get_m() == k) {
            // X supposed to be generator matrix G
            if (X.rank() != k) throw std::invalid_argument("G must have full rank " + std::to_string(k));
            G = X;
            // G.rref();
            H = G.basis_of_nullspace();
            H.rref();
            infoset.resize(n);
            std::iota(infoset.begin(), infoset.end(), 0);
            size_t i = 0;
            for (size_t j = 0; j < n - k; ++j) {
                auto u = unit_vector<T>(n - k, j);
                while (H.get_col(i) != u) ++i;
                auto it = std::ranges::find(infoset, i);
                infoset.erase(it);
            }
        } else if (X.get_m() == n - k) {
            // X suppoed to be parity check matrix H
            if (X.rank() != n - k) throw std::invalid_argument("G must have full rank " + std::to_string(n - k));
            H = X;
            // H.rref();
            G = H.basis_of_nullspace();
            G.rref();
            size_t i = 0;
            for (size_t j = 0; j < k; ++j) {
                auto u = unit_vector<T>(k, j);
                while (G.get_col(i) != u) ++i;
                infoset.push_back(i);
            }
        }
        HT = transpose(H);
        size_t j = 0;
        for (auto it = infoset.cbegin(); it != infoset.cend(); ++it) {
            // Mi.set_col(G.get_col(i), *it);
            Mi.set_submatrix(G.get_submatrix(0, *it, k, 1), 0, j);
            ++j;
        }
        Mi.invert();
    }

    /*
    LinearCodeBackend(const LinearCodeBackend& other)
        : n(other.n),
          k(other.k),
          weight_enumerator(other.weight_enumerator),
          dmin(other.dmin),
          Mi(other.Mi),
          infoset(other.infoset),
          G(other.G),
          H(other.H),
          HT(other.HT),
          standard_array(other.standard_array),
          codewords(other.codewords),
          size(other.size) {}
    */

    //~LinearCodeBackend() override = default;
    CodewordIterator<T> cbegin() const { return CodewordIterator<T>(const_cast<LinearCodeBackend*>(this), 0); }

    CodewordIterator<T> cend() const { return CodewordIterator<T>(const_cast<LinearCodeBackend*>(this), size); }

    size_t get_n() const { return n; }
    size_t get_k() const { return k; }

    const Polynomial<InfInt>& get_weight_enumerator() {
        if (weight_enumerator.is_empty()) {
            if (get_k() <= get_n() - get_k()) {  // calculate directly
                std::clog << "--> Calculating weight enumerator, this requires iterating through "
                          << sqm<size_t>(T::get_size(), get_k()) << " codewords" << std::endl;
                Vector<InfInt> temp(get_n() + 1);
                for (auto it = cbegin(); it != cend(); ++it) {
                    const size_t w = wH(*it);
                    weight_enumerator.add_to_coeff(1, w);
                }
                // weight_enumerator = Polynomial<InfInt>(temp);
            } else {  // calculate based on dual code and MacWilliams identity
                std::clog << "--> Using MacWilliam's identity for: " << std::endl;
                auto C = LinearCode<T>(get_n(), get_n() - get_k(), H);
                weight_enumerator = MacWilliamsIdentity<T>(C.get_weight_enumerator(), get_n(), get_n() - get_k());
            }
        }
        return weight_enumerator;
    }

    bool is_weight_enumerator_known() const { return !weight_enumerator.is_empty(); }

    bool is_dmin_known() const { return dmin != 0; }

    size_t get_dmin() {
        if (dmin != 0) return dmin;

        // if weight enumerator is calculated, use it...
        if (!weight_enumerator.is_empty()) {
            for (size_t i = 1; i <= weight_enumerator.get_degree(); ++i) {
                if (weight_enumerator[i] != 0) {
                    dmin = i;
                    return dmin;
                }
            }
            // ... otherwise:
        } else {
            const size_t n = get_n();
            const size_t k = get_k();

            if (k == 1) {
                dmin = get_G().get_row(0).wH();
                return dmin;
            }

            std::clog << "--> Calculating dmin, this requires finding minimal number of linearly dependent columns in H"
                      << std::endl;
            // find min. number of linearly dependent columns of H
            for (size_t d = 1; d <= n - k + 1; ++d) {
                /*
                if (k > UpperBound<T>(n, d)) {
                    dmin = d - 1;
                    return dmin;
                }
                */
                Matrix<T> M(n - k, d);
                std::vector<bool> selection(n);  // zero-initialized
                std::fill(selection.begin() + n - d, selection.end(), true);

                do {
                    size_t i = 0;
                    for (size_t j = 0; j < n; ++j) {
                        if (selection[j] == true) {
                            M.set_submatrix(H.get_submatrix(0, j, n - k, 1), 0, i);
                            ++i;
                        }
                    }
                    if (M.rank() < d) {
                        dmin = d;
                        return dmin;
                    }
                } while (next_permutation(selection.begin(), selection.end()));
            }
        }
        assert(false && "should never be here");
        return 0;
    }

    size_t get_tmax() const { return floor((dmin - 1) / 2); }
    const Matrix<T>& get_G() const { return G; }
    const Matrix<T>& get_Mi() const { return Mi; }
    const std::vector<size_t>& get_infoset() const { return infoset; }
    const Matrix<T>& get_H() const { return H; }
    const Matrix<T>& get_HT() const { return HT; }
    const InfInt& get_size() const { return size; }

    bool is_perfect() {
        return fabsl(HammingUpperBound<T>(get_n(), get_dmin()) - get_k()) <
               10 * std::numeric_limits<long double>::epsilon();
    }

    bool is_MDS() { return SingletonUpperBound(get_n(), get_dmin()) == get_k(); }

    bool is_equidistant() {
        return fabsl(PlotkinUpperBound<T>(get_n(), get_dmin()) - get_k()) <
               10 * std::numeric_limits<long double>::epsilon();
    }

    bool is_weakly_self_dual() const { return G * transpose(G) == Matrix<T>(get_k(), get_k()); }
    bool is_dual_containing() const { return H * HT == Matrix<T>(get_n() - get_k(), get_n() - get_k()); }
    bool is_self_dual() const { return 2 * get_k() == get_n() && is_weakly_self_dual() && is_dual_containing(); }

    void calculate_standard_array() {
        std::clog << "Calculating standard array!" << std::endl;

        const size_t n = get_n();
        const size_t k = get_k();
        size_t count = 1;
        standard_array.resize(sqm<size_t>(T::get_size(), n - k));
        tainted.resize(sqm<size_t>(T::get_size(), n - k));
        tainted_burst.resize(sqm<size_t>(T::get_size(), n - k));
        standard_array[0] = Vector<T>(n);
        tainted[0] = false;
        tainted_burst[0] = false;

        for (size_t t = 1; t <= n; ++t) {
            bool breakflag = true;

            std::vector<bool> errpos(n);
            std::fill(errpos.begin(), errpos.begin() + t, true);

            if constexpr (T::get_size() == 2) {
                do {
                    Vector<T> e(n);
                    for (size_t j = 0; j < n; ++j)
                        if (errpos[j]) e.set_component(j, T(1));
                    auto s = e * HT;                           // calculate syndrome...
                    const size_t coset_index = s.asInteger();  // ... and interpret it as binary number
                    if (standard_array[coset_index].is_empty()) {
                        standard_array[coset_index] = std::move(e);
                        ++count;
                        // std::cout << t << ": " << count << std::endl;
                        if (count < standard_array.size()) {
                            breakflag = false;
                        } else {
                            breakflag = true;
                            break;
                        }
                    } else {
                        if (wH(e) == wH(standard_array[coset_index])) {
                            tainted[coset_index] = true;
                            const size_t cb_e = cyclic_burst_length(e);
                            const size_t cb_current = cyclic_burst_length(standard_array[coset_index]);
                            if (cb_e < cb_current)
                                standard_array[coset_index] = e;
                            else if (cb_e == cb_current)
                                tainted_burst[coset_index] = true;
                        }
                    }
                } while (std::prev_permutation(errpos.begin(), errpos.end()));
            } else {
                auto total = sqm<InfInt>(T::get_size() - 1, t);
                do {
                    for (InfInt i = 0; i < total; ++i) {
                        std::vector<T> c(t);
                        InfInt temp = i;
                        for (size_t j = 0; j < t; ++j) {
                            c[j] = T((InfInt(1) + temp % (T::get_size() - 1)).toInt());
                            temp /= (T::get_size() - 1);
                        }

                        Vector<T> e(n);
                        size_t ell = 0;
                        for (size_t j = 0; j < n; ++j) {
                            if (errpos[j]) {
                                e.set_component(j, c[ell]);
                                ++ell;
                            }
                        }

                        auto s = e * HT;                           // calculate syndrome...
                        const size_t coset_index = s.asInteger();  // ... and interpret it as binary number
                        if (standard_array[coset_index].is_empty()) {
                            standard_array[coset_index] = std::move(e);
                            ++count;
                            if (count < standard_array.size()) {
                                breakflag = false;
                            } else {
                                breakflag = true;
                                break;
                            }
                        } else {
                            if (wH(e) == wH(standard_array[coset_index])) {
                                tainted[coset_index] = true;
                                const size_t cb_e = cyclic_burst_length(e);
                                const size_t cb_current = cyclic_burst_length(standard_array[coset_index]);
                                if (cb_e < cb_current)
                                    standard_array[coset_index] = e;
                                else if (cb_e == cb_current)
                                    tainted_burst[coset_index] = true;
                            }
                        }
                    }

                } while (std::prev_permutation(errpos.begin(), errpos.end()));
            }
            if (breakflag) break;
        }

        /*
        for (size_t i = 0; i < standard_array.size(); ++i) {
            std::cout << standard_array[i] << ": " << wH(standard_array[i]) << " --- " << tainted[i] << std::endl;
        }
        */
    }

    Vector<T> dec_BD(const Vector<T> r, const Code<T>& C, BD_t type = boosted_BD) {
        // calculate standard array if necessary
        if (standard_array.empty()) calculate_standard_array();

        auto s = r * HT;                 // calculate syndrome...
        const size_t i = s.asInteger();  // ... and interpret it as binary number
        if (tainted[i] || (type == BD && wH(standard_array[i]) > get_tmax())) {  // decoding failure
            throw decoding_failure(
                "Linear code decoder failed, coset of syndrome empty or tainted (ambiguous leader)!");
        } else {  // correct decoding or decoding error
            return r - standard_array[i];
        }
    }

    Vector<T> dec_ML(const Vector<T> r, const Code<T>& C) {
        // calculate standard array if necessary
        if (standard_array.empty()) calculate_standard_array();

        auto s = r * HT;                 // calculate syndrome...
        const size_t i = s.asInteger();  // ... and interpret it as binary number
        return r - standard_array[i];
    }

    Vector<T> dec_burst(const Vector<T> r, const Code<T>& C) {
        // calculate standard array if necessary
        if (standard_array.empty()) calculate_standard_array();

        auto s = r * HT;                 // calculate syndrome...
        const size_t i = s.asInteger();  // ... and interpret it as binary number
        if (tainted_burst[i]) {          // decoding failure
            throw decoding_failure(
                "Linear code decoder (burst) failed, coset of syndrome empty or tainted (ambiguous burst leader)!");
        } else {  // correct decoding or decoding error
            return r - standard_array[i];
        }
    }

    void calculate_codewords() {
        if (sqm<InfInt>(T::get_size(), get_k()) > std::numeric_limits<unsigned int>::max()) {
            throw std::out_of_range("code too big (more than " +
                                    std::to_string(std::numeric_limits<unsigned int>::max()) +
                                    " codewords) to compute all of them");
        }
        codewords = get_G().get_span();
    }

    bool is_codewords_calculated() const { return !codewords.empty(); }

    std::vector<Vector<T>>& get_codewords() {
        if (!is_codewords_calculated()) calculate_codewords();

        return codewords;
    }

    void get_info(std::ostream& os) override {
        if (os.iword(index) > 0) {
            os << std::endl;
            os << "G = " << std::endl;
            os << G << std::endl;
            os << "H = " << std::endl;
            os << H;
        }
        if (os.iword(index) > 1) {
            os << std::endl;
            auto A = get_weight_enumerator();
            os << "A(x) = " << A << std::setfill(' ') << std::endl;
            size_t dmin = get_dmin();
            os << "dmin = " << dmin;
        }
        if (os.iword(index) > 0) {
            os << std::endl;
            os << "Linear code properties: {" << std::flush;
        }
        if (os.iword(index) > 1) {
            if (is_MDS()) {
                os << " MDS" << std::flush;
            }
            if (is_perfect()) {
                os << " perfect" << std::flush;
            }
            if (is_equidistant()) {
                os << " equidistant" << std::flush;
            }
        }
        if (os.iword(index) > 0) {
            if (!is_self_dual() && is_weakly_self_dual()) {
                os << " weakly_self-dual" << std::flush;
            }
            if (!is_self_dual() && is_dual_containing()) {
                os << " dual-containing" << std::flush;
            }
            if (is_self_dual()) {
                os << " self-dual" << std::flush;
            }
            os << " }";
        }
    }

    Backend* clone() const override { return new LinearCodeBackend<T>(*this); }

   private:
    size_t n;
    size_t k;
    Polynomial<InfInt> weight_enumerator;
    size_t dmin = 0;
    Matrix<T> Mi;
    std::vector<size_t> infoset;
    Matrix<T> G;
    Matrix<T> H;
    Matrix<T> HT;
    std::vector<Vector<T>> standard_array;
    std::vector<bool> tainted;
    std::vector<bool> tainted_burst;
    std::vector<Vector<T>> codewords;
    const InfInt size;
};

template <class T>
class RepetitionCodeBackend : public Backend {
   public:
    RepetitionCodeBackend(bool odd) : odd(odd) {}
    //~RepetitionCodeBackend() override = default;

    Vector<T> dec_BD(const Vector<T>& r, const Code<T>& C) {
        std::vector<size_t> histogram(T::get_size());
        for (size_t i = 0; i < C.get_n(); ++i) {
            ++histogram[r[i].get_label()];
        }

        auto maxima = find_maxima(histogram);
        if (maxima.size() == 1) return Vector<T>(C.get_n(), T(maxima.front()));

        throw decoding_failure("Repetition code decoder failed, no unambiguous majority decision!");
    }

    void get_info(std::ostream& os) override {
        os << std::endl << "Repetition code properties: { " << (odd ? "odd" : "even") << " length }";
    }

    Backend* clone() const override { return new RepetitionCodeBackend<T>(odd); }

   private:
    const bool odd;
};

template <class T>
class SingleParityCheckCodeBackend : public Backend {
   public:
    SingleParityCheckCodeBackend() = default;
    //~SingleParityCheckCodeBackend() override = default;

    void get_info(std::ostream& os) override {
        os << std::endl << "Single parity check code properties: { even_parity }";
    }

    Backend* clone() const override { return new SingleParityCheckCodeBackend<T>(); }
};

template <class T>
class HammingCodeBackend : public Backend {
   public:
    HammingCodeBackend(size_t s) : s(s) {
        if (s < 2) throw std::invalid_argument("Hamming codes must have parameter s>=2!");
    }
    //~HammingCodeBackend() override = default;
    size_t get_s() const { return s; }

    Vector<T> dec_BD(const Vector<T>& r, const Code<T>& C) {
        auto HT = C.get_HT();
        auto c_hat = r;
        auto s = r * HT;
        if (wH(s) == 0) return r;
        for (size_t i = 0; i < C.get_n(); ++i) {
            for (size_t j = 1; j < T::get_size(); ++j) {
                T a(j);
                if (s == a * HT.get_row(i)) {
                    c_hat.set_component(i, c_hat[i] - a);
                    return c_hat;
                }
            }
        }

        // can never be here, Hamming codes are perfect
        throw decoding_failure("Hamming code decoder failed, syndrome is not a column of the parity check matrix!");
    }

    void get_info(std::ostream& os) override { os << std::endl << "Hamming code properties: { s = " << s << " }"; }
    Backend* clone() const override { return new HammingCodeBackend<T>(s); }

   private:
    const size_t s;
};

template <class T>
class SimplexCodeBackend : public Backend {
   public:
    SimplexCodeBackend(size_t s) : s(s) {};
    //~SimplexCodeBackend() override = default;
    size_t get_s() const { return s; }

    void get_info(std::ostream& os) override { os << std::endl << "Simplex code properties: { s = " << s << " }"; }
    Backend* clone() const override { return new SimplexCodeBackend<T>(s); }

   private:
    const size_t s;
};

class CordaroWagnerCodeBackend : public Backend {
   public:
    CordaroWagnerCodeBackend(size_t n) : n(n) {
        if (n < 2) throw std::invalid_argument("Cordaro-Wagner codes must have length n>=2!");
        if ((n - 1) % 3 == 0) {  // n=3r-1
            r = (n - 1) / 3;
        } else if (n % 3 == 0) {  // n=3r
            r = n / 3;
        } else {  // n=3r+1
            r = (n + 1) / 3;
        }
    }
    //~CordaroWagnerCodeBackend() override = default;
    size_t get_r() const { return r; }
    void get_info(std::ostream& os) override {
        os << std::endl << "Cordaro--Wagner code properties: { r = " << r << " }";
    }
    Backend* clone() const override { return new CordaroWagnerCodeBackend(n); }

   private:
    const size_t n;
    size_t r;
};

template <class T>
class PolynomialCodeBackend : public Backend {
   public:
    PolynomialCodeBackend(size_t k, const Polynomial<T>& gamma, int8_t cyclic = -1)
        : k(k), gamma(gamma), cyclic(cyclic), Mi(k, k) {
        const size_t n = k + gamma.get_degree();
        G = ToeplitzMatrix(
            pad_back(pad_front(Vector<T>(gamma), k + gamma.get_degree()), 2 * k + gamma.get_degree() - 1), k,
            k + gamma.get_degree());
        auto M = transpose(rref(G));
        size_t i = 0;
        for (size_t j = 0; j < k; ++j) {
            auto u = unit_vector<T>(k, j);
            while (M.get_row(i) != u) ++i;
            Mi.set_submatrix(G.get_submatrix(0, i, k, 1), 0, j);
            infoset.push_back(i);
        }
        Mi.invert();
    }
    //PolynomialCodeBackend(const PolynomialCodeBackend& other)
    //    : k(other.k), gamma(other.gamma), cyclic(other.cyclic), G(other.G), Mi(other.Mi), infoset(other.infoset) {}
    //~PolynomialCodeBackend() override = default;
    const Polynomial<T>& get_gamma() const { return gamma; }

    bool is_cyclic() {
        if (cyclic == -1) {
            Polynomial<T> p;
            p.set_coeff(0, -T(1));
            p.set_coeff(gamma.get_degree() + k, T(1));
            p %= gamma;
            cyclic = p.is_zero();
        }
        return cyclic;
    }

    size_t get_ell(LinearCodeBackend<T>* backend) {
        // calculate standard array if necessary
        if (backend->standard_array.empty()) backend->calculate_standard_array();

        const size_t untainted = std::count(backend->tainted_burst.cbegin(), backend->tainted_burst.cend(), false);

        size_t res = 0;
        do {
            ++res;
        } while (untainted >= backend->get_n() * (T::get_size() - 1) * sqm<size_t>(T::get_size(), res - 1) + 1);
        --res;
        return res;
    }

    const Matrix<T>& get_G() const { return G; }
    const Matrix<T>& get_Mi() const { return Mi; }

    void get_info(std::ostream& os) override {
        os << std::endl;
        os << "Polynomial code properties: { gamma = " << gamma;
        if (is_cyclic()) {
            os << " cyclic";
        }
        os << " }";
    }

    Backend* clone() const override { return new PolynomialCodeBackend<T>(*this); }

   private:
    const size_t k;
    const Polynomial<T> gamma;
    int8_t cyclic;
    Matrix<T> G;
    Matrix<T> Mi;
    std::vector<size_t> infoset;
};

template <class T>
class ExtendedCodeBackend : public Backend {
   public:
    ExtendedCodeBackend(const Code<T>& C, size_t i, const Vector<T>& v) : C(C), i(i), v(v) {}
    //~ExtendedCodeBackend() override = default;
    const Code<T>& get_C() const { return C; }
    size_t get_i() const { return i; }
    const Vector<T>& get_v() { return v; }

    Vector<T> dec_BD(const Vector<T>& rp, const Code<T>& C) {
        Vector<T> r = concatenate(rp.get_subvector(0, i), rp.get_subvector(i + 1, C.get_n() - i - 1));
        Vector<T> c_hat;
        try {
            c_hat = this->C.dec_BD(r);
        } catch (const std::exception& e) {
            std::string message("Extended code decoder failed due to failure of base code: ");
            message += e.what();
            throw decoding_failure(message.c_str());
        }
        auto u_hat = this->C.encinv(c_hat);

        auto cp_hat = C.enc(u_hat);
        if (dH(rp, cp_hat) <= C.get_tmax()) return cp_hat;

        std::string message("Extended code decoder failed, too many (more than ");
        message += std::to_string(C.get_tmax());
        message += ") errors";
        throw decoding_failure(message.c_str());
    }

    void get_info(std::ostream& os) override {
        os << std::endl;
        os << "Extended code properties: { C = " << std::endl
           << C << std::endl
           << "i = " << i << " v = " << v << " } // end extended code properties";
    }

    Backend* clone() const override { return new ExtendedCodeBackend<T>(C, i, v); }

   private:
    Code<T> C;
    const size_t i;
    const Vector<T> v;
};

template <class T>
class AugmentedCodeBackend : public Backend {
   public:
    AugmentedCodeBackend(const Code<T>& C, size_t j, const Vector<T>& w) : C(C), j(j), w(w) {}
    //~AugmentedCodeBackend() override = default;
    const Code<T>& get_C() const { return C; }
    size_t get_j() const { return j; }
    const Vector<T>& get_w() { return w; }

    void get_info(std::ostream& os) override {
        os << std::endl;
        os << "Augmented code properties: { C = " << std::endl
           << C << std::endl
           << "j = " << j << " w = " << w << " } // end augmented code properties";
    }

    Backend* clone() const override { return new AugmentedCodeBackend<T>(C, j, w); }

   private:
    Code<T> C;
    const size_t j;
    const Vector<T> w;
};

template <class T>
class LDCCodeBackend : public Backend {
   public:
    LDCCodeBackend(const Code<T>& U, const Code<T>& V) : U(U), V(V) {}
    //~LDCCodeBackend() override = default;

    const Code<T>& get_U() const { return U; }
    const Code<T>& get_V() const { return V; }

    Vector<T> dec_Dumer(const Vector<T>& r) const {
        auto rl = r.get_subvector(0, U.get_n());
        auto rr = r.get_subvector(U.get_n(), U.get_n());

        Vector<T> cr_hat;
        try {
            cr_hat = V.dec_Dumer(rl + rr);
        } catch (const std::exception& e) {
            std::string message("Recursive Dumer code decoder failed due to failure of code V: ");
            message += e.what();
            throw decoding_failure(message.c_str());
        }

        bool flag_1 = true;
        Vector<T> cl_hat_1;
        try {
            cl_hat_1 = U.dec_Dumer(rl);
        } catch (const std::exception& e) {
            flag_1 = false;
        }

        bool flag_2 = true;
        Vector<T> cl_hat_2;
        try {
            cl_hat_2 = U.dec_Dumer(rr - cr_hat);
        } catch (const std::exception& e) {
            flag_2 = false;
        }

        if (flag_1 && flag_2) {
            auto c_hat_1 = concatenate(cl_hat_1, cl_hat_1 + cr_hat);
            auto c_hat_2 = concatenate(cl_hat_2, cl_hat_2 + cr_hat);
            if (dH(r, c_hat_1) < dH(r, c_hat_2)) {
                return c_hat_1;
            } else {
                return c_hat_2;
            }
        } else if (flag_1 && !flag_2) {
            return concatenate(cl_hat_1, cl_hat_1 + cr_hat);
        } else if (!flag_1 && flag_2) {
            return concatenate(cl_hat_2, cl_hat_2 + cr_hat);
        } else {
            throw decoding_failure(
                "Recursive Dumer code decoder failed due to failure of both decoding attempts of U!");
        }
    }

    Vector<Fp<2>> dec_Dumer_soft(const Vector<double>& llrs) const {
        auto llrs_l = llrs.get_subvector(0, llrs.get_n() / 2);
        auto llrs_r = llrs.get_subvector(llrs.get_n() / 2, llrs.get_n() / 2);

        // calculcate LLRs for V decoding (boxplus)
        Vector<double> llrs_V(V.get_n());
        for (size_t i = 0; i < llrs_V.get_n(); ++i)
            // exact
            // llrs_V.push_back(2.0 * atanh(tanh(llrs_l[i] / 2.0) * tanh(llrs_r[i] / 2.0)));
            // approximation
            llrs_V.set_component(i, copysign(1.0, llrs_l[i] * llrs_r[i]) * std::min({abs(llrs_l[i]), abs(llrs_r[i])}));

        Vector<T> cr_hat;
        if (auto backend = V.template get_backend<LDCCodeBackend<Fp<2>>>())
            cr_hat = backend->dec_Dumer_soft(llrs_V);
        else
            cr_hat = V.dec_ML_soft(llrs_V);

        // calculcate LLRs for U decoding
        Vector<double> llrs_U(U.get_n());
        for (size_t i = 0; i < llrs_U.get_n(); ++i)
            llrs_U.set_component(i, llrs_l[i] + pow(-1.0, cr_hat[i] == Fp<2>(0) ? 0.0 : 1.0) * llrs_r[i]);

        Vector<Fp<2>> cl_hat;
        if (auto backend = U.template get_backend<LDCCodeBackend<Fp<2>>>())
            cl_hat = backend->dec_Dumer_soft(llrs_U);
        else
            cl_hat = U.dec_ML_soft(llrs_U);

        return concatenate(cl_hat, cl_hat + cr_hat);
    }

    void get_info(std::ostream& os) override {
        os << std::endl;
        std::ios old_state(nullptr);
        auto old = os.iword(index);
        os << showbasic << "LDC code properties: { U = " << U << " V = " << V << " }";
        os.iword(index) = old;
    }

    Backend* clone() const override { return new LDCCodeBackend(U, V); }

   private:
    Code<T> U;
    Code<T> V;
};

class RMCodeBackend : public Backend {
   public:
    RMCodeBackend(size_t r, size_t m) : r(r), m(m) {}
    //~RMCodeBackend() override = default;
    size_t get_r() const { return r; }
    size_t get_m() const { return m; }

    void get_info(std::ostream& os) override {
        os << std::endl;
        os << "RM code properties: { r = " << r << " m = " << m << " }";
    }

    Backend* clone() const override { return new RMCodeBackend(r, m); }

   private:
    const size_t r;
    const size_t m;
};

template <class T>
class GRSCodeBackend : public Backend {
   public:
    GRSCodeBackend(const Vector<T>& a, const Vector<T>& d, size_t k)
        : a(a),
          d(d),
          k(k),
          G(VandermondeMatrix<T>(a, k) * DiagonalMatrix<T>(d)),
          Mi(G.get_submatrix(0, 0, k, k).invert()) {
        if (a.get_n() != d.get_n())
            throw std::invalid_argument("GRS codes must of code locators a and column multipliers d of same length!");
        for (size_t i = 0; i < a.get_n(); ++i) {
            for (size_t j = i + 1; j < a.get_n(); ++j) {
                if (a[i] == a[j]) throw std::invalid_argument("GRS codes must have pairwise distinct code locators!");
            }
        }
        for (size_t i = 0; i < a.get_n(); ++i) {
            if (d[i] == T(0)) throw std::invalid_argument("GRS codes must have nonzero column multipliers!");
        }
    }
    //GRSCodeBackend(const GRSCodeBackend<T>& other) : a(other.a), d(other.d), k(other.k), G(other.G), Mi(other.Mi) {}
    //~GRSCodeBackend() override = default;
    const Vector<T>& get_locators() const { return a; }
    const Vector<T>& get_multipliers() const { return d; }

    const Matrix<T>& get_G() const { return G; }
    const Matrix<T>& get_Mi() const { return Mi; }

    bool is_singly_extended() const {
        for (size_t i = 0; i < a.get_n(); ++i) {
            if (a[i] == T(0)) return true;
        }
        return false;
    }

    bool is_primitive() const {
        if (a.get_n() != T::get_size() - 1) return false;
        for (size_t i = 0; i < a.get_n(); ++i) {
            if (a[i] == T(0)) return false;
        }
        return true;
    }

    bool is_narrow_sense() const { return a == d; }

    bool is_normalized() const {
        for (size_t i = 0; i < a.get_n(); ++i) {
            if (d[i] != T(1)) return false;
        }
        return true;
    }

    void get_info(std::ostream& os) override {
        os << std::endl;
        os << "GRS code properties: { a = " << a << " d = " << d;
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

    Backend* clone() const override { return new GRSCodeBackend<T>(*this); }

   private:
    const Vector<T> a;
    const Vector<T> d;
    const size_t k;
    const Matrix<T> G;
    const Matrix<T> Mi;
};

template <class T>
class RSCodeBackend : public Backend {
   public:
    RSCodeBackend(T alpha, size_t b) : alpha(alpha), b(b) {}
    //~RSCodeBackend() override = default;
    T get_alpha() const { return alpha; }
    size_t get_b() const { return b; }

    void get_info(std::ostream& os) override {
        os << std::endl;
        os << "RS code properties: { alpha = " << alpha << " b = " << b << " }";
    }

    Backend* clone() const override { return new RSCodeBackend<T>(alpha, b); }

   private:
    const T alpha;
    const size_t b;
};

template <class T, class SUPER = T>
class SSCodeBackend : public Backend {
   public:
    SSCodeBackend(Code<T, SUPER> SuperC, size_t d) : SuperC(SuperC), d(d) {}
    //~SSCodeBackend() override = default;
    const Code<T, SUPER>& get_supercode() const { return SuperC; }
    size_t get_design_distance() const { return d; }

    void get_info(std::ostream& os) override {
        os << std::endl;
        std::ios old_state(nullptr);
        auto old = os.iword(index);
        os << "Subfield-subcode properties: { supercode = " << showbasic << SuperC;
        os.iword(index) = old;
        os << " design distance = " << d << " }";
    }

    Backend* clone() const override { return new SSCodeBackend<T, SUPER>(SuperC, d); }

   private:
    Code<T, SUPER> SuperC;
    const size_t d;
};

}  // namespace details

template <class T, class SUPER = T>
Code<T, SUPER> EmptyCode();

Code<Fp<2>> RMCode(size_t r, size_t m);

template <class T, bool t = is_finite_field<T>(), typename = std::enable_if_t<t>>
Code<T> GRSCode(const Vector<T>& a, const Vector<T>& d, size_t k);

template <class T, uint8_t m, bool t = is_finite_field<T>(), typename = std::enable_if_t<t>>
Code<SF<T, m>, T> SSCode(const Code<T>& SuperC, const Matrix<int>& M, Matrix<T>* Bp = nullptr);

template <class T, class SUPER>
std::ostream& operator<<(std::ostream& os, Code<T, SUPER>& rhs);

template <class T, class SUPER>
class Code {
    friend std::ostream& operator<< <>(std::ostream& os, Code<T, SUPER>& rhs);

   public:
    Code() = default;

    Code(const Code<T, SUPER>& other) {
        for (auto it = other.backend_collection.cbegin(); it != other.backend_collection.cend(); ++it) {
            backend_collection.push_back((*it)->clone());
        }
    }

    Code(Code<T, SUPER>&& other) noexcept { std::swap(backend_collection, other.backend_collection); }

    ~Code() { clear(); }

    Code& operator=(const Code<T>& rhs) {
        if (this != rhs) {
            clear();
            for (auto it = rhs.backend_collection.cbegin(); it != rhs.backend_collection.cend(); ++it) {
                backend_collection.push_back((*it)->clone());
            }
        }
        return *this;
    }

    Code& operator=(Code<T>&& rhs) noexcept {
        if (this != rhs) *this = std::move(rhs);
        return *this;
    }

    CodewordIterator<T> cbegin() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to get cbegin iterator: code does not have LinearCodeBackend");
        return CodewordIterator<T>((LinearCodeBackend<T>*)(backend), 0);
    }

    CodewordIterator<T> cend() const {
        static auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to get cend iterator: code does not have LinearCodeBackend");
        auto end = CodewordIterator<T>((LinearCodeBackend<T>*)(get_backend<LinearCodeBackend<T>>()), get_size());
        return end;
    }

    template <class B>
    void add_backend(B* b)
        requires(std::is_base_of_v<Backend, B>)
    {
        auto backend = get_backend<B>();
        if (!backend) backend_collection.push_back(b);
    }

    template <class B>
    B* get_backend() const
        requires(std::is_base_of_v<Backend, B>)
    {
        for (auto it = backend_collection.cbegin(); it != backend_collection.cend(); ++it) {
            auto backend = dynamic_cast<B*>(*it);
            if (backend) return backend;
        }
        return nullptr;
    }

    size_t get_n() const {
        if (get_backend<EmptyCodeBackend>()) return 0;
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to calculate n: code does not have LinearCodeBackend");
        return backend->get_n();
    }

    size_t get_k() const {
        if (get_backend<EmptyCodeBackend>()) return 0;
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to calculate k: code does not have LinearCodeBackend");
        return backend->get_k();
    }

    size_t get_R() const {
        if (get_backend<EmptyCodeBackend>()) return 0;
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to calculate R: code does not have LinearCodeBackend");
        return backend->get_k() / backend->get_n();
    }

    InfInt get_size() const {
        if (get_backend<EmptyCodeBackend>()) return 0;
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to calculate size: code does not have LinearCodeBackend");
        return backend->get_size();
    }

    Polynomial<InfInt> get_weight_enumerator() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to access weight enumerator: code does not have LinearCodeBackend");
        return backend->get_weight_enumerator();
    }

    size_t get_dmin() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to access dmin: code does not have LinearCodeBackend");
        return backend->get_dmin();
    }

    size_t get_tmax() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to access tmax: code does not have LinearCodeBackend");
        return backend->get_tmax();
    }

    const Matrix<T>& get_G() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to access generator matrix G: code does not have LinearCodeBackend");
        return backend->get_G();
    }

    long double P_word(double pe) {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to calculate Pword: code does not have LinearCodeBackend");

        const size_t n = get_n();
        const size_t tmax = get_tmax();
        long double res = 0.0;
        for (size_t i = tmax + 1; i <= n; ++i) {
            res += bin<InfInt>(n, i).toUnsignedLongLong() * powl(pe, i) * powl(1 - pe, n - i);
        }
        return res;
    }

    long double P_error(double pe) {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to calculate Perror: code does not have LinearCodeBackend");

        const size_t n = get_n();
        const size_t tmax = get_tmax();
        long double res = 0.0;
        for (size_t h = 1; h <= n; ++h) {
            InfInt sum = 0;
            for (size_t s = 0; s <= tmax; ++s) {
                for (size_t ell = 1; ell <= n; ++ell) {
                    sum += (get_weight_enumerator())[ell] * N(ell, h, s);
                }
            }
            res += powl(pe / (T::get_size() - 1), h) * powl(1 - pe, n - h) * sum.toUnsignedLongLong();
        }
        return res;
    }

    long double P_failure(double pe) noexcept {
        if (fabsl(P_word(pe) - P_error(pe)) < 10 * std::numeric_limits<long double>::epsilon())
            return 0;
        else
            return P_word(pe) - P_error(pe);
    }

    long double Bhattacharyya(double pe) {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to calculate Bhattacharyya bound: code does not have LinearCodeBackend");

        auto A = get_weight_enumerator();

        long double gamma = 2 * sqrt(pe * (1 - pe));

        long double res = 0;
        for (size_t i = get_dmin(); i <= A.get_degree(); ++i) {
            long double sum = 0;
            for (size_t j = ceill(i / 2.0); j <= i; ++j) {
                sum += bin<unsigned long long>(i, j) * powl(pe, j) * pow(1 - pe, i - j);
            }
            res += A[i].toUnsignedLongLong() * sum;
        }

        return res;
    }

    long double Bhattacharyya_soft(double b, double sigma) {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to calculate Bhattacharyya bound (soft): code does not have LinearCodeBackend");

        auto A = get_weight_enumerator();

        long double gamma = expl(-powl(b, 2.0) / (8.0 * powl(sigma, 2.0)));

        long double res = 0;
        for (size_t i = get_dmin(); i <= A.get_degree(); ++i) {
            res += A[i].toUnsignedLongLong() * powl(gamma, i);
        }

        return res;
    }

    void calculate_codewords() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to calculate codewords: code does not have LinearCodeBackend");
        backend->calculate_codewords();
    }

    const std::vector<Vector<T>>& get_codewords() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to get codewords: code does not have LinearCodeBackend");
        return backend->get_codewords();
    }

    void calculate_standard_array() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to calculate standard array: code does not have LinearCodeBackend");
        backend->calculate_standard_array();
    }

    Vector<T> dec_BD(const Vector<T>& r, BD_t type = boosted_BD) const {
        if (r.get_n() != get_n())
            throw std::invalid_argument("Can only decode received vectors of length " + std::to_string(get_n()));

        // universe code
        if (get_n() == get_k()) return r;

        if (auto backend = get_backend<RepetitionCodeBackend<T>>()) return backend->dec_BD(r, *this);

        if (auto backend = get_backend<HammingCodeBackend<T>>()) return backend->dec_BD(r, *this);

        if (auto backend = get_backend<ExtendedCodeBackend<T>>()) return backend->dec_BD(r, *this);

        if (auto backend = get_backend<LinearCodeBackend<T>>()) return backend->dec_BD(r, *this, type);

        throw decoding_failure("No BD decoder backend applicable!");
    }

    Vector<T> dec_ML(const Vector<T>& r) const {
        if (r.get_n() != get_n())
            throw std::invalid_argument("Can only decode received vectors of length " + std::to_string(get_n()));

        // universe code
        if (get_n() == get_k()) return r;

        if (auto backend = get_backend<LinearCodeBackend<T>>()) return backend->dec_ML(r, *this);

        throw decoding_failure("No hard input ML decoder backend applicable!");
    }

    Vector<Fp<2>> dec_ML_soft(const Vector<double>& llrs) const {
        const size_t n = get_n();

        if (llrs.get_n() != n)
            throw std::invalid_argument("Can only decode vectors of length " + std::to_string(get_n()));

        // universe code
        if (n == get_k()) {
            Vector<Fp<2>> c_hat(n);
            for (size_t i = 0; i < n; ++i) c_hat.set_component(i, llrs[i] > 0 ? Fp<2>(0) : Fp<2>(1));
            return c_hat;
        }

        double max_sum = std::numeric_limits<double>::lowest();
        Vector<T> c_hat;
        for (auto it = this->cbegin(); it != this->cend(); ++it) {
            double sum = 0;

            for (size_t i = 0; i < llrs.get_n(); ++i) sum -= ((*it)[i] == Fp<2>(0) ? 0.0 : 1.0) * llrs[i];
            if (sum > max_sum) {
                max_sum = sum;
                c_hat = *it;
            }
        }
        assert(!c_hat.is_empty());
        return c_hat;
    }

    Vector<T> dec_burst(const Vector<T>& r) const {
        if (r.get_n() != get_n())
            throw std::invalid_argument("Can only decode received vectors of length " + std::to_string(get_n()));

        // universe code
        if (get_n() == get_k()) return r;

        if (auto backend = get_backend<LinearCodeBackend<T>>()) return backend->dec_burst(r, *this);

        throw decoding_failure("No burst decoder backend applicable!");
    }

    Vector<Fp<2>> dec_Dumer(const Vector<Fp<2>>& r) const {
        if (r.get_n() != get_n())
            throw std::invalid_argument("Can only decode received vectors of length " + std::to_string(get_n()));

        if (auto backend = get_backend<LDCCodeBackend<Fp<2>>>()) return backend->dec_Dumer(r);
        throw std::invalid_argument("Trying to apply Dumer algorithm: code does not have LDCCodeBackend");
    }

    Vector<Fp<2>> dec_Dumer_soft(const Vector<double>& llrs) const {
        if (llrs.get_n() != get_n())
            throw std::invalid_argument("Can only decode vectors of length " + std::to_string(get_n()));

        if (auto backend = get_backend<LDCCodeBackend<T>>()) return backend->dec_Dumer_soft(llrs);
        throw std::invalid_argument("Trying to apply soft input Dumer algorithm: code does not have LDCCodeBackend");
    }

    Vector<T> enc(const Vector<T>& u) const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to encode: code does not have LinearCodeBackend");
        return u * backend->get_G();
    }

    Vector<T> encinv(const Vector<T>& c) const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to invert encoding: code does not have LinearCodeBackend");
        Vector<T> c_sub(get_k());
        size_t i = 0;
        for (auto it = backend->get_infoset().cbegin(); it != backend->get_infoset().cend(); ++it) {
            c_sub.set_component(i, c[*it]);
            ++i;
        }
        return c_sub * backend->get_Mi();
    }

    const Matrix<T>& get_polynomial_G() const {
        auto backend = get_backend<PolynomialCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to access polynomial generator matrix G: code does not have PolynomialCodeBackend");
        return backend->get_G();
    }

    Vector<T> polynomial_enc(const Vector<T>& u) const {
        auto backend = get_backend<PolynomialCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to encode with polynomial G: code does not have PolynomialCodeBackend");
        return u * backend->get_G();
    }

    Vector<T> polynomial_encinv(const Vector<T>& c) const {
        auto backend = get_backend<PolynomialCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to invert encoding: code does not have PolynomialCodeBackend");
        Vector<T> c_sub(get_k());
        size_t i = 0;
        for (auto it = backend->get_infoset().cbegin(); it != backend->get_infoset().cend(); ++it) {
            c_sub.set_component(i, c[*it]);
            ++i;
        }
        return c_sub * backend->get_Mi();
    }

    Polynomial<T> polynomial_enc(const Polynomial<T>& ux) const {
        auto backend = get_backend<PolynomialCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to encode with polynomial G: code does not have PolynomialCodeBackend");
        return ux * backend->get_gamma();
    }

    Polynomial<T> polynomial_encinv(const Polynomial<T>& cx) const {
        auto backend = get_backend<PolynomialCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to encode with polynomial G: code does not have PolynomialCodeBackend");
        return cx / backend->get_gamma();
    }

    const Matrix<T>& get_canonical_G() const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to access canonical generator matrix G: code does not have GRSCodeBackend");
        return backend->get_G();
    }

    Vector<T> canonical_enc(const Vector<T>& u) const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to encode with canonical G: code does not have GRSCodeBackend");
        return u * backend->get_G();
    }

    Vector<T> canonical_enc(const Polynomial<T>& ux) const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to encode with canonical G: code does not have GRSCodeBackend");
        Vector<T> c(get_n());
        for (size_t i = 0; i < get_n(); ++i) {
            c.set_component(i, backend->get_multipliers()[i] * ux(backend->get_locators()[i]));
        }
        return c;
    }

    Vector<T> canonical_encinv(const Vector<T>& c) const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to invert encoding: code does not have GRSCodeBackend");
        auto c_sub = c.get_subvector(0, get_k());
        return c_sub * backend->get_Mi();
    }

    Code<SF<T, 1>, T> get_SSCode_with_dimension(size_t kp) {
        if (kp > get_k())
            throw std::invalid_argument(
                "Trying to construct a subfield subcode with fixed dimension: dimension of subcode cannot be "
                "larger "
                "than dimension of original code");

        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to construct a subfield subcode with fixed dimension: code does not have GRSCodeBackend");

        Matrix<uint8_t> M(T::get_m(), get_n());
        for (size_t i = 0; i < get_n(); ++i) {
            M(0, i) = 1;
        }

        Matrix<T> B;
        auto SSC = SSCode<T, 1>(*this, M, &B);

        size_t max = 0;
        size_t d;
        Matrix<T> B_sub;
        for (size_t i = 0; i < B.get_m() - kp; ++i) {
            // sort only rows below current block (rows of a block have their leading component in the same column),
            // i.e., columns i_start, ..., m-1
            size_t i_start = i + 1;
            for (size_t mu = i + 1; mu < B.get_m(); ++mu) {
                if (s(B, mu) == s(B, i)) ++i_start;
            }

            // sort (using bubble sort) starting from row i_start according to the position of the trailing
            // component
            auto B_sorted = B;

            size_t m = B_sorted.get_m();
            bool swapped = false;
            do {
                for (size_t mu = i_start; mu < m - 1; ++mu) {
                    if (t(B_sorted, mu) > t(B_sorted, mu + 1)) {
                        B_sorted.swap_rows(mu, mu + 1);
                        swapped = true;
                    }
                }
                --m;
                if (m == 1) break;
            } while (swapped);

            // t trajectory must not move to the left, t_block gives the column of the rightmost nonzero component
            // of the block from row i to row i+kp-1
            size_t t_block = 0;
            for (size_t mu = i; mu < i + kp; ++mu) {
                if (t(B_sorted, mu) > t_block) t_block = t(B_sorted, mu);
            }

            // zerocols is the number of zero columns at the left and right of the block and we are looking for a
            // block (starting at inde i) that maximizes this value
            size_t zerocols = s(B_sorted, i) + B_sorted.get_n() - t_block - 1;
            if (zerocols > max) {
                max = zerocols;
                d = zerocols + 1;  // designed distance, distance of universe code UC plus value
                B_sub = B_sorted.get_submatrix(
                    i, 0, kp, get_n());  // submatrix of B that with largest distance under constraint kp
            }
        }

        if (B_sub.get_m() == 0) return EmptyCode<SF<T, 1>, T>();

        auto C = LinearCode<SF<T, 1>, T>(get_n(), B_sub.get_m(), Matrix<SF<T, 1>>(B_sub * get_G()));

        if constexpr (SF<T, 1>::get_size() == 2) {
            if (C.get_k() == 1) C.add_backend(new RepetitionCodeBackend<SF<T, 1>>(get_n()));
        }

        // C.add_backend(new SSCodeBackend<T>(*this, d));

        return C;
    }

    /*
    Code<T, SUPER> get_SSCode_with_dimension(size_t kp) const {
        Matrix<uint8_t> M(T::get_m(), get_n());
        for (size_t i = 0; i < get_n(); ++i) {
            M(0, i) = 1;
        }
        return get_SSCode_with_dimension<T, SUPER>(kp, M);
    }
    */

    const Matrix<T>& get_H() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to access parity check matrix H: code does not have LinearCodeBackend");
        return backend->get_H();
    }

    const Matrix<T>& get_HT() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to access transposed parity check matrix HT: code does not have LinearCodeBackend");
        return backend->get_HT();
    }

    bool is_perfect() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is perfect: code does not have LinearCodeBackend");
        return backend->is_perfect();
    }

    bool is_MDS() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to check whether code is MDS: code does not have LinearCodeBackend");
        return backend->is_MDS();
    }

    bool is_equidistant() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to check whether code is MDS: code does not have LinearCodeBackend");
        return backend->is_equidistant();
    }

    bool is_weakly_self_dual() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is weakly self-dual: code does not have LinearCodeBackend");
        return backend->is_weakly_self_dual();
    }

    bool is_dual_containing() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is dual-containing: code does not have LinearCodeBackend");
        return backend->is_dual_containing();
    }

    bool is_self_dual() const {
        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is self-dual: code does not have LinearCodeBackend");
        return backend->is_self_dual();
    }

    bool is_polynomial() {
        if (get_backend<EmptyCodeBackend>()) return false;
        if (get_backend<PolynomialCodeBackend<T>>()) return true;

        std::clog << "--> Checking whether the code is polynomial, this requires iterating through "
                  << sqm<InfInt>(T::get_size(), get_k()) << std::setfill(' ') << " codewords (worst case)" << std::endl;

        auto backend = get_backend<LinearCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is polynomial: code does not have LinearCodeBackend");
        auto d = Polynomial<T>({0});
        for (auto it = cbegin(); it != cend(); ++it) {
            d = GCD(d, Polynomial<T>(*it));
            if (d.get_degree() < backend->get_n() - backend->get_k() && d != Polynomial<T>({0})) return false;
        }
        if (d.get_trailing_degree() > 0) return false;

        d = normalize(d);  // degree at this point can only be n-k

        size_t k = backend->get_k();
        if (backend->get_n() == 1)
            backend_collection.push_back(new PolynomialCodeBackend<T>(k, d, 1));
        else
            backend_collection.push_back(new PolynomialCodeBackend<T>(k, d));
        return true;
    }

    size_t get_s() const {
        auto backend = get_backend<HammingCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to access Hamming code parameter s: code does not have HammingCodeBackend");
        return backend->get_s();
    }

    size_t get_CWr() const {
        auto backend = get_backend<CordaroWagnerCodeBackend>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to access Cordaro-Wagner code parameter r: code does not have CordaroWagnerCodeBackend");
        return backend->get_r();
    }

    size_t get_RMr() const {
        auto backend = get_backend<RMCodeBackend>();
        if (!backend)
            throw std::invalid_argument("Trying to access RM code parameter r: code does not have RMCodeBackend");
        return backend->get_r();
    }

    size_t get_RMm() const {
        auto backend = get_backend<RMCodeBackend>();
        if (!backend)
            throw std::invalid_argument("Trying to access RM code parameter m: code does not have RMCodeBackend");
        return backend->get_m();
    }

    const Polynomial<T>& get_gamma() const {
        auto backend = get_backend<PolynomialCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to access generator polynomial gamma: code does not have PolynomialCodeBackend");
        return backend->get_gamma();
    }

    bool is_cyclic() const {
        auto backend = get_backend<PolynomialCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is cyclic: code does not have PolynomialCodeBackend");
        return backend->is_cyclic();
    }

    size_t get_ell() const {
        auto poly_backend = get_backend<PolynomialCodeBackend<T>>();
        if (!poly_backend)
            throw std::invalid_argument(
                "Trying to calculate cyclic burst error correction performance: code does not have "
                "PolynomialCodeBackend");
        auto lin_backend = get_backend<LinearCodeBackend<T>>();
        if (!lin_backend)
            throw std::invalid_argument(
                "Trying to calculate cyclic burst error correction performance: code does not have "
                "LinearCodeBackend");
        return poly_backend->get_ell(lin_backend);
    }

    const Vector<T>& get_locators() const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend) throw std::invalid_argument("Trying to access code locators: code does not have GRSCodeBackend");
        return backend->get_locators();
    }

    const Vector<T>& get_multipliers() const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to access column multipliers: code does not have GRSCodeBackend");
        return backend->get_multipliers();
    }

    bool is_singly_extended() const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is singly-extended: code does not have GRSCodeBackend");
        return backend->is_singly_extended();
    }

    bool is_primitive() const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to check whether code is primitive: code does not have GRSCodeBackend");
        return backend->is_primitive();
    }

    bool is_narrow_sense() const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is narrow-sense: code does not have GRSCodeBackend");
        return backend->is_narrow_sense();
    }

    bool is_normalized() const {
        auto backend = get_backend<GRSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument(
                "Trying to check whether code is normalized: code does not have GRSCodeBackend");
        return backend->is_normalized();
    }

    T get_alpha() const {
        auto backend = get_backend<RSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to access RS code parameter alpha: code does not have RSCodeBackend");
        return backend->get_alpha();
    }

    size_t get_b() const {
        auto backend = get_backend<RSCodeBackend<T>>();
        if (!backend)
            throw std::invalid_argument("Trying to access RS code parameter b: code does not have RSCodeBackend");
        return backend->get_b();
    }

    Code<SUPER> get_supercode() const {
        if (get_backend<EmptyCodeBackend>())

            throw std::invalid_argument("Empty code does not have a well-defined supercode");
        auto backend = get_backend<SSCodeBackend<SUPER>>();
        if (!backend) throw std::invalid_argument("Trying to access supercode: code does not have SSCodeBackend");
        return backend->get_supercode();
    }

    size_t get_design_distance() const {
        if (get_backend<EmptyCodeBackend>()) return 0;
        auto backend = get_backend<SSCodeBackend<SUPER>>();
        if (!backend) throw std::invalid_argument("Trying to access design distance: code does not have SSCodeBackend");
        return backend->get_design_distance();
    }

    template <class U = T, uint8_t m>  // "feed forward" template parameter of class template
    Code<SF<U, m>, U> get_subfield_subcode(size_t kp = 0) const {
        Matrix<int> M(U::get_m() / m, get_n());
        for (size_t i = 0; i < get_n(); ++i) {
            M(0, i) = 1;
        }

        return get_subfield_subcode<U, m>(kp, M);
    }

    template <class U = T, uint8_t m>  // "feed forward" template parameter of class template
    Code<SF<U, m>, U> get_subfield_subcode(size_t kp, const Matrix<int>& M) const {
        const size_t n = get_n();
        const size_t k = get_k();

        if (n != k && get_backend<RSCodeBackend<T>>()) {
            auto gamma = get_gamma();
            auto alpha = get_alpha();
            const size_t b = get_b();

            std::vector<T> R;  // roots of gamma
            for (size_t j = 0; j < n - k; ++j) {
                R.push_back(alpha ^ (j - b + 1));
            }
            std::vector<Polynomial<T>> minpolys;  // minimal polynomials of roots
            for (auto it = R.cbegin(); it != R.cend(); ++it) {
                Polynomial<T> minpoly({-*it, 1});  // build up min. polynomial over superfield
                size_t exponent = SF<T, m>::get_size();
                // careful: in C++, != and - operators bind stronger than ^ operator
                while (*it != (*it ^ exponent)) {
                    minpoly *= Polynomial<T>({-(*it ^ exponent), 1});
                    exponent *= SF<T, m>::get_size();
                }  // m has coefficients from subfield but their data type is superfield...
                minpolys.emplace_back(minpoly);
            }
            auto temp = LCM(minpolys);
            auto gammap = Polynomial<SF<T, m>>(temp);  // least common multiple of minimal polynomials

            auto quotient = temp / gamma;
            auto B = ToeplitzMatrix(pad_back(pad_front(Vector<T>(quotient), k), 2 * k - quotient.get_degree() - 1),
                                    k - quotient.get_degree(), k);

            if (n - gammap.get_degree() == 0 || kp > n - gammap.get_degree()) return EmptyCode<SF<T, m>, T>();

            auto C = PolynomialCode<SF<T, m>, T>(kp == 0 ? n - gammap.get_degree() : kp, gammap);

            if constexpr (SF<T, m>::get_size() == 2) {
                if (C.get_k() == 1) C.add_backend(new RepetitionCodeBackend<SF<T, m>>(n));
            }

            const size_t s = gammap.get_degree();
            if (n == (sqm<size_t>(SF<T, m>::get_size(), s) - 1) / (SF<T, m>::get_size() - 1)) {
                C.add_backend(new HammingCodeBackend<SF<T, m>>(s));
            }

            auto SuperCp = RSCode<T>(alpha, b, n - GCD(gamma, temp).get_degree());
            C.add_backend(new SSCodeBackend<T>(SuperCp, SuperCp.get_dmin()));

            // C.add_backend(new SSCodeBackend<T>(*this, n - k + 1));

            return C;
        } else if (get_backend<GRSCodeBackend<T>>()) {
            auto G = get_canonical_G();

            std::cout << std::endl << G << std::endl;

            const size_t q = T::get_m() / SF<T, m>::get_m();
            Matrix<SF<T, m>> Mt(q * n, q * k);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < k; ++j) {
                    auto S = T_matrix(G(j, i)) * R_matrix();
                    // set first column to zero ($[\cdot]_0$)
                    // S.set_submatrix(ZeroMatrix<SF<T, m>>(q, 1), 0, 0);
                    for (size_t ell = 0; ell < q; ++ell) {  // if M(ell, i) is nonzero component must be zero
                        if (M(ell, i)) S.set_submatrix(ZeroMatrix<SF<T, m>>(q, 1), 0, ell);
                    }
                    S.transpose();
                    Mt.set_submatrix(S, q * i, q * j);
                }
            }

            auto Bt = Mt.basis_of_nullspace();
            std::cout << std::endl << Bt << std::endl;

            if (kp == 0) kp = Bt.get_m();

            size_t ds = 0;
            size_t leading = 0;
            size_t trailing = 0;
            Matrix<SF<T, m>> E;

            for (size_t d = 1; d <= n; ++d) {
                for (size_t l = 0; l <= d - 1; ++l) {
                    size_t r = d - 1 - l;
                    auto L = Bt.get_submatrix(0, 0, Bt.get_m(), l * T::get_m());
                    auto R = Bt.get_submatrix(0, Bt.get_n() - r * T::get_m(), Bt.get_m(), r * T::get_m());
                    auto LR = horizontal_join(L, R);  // left and right column submatrices

                    auto A = horizontal_join(LR, IdentityMatrix<SF<T, m>>(LR.get_m()));  // augmented matrix
                    A.rref();
                    LR = A.get_submatrix(0, 0, LR.get_m(), LR.get_n());
                    size_t rank = LR.rank();

                    size_t rank_def = LR.get_m() - rank;
                    if (rank_def >= kp) {
                        std::cout << rank_def << " --> d = " << d << " (l = " << l << ", r = " << r << ")" << std::endl;
                        ds = d;
                        leading = l;
                        trailing = r;
                        E = A.get_submatrix(0, LR.get_n(), LR.get_m(), LR.get_m());
                    }
                }
            }

            if (ds == 0) return EmptyCode<SF<U, m>, U>();

            Bt = E * Bt;

            Matrix<T> B(Bt.get_m(), Bt.get_n() / q);
            for (size_t i = 0; i < B.get_m(); ++i) {
                for (size_t j = 0; j < B.get_n(); ++j) {
                    B(i, j) = T(Vector<SF<T, m>>(Bt.get_submatrix(i, q * j, 1, q)));
                }
            }
            std::cout << std::endl << B << std::endl;

            auto B_sub = B.get_submatrix(B.get_m() - kp, 0, kp, B.get_n());

            // std::cout << showall << LinearCode<T>(n, B_sub.get_m(), B_sub) << std::endl;
            std::cout << std::endl << B_sub << std::endl;

            auto Gp = B_sub * G;

            for (size_t j = 0; j < Gp.get_n(); ++j) {
                U div;
                for (size_t ell = 0; ell < U::get_m() / m; ++ell) {
                    if (M(ell, j) != 0) {
                        div = sqm<U>(U(U::get_characteristic()), ell);
                        break;
                    }
                }
                for (size_t i = 0; i < Gp.get_m(); ++i) {
                    Gp(i, j) /= div;
                }
            }

            auto C = LinearCode<SF<U, m>, U>(get_n(), B_sub.get_m(), Matrix<SF<U, m>>(Gp));

            if constexpr (SF<U, m>::get_size() == 2) {
                if (C.get_k() == 1) C.add_backend(new RepetitionCodeBackend<SF<U, m>>(get_n()));
            }

            auto a = get_locators();
            auto d = get_multipliers();
            Vector<T> dp(get_n());
            for (size_t i = 0; i < get_n(); ++i) {
                dp.set_component(i, d[i] * sqm<T>(a[i], leading));
            }
            auto SuperCp = GRSCode<T>(a, dp, get_n() - leading - trailing);

            C.add_backend(new SSCodeBackend<T>(SuperCp, SuperCp.get_dmin()));

            // C.add_backend(new SSCodeBackend<U>(*this, d));

            return C;
        } else {
            throw std::invalid_argument("Can only construct subfield-subcodes of RS and GRS codes!");
        }
    }

   private:
    void clear() {
        for (auto it = backend_collection.begin(); it != backend_collection.end(); ++it) {
            delete *it;
        }
        backend_collection.clear();
    }

    template <class V>
    size_t s(const Matrix<V>& M, size_t i) const {
        for (size_t j = 0; j < M.get_n(); ++j) {
            if (M(i, j) != V(0)) return j;
        }
        return 0;
    }

    template <class V>
    size_t t(const Matrix<V>& M, size_t i) const {
        for (size_t j = M.get_n() - 1; j >= 0; --j) {
            if (M(i, j) != V(0)) {
                return j;
            }
        }
        return M.get_n() - 1;
    }

    Matrix<SF<T, 1>> T_matrix(const T& b) const {
        const size_t q = T::get_m() / SF<T, 1>::get_m();
        auto v = Vector<SF<T, 1>>(q - 1);
        v = concatenate(v, b.template as_vector<SF<T, 1>>());
        v = concatenate(v, Vector<SF<T, 1>>(q - 1));
        auto res = ToeplitzMatrix<SF<T, 1>>(v, q, 2 * q - 1);
        return res;
    }

    Matrix<SF<T, 1>> Rj_matrix(size_t j) const {
        const size_t q = T::get_m() / SF<T, 1>::get_m();
        auto I = IdentityMatrix<SF<T, 1>>(j);
        Matrix<SF<T, 1>> v(1, q);
        v(0, 0) = 1;
        auto C = transpose(CompanionMatrix<SF<T, 1>>(T::template get_modulus<SF<T, 1>>()));
        return diagonal_join(I, vertical_join(v, C));
    }

    Matrix<SF<T, 1>> R_matrix() const {
        const size_t q = T::get_m() / SF<T, 1>::get_m();
        auto res = Rj_matrix(q - 2);
        for (size_t i = q - 2; i > 0; --i) res *= Rj_matrix(i - 1);
        return res;
    }

    InfInt N(size_t ell, size_t h, size_t s) const noexcept {
        const size_t n = get_n();

        if (T::get_size() == 2) {
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

            bool breakflag = false;
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

    std::vector<Backend*> backend_collection;
};

template <class T, class SUPER>
std::ostream& operator<<(std::ostream& os, Code<T, SUPER>& rhs) {
    if (rhs.template get_backend<EmptyCodeBackend>()) {
        os << "[F_" << T::get_size() << "; " << 0 << ", " << 0 << "]";
        return os;
    }

    if (os.iword(index) > 1) {
        rhs.is_polynomial();  // no output, this adds polynomial backend resulting in output below
    }

    os << "[F_" << T::get_size() << "; " << rhs.get_n() << ", " << rhs.get_k() << "]";
    if (auto backend = rhs.template get_backend<LinearCodeBackend<T>>()) {
        if (backend->is_dmin_known()) {
            os << ", dmin = " << backend->get_dmin();
        }
    }
    if (os.iword(index) > 0) {
        os << std::endl;
        os << "F_" << T::get_size() << " = " << T::get_info();

        for (auto it = rhs.backend_collection.begin(); it != rhs.backend_collection.end(); ++it) {
            (*it)->get_info(os);
        }
    }

    return os;
}

template <class T, class SUPER = T>
std::ostream& operator<<(std::ostream& os, Code<T, SUPER>&& rhs) {
    os << rhs;
    return os;
}

template <class T, class SUPER>
Code<T, SUPER> EmptyCode() {
    auto C = Code<T, SUPER>();
    C.add_backend(new EmptyCodeBackend());
    return C;
}

template <class T, class SUPER>
Code<T, SUPER> UniverseCode(size_t n) {
    auto C = Code<T>();
    auto gamma = Polynomial<T>();
    gamma.set_coeff(0, T(1));
    gamma.set_coeff(n, T(1));
    C.add_backend(new PolynomialCodeBackend<T>(n, gamma, 1));
    auto weight_enumerator = Polynomial<InfInt>();
    for (size_t i = 0; i <= n; ++i) weight_enumerator.set_coeff(i, bin<InfInt>(n, i));
    C.add_backend(new LinearCodeBackend<T>(n, n, IdentityMatrix<T>(n), 1, weight_enumerator));
    return C;
}

template <class T, class SUPER = T>
Code<T, SUPER> LinearCode(size_t n, size_t k, const Matrix<T>& X) {
    auto C = Code<T, SUPER>();
    C.add_backend(new LinearCodeBackend<T>(n, k, X));
    return C;
}

/*
// not needed
template <class T, class SUPER = T>
Code<T, SUPER> LinearCode(size_t n, size_t k, Matrix<T>&& X) {
    auto C = Code<T, SUPER>();
    C.add_backend(new LinearCodeBackend<T>(n, k, X));
    return C;
}
*/

template <class T>
Code<T> RepetitionCode(size_t n) {
    auto C = Code<T>();
    auto G = Matrix<T>(1, n, T(1));
    auto weight_enumerator = Polynomial<InfInt>();
    weight_enumerator.set_coeff(0, 1);
    weight_enumerator.set_coeff(n, 1);
    C.add_backend(new LinearCodeBackend<T>(n, 1, G, n, weight_enumerator));
    C.add_backend(new RepetitionCodeBackend<T>(n % 2));
    return C;
}

template <class T>
Code<T> SingleParityCheckCode(size_t n) {
    return dual(RepetitionCode<T>(n));
}

template <class T>
Code<T> HammingCode(size_t s) {
    auto C = Code<T>();
    const size_t n = (sqm<size_t>(T::get_size(), s) - 1) / (T::get_size() - 1);
    const size_t k = n - s;
    auto H = Matrix<T>(n, s);
    size_t i = 0;
    // topmost element loop
    for (size_t top = 0; top < s; ++top) {
        const auto v = IdentityMatrix<T>(s - top - 1).rowspace();
        for (size_t j = 0; j < v.size(); ++j) {
            H(n - i - 1, top) = T(1);
            H.set_submatrix(Matrix(v[v.size() - j - 1]), n - i - 1, top + 1);
            ++i;
        }
    }
    H.transpose();
    if constexpr (T::get_size() == 2) {
        Polynomial<InfInt> temp;                                       // weight enumerator of the dual...
        temp.set_coeff(0, 1);                                          // ... code, a binary simplex code...
        temp.set_coeff(sqm<size_t>(2, s - 1), sqm<InfInt>(2, s) - 1);  // ... easy to calculate using ML
        C.add_backend(new LinearCodeBackend<T>(n, k, H, 3, MacWilliamsIdentity<T>(temp, n, n - k)));
        if (s == 2) C.add_backend(new RepetitionCodeBackend<T>(n));
    } else {
        C.add_backend(new LinearCodeBackend<T>(n, k, H, 3));
    }
    C.add_backend(new HammingCodeBackend<T>(s));
    return C;
}

template <class T>
Code<T> SimplexCode(size_t s) {
    if (s == 0)
        return EmptyCode<T>();
    else if (s == 1)
        return RepetitionCode<T>(1);
    else
        return dual(HammingCode<T>(s));
}

Code<Fp<2>> CordaroWagnerCode(size_t n) {
    auto C = Code<Fp<2>>();

    auto* cwbackend = new CordaroWagnerCodeBackend(n);
    const size_t k = 2;
    const size_t h = cwbackend->get_r();
    size_t i = 0;
    if (n == 3 * h - 1) {
        i = h - 1;
    } else if (n == 3 * h) {
        i = h;
    } else if (n == 3 * h + 1) {
        i = h + 1;
    }
    const size_t j = h;
    Matrix<Fp<2>> G(2, n);
    const Matrix<Fp<2>> type1_column(2, 1, {1, 0});
    for (size_t s = 0; s < h; ++s) {
        G.set_submatrix(type1_column, 0, s);
    }
    const Matrix<Fp<2>> type2_column(2, 1, {0, 1});
    for (size_t s = h; s < h + i; ++s) {
        G.set_submatrix(type2_column, 0, s);
    }
    const Matrix<Fp<2>> type3_column(2, 1, {1, 1});
    for (size_t s = h + i; s < h + i + j; ++s) {
        G.set_submatrix(type3_column, 0, s);
    }
    C.add_backend(new LinearCodeBackend<Fp<2>>(n, 2, G, floor(2.0 * n / 3.0) - 1));
    C.add_backend(cwbackend);
    return C;
}

template <class T>
Code<T> dual(const Code<T>& C) {
    if (auto backend = C.template get_backend<EmptyCodeBackend>())
        throw std::invalid_argument("Trying to construct dual code of empty code!");

    if (C.get_n() == C.get_k()) return EmptyCode<T>();

    if (auto backend = C.template get_backend<LinearCodeBackend<T>>()) {
        Code<T> D;
        const size_t n = C.get_n();
        const size_t k = C.get_k();
        auto H = C.get_H();

        if (auto backend = C.template get_backend<RMCodeBackend>()) {
            const size_t r = backend->get_r();
            const size_t m = backend->get_m();
            return RMCode(m - r - 1, m);
        }

        if (backend->is_weight_enumerator_known()) {
            auto A = MacWilliamsIdentity<T>(C.get_weight_enumerator(), n, k);
            size_t dmin = 1;
            for (size_t i = 1; i <= A.get_degree(); ++i) {
                if (A[i] != 0) {
                    dmin = i;
                    break;
                }
            }
            D.add_backend(new LinearCodeBackend<T>(n, n - k, H, dmin, A));
        } else {
            D.add_backend(new LinearCodeBackend<T>(n, n - k, H));
        }

        if (auto backend = C.template get_backend<PolynomialCodeBackend<T>>()) {
            if (C.is_cyclic()) {
                auto gamma = C.get_gamma();
                Polynomial<T> xnm1;
                xnm1.set_coeff(0, T(1));
                xnm1.set_coeff(n, T(1));
                D.add_backend(new PolynomialCodeBackend<T>(n - k, xnm1 / gamma, 1));
            }
        }

        if (auto backend = C.template get_backend<RepetitionCodeBackend<T>>())
            D.add_backend(new SingleParityCheckCodeBackend<T>());

        if (auto backend = C.template get_backend<SingleParityCheckCodeBackend<T>>())
            D.add_backend(new RepetitionCodeBackend<T>(n));

        if (auto backend = C.template get_backend<HammingCodeBackend<T>>())
            D.add_backend(new SimplexCodeBackend<T>(backend->get_s()));

        if (auto backend = C.template get_backend<SimplexCodeBackend<T>>())
            D.add_backend(new HammingCodeBackend<T>(backend->get_s()));

        if (auto backend = C.template get_backend<GRSCodeBackend<T>>()) {
            auto DG = DiagonalMatrix<T>(C.get_multipliers());
            // auto Vp = VandermondeMatrix<T>(get_locators(), get_n() - get_k());
            auto W = VandermondeMatrix<T>(C.get_locators(), n - 1);
            // auto dp = W.basis_of_nullspace().get_row(0); // less efficient alternative
            auto dp = transpose(vertical_join(
                                    inverse(W.get_submatrix(0, 0, n - 1, n - 1)) * W.get_submatrix(0, n - 1, n - 1, 1),
                                    -IdentityMatrix<T>(1)  // dn−1 = −1
                                    ))
                          .get_row(0);
            auto Dp = DiagonalMatrix<T>(dp);
            auto DH = Dp * inverse(DG);
            auto d = DH.diagonal();
            D.add_backend(new GRSCodeBackend<T>(C.get_locators(), d, n - k));
        }

        return D;

    } else {
        throw std::invalid_argument("Trying to construct dual code of non-linear code!");
    }
}

template <class T>
Code<T> puncture(const Code<T>& C, size_t i) {
    const size_t n = C.get_n();
    const size_t k = C.get_k();

    if (i >= n) throw std::invalid_argument("Trying to puncture a code at an invalid index!");

    if (n == 1) return EmptyCode<T>();

    if (auto backend = C.template get_backend<ExtendedCodeBackend<T>>())
        if (i == backend->get_i()) return backend->get_C();

    if (auto backend = C.template get_backend<RepetitionCodeBackend<T>>()) return RepetitionCode<T>(n - 1);

    if (auto backend = C.template get_backend<GRSCodeBackend<T>>()) {
        auto ap = C.get_locators();
        ap.delete_components({i});
        auto dp = C.get_multipliers();
        dp.delete_components({i});
        return GRSCode<T>(ap, dp, k);
    }

    auto G = C.get_G();
    Matrix<T> Gp;
    if (i == 0) {
        Gp = G.get_submatrix(0, 1, k, n - 1);
    } else if (i == n - 1) {
        Gp = G.get_submatrix(0, 0, k, n - 1);
    } else {
        auto L = G.get_submatrix(0, 0, k, i);
        auto R = G.get_submatrix(0, i + 1, k, n - i - 1);
        Gp = horizontal_join(L, R);
    }

    /* check if one of the rows is zero or a multiple of another */
    size_t kp = k;
    bool breakflag = false;
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = i + 1; j < k; ++j) {
            if (Gp.get_row(j).is_zero()) {
                Gp.delete_rows({j});
                kp -= 1;
                breakflag = true;
                break;
            }
            for (int a = 0; a < T::get_size(); ++a) {
                if (Gp.get_row(i) == T(a) * Gp.get_row(j)) {
                    Gp.delete_rows({j});
                    kp -= 1;
                    breakflag = true;
                    break;
                }
            }
            if (breakflag) break;
        }
        if (breakflag) break;
    }

    if (n - 1 == kp)
        return UniverseCode<T>(n - 1);
    else
        return LinearCode<T>(n - 1, kp, Gp);
}

template <class T>
Code<T> extend(const Code<T>& C, size_t i, const Vector<T>& v) {
    const size_t n = C.get_n();
    const size_t k = C.get_k();

    if (i > n) throw std::invalid_argument("Trying to extend a code at an invalid index!");
    if (v.get_n() != k) throw std::invalid_argument("Trying to extend a code with an invalid generator matrix column!");

    if (auto backend = C.template get_backend<RepetitionCodeBackend<T>>()) {
        if (v.get_n() == 1 && v[0] == T(1)) {
            auto Cp = RepetitionCode<T>(n + 1);
            Cp.add_backend(new ExtendedCodeBackend(C, i, v));
            return Cp;
        }
    }

    auto G = C.get_G();
    auto L = G.get_submatrix(0, 0, k, i);
    auto M = Matrix(v);
    M.transpose();
    auto R = G.get_submatrix(0, i, k, n - i);
    auto Gp = horizontal_join(horizontal_join(L, M), R);

    auto Cp = Code<T>();
    bool flag = true;
    if constexpr (T::get_size() == 2) {
        for (size_t i = 0; i < Gp.get_m(); ++i) {
            if (wH(Gp.get_row(i)) % 2) {
                flag = false;
                break;
            }
        }
    }
    if (T::get_size() == 2 && flag)
        Cp.add_backend(new LinearCodeBackend<T>(n + 1, k, Gp, C.get_dmin() + 1));
    else
        Cp.add_backend(new LinearCodeBackend<T>(n + 1, k, Gp));

    Cp.add_backend(new ExtendedCodeBackend(C, i, v));

    return Cp;
}

template <class T>
Code<T> expurgate(const Code<T>& C, size_t j) {
    const size_t n = C.get_n();
    const size_t k = C.get_k();

    if (j >= k) throw std::invalid_argument("Trying to expurgate a code at an invalid index!");

    if (k == 1) return EmptyCode<T>();

    if (auto backend = C.template get_backend<AugmentedCodeBackend<T>>())
        if (j == backend->get_j()) return backend->get_C();

    if (auto backend = C.template get_backend<RSCodeBackend<T>>()) {
        if (C.get_G() == C.get_canonical_G()) {
            auto alpha = C.get_alpha();
            const size_t b = C.get_b();
            Polynomial<T> f = {-sqm(alpha, -b + n - j), 1};
            if (j == k - 1)
                return RSCode<T>(alpha, b, k - 1);
            else if (j == 0)
                return RSCode<T>(alpha, b + 1, k - 1);
            else
                return PolynomialCode<T>(k - 1, f * C.get_gamma());
        }
    }

    if (auto backend = C.template get_backend<GRSCodeBackend<T>>()) {
        if (C.get_G() == C.get_canonical_G()) {
            auto a = C.get_locators();
            auto d = C.get_multipliers();
            if (j == k - 1) {
                return GRSCode<T>(a, d, k - 1);
            } else if (j == 0) {
                auto dp = d;
                for (size_t i = 0; i < n; ++i) {
                    dp.set_component(i, dp[i] * a[i]);
                }
                return GRSCode<T>(a, dp, k - 1);
            }
        }
    }

    auto G = C.get_G();
    Matrix<T> Gp;
    if (j == 0) {
        Gp = G.get_submatrix(1, 0, k - 1, n);
    } else if (j == k - 1) {
        Gp = G.get_submatrix(0, 0, k - 1, n);
    } else {
        auto U = G.get_submatrix(0, 0, j, n);
        auto L = G.get_submatrix(j + 1, 0, k - j - 1, n);
        Gp = vertical_join(U, L);
    }

    return LinearCode<T>(n, k - 1, Gp);
}

template <class T>
Code<T> augment(const Code<T>& C, size_t j, const Vector<T>& w) {
    const size_t n = C.get_n();
    const size_t k = C.get_k();

    if (j > k) throw std::invalid_argument("Trying to augment a code at an invalid index!");
    if (j >= n)
        throw std::invalid_argument(
            "Trying to augment a code so that its dimension would become larger than its length!");
    if (w.get_n() != n) throw std::invalid_argument("Trying to augment a code with an invalid generator matrix row!");
    if (wH(w * transpose(C.get_H())) == 0) throw std::invalid_argument("Trying to augment a code with a codeword!");

    if (auto backend = C.template get_backend<RSCodeBackend<T>>()) {
        auto alpha = C.get_alpha();
        const size_t b = C.get_b();
        if (j == k) {
            bool flag = true;
            for (size_t i = 0; i < n; ++i) {
                if (w[i] != sqm(alpha, (b + k) * i)) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                auto Cp = RSCode<T>(alpha, b, k + 1);
                Cp.add_backend(new AugmentedCodeBackend(C, j, w));
                return Cp;
            }
        }
    }

    if (auto backend = C.template get_backend<GRSCodeBackend<T>>()) {
        auto a = C.get_locators();
        auto d = C.get_multipliers();
        if (j == k) {
            bool flag = true;
            for (size_t i = 0; i < n; ++i) {
                if (w[i] != d[i] * sqm(a[i], j)) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                auto Cp = GRSCode<T>(a, d, k + 1);
                Cp.add_backend(new AugmentedCodeBackend(C, j, w));
                return Cp;
            }
        }
    }

    auto G = C.get_G();
    auto U = G.get_submatrix(0, 0, j, n);
    auto M = Matrix(w);
    auto L = G.get_submatrix(j, 0, k - j, n);
    auto Gp = vertical_join(vertical_join(U, M), L);

    auto Cp = LinearCode<T>(n, k + 1, Gp);
    Cp.add_backend(new AugmentedCodeBackend(C, j, w));
    return Cp;
}

template <class T>
Code<T> shorten(const Code<T>& C, size_t i, size_t j) {
    return puncture(expurgage(C, j), i);
}

template <class T>
Code<T> lengthen(const Code<T>& C, size_t i, const Vector<T>& v, size_t j, const Vector<T>& w) {
    return extend(augment(C, j, w), i, v);
}

template <class T>
Code<T> ldc(const Code<T>& U, const Code<T>& V) {
    const size_t n = U.get_n();

    if (n != V.get_n())
        throw std::invalid_argument("Trying to apply lens-doubling construction to codes of different lengths!");

    Matrix<T> G(U.get_k() + V.get_k(), 2 * n);
    G.set_submatrix(U.get_G(), 0, 0);
    G.set_submatrix(U.get_G(), 0, n);
    G.set_submatrix(V.get_G(), U.get_k(), n);

    auto C = Code<T>();
    if constexpr (T::get_size() == 2)
        C.add_backend(
            new LinearCodeBackend<T>(2 * n, U.get_k() + V.get_k(), G, std::min({2 * U.get_dmin(), V.get_dmin()})));
    else
        C.add_backend(new LinearCodeBackend<T>(2 * n, U.get_k() + V.get_k(), G));

    C.add_backend(new LDCCodeBackend<T>(U, V));

    return C;
}

Code<Fp<2>> RMCode(size_t r, size_t m) {
    if (r > m) throw std::invalid_argument("Trying to construct RM with r > m!");

    if (r == m) {
        return UniverseCode<Fp<2>>(sqm(2, m));
    } else if (r == 0) {
        return RepetitionCode<Fp<2>>(sqm(2, m));
    } else {
        auto C = ldc(RMCode(r, m - 1), RMCode(r - 1, m - 1));
        if (r == m - 1) C.add_backend(new SingleParityCheckCodeBackend<Fp<2>>());
        C.add_backend(new RMCodeBackend(r, m));
        return C;
    }
}

template <class T, class SUPER = T>
Code<T, SUPER> PolynomialCode(size_t k, const Polynomial<T>& gamma) {
    auto C = Code<T, SUPER>();
    const size_t n = k + gamma.get_degree();
    auto G =
        ToeplitzMatrix(pad_back(pad_front(Vector<T>(gamma), k + gamma.get_degree()), 2 * k + gamma.get_degree() - 1), k,
                       k + gamma.get_degree());
    C.add_backend(new LinearCodeBackend<T>(n, k, G));
    C.add_backend(new PolynomialCodeBackend<T>(k, gamma));
    return C;
}

template <class T>
Code<T> GolayCode() {
    static_assert(T::get_size() == 2 || T::get_size() == 3, "Golay codes only defined for fields of size 2 or 3");

    size_t k;
    if constexpr (T::get_size() == 2)
        k = 12;
    else if (T::get_size() == 3)
        k = 6;

    Polynomial<T> gamma;
    if constexpr (T::get_size() == 2) {
        gamma = Polynomial<T>({1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1});
    } else if (T::get_size() == 3) {
        gamma = Polynomial<T>({2, 0, 1, 2, 1, 1});
    }

    return PolynomialCode<T>(k, gamma);
}

template <class T, bool t, typename>
Code<T> GRSCode(const Vector<T>& a, const Vector<T>& d, size_t k) {
    auto C = Code<T>();

    const size_t n = a.get_n();
    auto G = VandermondeMatrix<T>(a, k) * DiagonalMatrix<T>(d);
    Polynomial<InfInt> A({InfInt(1)});  // MDS, weight enumerator known
    for (size_t i = n - k + 1; i <= n; ++i) {
        InfInt s = 0;
        for (size_t j = 0; j <= i - (n - k + 1); ++j) {
            s += InfInt(j % 2 ? -1 : 1) * bin<InfInt>(i - 1, j) * sqm<InfInt>(T::get_size(), i - j - (n - k + 1));
        }
        A.set_coeff(i, bin<InfInt>(n, i) * (T::get_size() - 1) * s);
    }
    C.add_backend(new LinearCodeBackend<T>(n, k, G, n - k + 1, A));

    if (k == 1) {
        auto G = C.get_G();
        bool repetition = true;
        for (size_t i = 0; i < n - 1; ++i) {
            if (G(0, i) != G(0, i + 1)) {
                repetition = false;
                break;
            }
        }
        if (repetition) {
            C.add_backend(new RepetitionCodeBackend<T>(n));
            C.add_backend(new HammingCodeBackend<T>(2));
        }
    }

    if (a[0] == T(1) && d[0] == T(1)) {
        auto alpha = a[1];
        auto order = alpha.get_multiplicative_order();
        if (order == n) {
            bool rs = true;
            size_t b = 0;
            for (size_t i = 0; i < n; ++i) {
                if ((alpha ^ i) == d[1]) {
                    b = i;
                    break;
                }
            }
            for (size_t i = 2; i < n; ++i) {
                if (a[i] != (alpha ^ i) || d[i] != (alpha ^ (b * i))) {
                    rs = false;
                    break;
                }
            }
            if (rs) {
                Polynomial<T> gamma = 1;
                for (size_t i = 0; i < n - k; ++i) {
                    gamma *= Polynomial<T>({-(alpha ^ (i - b + 1)), 1});
                }
                C.add_backend(new PolynomialCodeBackend(k, gamma, true));
                C.add_backend(new RSCodeBackend<T>(alpha, b));
            }
        }
    }
    C.add_backend(new GRSCodeBackend(a, d, k));

    return C;
}

template <class T, bool t = is_finite_field<T>()>
Code<T> RSCode(T alpha, size_t b, size_t k)
    requires(t)
{
    const size_t n = alpha.get_multiplicative_order();
    if (k > n) throw std::invalid_argument("Trying to construct RS code with k>n!");
    Vector<T> a(n);
    Vector<T> d(n);
    for (size_t i = 0; i < n; ++i) {
        a.set_component(i, alpha ^ i);
        d.set_component(i, alpha ^ (b * i));
    }
    auto C = GRSCode<T>(a, d, k);
    return C;
}

namespace details {

template <class F>
Matrix<SF<F, 1>> T_matrix(const F& b) {
    const size_t q = F::get_m() / SF<F, 1>::get_m();
    auto v = Vector<SF<F, 1>>(q - 1);
    v = concatenate(v, b.template as_vector<SF<F, 1>>());
    v = concatenate(v, Vector<SF<F, 1>>(q - 1));
    auto res = ToeplitzMatrix<SF<F, 1>>(v, q, 2 * q - 1);
    return res;
}

template <class F>
Matrix<SF<F, 1>> Rj_matrix(size_t j) {
    const size_t q = F::get_m() / SF<F, 1>::get_m();
    auto I = IdentityMatrix<SF<F, 1>>(j);
    Matrix<SF<F, 1>> v(1, q);
    v(0, 0) = 1;
    auto C = transpose(CompanionMatrix<SF<F, 1>>(F::template get_modulus<SF<F, 1>>()));
    return diagonal_join(I, vertical_join(v, C));
}

template <class F>
Matrix<SF<F, 1>> R_matrix() {
    const size_t q = F::get_m() / SF<F, 1>::get_m();
    auto res = Rj_matrix<F>(q - 2);
    for (size_t i = q - 2; i > 0; --i) res *= Rj_matrix<F>(i - 1);
    return res;
}

}  // namespace details

// Note: For RS codes, Bp points to B matrix for polynomial encoding. For non-GRS codes it
// points to B matrix for systematic encoding.
template <class T, uint8_t m, bool t, typename>
Code<SF<T, m>, T> SSCode(const Code<T>& SuperC, const Matrix<int>& M, Matrix<T>* Bp) {
    // ToDo, branch is turned off!!!
    if (false && SuperC.template get_backend<RSCodeBackend<T>>()) {
        const size_t n = SuperC.get_n();
        const size_t k = SuperC.get_k();
        auto gamma = SuperC.get_gamma();
        auto alpha = SuperC.get_alpha();
        const size_t b = SuperC.get_b();
        std::vector<T> R;  // roots of gamma
        for (size_t j = 0; j < n - k; ++j) {
            R.push_back(alpha ^ (j - b + 1));
        }
        std::vector<Polynomial<T>> minpolys;  // minimal polynomials of roots
        for (auto it = R.cbegin(); it != R.cend(); ++it) {
            Polynomial<T> minpoly({-*it, 1});  // build up min. polynomial over superfield
            size_t exponent = SF<T, m>::get_size();
            // careful: in C++, != and - operators bind stronger than ^ operator
            while (*it != (*it ^ exponent)) {
                minpoly *= Polynomial<T>({-(*it ^ exponent), 1});
                exponent *= SF<T, m>::get_size();
            }  // m has coefficients from subfield but their data type is superfield...
            minpolys.emplace_back(minpoly);
        }
        auto temp = LCM(minpolys);
        auto gammap = Polynomial<SF<T, m>>(temp);  // least common multiple of minimal polynomials

        if (Bp) {
            auto b = temp / gamma;
            *Bp =
                ToeplitzMatrix(pad_back(pad_front(Vector<T>(b), k), 2 * k - b.get_degree() - 1), k - b.get_degree(), k);
        }

        if (n - gammap.get_degree() == 0) return EmptyCode<SF<T, m>, T>();

        auto C = PolynomialCode<SF<T, m>, T>(n - gammap.get_degree(), gammap);

        if constexpr (SF<T, m>::get_size() == 2) {
            if (C.get_k() == 1) C.add_backend(new RepetitionCodeBackend<SF<T, m>>(n));
        }

        const size_t s = gammap.get_degree();
        if (n == (sqm<size_t>(SF<T, m>::get_size(), s) - 1) / (SF<T, m>::get_size() - 1)) {
            C.add_backend(new HammingCodeBackend<SF<T, m>>(s));
        }

        C.add_backend(new SSCodeBackend<T>(SuperC, n - k + 1));

        return C;
    }
    if (SuperC.template get_backend<GRSCodeBackend<T>>()) {
        const size_t n = SuperC.get_n();
        const size_t k = SuperC.get_k();
        auto G = SuperC.get_canonical_G();

        const size_t q = T::get_m() / SF<T, m>::get_m();
        Matrix<SF<T, m>> Mt(q * n, q * k);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                auto S = T_matrix<T>(G(j, i)) * R_matrix<T>();
                // set first column to zero ($[\cdot]_0$)
                // S.set_submatrix(ZeroMatrix<SF<T, m>>(q, 1), 0, 0);
                for (size_t ell = 0; ell < q; ++ell) {  // if M(ell, i) is nonzero component must be go
                    if (M(ell, i)) {
                        S.set_submatrix(ZeroMatrix<SF<T, m>>(q, 1), 0, ell);
                    }
                }
                S.transpose();
                Mt.set_submatrix(S, q * i, q * j);
            }
        }

        auto Bt = Mt.basis_of_nullspace().rref();

        Matrix<T> B(Bt.get_m(), Bt.get_n() / q);
        for (size_t i = 0; i < B.get_m(); ++i) {
            for (size_t j = 0; j < B.get_n(); ++j) {
                B(i, j) = T(Vector<SF<T, m>>(Bt.get_submatrix(i, q * j, 1, q)));
            }
        }

        auto Gp = B * G;
        // std::cout << Gp << std::endl;

        for (size_t j = 0; j < Gp.get_n(); ++j) {
            T div;
            for (size_t ell = 0; ell < q; ++ell) {
                if (M(ell, j) != 0) {
                    div = sqm<T>(T(T::get_characteristic()), ell);
                    break;
                }
            }
            for (size_t i = 0; i < Gp.get_m(); ++i) {
                Gp(i, j) /= div;
            }
        }

        // std::cout << Gp << std::endl;

        if (Bp) *Bp = B;

        if (B.get_m() == 0) return EmptyCode<SF<T, m>, T>();

        auto C = LinearCode<SF<T, m>, T>(n, B.get_m(), Matrix<SF<T, m>>(Gp));

        if constexpr (SF<T, 1>::get_size() == 2) {
            if (C.get_k() == 1) C.add_backend(new RepetitionCodeBackend<SF<T, 1>>(n));
        }

        size_t leading = 0;
        while (true) {
            size_t w = B.get_submatrix(0, 0, B.get_m(), leading + 1).wH();
            if (w == 0) {
                ++leading;
            } else {
                break;
            }
        }

        size_t trailing = 0;
        while (true) {
            size_t w = B.get_submatrix(0, B.get_n() - trailing - 1, B.get_m(), trailing + 1).wH();
            if (w == 0) {
                ++trailing;
            } else {
                break;
            }
        }

        auto a = SuperC.get_locators();
        auto d = SuperC.get_multipliers();
        Vector<T> dp(n);
        for (size_t i = 0; i < n; ++i) {
            dp.set_component(i, d[i] * sqm<T>(a[i], leading));
        }
        auto SuperCp = GRSCode<T>(a, dp, n - leading - trailing);

        // std::cout << SuperC.get_dmin() + leading + trailing << std::endl;
        // C.add_backend(new SSCodeBackend<T>(SuperC, SuperC.get_dmin() + leading + trailing));
        C.add_backend(new SSCodeBackend<T>(SuperCp, SuperCp.get_dmin()));

        return C;
    } else {
        throw std::invalid_argument("Can only construct subfield-subcodes of RS and GRS codes!");
    }
}

template <class T>
class Encoder : public Block<Vector<T>> {
   public:
    Encoder(const Code<T>& C) : C(C) {}

    Vector<T> operator()(const Vector<T>& in) noexcept { return C.enc(in); }

   private:
    const Code<T>& C;
};

template <class T>
class Dec_BD : public Block<Vector<T>> {
   public:
    Dec_BD(const Code<T>& C, BD_t type = boosted_BD) : C(C), type(type) {}

    Vector<T> operator()(const Vector<T>& in) { return C.dec_BD(in, type); }

   private:
    const Code<T>& C;
    const BD_t type;
};

template <class T>
class Dec_ML : public Block<Vector<T>> {
   public:
    Dec_ML(const Code<T>& C) : C(C) {}

    Vector<T> operator()(const Vector<T>& in) noexcept { return C.dec_ML(in); }

   private:
    const Code<T>& C;
};

template <class T>
class Dec_ML_soft : public Block<Vector<double>, Vector<T>> {
   public:
    Dec_ML_soft(const Code<T>& C) : C(C) {}

    Vector<T> operator()(const Vector<double>& in) noexcept { return C.dec_ML_soft(in); }

   private:
    const Code<T>& C;
};

class Dec_RM : public Block<Vector<Fp<2>>> {
   public:
    Dec_RM(const Code<Fp<2>>& C) : C(C) {}

    Vector<Fp<2>> operator()(const Vector<Fp<2>>& in) override { return C.dec_Dumer(in); }

   private:
    const Code<Fp<2>>& C;
};

class Dec_RM_soft : public Block<Vector<double>, Vector<Fp<2>>> {
   public:
    Dec_RM_soft(const Code<Fp<2>>& C) : C(C) {}

    Vector<Fp<2>> operator()(const Vector<double>& in) noexcept override { return C.dec_Dumer_soft(in); }

   private:
    const Code<Fp<2>>& C;
};

template <class T>
class Encoder_Inverse : public Block<Vector<T>> {
   public:
    Encoder_Inverse(const Code<T>& C) : C(C) {}

    Vector<T> operator()(const Vector<T>& in) noexcept { return C.encinv(in); }

   private:
    const Code<T>& C;
};

}  // namespace ECC

#endif
