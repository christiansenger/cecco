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
    - rvalue unary operator-
*/

#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <utility>
#include <vector>

#include "fields.hpp"
#include "helpers.hpp"
#include "matrices.hpp"

namespace ECC {

template <class T>
class Polynomial;
template <class T>
bool operator==(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept;
template <class T>
bool operator!=(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept;
template <class T>
Polynomial<T> derivative(const Polynomial<T>& poly, size_t s);
template <class T>
Polynomial<T> derivative(Polynomial<T>&& poly, size_t s);
template <class T>
Polynomial<T> reciprocal(const Polynomial<T>& poly);
template <class T>
Polynomial<T> reciprocal(Polynomial<T>&& poly);
template <class T>
std::ostream& operator<<(std::ostream& os, const Polynomial<T>& rhs) noexcept;

template <class T>
class Polynomial {
    friend bool operator== <>(const Polynomial& lhs, const Polynomial& rhs) noexcept;
    friend bool operator!= <>(const Polynomial& lhs, const Polynomial& rhs) noexcept;
    friend Polynomial derivative<>(const Polynomial& poly, size_t s);
    friend Polynomial derivative<>(Polynomial&& poly, size_t s);
    friend Polynomial reciprocal<>(const Polynomial& poly);
    friend Polynomial reciprocal<>(Polynomial&& poly);
    friend std::ostream& operator<< <>(std::ostream& os, const Polynomial& rhs) noexcept;

   public:
    /* constructors */
    Polynomial() noexcept : data(0) {}
    Polynomial(int e) noexcept : data(1) { data.back() = T(e); }
    Polynomial(const T& e) noexcept : data(1) { data.back() = e; }
    Polynomial(const Vector<T>& v) noexcept;
    Polynomial(const std::initializer_list<T>& l) noexcept;
    Polynomial(const Polynomial& other) noexcept : data(other.data), cache(other.cache) {}
    template <class S>
    Polynomial(Polynomial&& other) noexcept requires (std::is_base_of_v<S, T>) : data(std::move(other.data)), cache(std::move(other.cache)) {}
    template <class S, typename = std::enable_if_t<std::is_base_of_v<S, T>>>
    Polynomial(const Polynomial<S>& other) noexcept;
    template <uint8_t m>
    Polynomial(const Polynomial<SF<T, m>>& other) noexcept;

    /* assignment operators */
    Polynomial& operator=(const T& rhs) noexcept;
    Polynomial& operator=(const Vector<T>& rhs) noexcept;
    Polynomial& operator=(const Polynomial& rhs) noexcept;
    Polynomial& operator=(Polynomial&& rhs) noexcept;

    /* non-modifying operations */
    Polynomial operator+() const noexcept { return *this; }
    Polynomial operator-() const noexcept;
    T operator()(const T& s) const;  // evaluation

    /* modifying operations */
    Polynomial& differentiate(size_t s);
    Polynomial& Hasse_differentiate(size_t s);

    /* operational assignments */
    Polynomial& operator+=(const Polynomial& rhs) noexcept;
    Polynomial& operator-=(const Polynomial& rhs) noexcept;
    Polynomial& operator*=(const Polynomial& rhs) noexcept;
    Polynomial& operator/=(const Polynomial& rhs);
    Polynomial& operator%=(const Polynomial& rhs);
    Polynomial& operator*=(const T& s) noexcept;
    Polynomial& operator/=(const T& s);

    /* randomization */
    void randomize(size_t d) noexcept;

    /* getters */
    size_t get_degree() const;
    size_t get_trailing_degree() const;
    size_t wH() noexcept { return cache("weight", *this, &Polynomial::calculateWeight); }

    /* coefficient access */
    void set_coeff(size_t i, const T& c);
    void add_to_coeff(const T& c, size_t i);
    T operator[](size_t i) const noexcept;

    /* properties */
    bool is_empty() const noexcept { return data.size() == 0; }
    bool is_zero() const {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is zero");
        return get_degree() == 0 && get_trailing_coefficient() == T(0);
    }
    bool is_one() const {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is one");
        return get_degree() == 0 && get_trailing_coefficient() == T(1);
    }
    bool is_monic() const {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is monic");
        return get_leading_coefficient() == T(1);
    }
    const T& get_trailing_coefficient() const;
    const T& get_leading_coefficient() const;

   private:
    std::vector<T> data;
    Cache<Polynomial, size_t> cache;

    size_t calculateWeight() noexcept;

    void prune() noexcept;
};

/* member functions for Polynomial */

template <class T>
inline Polynomial<T>::Polynomial(const Vector<T>& v) noexcept : data(v.get_n()) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = v[i];
    }
    prune();
}

template <class T>
inline Polynomial<T>::Polynomial(const std::initializer_list<T>& l) noexcept : data(l) {
    // std::reverse(data.begin(), data.end());
    prune();
}

template <class T>
template <class S, typename>
inline Polynomial<T>::Polynomial(const Polynomial<S>& other) noexcept {
    Vector<S> coefficients(other);
    auto M = coefficients.template as_matrix<T>();
    // std::cout << M << std::endl;
    bool breakflag = false;
    for (size_t i = 1; i < M.get_m(); ++i) {
        for (size_t j = 0; j < M.get_n(); ++j) {
            if (M(i, j) != T(0)) {
                std::cout << "Warning: data loss while converting superfield polynomial to (prime) subfield polynomial!"
                          << std::endl;
                breakflag = true;
                break;
            }
        }
        if (breakflag) break;
    }
    auto firstrow = M.get_row(0);
    *this = Polynomial(firstrow);
}

template <class T>
template <uint8_t m>
inline Polynomial<T>::Polynomial(const Polynomial<SF<T, m>>& other) noexcept {
    for (size_t i = 0; i <= other.get_degree(); ++i) {
        data.push_back(T(other[i]));
    }
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator=(const T& rhs) noexcept {
    data.resize(1);
    data.back() = rhs;

    if (rhs != T(0)) {
        cache.set("weight", (size_t)1);
    } else {
        cache.set("weight", (size_t)0);
    }
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator=(const Vector<T>& rhs) noexcept {
    data.resize(rhs.get_degree() + 1);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = rhs[i];
    }
    cache.invalidate();
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator=(const Polynomial<T>& rhs) noexcept {
    if (*this == rhs) return *this;
    data = rhs.data;
    cache = rhs.cache;
    // std::cout << "polynomial copy assignment" << std::endl;
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator=(Polynomial<T>&& rhs) noexcept {
    if (*this == rhs) return *this;
    data = std::move(rhs.data);
    cache = std::move(rhs.cache);
    // std::cout << "polynomial move assignment" << std::endl;
    return *this;
}

template <class T>
inline Polynomial<T> Polynomial<T>::operator-() const noexcept {
    Polynomial res(*this);
    std::for_each(res.data.begin(), res.data.end(), [](T& v) { v = -v; });
    return res;
}

template <class T>
inline T Polynomial<T>::operator()(const T& s) const {
    if (data.size() == 0) throw std::invalid_argument("trying to evaluate empty polynomial");

    if (data.size() == 1) return data.front();

    T value(data.back() * s);
    for (auto it = data.crbegin() + 1; it != std::prev(data.crend()); ++it) {
        value += *it;
        value *= s;
    }
    value += data.front();

    return value;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::differentiate(size_t s) {
    if (data.size() == 0) throw std::invalid_argument("trying to differentiate empty polynomial");

    if (s == 0) return *this;
    const size_t d = data.size() - 1;
    if (d == 0 || s > d) {
        data.resize(1);
        data[0] = T(0);
        return *this;
    }
    for (size_t i = 0; i <= d - s; ++i) {
        data[i] = fac<size_t>(i + s) / fac<size_t>(i) * data[i + s];
    }
    data.resize(data.size() - s);
    prune();
    cache.invalidate();
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::Hasse_differentiate(size_t s) {
    if (data.size() == 0) throw std::invalid_argument("trying to Hasse differentiate empty polynomial");

    if (s == 0) return *this;
    const size_t d = data.size() - 1;
    if (d == 0 || s > d) {
        data.resize(1);
        data[0] = T(0);
        return *this;
    }
    for (size_t i = 0; i <= d - s; ++i) {
        data[i] = bin<size_t>(i + s, s) * data[i + s];
    }
    data.resize(data.size() - s);
    prune();
    cache.invalidate();
    return *this;

    /*
    if (data.size() == 1) {
        data[0] = T(0);
    } else {
        for (size_t i = 0; i < s; ++i) {
            if (data.size() == 1) {
                data[0] = T(0);
                break;
            }
            data.erase(data.begin());
            for (size_t j = 0; j < data.size(); ++j) {
                data[j] *= j + 1;
            }
        }
    }
    cache.invalidate();
    return *this;
    */
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator+=(const Polynomial<T>& rhs) noexcept {
    if (data.size() < rhs.data.size()) {
        data.resize(rhs.data.size());
    }
    for (size_t i = 0; i < rhs.data.size(); ++i) {
        data[i] += rhs.data[i];
    }
    cache.invalidate();
    prune();
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator-=(const Polynomial<T>& rhs) noexcept {
    if (data.size() < rhs.data.size()) {
        data.resize(rhs.data.size());
    }
    for (size_t i = 0; i < rhs.data.size(); ++i) {
        data[i] -= rhs.data[i];
    }
    cache.invalidate();
    prune();
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator*=(const Polynomial<T>& rhs) noexcept {
    Polynomial res;
    res.data.resize(data.size() + rhs.data.size() - 1);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < rhs.data.size(); ++j) {
            res.add_to_coeff(data[i] * rhs.data[j], i + j);
        }
    }
    data = std::move(res.data);
    cache.invalidate();
    prune();  // ToDo: check if necessary!!
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator/=(const Polynomial<T>& rhs) {
    if (rhs.is_zero()) throw std::invalid_argument("division by zero (polynomial)");
    auto res = polynomialLongDivision(*this, rhs);
    *this = res.first;
    cache.invalidate();
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator%=(const Polynomial<T>& rhs) {
    if (rhs.is_zero()) throw std::invalid_argument("division by zero (polynomial)");

    if (get_degree() == rhs.get_degree()) {  // simply subtract polynomial modulus
        const size_t degree = get_degree();
        const auto scalar = get_leading_coefficient() / rhs.get_leading_coefficient();
        for (size_t i = 0; i < get_degree(); ++i) {
            data[i] -= scalar * rhs[i];
        }
        set_coeff(get_degree(), 0);
    } else {  // full-blown polynomial long division
        auto res = polynomialLongDivision(*this, rhs);
        *this = res.second;
        cache.invalidate();
    }

    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator*=(const T& s) noexcept {
    if (s == T(0)) {
        cache.set("weight", (size_t)0);
        data.resize(1);
        data.back() = T(0);
    } else {
        std::for_each(data.begin(), data.end(), [&s](T& v) { v *= s; });
    }
    return *this;
}

template <class T>
inline Polynomial<T>& Polynomial<T>::operator/=(const T& s) {
    if (s == T(0)) throw std::invalid_argument("division by zero (polynomial)");
    std::for_each(data.begin(), data.end(), [&s](T& v) { v /= s; });
    return *this;
}

template <class T>
inline void Polynomial<T>::randomize(size_t d) noexcept {
    static std::uniform_int_distribution<int> dist(0, d);
    data.resize(d + 1);
    std::for_each(data.begin(), data.end() - 1, std::mem_fn(&T::randomize));
    do {
        data.back().randomize();
    } while (data.back() == T(0));
    cache.invalidate();
}

template <class T>
inline size_t Polynomial<T>::get_degree() const {
    if (is_empty()) throw std::invalid_argument("calculating degree of empty polynomial");
    return (data.size() - 1);
}

template <class T>
inline size_t Polynomial<T>::get_trailing_degree() const {
    if (is_empty()) throw std::invalid_argument("calculating trailing degree of empty polynomial");
    const size_t d = get_degree();
    if (d == 0) return 0;
    size_t td = 0;
    for (size_t i = 0; i <= d; ++i) {
        if (data[i] == T(0))
            ++td;
        else
            break;
    }
    return td;
}

template <class T>
inline void Polynomial<T>::set_coeff(size_t i, const T& c) {
    cache.invalidate();

    if (i == data.size() - 1) {
        data.back() = c;
        if (c == T(0)) prune();
        return;
    }

    if (data.size() != 0 && i < data.size() - 1) {
        data[i] = c;
        return;
    }

    if (c != T(0)) {
        data.resize(i + 1);
        data.back() = c;
    }
}

template <class T>
inline void Polynomial<T>::add_to_coeff(const T& c, size_t i) {
    if (c == T(0)) return;

    cache.invalidate();

    if (i == data.size() - 1) {
        data.back() += c;
        if (data.back() == T(0)) prune();
        return;
    }

    if (data.size() != 0 && i < data.size() - 1) {
        data[i] += c;
        return;
    }

    data.resize(i + 1);
    data.back() = c;
}

template <class T>
inline T Polynomial<T>::operator[](size_t i) const noexcept {
    // if (i >= data.size()) throw std::invalid_argument("trying to access non-existent element");
    if (i >= data.size()) return T(0);
    return data[i];
}

template <class T>
inline const T& Polynomial<T>::get_trailing_coefficient() const {
    if (is_empty())
        throw std::invalid_argument(
            "trying to access non-existent element (trailing "
            "coefficient)");
    return data[get_trailing_degree()];
}

template <class T>
inline const T& Polynomial<T>::get_leading_coefficient() const {
    if (is_empty())
        throw std::invalid_argument(
            "trying to access non-existent element (leading "
            "coefficient)");
    return data.back();
}

template <class T>
inline size_t Polynomial<T>::calculateWeight() noexcept {
    return data.size() - std::count(data.cbegin(), data.cend(), T(0));
}

template <class T>
inline void Polynomial<T>::prune() noexcept {
    if (data.size() == 0) return;

    const auto leading_coefficient = std::find_if(data.crbegin(), data.crend(), [](const T& e) { return e != T(0); });
    if (leading_coefficient != data.crend()) {
        data.resize(data.size() - std::distance(data.crbegin(), leading_coefficient));
    } else {
        data.resize(1);
        data.back() = T(0);
    }
}

/* free functions wrt. Polynomial */

/*
 * polynomial + polynomial
 */

template <class T>
Polynomial<T> operator+(Polynomial<T> lhs, const Polynomial<T>& rhs) noexcept {
    lhs += rhs;
    return lhs;
}

template <class T>
Polynomial<T> operator+(const Polynomial<T>& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(rhs));
    res += lhs;
    return res;
}

/*
 * polynomial - polynomial
 */

template <class T>
Polynomial<T> operator-(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(lhs);
    res -= rhs;
    return res;
}

template <class T>
Polynomial<T> operator-(const Polynomial<T>& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(rhs));
    res -= lhs;
    return -res;
}

template <class T>
Polynomial<T> operator-(Polynomial<T>&& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

template <class T>
Polynomial<T> operator-(Polynomial<T>&& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

/*
 * polynomial * polynomial
 */

template <class T>
Polynomial<T> operator*(Polynomial<T> lhs, const Polynomial<T>& rhs) noexcept {
    lhs *= rhs;
    return lhs;
}

template <class T>
Polynomial<T> operator*(const Polynomial<T>& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(rhs));
    res *= lhs;
    return res;
}

/*
 * polynomial * N
 */

template <class T>
Polynomial<T> operator*(const Polynomial<T>& lhs, size_t n) noexcept {
    if (T::get_characteristic() != 0) {
        n = n % T::get_characteristic();
    }
    Polynomial<T> res = {0};
    for (size_t i = 0; i < n; ++i) {
        res += lhs;
    }
    return res;
}

/*
 * N * polynomial
 */

template <class T>
Polynomial<T> operator*(size_t n, const Polynomial<T>& rhs) noexcept {
    return rhs * n;
}

/*
 * polynomial * T
 */

template <class T>
Polynomial<T> operator*(const Polynomial<T>& lhs, const T& rhs) noexcept {
    Polynomial<T> res(lhs);
    res *= rhs;
    return res;
}

template <class T>
Polynomial<T> operator*(const Polynomial<T>& lhs, T&& rhs) noexcept {
    Polynomial<T> res(lhs);
    res *= rhs;
    return res;
}

template <class T>
Polynomial<T> operator*(Polynomial<T>&& lhs, const T& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

template <class T>
Polynomial<T> operator*(Polynomial<T>&& lhs, T&& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

/*
 * T * polynomial
 */

template <class T>
Polynomial<T> operator*(const T& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(rhs);
    res *= lhs;
    return res;
}

template <class T>
Polynomial<T> operator*(const T& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(rhs));
    res *= lhs;
    return res;
}

template <class T>
Polynomial<T> operator*(T&& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(rhs);
    res *= lhs;
    return res;
}

template <class T>
Polynomial<T> operator*(T&& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(rhs));
    res *= lhs;
    return res;
}

/*
 * polynomial / polynomial
 */

template <class T>
Polynomial<T> operator/(Polynomial<T> lhs, const Polynomial<T>& rhs) noexcept {
    lhs /= rhs;
    return lhs;
}

/*
 * polynomial % polynomial
 */

template <class T>
Polynomial<T> operator%(Polynomial<T> lhs, const Polynomial<T>& rhs) noexcept {
    lhs %= rhs;
    return lhs;
}

/*
 * polynomial / T
 */

template <class T>
Polynomial<T> operator/(const Polynomial<T>& lhs, const T& rhs) noexcept {
    Polynomial<T> res(lhs);
    res /= rhs;
    return res;
}

template <class T>
Polynomial<T> operator/(Polynomial<T>&& lhs, const T& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res /= rhs;
    return res;
}

/*
 * polynomial polylongdiv polynomial
 */

template <class T>
std::pair<Polynomial<T>, Polynomial<T>> polynomialLongDivision(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    if (rhs.get_degree() == 0) {
        if (rhs[0] == T(0)) throw std::invalid_argument("polynomial long division by zero polynomial");
        return std::make_pair(lhs / rhs[0], Polynomial<T>());
    }

    if (lhs.get_degree() == 0) return std::make_pair(Polynomial<T>(), Polynomial<T>());

    if (lhs.get_degree() < rhs.get_degree()) return std::make_pair(Polynomial<T>(), rhs);

    Polynomial<T> q;
    Polynomial<T> r(lhs);

    while (r.get_degree() >= rhs.get_degree()) {
        T t = r[r.get_degree()] / rhs[rhs.get_degree()];
        // std::cout << r[r.get_degree()] << ", " << rhs[rhs.get_degree()] << ": " << t << std::endl;
        size_t i = r.get_degree() - rhs.get_degree();
        q.add_to_coeff(t, i);
        for (size_t j = 0; j <= rhs.get_degree(); ++j) {
            r.add_to_coeff(-(t * rhs)[j], i + j);
        }
    }

    return std::make_pair(std::move(q), std::move(r));
}

template <class T>
std::pair<Polynomial<T>, Polynomial<T>> polynomialLongDivision(Polynomial<T>&& lhs, const Polynomial<T>& rhs) {
    if (rhs.get_degree() == 0) {
        if (rhs[0] == T(0)) throw std::invalid_argument("polynomial long division by zero polynomial");
        return std::make_pair(lhs / rhs[0], Polynomial<T>());
    }

    if (lhs.get_degree() == 0) return std::make_pair(Polynomial<T>(), Polynomial<T>());

    if (lhs.get_degree() < rhs.get_degree()) return std::make_pair(Polynomial<T>(), rhs);

    Polynomial<T> q;
    Polynomial<T> r(std::move(lhs));

    while (r.get_degree() >= rhs.get_degree()) {
        T t = r[r.get_degree()] / rhs[rhs.get_degree()];

        size_t i = r.get_degree() - rhs.get_degree();
        q.add_to_coeff(t, i);
        for (size_t j = 0; j <= rhs.get_degree(); ++j) {
            r.add_to_coeff(-(t * rhs)[j], i + j);
        }
    }

    return std::make_pair(std::move(q), std::move(r));
}

/*
 * Careful: This seemingly convenient overload is actually pretty
 * dangerous:
 *
 * The usual operator precedence is violated, i.e., b*a^p is (wrongly)
 * evaluated as (b*a)^a instead of the expected b*(a^p).
 */
template <class T>
Polynomial<T> operator^(const Polynomial<T>& base, int exponent) noexcept {
    return sqm<Polynomial<T>>(base, exponent);
}

template <class T>
Polynomial<T> derivative(const Polynomial<T>& poly, size_t s) {
    Polynomial<T> res(poly);
    res.differentiate(s);
    return res;
}

template <class T>
Polynomial<T> derivative(Polynomial<T>&& poly, size_t s) {
    Polynomial<T> res(std::move(poly));
    res.differentiate(s);
    return res;
}

template <class T>
Polynomial<T> Hasse_derivative(const Polynomial<T>& poly, size_t s) {
    Polynomial<T> res(poly);
    res.Hasse_differentiate(s);
    return res;
}

template <class T>
Polynomial<T> Hasse_derivative(Polynomial<T>&& poly, size_t s) {
    Polynomial<T> res(std::move(poly));
    res.Hasse_differentiate(s);
    return res;
}

template <class T>
Polynomial<T> reciprocal(const Polynomial<T>& poly) {
    Polynomial res(poly);
    std::reverse(res.data.begin(), res.data.end());
    return res;
}

template <class T>
Polynomial<T> reciprocal(Polynomial<T>&& poly) {
    Polynomial res(std::move(poly));
    std::reverse(res.data.begin(), res.data.end());
    return res;
}

template <class T>
Polynomial<T> normalize(const Polynomial<T>& poly) {
    if (poly.is_zero() || poly.is_monic()) return poly;
    Polynomial res = poly / poly.get_leading_coefficient();
    return res;
}

template <class T>
Polynomial<T> normalize(Polynomial<T>&& poly) {
    if (poly.is_zero() || poly.is_monic()) return std::move(poly);
    Polynomial res(std::move(poly));
    res /= res.get_leading_coefficient();
    return res;
}

template <class T>
bool operator==(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept {
    return lhs.data == rhs.data;
}

template <class T>
bool operator!=(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept {
    return lhs.data != rhs.data;
}

template <class T>
Polynomial<T> monomial(size_t i) noexcept {
    Polynomial<T> res;
    res.set_coeff(i, T(1));
    return res;
}

template <class T>
Polynomial<T> ZeroPolynomial() noexcept {
    return Polynomial<T>(0);
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Polynomial<T>& rhs) noexcept {
    if (rhs.data.size() == 0) {
        os << "()";
        return os;
    }

    bool next_negative = false;
    for (size_t i = 0; i < rhs.data.size(); ++i) {
        auto coeff = rhs.data[i];
        if (next_negative) coeff *= -T(1);
        if (coeff != T(0) || rhs.data.size() == 1) {
            if (coeff != T(1) || i == 0) {
                os << coeff;
            }
            if (i > 0) {
                os << "x";
                if (i > 1) os << "^" << i;
            }
            if (i < rhs.data.size() - 1) {
                // if (rhs.data[i + 1].has_positive_sign()) {
                if (true) {  // todo
                    next_negative = false;
                    os << " + ";
                } else {
                    next_negative = true;
                    os << " - ";
                }
            }
        }
    }

    return os;
}

// extended Euclidean algorithm
template <class T>
Polynomial<T> GCD(Polynomial<T> a, Polynomial<T> b, Polynomial<T>* s = nullptr, Polynomial<T>* t = nullptr) noexcept {
    if (a.get_degree() < b.get_degree()) {
        auto temp = a;
        a = b;
        b = temp;
    }

    if (s != nullptr && t != nullptr) {  // extended EA
        *s = Polynomial<T>({1});
        *t = Polynomial<T>({0});
        Polynomial<T> u = Polynomial<T>({0});
        Polynomial<T> v = Polynomial<T>({1});
        // while (b.get_degree() > 0) {
        while (b != Polynomial<T>(0)) {  // changed 25.07.2023, verify!
            const Polynomial<T> q = a / b;
            Polynomial<T> b1 = b;
            b = a - q * b;
            a = std::move(b1);
            Polynomial<T> u1 = u;
            u = *s - q * u;
            *s = std::move(u1);
            Polynomial<T> v1 = v;
            v = *t - q * v;
            *t = std::move(v1);
        }
    } else {                             // "normal" EA
                                         // while (b.get_degree() > 0) {
        while (b != Polynomial<T>(0)) {  // changed 25.07.2023, verify!
            const Polynomial<T> q = a / b;
            Polynomial<T> b1 = b;
            b = a - q * b;
            a = std::move(b1);
        }
    }
    return a;
}

template <class T>
Polynomial<T> GCD(const std::vector<Polynomial<T>>& polys) {
    Polynomial<T> res = polys.front();
    for (auto it = polys.cbegin() + 1; it != polys.cend(); ++it) {
        res = GCD<T>(res, *it);
    }
    return res;
}

template <class T>
Polynomial<T> LCM(Polynomial<T> a, Polynomial<T> b) noexcept {
    return (a * b) / GCD(a, b);
}

template <class T>
Polynomial<T> LCM(const std::vector<Polynomial<T>>& polys) {
    Polynomial<T> res = polys.front();
    for (auto it = polys.cbegin() + 1; it != polys.cend(); ++it) {
        res = LCM<T>(res, *it);
    }
    return res;
}

}  // namespace ECC

#endif
