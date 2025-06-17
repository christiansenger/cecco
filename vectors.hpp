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

#ifndef VECTORS_HPP
#define VECTORS_HPP

#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <vector>
#include <complex>

#include "fields.hpp"
#include "helpers.hpp"

namespace ECC {

class Base;

template <class S, uint8_t m>
class SF;
template <class T>
class Matrix;
template <class T>
class Vector;
template <class T>
class Polynomial;
template <class T>
bool operator==(const Vector<T>& lhs, const Vector<T>& rhs);
template <class T>
bool operator!=(const Vector<T>& lhs, const Vector<T>& rhs);
template <class T>
Vector<T> unit_vector(size_t length, size_t i);
template <class T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& rhs) noexcept;
template <class T>
size_t wH(Vector<T>& v) noexcept;
template <class T>
size_t dH(const Vector<T>& lhs, const Vector<T>& rhs);
double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs);

template <class T>
class Vector {
    friend bool operator== <>(const Vector& lhs, const Vector& rhs);
    friend bool operator!= <>(const Vector& lhs, const Vector& rhs);
    friend Vector unit_vector<>(size_t length, size_t i);
    friend std::ostream& operator<< <>(std::ostream& os, const Vector& rhs) noexcept;
    friend size_t wH<>(Vector& v) noexcept;
    friend size_t dH<>(const Vector& lhs, const Vector& rhs);
    friend double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs);

   public:
    /* constructors */
    Vector() noexcept : data(0) {}
    Vector(size_t n) noexcept : data(n) {}
    Vector(size_t n, const T& l) noexcept : data(n) { std::fill(data.begin(), data.end(), l); }
    Vector(const std::initializer_list<T>& l) noexcept : data(l) {}
    Vector(const Vector& other) noexcept;
    template <class S, typename = std::enable_if_t<std::is_base_of_v<S, T>>>
    Vector(Vector&& other) noexcept;
    template <class S, typename = std::enable_if_t<std::is_base_of_v<T, S>>>
    Vector(Matrix<S> mat);
    template <class S, typename = std::enable_if_t<std::is_base_of_v<S, T>>>
    Vector(const Vector<S>& other, size_t ell = 0) noexcept;
    template <uint8_t m>
    Vector(const Vector<SF<T, m>>& other) noexcept;
    Vector(const Polynomial<T>& poly) noexcept;

    /* assignment operators */
    Vector& operator=(const Vector& rhs) noexcept;
    Vector& operator=(Vector&& rhs) noexcept;
    Vector& operator=(const T& rhs) noexcept;

    /* non-modifying operations */
    Vector operator+() const noexcept { return *this; }
    Vector operator-() const noexcept;

    /* modifying operations */
    Vector& operator+=(const Vector& rhs);
    Vector& operator-=(const Vector& rhs);
    Vector& operator*=(const T& s) noexcept;
    Vector& operator/=(const T& s);
    Vector& concatenate(const Vector& rhs) noexcept;
    Vector& delete_components(std::vector<size_t> v);

    /* randomization */
    template <bool b = std::is_base_of_v<Base, T>>
    void randomize() noexcept requires (b) {
        std::for_each(data.begin(), data.end(), std::mem_fn(&T::randomize));
        cache.invalidate();
    }

    /* getters */
    size_t get_n() const noexcept { return data.size(); }
    size_t wH() noexcept { return cache("weight", *this, &Vector::calculateWeight); }
    size_t burst_length() const noexcept;
    size_t cyclic_burst_length() const noexcept;

    /* component access */
    void set_component(size_t i, const T& c);
    const T& operator[](size_t i) const;
    Vector get_subvector(size_t i, size_t w) const&;
    Vector& get_subvector(size_t i, size_t w) &&;
    Vector& set_subvector(const Vector& v, size_t i);

    /* rotations */
    Vector& rotateLeft(size_t i) noexcept;
    Vector& rotateRight(size_t i) noexcept;

    /* reverse */
    Vector& reverse() noexcept;

    /* properties */
    bool is_empty() const noexcept { return data.size() == 0; }
    bool is_zero() const noexcept;
    bool is_pairwisedistinct() const noexcept;

    /* interpret as integer */
    // template <class S, class U = T, typename std::enable_if<!std::is_arithmetic<U>::value>::type>
    template <bool b = is_finite_field<T>()>
    size_t asInteger() const noexcept requires (b) {
        size_t res = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            res += data[data.size() - i - 1].get_label() * sqm<size_t>(T::get_size(), i);
        }
        return res;
    }

    /* vector as matrix over subfield */
    // template <class S, class U = T, typename std::enable_if<!std::is_arithmetic<U>::value>::type>
    template <class S, bool b = is_finite_field<T>()>
    Matrix<S> as_matrix() const noexcept requires (b) {
        Matrix<S> res(T::get_m() / S::get_m(), data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            Matrix<S> temp(T(data[i]).template as_vector<S>());
            temp.transpose();
            res.set_submatrix(temp, 0, i);
        }
        return res;
    }

   private:
    std::vector<T> data;
    Cache<Vector, size_t> cache;

    size_t calculateWeight();

    T& at(size_t i);
};

/* member functions for Vector */

template <class T>
inline Vector<T>::Vector(const Vector<T>& other) noexcept : data(other.data), cache(other.cache) {
    // std::cout << "vector copy" << std::endl;
}

template <class T>
template <class S, typename>
inline Vector<T>::Vector(Vector<T>&& other) noexcept : data(std::move(other.data)), cache(std::move(other.cache)) {
    // std::cout << "vector move" << std::endl;
}

template <class T>
template <class S, typename>
inline Vector<T>::Vector(Matrix<S> mat) {
    if (T::get_m() % mat.get_m() != 0)
        throw std::invalid_argument("trying to construct base field vector from subfield matrix of incompatible size");
    mat.transpose();
    Vector<T> res(mat.get_m());
    for (size_t i = 0; i < mat.get_m(); ++i) {
        res.set_component(i, T(mat.get_row(i)));
    }
    data = res.data;
    cache = res.cache;
}

template <class T>
template <class S, typename>
inline Vector<T>::Vector(const Vector<S>& other, size_t ell) noexcept {
    data.resize(other.get_n());
    for (size_t i = 0; i < other.get_n(); ++i) {
        auto temp = other[i].template as_vector<T>();
        data[i] = temp[ell];
        for (size_t j = 0; j < T::get_m() / S::get_m(); ++j) {
            if (j == ell) continue;
            if (temp[j] != T(0)) {
                std::cout << "Warning: data loss while converting superfield vector to (prime) subfield vector!"
                          << std::endl;
                break;
            }
        }
    }
}

template <class T>
template <uint8_t m>
inline Vector<T>::Vector(const Vector<SF<T, m>>& other) noexcept {
    data.resize(other.get_n());
    for (size_t i = 0; i < other.get_n(); ++i) {
        data[i] = T(other[i]);
    }
}

template <class T>
inline Vector<T>::Vector(const Polynomial<T>& poly) noexcept {
    Vector<T> res(poly.get_degree() + 1);
    for (size_t i = 0; i <= poly.get_degree(); ++i) {
        res.set_component(i, poly[i]);
    }
    data = res.data;
    cache = res.cache;
}

template <class T>
inline Vector<T>& Vector<T>::operator=(const Vector<T>& rhs) noexcept {
    if (this->data.size() == rhs.data.size() && *this == rhs) return *this;
    data = rhs.data;
    cache = rhs.cache;
    // std::cout << "vector copy assignment" << std::endl;
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::operator=(Vector<T>&& rhs) noexcept {
    if (this->data.size() == rhs.data.size() && *this == rhs) return *this;
    data = std::move(rhs.data);
    cache = std::move(rhs.cache);
    // std::cout << "vector move assignment" << std::endl;
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::operator=(const T& rhs) noexcept {
    std::fill(data.begin(), data.end(), rhs);
    cache.invalidate();
    return *this;
}

template <class T>
inline Vector<T> Vector<T>::operator-() const noexcept {
    Vector res(*this);
    std::for_each(res.data.begin(), res.data.end(), [](T& v) { v = -v; });
    return res;  // move elision
}

template <class T>
inline Vector<T>& Vector<T>::operator+=(const Vector<T>& rhs) {
    if (data.size() != rhs.data.size()) throw std::invalid_argument("trying to add two vectors of different lengths");
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += rhs.data[i];
    }
    cache.invalidate();
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::operator-=(const Vector<T>& rhs) {
    if (data.size() != rhs.data.size())
        throw std::invalid_argument(
            "trying to subtract two vectors of different "
            "lengths");
    ;
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= rhs.data[i];
    }
    cache.invalidate();
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::operator*=(const T& s) noexcept {
    if (s == T(0)) {
        std::fill(data.begin(), data.end(), T(0));
        cache.set("weight", (size_t)0);
    } else {
        std::for_each(data.begin(), data.end(), [&s](T& v) { v *= s; });
    }
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::operator/=(const T& s) {
    if (s == T(0)) throw std::invalid_argument("trying to divide components of vector by zero");
    std::for_each(data.begin(), data.end(), [&s](T& v) { v /= s; });
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::concatenate(const Vector<T>& rhs) noexcept {
    Vector temp(data.size() + rhs.data.size());
    temp.set_subvector(*this, 0);
    temp.set_subvector(rhs, data.size());
    data = std::move(temp.data);
    cache.invalidate();
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::delete_components(std::vector<size_t> v) {
    std::sort(v.begin(), v.end(), std::greater<>());
    for (auto it = v.cbegin(); it != v.cend(); ++it) {
        if (*it >= data.size()) throw std::invalid_argument("trying to delete non-existent component");
        Vector left = this->get_subvector(0, *it);
        Vector right = this->get_subvector(*it + 1, data.size() - (*it + 1));
        (*this) = left.concatenate(right);
    }
    return *this;
}

template <class T>
inline size_t Vector<T>::burst_length() const noexcept {
    size_t L = data.size();
    size_t R;
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] != T(0)) {
            L = i;
            break;
        }
    }
    if (L == data.size()) return 0;
    for (size_t i = data.size() - 1; i >= L; --i) {
        if (data[i] != T(0)) {
            R = i;
            break;
        }
    }
    return R - L + 1;
}

template <class T>
inline size_t Vector<T>::cyclic_burst_length() const noexcept {
    auto b = burst_length();
    return std::min(b, data.size() - b + 2);
}

template <class T>
inline void Vector<T>::set_component(size_t i, const T& c) {
    if (i >= data.size()) throw std::invalid_argument("trying to access non-existent element");
    data[i] = c;
    if (c != T(0)) cache.invalidate();
}

template <class T>
inline const T& Vector<T>::operator[](size_t i) const {
    if (i >= data.size()) throw std::invalid_argument("trying to access non-existent element");
    return data[i];
}

template <class T>
inline Vector<T> Vector<T>::get_subvector(size_t i, size_t w) const& {
    if (i + w > data.size())
        throw std::invalid_argument(
            "trying to extract a subvector with incompatible "
            "length");
    Vector res(w);
    for (size_t j = 0; j < w; ++j) {
        res.data[j] = data[i + j];
    }
    return res;
}

template <class T>
inline Vector<T>& Vector<T>::get_subvector(size_t i, size_t w) && {
    if (i + w > data.size())
        throw std::invalid_argument(
            "trying to extract a subvector with incompatible "
            "length");
    data.resize(i + w);
    data.erase(data.begin(), data.begin() + i);
    cache.invalidate();
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::set_subvector(const Vector& v, size_t i) {
    if (i + v.get_n() > data.size())
        throw std::invalid_argument(
            "trying to replace subvector with "
            "vector of incompatible length");
    for (size_t j = 0; j < v.get_n(); ++j) {
        data[i + j] = v.data[j];
    }
    cache.invalidate();
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::rotateLeft(size_t i) noexcept {
    std::rotate(data.begin(), data.begin() + i, data.end());
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::rotateRight(size_t i) noexcept {
    std::rotate(data.rbegin(), data.rbegin() + i, data.rend());
    return *this;
}

template <class T>
inline Vector<T>& Vector<T>::reverse() noexcept {
    std::reverse(data.begin(), data.end());
    return *this;
}

template <class T>
inline bool Vector<T>::is_zero() const noexcept {
    for (auto it = data.cbegin(); it != data.cend(); ++it) {
        if (*it != T(0)) return false;
    }
    return true;
}

template <class T>
inline bool Vector<T>::is_pairwisedistinct() const noexcept {
    auto S = data;
    std::ranges::sort(S, [](const T& a, const T& b) { return a.get_label() < b.get_label(); });
    auto last = std::unique(S.begin(), S.end());
    S.erase(last, S.end());
    return data.size() == S.size();
}

template <class T>
inline size_t Vector<T>::calculateWeight() {
    return data.size() - std::count(data.cbegin(), data.cend(), T(0));
}

template <class T>
inline T& Vector<T>::at(size_t i) {
    if (i >= data.size()) throw std::invalid_argument("trying to access non-existent element");
    cache.invalidate();
    return data[i];
}

/* free functions wrt. Vector */

/*
 * vector + vector
 */

template <class T>
Vector<T> operator+(Vector<T> lhs, const Vector<T>& rhs) noexcept {
    lhs += rhs;
    return lhs;
}

template <class T>
Vector<T> operator+(const Vector<T>& lhs, Vector<T>&& rhs) noexcept {
    Vector res(std::move(rhs));
    res += lhs;
    return res;
}

/*
 * vector - vector
 */

template <class T>
Vector<T> operator-(const Vector<T>& lhs, const Vector<T>& rhs) noexcept {
    Vector res(lhs);
    res -= rhs;
    return res;
}

template <class T>
Vector<T> operator-(const Vector<T>& lhs, Vector<T>&& rhs) noexcept {
    Vector res(std::move(rhs));
    res -= lhs;
    return -res;
}

template <class T>
Vector<T> operator-(Vector<T>&& lhs, const Vector<T>& rhs) noexcept {
    Vector res(std::move(lhs));
    res -= rhs;
    return res;
}

template <class T>
Vector<T> operator-(Vector<T>&& lhs, Vector<T>&& rhs) noexcept {
    Vector res(std::move(lhs));
    res -= rhs;
    return res;
}

/*
 * vector * T
 */

template <class T>
Vector<T> operator*(const Vector<T>& lhs, const T& rhs) noexcept {
    Vector res(lhs);
    res *= rhs;
    return res;
}

template <class T>
Vector<T> operator*(const Vector<T>& lhs, T&& rhs) noexcept {
    Vector res(lhs);
    res *= rhs;
    return res;
}

template <class T>
Vector<T> operator*(Vector<T>&& lhs, const T& rhs) noexcept {
    Vector res(std::move(lhs));
    res *= rhs;
    return res;
}

template <class T>
Vector<T> operator*(Vector<T>&& lhs, T&& rhs) noexcept {
    Vector res(std::move(lhs));
    res *= rhs;
    return res;
}

/*
 * T * vector
 */

template <class T>
Vector<T> operator*(const T& lhs, const Vector<T>& rhs) noexcept {
    Vector res(rhs);
    res *= lhs;
    return res;
}

template <class T>
Vector<T> operator*(const T& lhs, Vector<T>&& rhs) noexcept {
    Vector res(std::move(rhs));
    res *= lhs;
    return res;
}

template <class T>
Vector<T> operator*(T&& lhs, const Vector<T>& rhs) noexcept {
    Vector res(rhs);
    res *= lhs;
    return res;
}

template <class T>
Vector<T> operator*(T&& lhs, Vector<T>&& rhs) noexcept {
    Vector res(std::move(rhs));
    res *= lhs;
    return res;
}

/*
 * vector / T
 */

template <class T>
Vector<T> operator/(const Vector<T>& lhs, const T& rhs) noexcept {
    Vector res(lhs);
    res /= rhs;
    return res;
}

template <class T>
Vector<T> operator/(const Vector<T>& lhs, T&& rhs) noexcept {
    Vector res(lhs);
    res /= rhs;
    return res;
}

template <class T>
Vector<T> operator/(Vector<T>&& lhs, const T& rhs) noexcept {
    Vector res(std::move(lhs));
    res /= rhs;
    return res;
}

template <class T>
Vector<T> operator/(Vector<T>&& lhs, T&& rhs) noexcept {
    Vector res(std::move(lhs));
    res /= rhs;
    return res;
}

/*
 * vector innerproduct vector
 */

template <class T>
T innerProduct(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate inner product of two "
            "vectors of different lengths");
    T res(0);
    for (size_t i = 0; i < lhs.get_n(); ++i) {
        res += lhs[i] * rhs[i];
    }
    return res;
}

template <class T>
Vector<T> rotateLeft(const Vector<T>& v, size_t i) {
    Vector res(v);
    res.rotateLeft(i);
    return res;
}

template <class T>
Vector<T> rotateLeft(Vector<T>&& v, size_t i) {
    Vector res(std::move(v));
    res.rotateLeft(i);
    return res;
}

template <class T>
Vector<T> rotateRight(const Vector<T>& v, size_t i) {
    Vector res(v);
    res.rotateRight(i);
    return res;
}

template <class T>
Vector<T> rotateRight(Vector<T>&& v, size_t i) {
    Vector res(std::move(v));
    res.rotateRight(i);
    return res;
}

template <class T>
Vector<T> reverse(const Vector<T>& v) {
    Vector res(v);
    res.reverse();
    return res;
}

template <class T>
Vector<T> reverse(Vector<T>&& v) {
    Vector res(std::move(v));
    res.reverse();
    return res;
}

template <class T>
Vector<T> concatenate(const Vector<T>& lhs, const Vector<T>& rhs) {
    Vector res(lhs);
    res.concatenate(rhs);
    return res;
}

template <class T>
Vector<T> concatenate(Vector<T>&& lhs, const Vector<T>& rhs) {
    Vector res(std::move(lhs));
    res.concatenate(rhs);
    return res;
}

template <class T>
Vector<T> delete_components(const Vector<T>& lhs, std::vector<size_t> v) {
    Vector res(lhs);
    res.delete_components(v);
    return res;
}

template <class T>
Vector<T> delete_components(Vector<T>&& lhs, std::vector<size_t> v) {
    Vector res(std::move(lhs));
    res.delete_components(v);
    return res;
}

template <class T>
bool operator==(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.data.size() != rhs.data.size())
        throw std::invalid_argument(
            "trying to compare (==) two vectors of different "
            "lengths");
    return lhs.data == rhs.data;
}

template <class T>
bool operator!=(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.data.size() != rhs.data.size())
        throw std::invalid_argument(
            "trying to compare (!=) two vectors of different "
            "lengths");
    return lhs.data != rhs.data;
}

template <class T>
Vector<T> pad_front(const Vector<T>& v, size_t n) {
    if (n <= v.get_n()) return v;
    Vector<T> res(n);
    res.set_subvector(v, n - v.get_n());
    return res;
}

template <class T>
Vector<T> pad_back(const Vector<T>& v, size_t n) {
    if (n <= v.get_n()) return v;
    Vector<T> res(n);
    res.set_subvector(v, 0);
    return res;
}

template <class T>
Vector<T> unit_vector(size_t length, size_t i) {
    if (i >= length) throw std::invalid_argument("trying to create invalid unit vector");
    Vector<T> res(length);
    res.at(i) = T(1);
    res.cache.set("weight", (size_t)1);
    return res;  // move elision
}

template <class T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& rhs) noexcept {
    os << "( ";
    for (auto it = rhs.data.cbegin(); it != rhs.data.cend(); ++it) {
        std::cout << *it;
        if (it != rhs.data.cend() - 1) {
            os << ", ";
        }
    }
    os << " )";
    return os;
}

template <class T>
size_t wH(Vector<T>& v) noexcept {
    return v.wH();
}

template <class T>
size_t wH(Vector<T>&& v) noexcept {
    return v.wH();
}

template <class T>
size_t burst_length(Vector<T>& v) noexcept {
    return v.burst_length();
}

template <class T>
size_t burst_length(Vector<T>&& v) noexcept {
    return v.burst_length();
}

template <class T>
size_t cyclic_burst_length(Vector<T>& v) noexcept {
    return v.cyclic_burst_length();
}

template <class T>
size_t cyclic_burst_length(Vector<T>&& v) noexcept {
    return v.cyclic_burst_length();
}

template <class T>
size_t dH(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.data.size() != rhs.data.size())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between two vectors of different "
            "lengths");
    return (lhs - rhs).wH();
}

double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs) {
    if (lhs.data.size() != rhs.data.size())
        throw std::invalid_argument(
            "trying to calculate euclidean distance between two vectors of different "
            "lengths");
    double res = 0;
    for (size_t i = 0; i < lhs.data.size(); ++i) {
        res += pow(abs(lhs.data[i] - rhs.data[i]), 2);
    }
    res = sqrt(res);
    return res;
}

template <class T>
Vector<T>& operator>>(const Vector<T>& lhs, Vector<T>& rhs) noexcept {
    rhs = lhs;
    return rhs;
}

template <class T>
Matrix<T>& operator>>(const Matrix<T>& lhs, Matrix<T>& rhs) noexcept {
    rhs = lhs;
    return rhs;
}

template <class S, uint8_t m>
Matrix<SF<S, m>>& operator>>(const Vector<S>& lhs, Matrix<SF<S, m>>& rhs) noexcept {
    rhs = lhs.template as_matrix<SF<S, m>>();
    return rhs;
}

template <class S, uint8_t m>
Vector<S>& operator>>(const Matrix<SF<S, m>>& lhs, Vector<S>& rhs) noexcept {
    rhs = lhs.template as_vector<S>();
    return rhs;
}

}  // namespace ECC

#endif
