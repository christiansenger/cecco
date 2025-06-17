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

#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <utility>
#include <variant>
#include <vector>

#include "InfInt.hpp"
#include "fields.hpp"

namespace ECC {

inline std::random_device os_seed;
inline const int seed = os_seed();
inline std::mt19937 gen(seed);

/*
template <size_t nesting_depth, class body>
constexpr void meta_for_loop(size_t begin, size_t end, body&& c) {
    static_assert(nesting_depth > 0);
    for (uint16_t i = begin; i != end; ++i) {
        if constexpr (nesting_depth == 1) {
            c(i);
        } else {
            auto bind_an_argument = [i, &c](auto... args) { c(i, args...); };
            meta_for_loop<nesting_depth - 1>(begin, end, bind_an_argument);
        }
    }
}
*/

template <class T>
auto find_maxima(const std::vector<T>& v) {
    std::vector<size_t> indices;

    for (auto it_max = std::max_element(v.begin(), v.end()); it_max != v.end();
         it_max = std::find(it_max + 1, v.end(), *it_max)) {
        auto index = std::distance(v.begin(), it_max);
        indices.push_back(index);
    }

    return indices;
}

// extended Euclidean algorithm
template <class T>
constexpr T GCD(T a, T b, T* s = nullptr, T* t = nullptr) noexcept {
    if (s != nullptr && t != nullptr) {  // extended EA
        *s = T(1);
        *t = T(0);
        T u = T(0);
        T v = T(1);
        while (b != T(0)) {
            const T q = a / b;
            T b1 = b;
            b = a - q * b;
            a = b1;
            T u1 = u;
            u = *s - q * u;
            *s = u1;
            T v1 = v;
            v = *t - q * v;
            *t = v1;
        }
    } else {  // "normal" EA
        while (b != T(0)) {
            const T q = a / b;
            T b1 = b;
            b = a - q * b;
            a = b1;
        }
    }
    return a;
}

template <class T, uint16_t p>
constexpr T modinv(T a) noexcept {
    T b = p;
    T s = 1;
    T t = 0;
    T u = 0;
    T v = 1;
    while (b != T(0)) {
        T q = a / b;
        T b1 = b;
        b = a - q * b;
        a = b1;
        T u1 = u;
        u = s - q * u;
        s = u1;
        T v1 = v;
        v = t - q * v;
        t = v1;
    }
    return s;
}

// factorial
template <class T = size_t>
T fac(T n) noexcept {
    T res = 1;
    while (n > 1) {
        res *= n;
        --n;
    }
    return res;
}

// binomial coefficient
template <class T = size_t>
T bin(const T& n, T k) noexcept {
    if (k == 0 || n == k) return 1;
    if (n == 0) return 0;
    if (n - k >= 0 && n - k < k) k = n - k;
    std::vector<T> t(k);
    t[0] = n - k + 1;
    for (T i = 1; i < k; ++i) t[i] = t[i - 1] * (n - k + 1 + i) / (i + 1);
    return t[k - 1];
}

/* // slower than option below
template <>
InfInt bin(InfInt n, InfInt k) noexcept {
    if (k == 0 || n == k) return 1;
    if (n == 0) return 0;
    if (n-k>=0 && n - k < k) k = n - k;
    std::vector<InfInt> t(k.toUnsignedLong());
    t[0] = n - k + 1;
    for (size_t i = 1; i < t.size(); ++i) t[i] = t[i - 1] * (n - k + 1 + i) / (i + 1);
    return t[(k - 1).toUnsignedLong()];
}
*/

template <>
InfInt bin(const InfInt& n, InfInt k) noexcept {
    if (k == 0 || n == k) return 1;
    if (n == 0) return 0;
    if (n - k >= 0 && n - k < k) k = n - k;
    InfInt numerator = 1;
    InfInt denominator = 1;
    for (InfInt i = 1; i <= k; ++i) {
        numerator *= n + 1 - i;
        denominator *= i;
    }
    return numerator / denominator;
}

// square-and-multiply
template <class T = InfInt>
constexpr T sqm(T b, int e) noexcept {
    if (e == 0) {
        return T(1);
    }
    if (e < 0) {
        b = T(1) / b;
        e = -e;
    }
    // square and multiply
    T temp(1);
    while (e > 0) {
        if (e % 2) {
            temp *= b;
        }
        b *= b;
        e /= 2;
    }
    return temp;
}

// double-and-add
template <class T = size_t>
constexpr T daa(T b, int m) noexcept {
    if (m == 0) {
        return T(0);
    }
    if (m < 0) {
        b = -b;
        m = -m;
    }
    // double and add
    T temp(0);
    while (m > 0) {
        if (m % 2) {
            temp += b;
        }
        b += b;
        m /= 2;
    }
    return temp;
}
constexpr double floor_constexpr(double x) {
    long int i = static_cast<int>(x);
    return (x < 0 && x != i) ? i - 1 : i;
}

constexpr bool isPrime(uint16_t i) noexcept {
    if (i == 1 || i == 2) {
        return true;
    }
    if (i == 0 || !(i & 1)) {  // zero or even (and not two)
        return false;
    }
    for (uint16_t j = 3; j < i; j += 2) {  // check all odd numbers
        if (i % j == 0) return false;
    }
    return true;
}

constexpr bool isDivisor(uint16_t i, uint16_t j) noexcept { return (i % j) == 0; }

template <class T, class MANDATORY_TYPE, class... OPTIONAL_TYPES>
class Cache {
   public:
    template <class TYPE>
    auto set(const std::string& key, const TYPE& value) {
        return data.insert_or_assign(key, value);
    }

    template <class TYPE>
    auto set(const std::string& key, TYPE&& value) {
        return data.insert_or_assign(key, std::forward<TYPE>(value));
    }

    bool isSet(const std::string& key) const {
        auto it = data.find(key);
        return static_cast<bool>(it != data.end());
    }

    bool invalidate(const std::string& key = "") {
        // erase all
        if (key.empty()) {
            data.clear();
            return true;
        }

        return data.erase(key);  // true if found, false if not
    }

    template <class TYPE>
    const TYPE& operator()(const std::string& key, T& object, TYPE (T::*calculate)()) {
        // check if requested key is already there...
        auto it = data.find(key);
        if (it == data.end()) {
            // ... if not: calculate
            const TYPE value = std::invoke(calculate, object);
            auto res = set(key, value);
            it = res.first;
        }
        return std::get<TYPE>(it->second);
    }

   private:
    std::map<std::string, std::variant<MANDATORY_TYPE, OPTIONAL_TYPES...>> data;
};

template <typename>
struct is_finite_field : std::false_type {};

}  // namespace ECC

#endif