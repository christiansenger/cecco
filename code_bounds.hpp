/**
 * @file code_bounds.hpp
 * @brief Bounds on code parameters
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 1.0.0
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026, Christian Senger <senger@inue.uni-stuttgart.de>
 *
 * Licensed for noncommercial use only, including academic teaching, research, and personal non-profit purposes.
 * Commercial use is prohibited without a separate commercial license. See the [LICENSE](../../LICENSE) file in the
 * repository root for full terms and how to request a commercial license.
 */

#ifndef CODE_BOUNDS_HPP
#define CODE_BOUNDS_HPP

#include "field_concepts_traits.hpp"
/*
// transitive
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "InfInt.hpp"
#include "helpers.hpp"
*/

namespace CECCO {

template <FiniteFieldType T>
long double HammingUpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");

    constexpr size_t q = T::get_size();
    try {
        const size_t tmax = (dmin - 1) / 2;
        InfInt h = 0;
        for (size_t i = 0; i <= tmax; ++i) h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);

        return n - std::log2(static_cast<long double>(h.toUnsignedLongLong())) / std::log2(static_cast<long double>(q));
    } catch (const InfIntException& e) {
        std::cerr << " [Hamming bound overflow]";
        return std::numeric_limits<long double>::infinity();
    }
}

namespace details {

template <FiniteFieldType T>
InfInt A(size_t n, size_t d, InfInt w) {
    const InfInt e = (d + 1) / 2;
    if (w < e) return 1;

    constexpr size_t q = T::get_size();
    const InfInt Q = q, N = n;
    InfInt res = 1;
    for (InfInt i = e; i <= w; ++i) res = res * ((N - w + i) * (Q - 1)) / i;

    return res;
}

}  // namespace details

template <FiniteFieldType T>
long double JohnsonUpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");

    constexpr size_t q = T::get_size();
    try {
        const size_t tmax = (dmin - 1) / 2;
        const InfInt s = dmin % 2;
        InfInt h = 0;
        for (size_t i = 0; i <= tmax; ++i) h += bin<InfInt>(n, i) * sqm<InfInt>(q - 1, i);

        const InfInt numerator = bin<InfInt>(n, tmax + 1) * sqm<InfInt>(q - 1, tmax + 1) -
                                 s * bin<InfInt>(dmin, tmax) * details::A<T>(n, dmin, dmin);

        const InfInt denominator = details::A<T>(n, dmin, tmax + 1);

        return n - std::log2(static_cast<long double>(h.toUnsignedLongLong()) +
                             static_cast<long double>(numerator.toUnsignedLongLong()) /
                                 static_cast<long double>(denominator.toUnsignedLongLong())) /
                       std::log2(static_cast<long double>(q));
    } catch (const InfIntException& e) {
        std::cerr << " [Johnson bound overflow]";
        return std::numeric_limits<long double>::infinity();
    }
}

template <FiniteFieldType T>
long double PlotkinUpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");

    constexpr size_t q = T::get_size();
    try {
        const InfInt Q = q, N = n, D = dmin;
        if (Q * D > N * (Q - 1)) {  // conventional
            return std::log2(static_cast<long double>((Q * D).toUnsignedLongLong()) /
                             static_cast<long double>((Q * D - N * (Q - 1)).toUnsignedLongLong())) /
                   std::log2(static_cast<long double>(q));
        } else {  // improved
            const InfInt Delta = N - Q * D / (Q - 1) + 1;
            const InfInt M = sqm<InfInt>(q, Delta.toUnsignedLongLong() + 1) * D / (Q * D - (N - Delta) * (Q - 1));

            return std::log2(static_cast<long double>(M.toUnsignedLongLong())) / std::log2(static_cast<long double>(q));
        }
    } catch (const InfIntException& e) {
        std::cerr << " [Plotkin bound overflow]";
        return std::numeric_limits<long double>::infinity();
    }
}

template <FiniteFieldType T>
long double EliasUpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");

    constexpr size_t q = T::get_size();
    try {
        long double minimum = std::numeric_limits<long double>::infinity();
        const InfInt Q = q, N = n, D = dmin;
        for (size_t w = 0; Q * w <= (Q - 1) * N; ++w) {
            const InfInt denominator = Q * w * w - InfInt(2) * (Q - 1) * N * w + (Q - 1) * N * D;
            if (denominator > 0) {
                InfInt h = 0;
                for (size_t i = 0; i <= w; ++i) h += bin<InfInt>(N, i) * sqm<InfInt>(Q - 1, i);

                long double temp = static_cast<long double>(((Q - 1) * N * D).toUnsignedLongLong()) /
                                   static_cast<long double>(denominator.toUnsignedLongLong());

                temp /= static_cast<long double>(h.toUnsignedLongLong());

                minimum = std::min(minimum, temp);
            }
        }

        return n + std::log2(minimum) / std::log2(static_cast<long double>(q));
    } catch (const InfIntException& e) {
        std::cerr << " [Elias bound overflow]";
        return std::numeric_limits<long double>::infinity();
    }
}

inline size_t SingletonUpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");
    return n - dmin + 1;
}

template <FiniteFieldType T>
size_t GriesmerUpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");

    constexpr size_t q = T::get_size();
    size_t k = 0;
    for (size_t kp = 1; kp <= n; ++kp) {
        size_t sum = 0;
        size_t qi = 1;
        for (size_t i = 0; i < kp; ++i) {
            sum += qi >= dmin ? 1 : (dmin + qi - 1) / qi;
            if (sum > n) break;
            if (qi < dmin) {
                if (qi > dmin / q)
                    qi = dmin;
                else
                    qi *= q;
            }
        }
        if (sum <= n)
            k = kp;
        else
            break;
    }

    return k;
}

template <FiniteFieldType T>
long double UpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");

    long double minimum = std::numeric_limits<long double>::infinity();
    for (size_t delta = 0; delta < std::min(n, dmin); ++delta) {
        const long double hamming = HammingUpperBound<T>(n - delta, dmin - delta);
        const long double johnson = JohnsonUpperBound<T>(n - delta, dmin - delta);
        const long double plotkin = PlotkinUpperBound<T>(n - delta, dmin - delta);
        const long double elias = EliasUpperBound<T>(n - delta, dmin - delta);
        const long double singleton = SingletonUpperBound(n - delta, dmin - delta);
        const long double griesmer = GriesmerUpperBound<T>(n - delta, dmin - delta);
        minimum = std::min({minimum, hamming, johnson, plotkin, elias, singleton, griesmer});
    }
    return minimum;
}

template <FiniteFieldType T>
size_t GilbertVarshamovLowerBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate lower bound with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate lower bound with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate lower bound with dmin>n");
    if (dmin == 1) return n;

    constexpr size_t q = T::get_size();
    try {
        InfInt sum = 0;
        for (size_t i = 0; i <= dmin - 2; ++i) sum += bin<InfInt>(n - 1, i) * sqm<InfInt>(q - 1, i);
        size_t r = 0;
        for (InfInt qr = 1; qr <= sum; qr *= q) ++r;

        return r <= n ? n - r : 0;
    } catch (const InfIntException& e) {
        std::cerr << " [Gilbert-Varshamov bound overflow]";
        return 0;
    }
}

template <FiniteFieldType T>
size_t BurstUpperBound(size_t n, size_t ell) {
    if (ell > n) throw std::invalid_argument("Burst bound: burst length ell must be at most n");
    constexpr size_t q = T::get_size();
    return std::floor(n - ell - std::log2(1 + (q - 1) * (n - ell) / q) / std::log2(q));
}

constexpr size_t ReigerBurstUpperBound(size_t n, size_t ell) noexcept {
    if (2 * ell > n) return 0;
    return n - 2 * ell;
}

}  // namespace CECCO

#endif
