/**
 * @file code_bounds.hpp
 * @brief Bounds on code parameters
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 1.0.1
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
    const size_t tmax = (dmin - 1) / 2;
    const InfInt N = n, Q = q;
    InfInt h = 0;
    InfInt term = 1;
    for (size_t i = 0; i <= tmax; ++i) {
        if (i > 0) term = term * (N - i + 1) / i * (Q - 1);
        h += term;
    }

    return n - log2(h) / std::log2(static_cast<long double>(q));
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
    const size_t tmax = (dmin - 1) / 2;
    const InfInt s = dmin % 2;
    const InfInt N = n, Q = q;
    InfInt h = 0;
    InfInt term = 1;
    for (size_t i = 0; i <= tmax; ++i) {
        if (i > 0) term = term * (N - i + 1) / i * (Q - 1);
        h += term;
    }
    term = term * (N - tmax) / (tmax + 1) * (Q - 1);

    const InfInt numerator = term - s * bin<InfInt>(dmin, tmax) * details::A<T>(n, dmin, dmin);

    const InfInt denominator = details::A<T>(n, dmin, tmax + 1);

    // exact form of h + numerator/denominator, scaled by the (positive) denominator
    const InfInt bracket = h * denominator + numerator;
    if (bracket <= 0) return std::numeric_limits<long double>::infinity();

    return n - (log2(bracket) - log2(denominator)) / std::log2(static_cast<long double>(q));
}

template <FiniteFieldType T>
long double PlotkinUpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");

    constexpr size_t q = T::get_size();
    const InfInt Q = q, N = n, D = dmin;
    if (Q * D > N * (Q - 1)) {  // conventional
        return (log2(Q * D) - log2(Q * D - N * (Q - 1))) / std::log2(static_cast<long double>(q));
    } else {  // improved
        const InfInt temp = (Q * D + (Q - 2)) / (Q - 1);
        const InfInt denominator = Q * D - (temp - 1) * (Q - 1);

        // N - temp + 2 >= 1 in this branch, so the conversion cannot overflow
        return static_cast<long double>((N - temp + 2).toUnsignedLongLong()) +
               (log2(D) - log2(denominator)) / std::log2(static_cast<long double>(q));
    }
}

template <FiniteFieldType T>
long double EliasUpperBound(size_t n, size_t dmin) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (dmin == 0) throw std::invalid_argument("Cannot calculate upper bounds with dmin=0!");
    if (dmin > n) throw std::invalid_argument("Cannot calculate upper bounds with dmin>n");

    constexpr size_t q = T::get_size();
    long double minimum = std::numeric_limits<long double>::infinity();
    const InfInt Q = q, N = n, D = dmin;
    InfInt h = 0;
    InfInt term = 1;
    for (size_t w = 0; Q * w <= (Q - 1) * N; ++w) {
        if (w > 0) term = term * (N - w + 1) / w * (Q - 1);
        h += term;
        const InfInt denominator = Q * w * w - InfInt(2) * (Q - 1) * N * w + (Q - 1) * N * D;
        if (denominator > 0) {
            const long double temp = log2((Q - 1) * N * D) - log2(denominator) - log2(h);
            minimum = std::min(minimum, temp);
        }
    }

    return n + minimum / std::log2(static_cast<long double>(q));
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
    InfInt sum = 0;
    for (size_t i = 0; i <= dmin - 2; ++i) sum += bin<InfInt>(n - 1, i) * sqm<InfInt>(q - 1, i);
    size_t r = 0;
    for (InfInt qr = 1; qr <= sum; qr *= q) ++r;

    return r <= n ? n - r : 0;
}

template <FiniteFieldType T>
size_t BurstUpperBound(size_t n, size_t ell) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (ell > n) throw std::invalid_argument("Burst bound: burst length ell must be at most n");
    constexpr size_t q = T::get_size();
    const long double x = 1.0L + static_cast<long double>(q - 1) * static_cast<long double>(n - ell) / q;
    return static_cast<size_t>(std::floor(n - ell - std::log2(x) / std::log2(static_cast<long double>(q))));
}

constexpr size_t ReigerBurstUpperBound(size_t n, size_t ell) {
    if (n == 0) throw std::invalid_argument("Cannot calculate upper bounds with n=0!");
    if (ell > n) throw std::invalid_argument("Burst bound: burst length ell must be at most n");
    return ell > n / 2 ? 0 : n - 2 * ell;
}

}  // namespace CECCO

#endif
