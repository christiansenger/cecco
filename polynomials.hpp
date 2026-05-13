/**
 * @file polynomials.hpp
 * @brief Polynomial arithmetic library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.2.9
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
 * Univariate polynomials over any @ref CECCO::ComponentType (finite fields, `double`,
 * `std::complex<double>`, signed integers). Algorithms that need division — long division,
 * GCD, LCM, derivatives, normalisation, irreducibility test — require @ref CECCO::FieldType
 * coefficients. Cross-field constructors between two finite fields of the same characteristic
 * route through @ref CECCO::details::largest_common_subfield_t. Polynomials are stored in
 * canonical form (leading zero coefficients pruned automatically) and evaluated by Horner.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F7 = Fp<7>;
 * Polynomial<F7> p = {1, 2, 3};                        // 1 + 2x + 3x²
 * Polynomial<F7> q = {4, 5};                           // 4 + 5x
 * auto r = p * q;                                      // multiplication
 * F7 v = p(F7(2));                                     // Horner-method evaluation
 * auto [quot, rem] = p.poly_long_div(q);               // long division
 * auto g = GCD(p, q);                                  // gcd via extended Euclid
 *
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, MOD{1, 1, 1}>;
 * Polynomial<F2> a = {1, 0, 1};                        // x² + 1 over F₂
 * Polynomial<F4> b(a);                                 // upcast F₂ ⊆ F₄
 * @endcode
 *
 * @section Performance_Features
 *
 * - Hamming weight is lazily computed and cached.
 * - Move-aware free arithmetic operators, so chained expressions don't copy intermediates.
 * - Horner's method gives O(n) polynomial evaluation.
 * - Canonical form (leading zeros pruned) is maintained by the mutating operations.
 *
 * @see @ref fields.hpp, @ref vectors.hpp, @ref matrices.hpp, @ref field_concepts_traits.hpp
 */

#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP

#include <initializer_list>

#include "matrices.hpp"

/*
// transitive
#include <algorithm>
#include <complex>
#include <concepts>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "field_concepts_traits.hpp"
#include "helpers.hpp"
*/

namespace CECCO {

template <ComponentType T>
class Vector;
template <ComponentType T>
class Polynomial;

template <ComponentType T>
constexpr bool operator==(const Polynomial<T>& lhs, const Polynomial<T>& rhs);
template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Polynomial<T>& rhs);

template <ComponentType T>
Polynomial<T> ZeroPolynomial();

/**
 * @brief Univariate polynomial p(x) = a₀ + a₁x + … + aₙxⁿ over a @ref CECCO::ComponentType
 *
 * @tparam T Coefficient type satisfying @ref CECCO::ComponentType (finite field, `double`,
 *           `std::complex<double>`, or signed integer including `InfInt`)
 *
 * Coefficients are stored low-to-high in a contiguous buffer; the leading-zero invariant is
 * maintained by an internal `prune()` so `data.size() - 1` is the degree (when non-empty and
 * not the zero polynomial). The empty polynomial (no coefficients) is *distinct* from the
 * zero polynomial — most queries throw `std::invalid_argument` on an empty polynomial.
 *
 * Methods that need division are gated by `requires FieldType<T>`; methods that compare
 * against zero (degree, Hamming weight, …) are gated by `requires ReliablyComparableType<T>`.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F7 = Fp<7>;
 * Polynomial<F7> p = {1, 2, 3};                    // 1 + 2x + 3x²
 * Polynomial<F7> q = {4, 5};                       // 4 + 5x
 * auto sum = p + q;
 * auto prod = p * q;
 * F7 v = p(F7(3));                                 // evaluation
 * auto [quot, rem] = p.poly_long_div(q);
 * size_t w = p.wH();                               // Hamming weight (cached)
 * @endcode
 */
template <ComponentType T>
class Polynomial {
    friend bool operator== <>(const Polynomial& lhs, const Polynomial& rhs);
    friend std::ostream& operator<< <>(std::ostream& os, const Polynomial& rhs);

   public:
    // Cache configuration for this class
    enum CacheIds { Weight = 0 };

    /** @name Constructors
     * @{
     */

    /// @brief Default constructor: empty polynomial (no coefficients; distinct from the zero polynomial)
    constexpr Polynomial() noexcept = default;

    /// @brief Constant polynomial from an `int` (constructs `T(e)`)
    constexpr Polynomial(int e) : data(1) { data.back() = T(e); }

    /// @brief Constant polynomial from a coefficient
    constexpr Polynomial(const T& e) : data(1) { data.back() = e; }

    /**
     * @brief From an initializer list of coefficients in ascending degree order
     *
     * `l[i]` becomes the coefficient of x^i. Trailing (high-degree) zeros are pruned to keep
     * canonical form: `Polynomial<int>{1, 2, 3}` represents 1 + 2x + 3x².
     */
    constexpr Polynomial(const std::initializer_list<T>& l) : data(l) { prune(); }

    constexpr Polynomial(const Polynomial& other) : data(other.data), cache(other.cache) {}
    constexpr Polynomial(Polynomial&& other) noexcept : data(std::move(other.data)), cache(std::move(other.cache)) {}

    /**
     * @brief Cross-field conversion between two finite fields of the same characteristic
     *
     * @tparam S Source field type (`Polynomial<S>`); must share characteristic with T
     *
     * Converts coefficient by coefficient via T's cross-field constructor, which routes through
     * @ref CECCO::details::largest_common_subfield_t and so handles disjoint construction towers.
     * Propagates `std::invalid_argument` if any coefficient is not representable in T.
     */
    template <FiniteFieldType S>
        requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
    Polynomial(const Polynomial<S>& other);

    /**
     * @brief From a coefficient vector: `v[i]` becomes the coefficient of x^i
     *
     * Resulting degree is `v.get_n() - 1` (after pruning). For cross-field conversion, convert
     * the vector first.
     */
    Polynomial(const Vector<T>& v);

    /** @} */

    /** @name Assignment Operators
     * @{
     */

    /// @brief Assign a scalar: replace `*this` with the constant polynomial @p rhs
    constexpr Polynomial& operator=(const T& rhs);

    constexpr Polynomial& operator=(const Polynomial<T>& rhs);
    constexpr Polynomial& operator=(Polynomial&& rhs) noexcept;

    /// @brief Cross-field assignment (same semantics as the cross-field constructor)
    template <FiniteFieldType S>
        requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
    Polynomial& operator=(const Polynomial<S>& other);

    /** @} */

    /** @name Unary Arithmetic Operations
     * @{
     */

    /// @brief Unary `+` (lvalue): returns a copy
    constexpr Polynomial operator+() const& { return *this; }
    /// @brief Unary `+` (rvalue): returns the rvalue itself
    constexpr Polynomial operator+() && noexcept { return std::move(*this); }

    /// @brief Unary `−` (lvalue): returns a new polynomial with each coefficient negated
    constexpr Polynomial operator-() const&;
    /// @brief Unary `−` (rvalue): negates coefficients in place
    constexpr Polynomial operator-() &&;

    /**
     * @brief Evaluate at @p s using Horner's method (O(n))
     *
     * @return p(s) = a₀ + a₁s + a₂s² + … + aₙsⁿ
     * @throws std::invalid_argument if `*this` is empty
     */
    T operator()(const T& s) const;

    /** @} */

    /** @name Compound Assignment Operations
     * @{
     */

    /// @brief Coefficient-wise addition; expands storage if @p rhs has higher degree, then prunes
    constexpr Polynomial& operator+=(const Polynomial& rhs);
    /// @brief Coefficient-wise subtraction; expands storage if @p rhs has higher degree, then prunes
    constexpr Polynomial& operator-=(const Polynomial& rhs);

    /**
     * @brief Polynomial multiplication by convolution (O(n²))
     *
     * @return Reference to `*this` with degree `deg(this) + deg(rhs)`
     * @throws std::invalid_argument if either operand is the empty polynomial
     */
    constexpr Polynomial& operator*=(const Polynomial& rhs);

    /**
     * @brief Polynomial division: `*this` becomes the quotient of `(*this) / rhs`
     *
     * @throws std::invalid_argument if @p rhs is the zero polynomial
     */
    Polynomial& operator/=(const Polynomial& rhs)
        requires FieldType<T>;

    /**
     * @brief Polynomial remainder: `*this` becomes `(*this) mod rhs`
     *
     * Specialised path when `deg(this) == deg(rhs)` (one subtraction); long division otherwise.
     *
     * @throws std::invalid_argument if @p rhs is the zero polynomial
     */
    Polynomial& operator%=(const Polynomial& rhs)
        requires FieldType<T>;

    /// @brief Multiply every coefficient by the scalar @p s
    constexpr Polynomial& operator*=(const T& s);

    /**
     * @brief Divide every coefficient by the scalar @p s
     *
     * @throws std::invalid_argument if @p s == T(0)
     * @note Round-trip `(p / s) * s == p` is only guaranteed when T satisfies @ref CECCO::FieldType
     * (otherwise integer rounding may corrupt coefficients).
     */
    Polynomial& operator/=(const T& s);

    /// @brief Multiply every coefficient by the integer @p n; reduces @p n modulo characteristic
    constexpr Polynomial& operator*=(size_t n)
        requires FieldType<T>;

    /**
     * @brief Long division: returns `(q, r)` with `*this == q · rhs + r` and `r == 0` or `deg(r) < deg(rhs)`
     *
     * @throws std::invalid_argument if @p rhs is the zero polynomial
     */
    std::pair<Polynomial<T>, Polynomial<T>> poly_long_div(const Polynomial<T>& rhs) const
        requires FieldType<T>;

    /** @} */

    /** @name Randomization
     * @{
     */

    /**
     * @brief Replace `*this` with a random polynomial of degree exactly @p d
     *
     * The leading coefficient is resampled until non-zero, so the result really has degree d.
     * Distribution per coefficient: finite-field types draw uniformly from the field; signed
     * integers from [−100, 100]; `double` and the parts of `std::complex<double>` from [−1, 1].
     */
    Polynomial& randomize(size_t d);

    /** @} */

    /** @name Differentiation
     * @{
     */

    /**
     * @brief In-place classical s-th derivative
     *
     * d^s/dx^s [∑ aᵢ xⁱ] = ∑_{i ≥ s} (i! / (i − s)!) aᵢ x^{i − s}. The s-th derivative of a
     * polynomial of degree < s is the zero polynomial. `s == 0` is the identity.
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    Polynomial& differentiate(size_t s)
        requires FieldType<T>;

    /**
     * @brief In-place s-th Hasse derivative (= classical s-th derivative divided by s!)
     *
     * D^{(s)}[∑ aᵢ xⁱ] = ∑_{i ≥ s} C(i, s) aᵢ x^{i − s}. Useful in characteristic p where the
     * classical derivative loses information whenever s ≥ p.
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    Polynomial& Hasse_differentiate(size_t s)
        requires FieldType<T>;

    /** @} */

    /** @name Information and Properties
     * @{
     */

    /**
     * @brief Degree max{i : aᵢ ≠ 0}
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    size_t degree() const
        requires ReliablyComparableType<T>;

    /// @brief Coefficient vector (length = number of stored coefficients; 0 for an empty polynomial)
    Vector<T> get_coefficients() const;

    /**
     * @brief Trailing degree min{i : aᵢ ≠ 0}
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    size_t trailing_degree() const
        requires ReliablyComparableType<T>;

    /**
     * @brief Coefficient of the lowest non-zero power
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    const T& trailing_coefficient() const
        requires ReliablyComparableType<T>;

    /**
     * @brief Coefficient of the highest non-zero power
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    const T& leading_coefficient() const;

    /// @brief True iff `*this` has no coefficients (distinct from the zero polynomial)
    constexpr bool is_empty() const noexcept { return data.empty(); }

    /**
     * @brief True iff `*this` is the zero polynomial (one coefficient, equal to T(0))
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    bool is_zero() const
        requires ReliablyComparableType<T>
    {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is zero");
        return data.size() == 1 && data[0] == T(0);
    }

    /**
     * @brief True iff `*this` is the constant polynomial 1
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    bool is_one() const
        requires ReliablyComparableType<T>
    {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is one");
        return degree() == 0 && trailing_coefficient() == T(1);
    }

    /**
     * @brief True iff the leading coefficient equals 1
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    bool is_monic() const
        requires ReliablyComparableType<T>
    {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is monic");
        return leading_coefficient() == T(1);
    }

    /**
     * @brief Test irreducibility by trial division against every monic polynomial of degree ≤ deg/2
     *
     * Cost grows as q^{deg/2} in field size — practical only for small fields and small degrees.
     *
     * @throws std::invalid_argument if `*this` is empty
     */
    constexpr bool is_irreducible() const
        requires FiniteFieldType<T>
    {
        const size_t d = degree();
        if (d == 0) return false;
        if (d == 1) return true;

        const size_t q = T::get_size();

        for (size_t i = 1; i <= d / 2; ++i) {
            size_t count = 1;
            for (size_t j = 0; j < i; ++j) count *= q;

            for (size_t k = 0; k < count; ++k) {
                size_t x = k;
                Vector<T> v;
                for (size_t j = 0; j < i; ++j) {
                    v = v.append(T(x % q));
                    x /= q;
                }
                auto p = Polynomial(v.append(T(1)));
                if (((*this) % p).is_zero()) return false;
            }
        }

        return true;
    }

    /// @brief Hamming weight: number of non-zero, non-erased coefficients; cached on first call
    size_t wH() const
        requires ReliablyComparableType<T>
    {
        return cache.template get_or_compute<Weight>([this] { return calculate_weight(); });
    }

    /** @} */

    /** @name Coefficient Access and Manipulation
     * @{
     */

    /**
     * @brief Add @p c to the coefficient of x^i; grows the polynomial if `i > degree`
     *
     * @param i Power of x
     * @param c Value to add (perfect-forwarded)
     */
    template <typename U>
    constexpr Polynomial& add_to_coefficient(size_t i, U&& c)
        requires std::convertible_to<std::decay_t<U>, T>;

    /**
     * @brief Set the coefficient of x^i to @p c; grows the polynomial if `i > degree`
     *
     * @param i Power of x
     * @param c New value (perfect-forwarded)
     */
    template <typename U>
    constexpr Polynomial& set_coefficient(size_t i, U&& c)
        requires std::convertible_to<std::decay_t<U>, T>;

    /// @brief Reciprocal: reverse coefficients, sending p(x) to xⁿ · p(1/x)
    constexpr Polynomial& reciprocal();

    /// @brief Make monic by dividing every coefficient by the leading one (no-op on zero or already-monic)
    constexpr Polynomial& normalize()
        requires FieldType<T>;

    /// @brief Coefficient of x^i; returns T(0) for `i > degree`
    constexpr T operator[](size_t i) const;

    /** @} */

   private:
    std::vector<T> data;

    /// @brief Cache for Hamming weight (invalidated by mutating operations)
    mutable details::Cache<details::CacheEntry<Weight, size_t>> cache;

    constexpr size_t calculate_weight() const
        requires ReliablyComparableType<T>;

    constexpr Polynomial& prune();
};

/* member functions for Polynomial */

template <ComponentType T>
template <FiniteFieldType S>
    requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
Polynomial<T>::Polynomial(const Polynomial<S>& other) {
    if (other.is_empty()) return;
    const size_t deg = other.degree();
    for (size_t i = 0; i <= deg; ++i) set_coefficient(i, T(other[i]));  // Uses enhanced cross-field constructors
}

template <ComponentType T>
Polynomial<T>::Polynomial(const Vector<T>& v) : data(v.get_n()) {
    for (size_t i = 0; i < data.size(); ++i) data[i] = v[i];
    prune();
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator=(const T& rhs) {
    data.resize(1);
    data.back() = rhs;
    if (rhs == T(0))
        cache.template set<Weight>(0);
    else
        cache.template set<Weight>(1);
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator=(const Polynomial<T>& rhs) {
    if (this == &rhs) return *this;
    if (rhs.data.empty()) {
        data.clear();
        cache.invalidate();
        return *this;
    }
    data = rhs.data;
    cache = rhs.cache;
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator=(Polynomial<T>&& rhs) noexcept {
    if (this == &rhs) return *this;
    if (rhs.data.empty()) {
        data.clear();
        cache.invalidate();
        return *this;
    }
    data = std::move(rhs.data);
    cache = std::move(rhs.cache);
    return *this;
}

template <ComponentType T>
template <FiniteFieldType S>
    requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
Polynomial<T>& Polynomial<T>::operator=(const Polynomial<S>& rhs) {
    if (rhs.data.empty()) {
        data.clear();
        cache.invalidate();
        return *this;
    }
    data.resize(0);
    for (size_t i = 0; i <= rhs.degree(); ++i)
        this->set_coefficient(i, T(rhs[i]));  // Uses enhanced cross-field constructors
    cache.invalidate();
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T> Polynomial<T>::operator-() const& {
    Polynomial res(*this);
    std::ranges::for_each(res.data, [](T& c) { c = -c; });
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> Polynomial<T>::operator-() && {
    std::ranges::for_each(data, [](T& c) { c = -c; });
    cache.invalidate();
    return std::move(*this);
}

template <ComponentType T>
T Polynomial<T>::operator()(const T& s) const {
    if (data.empty()) throw std::invalid_argument("trying to evaluate empty polynomial");
    if (data.size() == 1) return data.front();

    // Use std::accumulate with Horner's method for polynomial evaluation
    return std::accumulate(data.crbegin() + 1, data.crend(), data.back(),
                           [&s](const T& acc, const T& coeff) { return acc * s + coeff; });
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator+=(const Polynomial<T>& rhs) {
    if (data.size() < rhs.data.size()) data.resize(rhs.data.size());
    std::transform(data.begin(), data.begin() + rhs.data.size(), rhs.data.begin(), data.begin(), std::plus<T>{});
    cache.invalidate();
    prune();
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator-=(const Polynomial<T>& rhs) {
    if (data.size() < rhs.data.size()) data.resize(rhs.data.size());
    std::transform(data.begin(), data.begin() + rhs.data.size(), rhs.data.begin(), data.begin(), std::minus<T>{});
    cache.invalidate();
    prune();
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator*=(const Polynomial<T>& rhs) {
    if (is_empty() || rhs.is_empty()) throw std::invalid_argument("trying to multiply with empty polynomial");
    Polynomial res;
    res.data.resize(data.size() + rhs.data.size() - 1);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < rhs.data.size(); ++j) res.add_to_coefficient(i + j, data[i] * rhs.data[j]);
    }
    data = std::move(res.data);
    cache.invalidate();
    prune();
    return *this;
}

template <ComponentType T>
Polynomial<T>& Polynomial<T>::operator/=(const Polynomial<T>& rhs)
    requires FieldType<T>
{
    if (rhs.is_zero()) throw std::invalid_argument("division by zero (polynomial)");
    *this = this->poly_long_div(rhs).first;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Polynomial<T>& Polynomial<T>::operator%=(const Polynomial<T>& rhs)
    requires FieldType<T>
{
    if (rhs.is_zero()) throw std::invalid_argument("division by zero (polynomial)");

    const size_t d = degree();
    if (d == rhs.degree()) {  // simply subtract polynomial modulus
        const auto scalar = leading_coefficient() / rhs.leading_coefficient();
        for (size_t i = 0; i < d; ++i) {
            data[i] -= scalar * rhs[i];
        }
        set_coefficient(d, 0);
        cache.invalidate();
        prune();
    } else {  // full-blown polynomial long division
        auto res = this->poly_long_div(rhs);
        *this = res.second;
        cache.invalidate();
    }

    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator*=(const T& s) {
    if (s == T(0)) {
        cache.template set<Weight>(0);
        data.resize(1);
        data.back() = T(0);
    } else {
        std::ranges::for_each(data, [&s](T& c) { c *= s; });
        cache.invalidate();
    }
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator*=(size_t n)
    requires FieldType<T>
{
    if (T::get_characteristic() != 0) {
        n = n % T::get_characteristic();
    }
    if (n == 0) {
        cache.template set<Weight>(0);
        data.resize(1);
        data.back() = T(0);
    } else {
        std::ranges::for_each(data, [&n](T& c) { c *= n; });
        cache.invalidate();
    }
    return *this;
}

template <ComponentType T>
Polynomial<T>& Polynomial<T>::operator/=(const T& s) {
    if (s == T(0)) throw std::invalid_argument("division by zero (polynomial)");
    std::ranges::for_each(data, [&s](T& c) { c /= s; });
    cache.invalidate();
    return *this;
}

template <ComponentType T>
std::pair<Polynomial<T>, Polynomial<T>> Polynomial<T>::poly_long_div(const Polynomial<T>& rhs) const
    requires FieldType<T>
{
    if (rhs.is_zero()) throw std::invalid_argument("polynomial long division by zero polynomial");

    const auto rhs_degree = rhs.degree();
    const auto rhs_lc = rhs.leading_coefficient();

    if (rhs_degree == 0) return std::make_pair(*this / rhs[0], Polynomial<T>({0}));

    if (degree() < rhs_degree) return std::make_pair(Polynomial<T>({0}), *this);

    Polynomial<T> q;
    Polynomial<T> r = *this;

    while (r.degree() >= rhs_degree) {
        const T t = r.leading_coefficient() / rhs_lc;
        const size_t i = r.degree() - rhs_degree;
        q.add_to_coefficient(i, t);
        for (size_t j = 0; j <= rhs_degree; ++j) r.add_to_coefficient(i + j, -(t * rhs[j]));
    }

    return std::make_pair(std::move(q), std::move(r));
}

template <ComponentType T>
Polynomial<T>& Polynomial<T>::randomize(size_t d) {
    data.resize(d + 1);
    if constexpr (FieldType<T>) {
        std::ranges::for_each(data, std::mem_fn(&T::randomize));
        do {
            data.back().randomize();
        } while (data.back() == T(0));
    } else if constexpr (std::same_as<T, double>) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::ranges::for_each(data, [&](double& val) { val = dist(gen()); });
        do {
            data.back() = dist(gen());
        } while (data.back() == T(0));
    } else if constexpr (std::same_as<T, std::complex<double>>) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::ranges::for_each(data,
                              [&](std::complex<double>& val) { val = std::complex<double>(dist(gen()), dist(gen())); });
        do {
            data.back() = std::complex<double>(dist(gen()), dist(gen()));
        } while (data.back() == T(0));
    } else if constexpr (SignedIntType<T>) {
        std::uniform_int_distribution<long long> dist(-100, 100);
        std::ranges::for_each(data, [&](T& val) { val = T(dist(gen())); });
        do {
            data.back() = dist(gen());
        } while (data.back() == T(0));
    }
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Polynomial<T>& Polynomial<T>::differentiate(size_t s)
    requires FieldType<T>
{
    if (data.empty()) throw std::invalid_argument("trying to differentiate empty polynomial");

    if (s == 0) return *this;
    const size_t d = data.size() - 1;
    if (d == 0 || s > d) {
        data.resize(1);
        data[0] = T(0);
        return *this;
    }
    if constexpr (requires { typename T::BASE_FIELD; }) {
        using B = typename T::BASE_FIELD;
        for (size_t i = 0; i <= d - s; ++i) {
            B coeff(1);
            for (size_t k = 1; k <= s; ++k) coeff *= B(i + k);
            data[i] = T(coeff) * data[i + s];
        }
    } else {
        for (size_t i = 0; i <= d - s; ++i) {
            T coeff(1);
            for (size_t k = 1; k <= s; ++k) coeff *= T(i + k);
            data[i] = coeff * data[i + s];
        }
    }
    data.resize(data.size() - s);
    prune();
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Polynomial<T>& Polynomial<T>::Hasse_differentiate(size_t s)
    requires FieldType<T>
{
    if (data.empty()) throw std::invalid_argument("trying to Hasse differentiate empty polynomial");

    if (s == 0) return *this;
    const size_t d = data.size() - 1;
    if (d == 0 || s > d) {
        data.resize(1);
        data[0] = T(0);
        return *this;
    }
    for (size_t i = 0; i <= d - s; ++i) data[i] = bin<size_t>(i + s, s) * data[i + s];
    data.resize(data.size() - s);
    prune();
    cache.invalidate();
    return *this;
}

template <ComponentType T>
size_t Polynomial<T>::degree() const
    requires ReliablyComparableType<T>
{
    if (is_empty()) throw std::invalid_argument("calculating degree of empty polynomial");
    return (data.size() - 1);
}

template <ComponentType T>
Vector<T> Polynomial<T>::get_coefficients() const {
    Vector<T> res(data.size());
    for (size_t i = 0; i < data.size(); ++i) res.set_component(i, data[i]);
    return res;
}

template <ComponentType T>
size_t Polynomial<T>::trailing_degree() const
    requires ReliablyComparableType<T>
{
    if (is_empty()) throw std::invalid_argument("calculating trailing degree of empty polynomial");
    const size_t d = degree();
    if (d == 0) return 0;

    const auto first_nonzero = std::find_if(data.begin(), data.end(), [](const T& coeff) { return coeff != T(0); });
    return std::distance(data.begin(), first_nonzero);
}

template <ComponentType T>
const T& Polynomial<T>::trailing_coefficient() const
    requires ReliablyComparableType<T>
{
    if (is_empty())
        throw std::invalid_argument(
            "trying to access non-existent element (trailing "
            "coefficient)");
    return data[trailing_degree()];
}

template <ComponentType T>
const T& Polynomial<T>::leading_coefficient() const {
    if (is_empty())
        throw std::invalid_argument(
            "trying to access non-existent element (leading "
            "coefficient)");
    return data.back();
}

template <ComponentType T>
template <typename U>
constexpr Polynomial<T>& Polynomial<T>::add_to_coefficient(size_t i, U&& c)
    requires std::convertible_to<std::decay_t<U>, T>
{
    T c_converted = std::forward<U>(c);
    if (c_converted == T(0)) return *this;

    cache.invalidate();
    if (data.empty() || i >= data.size()) {
        // automatic growth, change zero coefficient beyond degree
        data.resize(i + 1);
        data.back() = std::move(c_converted);
    } else if (i == data.size() - 1) {
        // change MSC
        data.back() += c_converted;
        if (data.back() == T(0)) prune();
    } else {
        // change some coefficient
        data[i] += c_converted;
    }
    return *this;
}

template <ComponentType T>
template <typename U>
constexpr Polynomial<T>& Polynomial<T>::set_coefficient(size_t i, U&& c)
    requires std::convertible_to<std::decay_t<U>, T>
{
    T new_value = std::forward<U>(c);
    if (data.empty() || i >= data.size()) {
        if (new_value == T(0)) return *this;
        cache.invalidate();
        data.resize(i + 1);
        data[i] = std::move(new_value);
    } else {
        if (data[i] == new_value) return *this;
        cache.invalidate();
        data[i] = std::move(new_value);
        if (i == data.size() - 1 && data.back() == T(0)) prune();
    }
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::reciprocal() {
    std::reverse(data.begin(), data.end());
    prune();  // a zero constant term in the original becomes a leading zero after reversal
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::normalize()
    requires FieldType<T>
{
    if (is_zero() || is_monic()) return *this;
    *this /= leading_coefficient();
    return *this;
}

template <ComponentType T>
constexpr T Polynomial<T>::operator[](size_t i) const {
    if (i >= data.size()) return T(0);
    return data[i];
}

template <ComponentType T>
constexpr size_t Polynomial<T>::calculate_weight() const
    requires ReliablyComparableType<T>
{
    size_t res = data.size() - std::count(data.cbegin(), data.cend(), T(0));

#ifdef CECCO_ERASURE_SUPPORT
    if constexpr (FieldType<T>) res -= std::count_if(data.cbegin(), data.cend(), [](T x) { return x.is_erased(); });
#endif

    return res;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::prune() {
    if (data.empty()) return *this;

    const auto lc = std::find_if(data.crbegin(), data.crend(), [](const T& e) { return e != T(0); });
    if (lc != data.crend()) {
        data.resize(data.size() - std::distance(data.crbegin(), lc));
    } else {
        data.resize(1);
        data.back() = T(0);
    }
    return *this;
}

/* free functions wrt. Polynomial */

template <ComponentType T>
constexpr Polynomial<T> operator+(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(lhs);
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator+(Polynomial<T>&& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(std::move(lhs));
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator+(const Polynomial<T>& lhs, Polynomial<T>&& rhs) {
    Polynomial<T> res(std::move(rhs));
    res += lhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator+(Polynomial<T>&& lhs, Polynomial<T>&& rhs) {
    Polynomial<T> res(std::move(lhs));
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator-(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(lhs);
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator-(Polynomial<T>&& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator-(const Polynomial<T>& lhs, Polynomial<T>&& rhs) {
    Polynomial<T> res(std::move(rhs));
    res = -res;
    res += lhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator-(Polynomial<T>&& lhs, Polynomial<T>&& rhs) {
    Polynomial<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(lhs);
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(Polynomial<T>&& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(const Polynomial<T>& lhs, Polynomial<T>&& rhs) {
    Polynomial<T> res(std::move(rhs));
    res *= lhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(Polynomial<T>&& lhs, Polynomial<T>&& rhs) {
    Polynomial<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator/(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(lhs);
    res /= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator/(Polynomial<T>&& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(std::move(lhs));
    res /= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator%(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(lhs);
    res %= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator%(Polynomial<T>&& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(std::move(lhs));
    res %= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(const Polynomial<T>& lhs, const T& rhs) {
    Polynomial<T> res(lhs);
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(Polynomial<T>&& lhs, const T& rhs) {
    Polynomial<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(const T& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> res(rhs);
    res *= lhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(const T& lhs, Polynomial<T>&& rhs) {
    Polynomial<T> res(std::move(rhs));
    res *= lhs;
    return res;
}

// polynomial / scalar (corresponding to operator/= with T)
template <ComponentType T>
constexpr Polynomial<T> operator/(const Polynomial<T>& lhs, const T& rhs) {
    Polynomial<T> res(lhs);
    res /= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator/(Polynomial<T>&& lhs, const T& rhs) {
    Polynomial<T> res(std::move(lhs));
    res /= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(const Polynomial<T>& lhs, size_t n) {
    Polynomial<T> res(lhs);
    res *= n;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(Polynomial<T>&& lhs, size_t n) {
    Polynomial<T> res(std::move(lhs));
    res *= n;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(size_t n, const Polynomial<T>& rhs) {
    Polynomial<T> res(rhs);
    res *= n;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(size_t n, Polynomial<T>&& rhs) {
    Polynomial<T> res(std::move(rhs));
    res *= n;
    return res;
}

template <ComponentType T>
std::pair<Polynomial<T>, Polynomial<T>> poly_long_div(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> temp(lhs);
    return temp.poly_long_div(rhs);
}

template <ComponentType T>
std::pair<Polynomial<T>, Polynomial<T>> poly_long_div(Polynomial<T>&& lhs, const Polynomial<T>& rhs) {
    Polynomial<T> temp(std::move(lhs));
    return temp.poly_long_div(rhs);
}
template <ComponentType T>
constexpr Polynomial<T> derivative(const Polynomial<T>& poly, size_t s)
    requires FieldType<T>
{
    Polynomial<T> res(poly);
    res.differentiate(s);
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> derivative(Polynomial<T>&& poly, size_t s)
    requires FieldType<T>
{
    Polynomial<T> res(std::move(poly));
    res.differentiate(s);
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> Hasse_derivative(const Polynomial<T>& poly, size_t s)
    requires FieldType<T>
{
    Polynomial<T> res(poly);
    res.Hasse_differentiate(s);
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> Hasse_derivative(Polynomial<T>&& poly, size_t s)
    requires FieldType<T>
{
    Polynomial<T> res(std::move(poly));
    res.Hasse_differentiate(s);
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> reciprocal(const Polynomial<T>& poly) {
    Polynomial<T> res(poly);
    res.reciprocal();
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> reciprocal(Polynomial<T>&& poly) {
    Polynomial<T> res(std::move(poly));
    res.reciprocal();
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> normalize(const Polynomial<T>& poly)
    requires FieldType<T>
{
    Polynomial<T> res(poly);
    res.normalize();
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> normalize(Polynomial<T>&& poly)
    requires FieldType<T>
{
    Polynomial<T> res(std::move(poly));
    res.normalize();
    return res;
}

template <ComponentType T>
constexpr bool operator==(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    return lhs.data == rhs.data;
}

template <ComponentType T>
constexpr bool operator!=(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
    return !(lhs == rhs);
}

template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Polynomial<T>& rhs) {
    if (rhs.data.empty()) {
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
                bool positive_sign;
                if constexpr (FieldType<T>) {
                    positive_sign = rhs.data[i + 1].has_positive_sign();
                } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                    positive_sign = rhs.data[i + 1].real() >= 0 || rhs.data[i + 1].imag() >= 0;
                } else {
                    positive_sign = rhs.data[i + 1] >= 0;
                }

                if (positive_sign) {
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

/**
 * @brief Monomial a · xⁱ
 *
 * @param i Power of x
 * @param a Coefficient (defaults to T(1))
 */
template <ComponentType T>
constexpr Polynomial<T> Monomial(size_t i, auto&& a = T(1))
    requires std::convertible_to<std::decay_t<decltype(a)>, T>
{
    Polynomial<T> res;
    res.set_coefficient(i, std::forward<decltype(a)>(a));
    return res;
}

/// @brief Constant polynomial 0 (cached)
template <ComponentType T>
Polynomial<T> ZeroPolynomial() {
    static const Polynomial<T> zero(0);
    return zero;
}

/// @brief Constant polynomial 1 (cached)
template <ComponentType T>
Polynomial<T> OnePolynomial() {
    static const Polynomial<T> one(1);
    return one;
}

/**
 * @brief Greatest common divisor of two polynomials via the (extended) Euclidean algorithm
 *
 * @param a First polynomial
 * @param b Second polynomial
 * @param s Optional out-pointer for the Bézout coefficient of @p a
 * @param t Optional out-pointer for the Bézout coefficient of @p b
 * @return gcd(a, b); if both @p s and @p t are non-null, additionally `gcd(a, b) = s·a + t·b`
 *
 * Operands are reordered internally so that the higher-degree one drives the recursion.
 */
template <ComponentType T>
Polynomial<T> GCD(Polynomial<T> a, Polynomial<T> b, Polynomial<T>* s = nullptr, Polynomial<T>* t = nullptr)
    requires FieldType<T>
{
    if (a.degree() < b.degree()) std::swap(a, b);

    if (s != nullptr && t != nullptr) {  // extended EA
        *s = Polynomial<T>({1});
        *t = Polynomial<T>({0});
        Polynomial<T> u = Polynomial<T>({0});
        Polynomial<T> v = Polynomial<T>({1});
        // while (b.degree() > 0) {
        while (!b.is_zero()) {
            const Polynomial<T> q = a / b;
            Polynomial<T> b1 = std::move(b);
            b = a - q * b1;
            a = std::move(b1);
            Polynomial<T> u1 = std::move(u);
            u = *s - q * u1;
            *s = std::move(u1);
            Polynomial<T> v1 = std::move(v);
            v = *t - q * v1;
            *t = std::move(v1);
        }
    } else {  // "normal" EA
              // while (b.degree() > 0) {
        while (!b.is_zero()) {
            const Polynomial<T> q = a / b;
            Polynomial<T> b1 = std::move(b);
            b = a - q * b1;
            a = std::move(b1);
        }
    }
    return a;
}

/**
 * @brief Greatest common divisor of a list of polynomials, computed iteratively
 *
 * @return gcd(p₁, p₂, …, pₙ)
 * @throws std::invalid_argument if @p polys is empty
 */
template <ComponentType T>
Polynomial<T> GCD(const std::vector<Polynomial<T>>& polys)
    requires FieldType<T>
{
    if (polys.empty())
        throw std::invalid_argument("trying to calculate polynomial GCD but didn't provide any polynomials");
    if (polys.size() == 1) return polys[0];

    Polynomial<T> res = polys.front();
    for (auto it = polys.cbegin() + 1; it != polys.cend(); ++it) {
        res = GCD<T>(std::move(res), *it);
    }
    return res;
}

/// @brief Least common multiple `lcm(a, b) = (a · b) / gcd(a, b)`
template <ComponentType T>
Polynomial<T> LCM(const Polynomial<T>& a, const Polynomial<T>& b)
    requires FieldType<T>
{
    return (a * b) / GCD(a, b);
}

/**
 * @brief Least common multiple of a list of polynomials, computed iteratively
 *
 * @return lcm(p₁, p₂, …, pₙ)
 * @throws std::invalid_argument if @p polys is empty
 */
template <ComponentType T>
Polynomial<T> LCM(const std::vector<Polynomial<T>>& polys)
    requires FieldType<T>
{
    if (polys.empty())
        throw std::invalid_argument("trying to calculate polynomial LCM but didn't provide any polynomials");
    if (polys.size() == 1) return polys[0];

    Polynomial<T> res = polys.front();
    for (auto it = polys.cbegin() + 1; it != polys.cend(); ++it) {
        res = LCM<T>(res, *it);
    }
    return res;
}

/**
 * @brief Polynomial exponentiation by square-and-multiply
 *
 * @return base^exponent
 *
 * @warning Does **not** follow C++ precedence for `^`: in `b * base ^ exponent` the parser
 * evaluates `(b * base) ^ exponent`. Parenthesise as `b * (base ^ exponent)`, or call
 * @ref CECCO::sqm directly.
 */
template <ComponentType T>
constexpr Polynomial<T> operator^(const Polynomial<T>& base, int exponent) {
    return sqm<Polynomial<T>>(base, exponent);
}

/**
 * @brief Coefficient vector [a₀, a₁, …, aₘ] of the Conway polynomial for 𝔽_{p^m}
 *
 * @tparam p Prime characteristic
 * @tparam m Extension degree
 * @return Coefficients low-to-high; empty vector if the (p, m) pair is not in the built-in table
 *
 * The built-in table covers all primes ≤ 97 and degrees ≤ ~13 (more for small p).
 */
template <uint16_t p, size_t m>
constexpr Vector<Fp<p>> ConwayCoefficients() {
    if constexpr (p == 2) {
        if constexpr (m == 1)
            return {1, 1};
        else if constexpr (m == 2)
            return {1, 1, 1};
        else if constexpr (m == 3)
            return {1, 1, 0, 1};
        else if constexpr (m == 4)
            return {1, 1, 0, 0, 1};
        else if constexpr (m == 5)
            return {1, 0, 1, 0, 0, 1};
        else if constexpr (m == 6)
            return {1, 1, 0, 1, 1, 0, 1};
        else if constexpr (m == 7)
            return {1, 1, 0, 0, 0, 0, 0, 1};
        else if constexpr (m == 8)
            return {1, 0, 1, 1, 1, 0, 0, 0, 1};
        else if constexpr (m == 9)
            return {1, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        else if constexpr (m == 10)
            return {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1};
        else if constexpr (m == 11)
            return {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        else if constexpr (m == 12)
            return {1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1};
        else if constexpr (m == 13)
            return {1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    } else if constexpr (p == 3) {
        if constexpr (m == 1)
            return {1, 1};
        else if constexpr (m == 2)
            return {2, 2, 1};
        else if constexpr (m == 3)
            return {1, 2, 0, 1};
        else if constexpr (m == 4)
            return {2, 0, 0, 2, 1};
        else if constexpr (m == 5)
            return {1, 2, 0, 0, 0, 1};
        else if constexpr (m == 6)
            return {2, 2, 1, 0, 2, 0, 1};
        else if constexpr (m == 7)
            return {1, 0, 2, 0, 0, 0, 0, 1};
        else if constexpr (m == 8)
            return {2, 2, 2, 0, 1, 2, 0, 0, 1};
    } else if constexpr (p == 5) {
        if constexpr (m == 1)
            return {3, 1};
        else if constexpr (m == 2)
            return {2, 4, 1};
        else if constexpr (m == 3)
            return {3, 3, 0, 1};
        else if constexpr (m == 4)
            return {2, 4, 4, 0, 1};
        else if constexpr (m == 5)
            return {3, 4, 0, 0, 0, 1};
    } else if constexpr (p == 7) {
        if constexpr (m == 1)
            return {4, 1};
        else if constexpr (m == 2)
            return {3, 6, 1};
        else if constexpr (m == 3)
            return {4, 0, 6, 1};
        else if constexpr (m == 4)
            return {3, 4, 5, 0, 1};
    } else if constexpr (p == 11) {
        if constexpr (m == 1)
            return {9, 1};
        else if constexpr (m == 2)
            return {2, 7, 1};
        else if constexpr (m == 3)
            return {9, 2, 0, 1};
        else if constexpr (m == 4)
            return {3, 4, 5, 0, 1};
    } else if constexpr (p == 13) {
        if constexpr (m == 1)
            return {11, 1};
        else if constexpr (m == 2)
            return {2, 12, 1};
        else if constexpr (m == 3)
            return {11, 2, 0, 1};
    } else if constexpr (p == 17) {
        if constexpr (m == 1)
            return {14, 1};
        else if constexpr (m == 2)
            return {3, 16, 1};
        else if constexpr (m == 3)
            return {14, 1, 0, 1};
    } else if constexpr (p == 19) {
        if constexpr (m == 1)
            return {17, 1};
        else if constexpr (m == 2)
            return {2, 18, 1};
        else if constexpr (m == 3)
            return {17, 4, 0, 1};
    } else if constexpr (p == 23) {
        if constexpr (m == 1)
            return {18, 1};
        else if constexpr (m == 2)
            return {5, 21, 1};
    } else if constexpr (p == 29) {
        if constexpr (m == 1)
            return {27, 1};
        else if constexpr (m == 2)
            return {2, 24, 1};
    } else if constexpr (p == 31) {
        if constexpr (m == 1)
            return {28, 1};
        else if constexpr (m == 2)
            return {3, 29, 1};
    } else if constexpr (p == 37) {
        if constexpr (m == 1)
            return {35, 1};
        else if constexpr (m == 2)
            return {2, 33, 1};
    } else if constexpr (p == 43) {
        if constexpr (m == 1)
            return {40, 1};
        else if constexpr (m == 2)
            return {3, 42, 1};
    } else if constexpr (p == 47) {
        if constexpr (m == 1)
            return {42, 1};
        else if constexpr (m == 2)
            return {5, 45, 1};
    } else if constexpr (p == 53) {
        if constexpr (m == 1)
            return {51, 1};
        else if constexpr (m == 2)
            return {2, 49, 1};
    } else if constexpr (p == 59) {
        if constexpr (m == 1)
            return {57, 1};
        else if constexpr (m == 2)
            return {2, 58, 1};
    } else if constexpr (p == 61) {
        if constexpr (m == 1)
            return {59, 1};
        else if constexpr (m == 2)
            return {2, 60, 1};
    } else if constexpr (p == 67) {
        if constexpr (m == 1)
            return {65, 1};
        else if constexpr (m == 2)
            return {2, 63, 1};
    } else if constexpr (p == 71) {
        if constexpr (m == 1)
            return {64, 1};
        else if constexpr (m == 2)
            return {7, 69, 1};
    } else if constexpr (p == 73) {
        if constexpr (m == 1)
            return {68, 1};
        else if constexpr (m == 2)
            return {5, 70, 1};
    } else if constexpr (p == 79) {
        if constexpr (m == 1)
            return {76, 1};
        else if constexpr (m == 2)
            return {3, 78, 1};
    } else if constexpr (p == 83) {
        if constexpr (m == 1)
            return {81, 1};
        else if constexpr (m == 2)
            return {2, 82, 1};
    } else if constexpr (p == 89) {
        if constexpr (m == 1)
            return {86, 1};
        else if constexpr (m == 2)
            return {3, 82, 1};
    } else if constexpr (p == 97) {
        if constexpr (m == 1)
            return {92, 1};
        else if constexpr (m == 2)
            return {5, 96, 1};
    }
    // ... support all fields with less than 10k elements
    return Vector<Fp<p>>();
}

/**
 * @brief Conway polynomial for 𝔽_{p^m} as a `Polynomial<Fp<p>>`
 *
 * Wraps @ref ConwayCoefficients; returns the empty polynomial if (p, m) is not tabulated.
 */
template <uint16_t p, size_t m>
constexpr Polynomial<Fp<p>> ConwayPolynomial() {
    return Polynomial<Fp<p>>(ConwayCoefficients<p, m>());
}

/**
 * @brief Sample a random monic irreducible polynomial of degree @p degree over T
 *
 * Tries random polynomials until @ref Polynomial::is_irreducible accepts one. Runtime is
 * bounded in expectation by the density of irreducibles, but unbounded in the worst case;
 * suitable as a `modulus` for @ref CECCO::Ext.
 */
template <FieldType T>
Polynomial<T> find_irreducible(size_t degree) {
    if (degree == 0) return Polynomial<T>(T(1));
    Polynomial<T> res;
    do {
        res.randomize(degree);
    } while (!res.is_irreducible());

    res.normalize();
    return res;
}

}  // namespace CECCO

#endif
