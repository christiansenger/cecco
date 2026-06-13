/**
 * @file fields.hpp
 * @brief Finite field arithmetic library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.3.13
 * @date 2026
 *
 * @copyright
 * Copyright (c) 2026, Christian Senger <senger@inue.uni-stuttgart.de>
 *
 * Licensed for noncommercial use only, including academic teaching, research, and personal non-profit purposes.
 * Commercial use is prohibited without a separate commercial license. See the [LICENSE](../../LICENSE) file in the
 * repository root for full terms and how to request a commercial license.
 *
 *
 * @section Description
 *
 * Finite-field arithmetic. The library provides:
 *
 * - prime fields 𝔽_p ≅ ℤ/pℤ via @ref CECCO::Fp;
 * - extension fields 𝔽_{q^m} ≅ B[x]/(f(x)) over a base field B and an irreducible monic modulus
 *   f(x), via @ref CECCO::Ext, with selectable LUT generation mode (runtime or compile-time);
 * - merged field towers via @ref CECCO::Iso, which exposes one logical field from several
 *   isomorphic representations and connects otherwise-disjoint construction towers;
 * - cross-field constructors that pick the bridge field with @ref CECCO::details::largest_common_subfield_t;
 * - rational numbers ℚ with selectable precision via @ref CECCO::Rationals.
 *
 * A *field tower* in this library is a sequence of constructed extensions. Mathematical
 * intermediate fields that were never constructed are not part of the tower. Two towers ending
 * at isomorphic fields can be glued together by wrapping the two endpoints in an `Iso`.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F2     = Fp<2>;
 * using F3     = Fp<3>;
 * using F4     = Ext<F2, {1, 1, 1}, LutMode::CompileTime>;  // 𝔽₂[x]/(x² + x + 1)
 * using F9     = Ext<F3, {2, 2, 1}, LutMode::CompileTime>;
 * using F16_a  = Ext<F2, {1, 0, 0, 1, 1}>;                  // RunTime LUTs (default)
 * using F16_b  = Ext<F4, {2, 1, 1}>;
 * using F16    = Iso<F16_a, F16_b>;                         // merge the two F16 towers
 * using F256_a = Ext<F2, {1, 1, 0, 1, 0, 0, 0, 1, 1}>;
 * using F256_b = Ext<F4, {2, 2, 2, 0, 1}>;
 * using F256_c = Ext<F16, {6, 13, 1}>;
 * using F256   = Iso<F256_a, F256_b, F256_c>;
 *
 * F9 a(5), b(7);
 * auto c = a * b + F9(1);                       // arithmetic
 * size_t ord = a.get_multiplicative_order();
 * Vector<F3> v = a.as_vector<F3>();             // coordinate vector over F3
 *
 * F16 d(1);
 * F256 e(d);                                    // upcast: F16 ⊆ F256
 * @endcode
 *
 * @section Performance_Features
 *
 * - LUT modes per @ref CECCO::Ext instantiation: `LutMode::RunTime` (default; faster
 *   compilation, lazy first-access initialisation) or `LutMode::CompileTime` (zero startup,
 *   tables baked into the binary). Mix freely within a tower.
 * - For extension fields with q ≥ @ref CECCO_COMPRESS_LUTS_FROM_Q, the addition and
 *   multiplication tables are stored compressed (upper triangle only), saving ~50 % memory.
 * - Pure CRTP — no virtual dispatch; concepts (@ref CECCO::FieldType, @ref CECCO::FiniteFieldType)
 *   enforce the interface at compile time.
 *
 * @note CompileTime mode can exceed compiler recursion / step limits for larger fields. If
 * needed, raise with `-fconstexpr-depth=4294967295 -fconstexpr-steps=4294967295` (clang) or
 * `-fconstexpr-ops-limit=4294967295` (g++). Rule of thumb: CompileTime up to ~150 elements,
 * RunTime above.
 *
 * @section Irreducible_Polynomial_Construction
 *
 * @ref CECCO::Ext requires the modulus to be a monic irreducible polynomial of degree ≥ 2,
 * coefficients given low-to-high (constant term first). To find one inside the library:
 *
 * @code{.cpp}
 * using B = Fp<3>;
 * auto p = find_irreducible<B>(4);  // monic, degree 4, irreducible over B
 * auto v = Vector(p);               // vector form, ready to paste as modulus into Ext<B, …>
 * @endcode
 *
 * Externally (Magma):
 * @code
 * p := 2; m := 6; F := GaloisField(p);
 * P<x> := PolynomialRing(F);
 * px := IrreduciblePolynomial(F, m);
 * Reverse(Coefficients(px));
 * @endcode
 *
 * @see @ref field_concepts_traits.hpp for the concepts and traits (FieldType, SubfieldOf, …)
 * @see @ref vectors.hpp, @ref matrices.hpp, @ref polynomials.hpp
 */

#ifndef FIELDS_HPP
#define FIELDS_HPP

/**
 * @def CECCO_ERASURE_SUPPORT
 * @brief Enables `erase()` / `unerase()` / `is_erased()` on every field type
 *
 * Define only if needed (error-correction with erasures); enabling it costs some performance.
 */
#ifdef DOXYGEN
#define CECCO_ERASURE_SUPPORT
#endif

/**
 * @def CECCO_USE_LUTS_FOR_FP
 * @brief Force prime fields to use lookup tables instead of modular arithmetic
 *
 * Almost always slower than direct mod-p arithmetic — leave undefined unless you have a reason.
 */
#ifdef DOXYGEN
#define CECCO_USE_LUTS_FOR_FP
#endif

/**
 * @def CECCO_COMPRESS_LUTS_FROM_Q
 * @brief Compression threshold for 2D lookup tables in extension fields
 *
 * For fields with at least this many elements, the addition and multiplication tables store
 * only the upper triangle (exploiting commutativity), saving ~50 % memory. Optimal value
 * depends on cache size.
 */
#define CECCO_COMPRESS_LUTS_FROM_Q 256

#include <span>

#include "polynomials.hpp"
/*
//transitive
#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "InfInt.hpp"
#include "field_concepts_traits.hpp"
#include "helpers.hpp"
#include "matrices.hpp"
*/

namespace CECCO {

template <ComponentType T>
class Vector;
template <ComponentType T>
class Polynomial;
template <ComponentType T>
class Matrix;
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
class Iso;
}  // namespace CECCO

namespace CECCO {

namespace details {
/**
 * @brief Tag base for the CRTP-protection idiom
 *
 * Inherited by every CECCO field type so that the templated @ref CECCO::operator+ and friends
 * (which dispatch on @ref CECCO::FieldType) cannot accidentally match unrelated operands like
 * `int + int`. The protected constructor blocks direct instantiation.
 */
class Base {
   protected:
    Base() = default;
};

/**
 * @brief CRTP base documenting the interface every field type must provide
 *
 * @tparam T Derived field type (CRTP parameter)
 *
 * Most members are `= delete`, present solely to advertise the interface — derived types
 * shadow them with their own definitions. The free arithmetic operators in this file template
 * on @ref CECCO::FieldType and dispatch directly to `T`'s compound assignments; there is no
 * virtual dispatch.
 *
 * Derived types must supply: assignment from `T`, `T&&`, and `int`; `operator==`; unary `-`;
 * the four compound assignments `+=`, `−=`, `*=`, `/=`; `randomize()` /
 * `randomize_force_change()`; the property queries `is_zero()`, `has_positive_sign()`,
 * `get_multiplicative_order()`, `get_additive_order()`. When @ref CECCO_ERASURE_SUPPORT is
 * defined, also `erase()` / `unerase()` / `is_erased()`.
 *
 * Provided here (not deleted): `operator!=` (delegates to `==`) and the two unary `operator+`
 * overloads (identity).
 */
template <class T>
class Field : public details::Base {
   protected:
    ~Field() noexcept = default;

   public:
    /// @name Assignment Operators
    /// @{

    /// @brief Assign from `int` — derived must implement
    T& operator=(int l) = delete;
    /// @brief Copy assignment — derived must implement
    T& operator=(const T& rhs) noexcept = delete;
    /// @brief Move assignment — derived must implement
    T& operator=(T&& rhs) noexcept = delete;

    /// @}
    /// @name Comparison
    /// @{

    /// @brief Inequality, defined as `!(*this == rhs)`
    constexpr bool operator!=(const T& rhs) const { return !(static_cast<const T&>(*this) == rhs); }

    /// @}
    /// @name Unary Operations
    /// @{

    /// @brief Unary `+` on an lvalue: returns a copy
    constexpr T operator+() const& { return static_cast<const T&>(*this); }
    /// @brief Unary `+` on an rvalue: returns the rvalue itself
    constexpr T&& operator+() && noexcept { return static_cast<T&&>(*this); }

    /// @brief Additive inverse on an lvalue — derived must implement
    T operator-() const& noexcept = delete;
    /// @brief Additive inverse on an rvalue (in place) — derived must implement
    T& operator-() && noexcept = delete;

    /// @}
    /// @name Compound Assignment
    /// @{

    /// @brief `*this += rhs` — derived must implement
    T& operator+=(const T& rhs) noexcept = delete;
    /// @brief `*this -= rhs` — derived must implement
    T& operator-=(const T& rhs) noexcept = delete;
    /// @brief `*this *= rhs` — derived must implement
    T& operator*=(const T& rhs) noexcept = delete;
    /// @brief `*this /= rhs`; throws `std::invalid_argument` if rhs is zero — derived must implement
    T& operator/=(const T& rhs) = delete;

    /// @}
    /// @name Randomization
    /// @{

    /// @brief Uniform random element of the field — derived must implement; may return the same value
    Field& randomize() = delete;
    /// @brief Like @ref randomize but guaranteed to differ from the current value — derived must implement
    Field& randomize_force_change() = delete;

    /// @}
    /// @name Properties
    /// @{

    /// @brief Smallest k > 0 with `this^k == 1`; throws `std::invalid_argument` if `*this` is zero
    size_t get_multiplicative_order() const = delete;
    /// @brief Smallest k > 0 with `k * *this == 0`; for finite fields of characteristic p this is p (or 1 for zero);
    /// for ℚ it is 1 for zero and 0 (infinite) otherwise
    size_t get_additive_order() const = delete;
    /// @brief True if the element is "positive" (always true for finite fields; sign of numerator for ℚ)
    bool has_positive_sign() const noexcept = delete;
    /// @brief True if `*this` is the additive identity
    bool is_zero() const noexcept = delete;

#ifdef CECCO_ERASURE_SUPPORT
    /// @brief Mark this element as erased (out-of-field marker for erasure decoding)
    /// @warning Erased elements must not participate in field arithmetic; correct use is the caller's responsibility
    Field& erase() noexcept = delete;
    /// @brief Clear the erasure flag, resetting to additive identity
    Field& unerase() noexcept = delete;
    /// @brief Test whether this element is currently erased
    bool is_erased() const noexcept = delete;
#endif
    /// @}
};

}  // namespace details

/**
 * @name Field Arithmetic Operators
 * @brief Free CRTP operators (`+`, `−`, `*`, `/`, `^`) for any @ref CECCO::FieldType
 *
 * Each operator constructs a result by calling `T`'s compound assignment, with rvalue overloads
 * that move from a temporary instead of copying. Scalar multiplication takes the integer on
 * either side. Division throws `std::invalid_argument` if the right operand is zero.
 * `operator^` exponentiates by square-and-multiply via @ref CECCO::sqm.
 *
 * @warning `operator^` does **not** follow C++ precedence for `^`: in `b * a ^ p` the parser
 * evaluates `(b * a) ^ p`. Parenthesise as `b * (a ^ p)`, or call @ref CECCO::sqm directly.
 * @{
 */

template <FieldType T>
constexpr T operator+(const T& lhs, const T& rhs) {
    T res(lhs);
    res += rhs;
    return res;
}

template <FieldType T>
constexpr T operator+(T&& lhs, const T& rhs) {
    T res(std::move(lhs));
    res += rhs;
    return res;
}

template <FieldType T>
constexpr T operator+(const T& lhs, T&& rhs) {
    T res(std::move(rhs));
    res += lhs;
    return res;
}

template <FieldType T>
constexpr T operator+(T&& lhs, T&& rhs) {
    T res(std::move(lhs));
    res += rhs;
    return res;
}

template <FieldType T>
constexpr T operator-(const T& lhs, const T& rhs) {
    T res(lhs);
    res -= rhs;
    return res;
}

template <FieldType T>
constexpr T operator-(T&& lhs, const T& rhs) {
    T res(std::move(lhs));
    res -= rhs;
    return res;
}

template <FieldType T>
constexpr T operator-(const T& lhs, T&& rhs) {
    T res(-std::move(rhs));
    res += lhs;
    return res;
}

template <FieldType T>
constexpr T operator-(T&& lhs, T&& rhs) {
    T res(std::move(lhs));
    res -= rhs;
    return res;
}

template <FieldType T>
constexpr T operator*(const T& lhs, const T& rhs) {
    T res(lhs);
    res *= rhs;
    return res;
}

template <FieldType T>
constexpr T operator*(T&& lhs, const T& rhs) {
    T res(std::move(lhs));
    res *= rhs;
    return res;
}

template <FieldType T>
constexpr T operator*(const T& lhs, T&& rhs) {
    T res(std::move(rhs));
    res *= lhs;
    return res;
}

template <FieldType T>
constexpr T operator*(T&& lhs, T&& rhs) {
    T res(std::move(lhs));
    res *= rhs;
    return res;
}

template <FieldType T>
constexpr T operator*(const T& lhs, uint16_t rhs) {
    T res(lhs);
    res *= rhs;
    return res;
}

template <FieldType T>
constexpr T operator*(uint16_t lhs, const T& rhs) {
    T res(rhs);
    res *= lhs;
    return res;
}

template <FieldType T>
constexpr T operator*(T&& lhs, int rhs) {
    T res(std::move(lhs));
    res *= rhs;
    return res;
}

template <FieldType T>
constexpr T operator*(int lhs, T&& rhs) {
    T res(std::move(rhs));
    res *= lhs;
    return res;
}

template <FieldType T>
T operator/(const T& lhs, const T& rhs) {
    T res(lhs);
    res /= rhs;
    return res;
}

template <FieldType T>
T operator/(T&& lhs, const T& rhs) {
    T res(std::move(lhs));
    res /= rhs;
    return res;
}

template <FieldType T>
constexpr T operator^(const T& base, int exponent) {
    return sqm<T>(base, exponent);
}

template <FieldType T>
constexpr T operator^(T&& base, int exponent) {
    return sqm<T>(std::move(base), exponent);
}

/** @} */

#ifdef CECCO_ERASURE_SUPPORT
inline constexpr std::string_view ERASURE_MARKER = "X";
#endif

/**
 * @brief Field of rational numbers ℚ = { p/q : p, q ∈ ℤ, q ≠ 0 } with selectable precision
 *
 * @tparam T Numerator/denominator type satisfying @ref CECCO::SignedIntType (default `InfInt`)
 *
 * Characteristic 0. Values are kept in lowest terms with positive denominator at all times,
 * so equality is `numerator_a * denominator_b == numerator_b * denominator_a`. Construction
 * with a zero denominator throws `std::invalid_argument`.
 *
 * Pick `T = InfInt` for true ℚ — a fixed-width `T` (e.g. `int`, `long long`) caps numerator
 * and denominator and silently overflows past that range.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * Rationals<> a(3, 4);
 * Rationals<> b(5, 6);
 * auto c = a + b;          // 19/12 (auto-simplified)
 * auto d = a / b;          // 9/10
 * std::cout << c;          // "19/12"
 * @endcode
 */
template <SignedIntType T = InfInt>  // use InfInt for infinite precision... and bad performance
class Rationals : public details::Field<Rationals<T>> {
   public:
    /**
     * @brief Construct n / d, simplified to lowest terms with positive denominator
     *
     * @param n Numerator (default 0)
     * @param d Denominator (default 1)
     * @throws std::invalid_argument if d == 0
     */
    Rationals(int n = 0, int d = 1);

    Rationals(const Rationals& other) = default;
    Rationals(Rationals&& other) = default;

    /// @brief Assign the integer @p l as `l / 1`
    constexpr Rationals& operator=(int l);

    constexpr Rationals& operator=(const Rationals& rhs) = default;
    Rationals& operator=(Rationals&& rhs);

    /// @brief Cross-multiplication equality
    constexpr bool operator==(const Rationals<T>& rhs) const {
#ifdef CECCO_ERASURE_SUPPORT
        if (is_erased() != rhs.is_erased()) return false;
#endif
        return numerator * rhs.get_denominator() == rhs.get_numerator() * denominator;
    }

    /// @brief Additive inverse (lvalue overload returns a copy with negated numerator)
    constexpr Rationals operator-() const&;
    /// @brief Additive inverse (rvalue overload negates in place)
    constexpr Rationals& operator-() &&;

    /// @brief `*this += rhs`, result kept in lowest terms
    constexpr Rationals& operator+=(const Rationals& rhs);
    /// @brief `*this -= rhs`, result kept in lowest terms
    constexpr Rationals& operator-=(const Rationals& rhs);
    /// @brief `*this *= rhs`, result kept in lowest terms
    constexpr Rationals& operator*=(const Rationals& rhs);
    /// @brief `*this /= rhs`; throws `std::invalid_argument` if rhs is zero
    Rationals& operator/=(const Rationals& rhs);

    /// @brief Random rational (bounded numerator and denominator), simplified
    Rationals& randomize();
    /// @brief Like @ref randomize but guaranteed to differ from the current value
    Rationals& randomize_force_change();

    /**
     * @brief Multiplicative order
     *
     * @return 1 for `1/1`, 2 for `−1/1`, 0 (interpreted as infinite) for everything else
     * @throws std::invalid_argument if `*this` is zero
     */
    size_t get_multiplicative_order() const;

    /// @brief Additive order: 1 for zero, 0 (infinite) otherwise (characteristic 0)
    size_t get_additive_order() const noexcept;

    /// @brief Human-readable description: `"rational number"`
    static const std::string get_info() {
        static const std::string info = "rational number";
        return info;
    }

    /// @brief Characteristic of ℚ: 0
    static constexpr size_t get_characteristic() noexcept { return 0; }

    /// @brief Sign predicate (true iff numerator and denominator share their sign)
    constexpr bool has_positive_sign() const noexcept {
        return (numerator >= 0 && denominator > 0) || (numerator <= 0 && denominator < 0);
    }

    /// @brief True iff numerator is zero
    constexpr bool is_zero() const noexcept { return numerator == 0; }

#ifdef CECCO_ERASURE_SUPPORT
    /**
     * @brief Mark this element as erased
     *
     * @warning Erased elements must not participate in field arithmetic; correct use is the
     * caller's responsibility (cf. @ref CECCO_ERASURE_SUPPORT).
     */
    constexpr Rationals& erase() noexcept;
    /// @brief Clear the erasure flag, resetting to additive identity 0/1
    constexpr Rationals& unerase() noexcept;
    /// @brief Test whether this element is currently erased (encoded as denominator == 0)
    constexpr bool is_erased() const noexcept { return denominator == 0; }
#endif

    /// @brief Numerator (sign carrier)
    constexpr const T& get_numerator() const noexcept { return numerator; }
    /// @brief Denominator (always positive after simplification)
    constexpr const T& get_denominator() const noexcept { return denominator; }

   private:
    T numerator;
    T denominator;

    /// @brief Reduce to lowest terms with positive denominator; called automatically
    constexpr void simplify();
};

/* member functions for Rationals */

template <SignedIntType T>
Rationals<T>::Rationals(int n, int d) : numerator(n), denominator(d) {
    if (d == 0) throw std::invalid_argument("denominator must not be zero");
    simplify();
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator=(int l) {
    numerator = l;
    denominator = 1;
    return *this;
}

template <SignedIntType T>
Rationals<T>& Rationals<T>::operator=(Rationals&& rhs) {
    if (this == &rhs) return *this;
    numerator = std::move(rhs.numerator);
    denominator = std::move(rhs.denominator);
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T> Rationals<T>::operator-() const& {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return Rationals().erase();
#endif
    Rationals res(*this);
    res.numerator = -res.numerator;
    return res;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator-() && {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return this->erase();
#endif
    numerator = -numerator;
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator+=(const Rationals& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    auto tn = numerator * rhs.get_denominator() + denominator * rhs.get_numerator();
    auto td = denominator * rhs.get_denominator();
    numerator = tn;
    denominator = td;
    simplify();
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator-=(const Rationals& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    auto tn = numerator * rhs.get_denominator() - denominator * rhs.get_numerator();
    auto td = denominator * rhs.get_denominator();
    numerator = tn;
    denominator = td;
    simplify();
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator*=(const Rationals& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    auto tn = numerator * rhs.get_numerator();
    auto td = denominator * rhs.get_denominator();
    numerator = tn;
    denominator = td;
    simplify();
    return *this;
}

template <SignedIntType T>
Rationals<T>& Rationals<T>::operator/=(const Rationals& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    if (rhs.numerator == 0) throw std::invalid_argument("division by zero");
    auto tn = numerator * rhs.get_denominator();
    auto td = denominator * rhs.get_numerator();
    numerator = tn;
    denominator = td;
    simplify();
    return *this;
}

template <SignedIntType T>
Rationals<T>& Rationals<T>::randomize() {
#ifdef CECCO_ERASURE_SUPPORT
    this->unerase();
#endif
    thread_local std::uniform_int_distribution<int> dist(-100, 100);
    numerator = dist(gen());
    do {
        denominator = dist(gen());
    } while (denominator == 0);
    simplify();
    return *this;
}

template <SignedIntType T>
Rationals<T>& Rationals<T>::randomize_force_change() {
#ifdef CECCO_ERASURE_SUPPORT
    this->unerase();
#endif
    thread_local std::uniform_int_distribution<int> dist(-100, 100);
    T n;
    T d;
    do {
        n = dist(gen());
        do {
            d = dist(gen());
        } while (d == 0);
    } while (T(n) * denominator == numerator * T(d));
    numerator = n;
    denominator = d;
    simplify();
    return *this;
}

template <SignedIntType T>
size_t Rationals<T>::get_multiplicative_order() const {
    if (numerator == 0)
        throw std::invalid_argument(
            "trying to calculate multiplicative order "
            "of additive neutral element");
    if (numerator == denominator)
        return 1;
    else if (numerator == -denominator)
        return 2;
    return 0;
}

template <SignedIntType T>
size_t Rationals<T>::get_additive_order() const noexcept {
    if (numerator == 0) {
        return 1;
    }
    return 0;
}

#ifdef CECCO_ERASURE_SUPPORT
template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::erase() noexcept {
    denominator = 0;
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::unerase() noexcept {
    if (is_erased()) {
        numerator = 0;
        denominator = 1;
    }
    return *this;
}
#endif

template <SignedIntType T>
constexpr void Rationals<T>::simplify() {
    if (numerator == 0) {
        denominator = 1;
        return;
    }

    auto d = GCD<T>(numerator, denominator);
    numerator /= d;
    denominator /= d;
    if (denominator < 0) {
        numerator *= -1;
        denominator *= -1;
    }
}

/**
 * @brief Print as `"numerator/denominator"` (or just `"numerator"` when denominator is 1)
 *
 * Single stream insertion for `std::setw` compatibility.
 */
template <SignedIntType T>
std::ostream& operator<<(std::ostream& os, const Rationals<T>& e) {
#ifdef CECCO_ERASURE_SUPPORT
    if (e.is_erased()) {
        os << ERASURE_MARKER;
    } else {
#endif
        std::string temp = std::to_string(e.get_numerator());
        if (e.get_denominator() != 1) temp += "/" + std::to_string(e.get_denominator());
        os << temp;
#ifdef CECCO_ERASURE_SUPPORT
    }
#endif
    return os;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
class Ext;

// The <255 (instead of <256) is important since we're using std::numeric_limits<>::max() as erasure flag
template <uint16_t p, uint8_t m = 1>
using label_t = typename std::conditional_t<sqm(p, m) < 255, uint8_t, uint16_t>;

namespace details {

/**
 * @brief Encode polynomial coefficients (low-to-high) as a base-q integer label
 *
 * @tparam q Base field size
 * @tparam m Extension degree (number of coefficients used)
 * @tparam SIZE Length of @p coeffs (must be ≥ m)
 */
template <uint16_t q, uint8_t m, size_t SIZE>
static size_t constexpr integer_from_coeffs(const std::array<size_t, SIZE>& coeffs) noexcept {
    static_assert(SIZE >= static_cast<size_t>(m), "array size must be at least the extension degree m");
    size_t res = coeffs[0];
    size_t t = q;
    for (uint8_t i = 1; i < m; ++i) {
        res += coeffs[i] * t;
        t *= q;
    }
    return res;
}

/**
 * @brief 1D lookup table for unary field operations (negation, inversion, order)
 *
 * @tparam LabelType Label integer type
 * @tparam FieldSize Number of field elements
 */
template <typename LabelType, size_t FieldSize>
struct Lut1D {
    constexpr LabelType operator()(size_t i) const noexcept { return values[i]; }
    std::array<LabelType, FieldSize> values{};
};

/**
 * @brief 2D lookup table for commutative binary operations, optionally compressed
 *
 * @tparam LabelType Label integer type
 * @tparam FieldSize Number of field elements
 *
 * For `FieldSize` ≥ @ref CECCO_COMPRESS_LUTS_FROM_Q, only the upper triangle is stored
 * (saving ~50 % memory); the access operators normalise the index pair so callers don't
 * see the difference.
 */
template <typename LabelType, size_t FieldSize>
struct Lut2D {
    /// @brief Mutable access (used during table construction)
    constexpr LabelType& operator()(size_t i, size_t j) {
        if (i > j) return operator()(j, i);
        if constexpr (FieldSize < CECCO_COMPRESS_LUTS_FROM_Q) {
            return values[i][j];
        } else {
            if (i > floor_constexpr(FieldSize / 2.0)) {
                return values[FieldSize - i][j - i];
            } else {
                return values[i][j];
            }
        }
    }

    /// @brief Const access (used during field operations)
    constexpr LabelType operator()(size_t i, size_t j) const noexcept {
        if (i > j) return operator()(j, i);
        if constexpr (FieldSize < CECCO_COMPRESS_LUTS_FROM_Q) {
            return values[i][j];
        } else {
            if (i > floor_constexpr(FieldSize / 2.0)) {
                return values[FieldSize - i][j - i];
            } else {
                return values[i][j];
            }
        }
    }

    /// @brief Backing storage; full square below the compression threshold, upper triangle above
    std::array < std::array<LabelType, FieldSize>,
        FieldSize<CECCO_COMPRESS_LUTS_FROM_Q ? FieldSize : static_cast<size_t>(floor_constexpr(FieldSize / 2.0) + 1)>
            values{};
};

/**
 * @brief Lookup table mapping each extension-field label to its base-field coefficient vector
 *
 * @tparam LabelType Coefficient label type
 * @tparam ExtensionDegree Number of coefficients per element (m)
 * @tparam FieldSize Number of field elements
 */
template <typename LabelType, size_t ExtensionDegree, size_t FieldSize>
struct Lut2Dcoeff {
    std::array<std::array<LabelType, ExtensionDegree>, FieldSize> values{};
};

/// @brief Build the table of multiplicative orders for all elements of 𝔽_Q
///
/// Each entry is the smallest k > 0 with `a^k == 1`; element 0 carries the sentinel order 0.
/// All orders divide Q − 1.
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_multiplicative_orders(const Lut2D<LabelType, FieldSize>& mul_lut) {
    Lut1D<LabelType, FieldSize> lut_mul_ord;

    lut_mul_ord.values[0] = 0;
    lut_mul_ord.values[1] = 1;
    for (LabelType i = 2; i < FieldSize; ++i) {
        LabelType temp = i;
        for (LabelType j = 1; j < FieldSize; ++j) {
            if (temp == 1) {
                lut_mul_ord.values[i] = j;
                break;
            }
            temp = mul_lut(temp, i);
        }
    }

    return lut_mul_ord;
}

/// @brief Build multiplicative-inverse table for 𝔽_p via extended Euclidean algorithm
///
/// Sentinel: `lut_inv[0] == 0`. Prime-field path; for extension fields use
/// @ref compute_multiplicative_inverses_search.
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_multiplicative_inverses_direct()
    requires(is_prime<LabelType>(FieldSize))
{
    Lut1D<LabelType, FieldSize> lut_inv;
    lut_inv.values[0] = 0;
    for (LabelType i = 1; i < FieldSize; ++i) {
        // int64_t avoids overflow in extended GCD for primes near 2^16.
        int64_t s = modinv<FieldSize, int64_t>(i);
        if (s <= -static_cast<int64_t>(FieldSize) || s >= static_cast<int64_t>(FieldSize))
            s %= static_cast<int64_t>(FieldSize);
        if (s < 0) s += static_cast<int64_t>(FieldSize);
        lut_inv.values[i] = static_cast<LabelType>(s);
    }
    return lut_inv;
}

/// @brief Build multiplicative-inverse table by searching the multiplication LUT
///
/// Used for extension fields, where extended Euclidean over the prime is not applicable.
/// Exploits the symmetry `a · b = 1 ⇒ b · a = 1` to fill two entries per match.
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_multiplicative_inverses_search(const Lut2D<LabelType, FieldSize>& mul_lut) {
    Lut1D<LabelType, FieldSize> lut_inv;

    lut_inv.values[0] = 0;
    for (LabelType i = 1; i < FieldSize; ++i) {
        if (lut_inv.values[i] != 0) continue;
        for (LabelType j = i; j < FieldSize; ++j) {
            if (mul_lut(i, j) == 1) {
                lut_inv.values[i] = j;
                lut_inv.values[j] = i;
                break;
            }
        }
    }

    return lut_inv;
}

/// @brief Build additive-inverse (negation) table for 𝔽_p
///
/// Direct formula: −0 = 0, −a = p − a otherwise. Prime-field path; extension fields use
/// @ref compute_additive_inverses_search.
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_additive_inverses_direct()
    requires(is_prime<LabelType>(FieldSize))
{
    Lut1D<LabelType, FieldSize> lut_neg;
    lut_neg.values[0] = 0;
    for (LabelType i = 1; i < FieldSize; ++i) lut_neg.values[i] = FieldSize - i;
    return lut_neg;
}

/// @brief Build additive-inverse table by searching the addition LUT
///
/// Used for extension fields. Symmetry `a + b = 0 ⇒ b + a = 0` fills two entries per match.
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_additive_inverses_search(const Lut2D<LabelType, FieldSize>& add_lut) {
    Lut1D<LabelType, FieldSize> lut_neg;
    lut_neg.values[0] = 0;
    for (LabelType i = 1; i < FieldSize; ++i) {
        if (lut_neg.values[i] != 0) continue;
        for (LabelType j = i; j < FieldSize; ++j) {
            if (add_lut(i, j) == 0) {
                lut_neg.values[i] = j;
                lut_neg.values[j] = i;
                break;
            }
        }
    }
    return lut_neg;
}

/// @brief Build the addition table for 𝔽_p: `lut_add(a, b) = (a + b) mod p`
///
/// Stores the upper triangle only (commutativity). Extension-field analogue:
/// @ref compute_polynomial_addition_table.
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_modular_addition_table()
    requires(is_prime<LabelType>(FieldSize))
{
    Lut2D<LabelType, FieldSize> lut_add;

    for (LabelType i = 0; i < FieldSize; ++i) {
        for (LabelType j = i; j < FieldSize; ++j) lut_add(i, j) = (i + j) % FieldSize;
    }

    return lut_add;
}

/// @brief Smallest primitive root of F* by direct power enumeration
///
/// For each candidate g ∈ {2, …, p−1} walks g, g², …, g^{p−2}; g is primitive iff none of those
/// equals 1. Evaluated at compile time via F's constexpr `operator*=`.
template <typename F>
constexpr typename F::label_t compute_primitive_root() {
    constexpr size_t p = F::get_p();
    if constexpr (p == 2) return typename F::label_t{1};

    for (uint16_t g_int = 2; g_int < p; ++g_int) {
        const F g(g_int);
        F power(1);
        bool is_primitive = true;
        for (uint16_t k = 1; k < p - 1; ++k) {
            power *= g;
            if (power == F(1)) {
                is_primitive = false;
                break;
            }
        }
        if (is_primitive) return static_cast<typename F::label_t>(g_int);
    }
    return typename F::label_t{0};  // unreachable for prime p
}

/// @brief Build the addition table for 𝔽_{q^m} via coefficient-wise base-field addition
///
/// Each element is a polynomial over 𝔽_q; addition reduces to base-field addition on the
/// coefficient vectors taken from @p lut_coeff.
template <typename LabelType, LabelType FieldSize, typename LutCoeffType, uint8_t ExtensionDegree,
          typename BaseFieldType>
constexpr auto compute_polynomial_addition_table(const LutCoeffType& lut_coeff) {
    Lut2D<LabelType, FieldSize> lut_add;

    for (LabelType i = 0; i < FieldSize; ++i) {
        lut_add(0, i) = i;
        lut_add(i, 0) = i;
    }

    for (LabelType i = 1; i < FieldSize; ++i) {
        for (LabelType j = i; j < FieldSize; ++j) {
            std::array<size_t, ExtensionDegree> temp{};

            for (uint8_t s = 0; s < ExtensionDegree; ++s) {
                temp[s] = BaseFieldType::lut_add(lut_coeff.values[i][ExtensionDegree - 1 - s],
                                                 lut_coeff.values[j][ExtensionDegree - 1 - s]);
            }

            lut_add(i, j) = integer_from_coeffs<BaseFieldType::get_size(), ExtensionDegree>(temp);
        }
    }

    return lut_add;
}

/// @brief Multiplicative order of @p element by repeated multiplication
///
/// @throws std::invalid_argument if @p element is zero
///
/// Single-element runtime path; batch precomputation lives in @ref compute_multiplicative_orders.
/// Order always divides Q − 1.
template <typename FieldType>
size_t calculate_multiplicative_order(const FieldType& element) {
    if (element == FieldType(0))
        throw std::invalid_argument(
            "trying to calculate multiplicative order "
            "of additive neutral element");

    size_t i = 1;
    FieldType temp = element;
    const FieldType one(1);
    for (;;) {
        if (temp == one) return i;
        temp *= element;
        ++i;
    }
}

/// @brief First element with multiplicative order Q − 1 (a generator of 𝔽_Q*)
template <typename LabelType, LabelType FieldSize>
constexpr LabelType find_generator(const Lut1D<LabelType, FieldSize>& lut_mul_ord) {
    for (LabelType i = 1; i < FieldSize; ++i)
        if (lut_mul_ord(i) == FieldSize - 1) return i;

    return LabelType{0};  // cannot happen
}

/// @brief Build the multiplication table for 𝔽_p: `lut_mul(a, b) = (a · b) mod p`
///
/// Stores the upper triangle only (commutativity). Extension-field analogue:
/// @ref compute_polynomial_multiplication_table.
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_modular_multiplication_table()
    requires(is_prime<LabelType>(FieldSize))
{
    Lut2D<LabelType, FieldSize> lut_mul;

    // uint32_t intermediate avoids overflow for primes near 2^16
    // (matches the cast in Fp::operator*=).
    for (LabelType i = 0; i < FieldSize; ++i) {
        for (LabelType j = i; j < FieldSize; ++j)
            lut_mul(i, j) = static_cast<LabelType>((static_cast<uint32_t>(i) * j) % FieldSize);
    }

    return lut_mul;
}

/// @brief Build the multiplication table for 𝔽_{q^m}: polynomial multiply, then reduce mod f(x)
///
/// `f(x) = Modulus` must be monic and irreducible over 𝔽_q. Reducibility is detected during
/// table construction (a non-zero label that maps to 0 is impossible in a field) and surfaces as
/// `std::invalid_argument`.
///
/// @throws std::invalid_argument if @p Modulus is not irreducible over the base field
template <typename LabelType, LabelType FieldSize, typename LutCoeffType, uint8_t ExtensionDegree,
          typename BaseFieldType, auto Modulus>
constexpr auto compute_polynomial_multiplication_table(const LutCoeffType& lut_coeff) {
    Lut2D<LabelType, FieldSize> lut_mul;
    constexpr size_t q = BaseFieldType::get_size();
    constexpr uint8_t m = ExtensionDegree;

    for (LabelType i = 0; i < FieldSize; ++i) {
        lut_mul(0, i) = 0;
        lut_mul(i, 0) = 0;
        lut_mul(1, i) = i;
        lut_mul(i, 1) = i;
    }

    for (LabelType i = 2; i < FieldSize; ++i) {
        const auto lhs = lut_coeff.values[i];

        // Calculate degree of lhs polynomial
        uint8_t lhs_deg = m - 1;
        for (uint8_t s = 0; s < m; ++s) {
            if (lhs[s] != 0)
                break;
            else
                --lhs_deg;
        }

        for (LabelType j = i; j < FieldSize; ++j) {
            const auto rhs = lut_coeff.values[j];

            // Calculate degree of rhs polynomial
            uint8_t rhs_deg = m - 1;
            for (uint8_t t = 0; t < m; ++t) {
                if (rhs[t] != 0)
                    break;
                else
                    --rhs_deg;
            }

            // Early termination for zero polynomials
            if (lhs_deg == 0 && lhs[m - 1] == 0) {
                lut_mul(i, j) = 0;
                continue;
            }
            if (rhs_deg == 0 && rhs[m - 1] == 0) {
                lut_mul(i, j) = 0;
                continue;
            }

            std::array<size_t, 2 * m - 1> temp{};

            // Polynomial multiplication
            for (uint8_t s = 0; s <= lhs_deg; ++s) {
                const auto S = lhs[m - 1 - s];
                if (S == 0) continue;
                for (uint8_t t = 0; t <= rhs_deg; ++t) {
                    const auto T = rhs[m - 1 - t];
                    if (T == 0) continue;  // Skip zero coefficient multiplication
                    temp[s + t] = BaseFieldType::lut_add(temp[s + t], BaseFieldType::lut_mul(S, T));
                }
            }

            // Reduction modulo irreducible polynomial
            for (uint8_t s = 0; s < m - 1; ++s) {
                const size_t flag_idx = 2 * m - 2 - s;
                const auto flag = temp[flag_idx];
                if (flag == 0) continue;
                for (uint8_t t = 0; t <= m; ++t) {
                    const auto mod_coeff = Modulus[m - t];
                    if (mod_coeff == 0) continue;
                    const size_t target_idx = flag_idx - t;
                    const auto neg_product = BaseFieldType::lut_neg(BaseFieldType::lut_mul(flag, mod_coeff));
                    temp[target_idx] = BaseFieldType::lut_add(temp[target_idx], neg_product);
                }
            }

            lut_mul(i, j) = integer_from_coeffs<q, m>(temp);
            if (lut_mul(i, j) == 0) {
                // In CompileTime mode this throw surfaces upstream as
                // "must be initialized by a constant expression" at LutHolder::lut.
                throw std::invalid_argument("extension field construction requires irreducible modulus");
            }
        }
    }

    return lut_mul;
}

/**
 * @brief Holds a LUT generated by `F()` (no dependencies); selects compile-time or lazy storage
 *
 * @tparam LutType Generated table type
 * @tparam F Generator function pointer
 * @tparam mode @ref LutMode::CompileTime or @ref LutMode::RunTime
 *
 * Used for self-contained tables (e.g. mod-p arithmetic in prime fields).
 */
template <typename LutType, LutType (*F)(), LutMode mode>
struct LutHolderNoProvider;

/// @brief CompileTime specialisation: table is a constexpr static member
template <typename LutType, LutType (*F)()>
struct LutHolderNoProvider<LutType, F, LutMode::CompileTime> {
    // If the compiler reports "must be initialized by a constant expression" here, the constexpr
    // step budget was likely exhausted: switch to LutMode::RunTime or raise -fconstexpr-steps /
    // -fconstexpr-ops-limit (see the file-header @note on CompileTime mode).
    static constexpr LutType lut = F();
    static constexpr const LutType& get_lut() { return lut; }
};
template <typename LutType, LutType (*F)()>
constexpr LutType LutHolderNoProvider<LutType, F, LutMode::CompileTime>::lut;

/// @brief RunTime specialisation: thread-safe lazy initialisation on first access
template <typename LutType, LutType (*F)()>
struct LutHolderNoProvider<LutType, F, LutMode::RunTime> {
    static void fill_lut() {
        get_lut();  // Ensure initialization
    }
    static const LutType& get_lut() {
        static LutType lut = F();  // Thread-safe lazy initialization
        return lut;
    }
};

/**
 * @brief Holds a LUT generated by `F(P)`; `P` provides a dependency table
 *
 * @tparam LutType Generated table type
 * @tparam ProviderLutType Dependency table type
 * @tparam P Provider accessor (returns the dependency table)
 * @tparam F Generator function consuming the provider
 * @tparam mode @ref LutMode::CompileTime or @ref LutMode::RunTime
 *
 * Used for extension-field tables that consume a coefficient table from the base field.
 */
template <typename LutType, typename ProviderLutType, const ProviderLutType& (*P)(),
          LutType (*F)(const ProviderLutType& (*)()), LutMode mode>
struct LutHolder;

/// @brief CompileTime specialisation: table baked into the binary
template <typename LutType, typename ProviderLutType, const ProviderLutType& (*P)(),
          LutType (*F)(const ProviderLutType& (*)())>
struct LutHolder<LutType, ProviderLutType, P, F, LutMode::CompileTime> {
    // If the compiler reports "must be initialized by a constant expression" here: either the
    // constexpr step budget was exhausted (switch to LutMode::RunTime or raise -fconstexpr-steps
    // / -fconstexpr-ops-limit; see the file-header @note) or a reducible modulus reached the
    // throw in compute_polynomial_multiplication_table.
    static constexpr LutType lut = F(P);
    static constexpr const LutType& get_lut() { return lut; }
};
template <typename LutType, typename ProviderLutType, const ProviderLutType& (*P)(),
          LutType (*F)(const ProviderLutType& (*)())>
constexpr LutType LutHolder<LutType, ProviderLutType, P, F, LutMode::CompileTime>::lut;

/// @brief RunTime specialisation: thread-safe lazy initialisation, dependency resolved on first access
template <typename LutType, typename ProviderLutType, const ProviderLutType& (*P)(),
          LutType (*F)(const ProviderLutType& (*)())>
struct LutHolder<LutType, ProviderLutType, P, F, LutMode::RunTime> {
    static void fill_lut() {
        get_lut();  // Ensure initialization
    }
    static const LutType& get_lut() {
        static LutType lut = F(P);  // Thread-safe lazy initialization
        return lut;
    }
};

}  // namespace details

/**
 * @brief Functor representing the field embedding φ: SUBFIELD → SUPERFIELD, with reverse lookup
 *
 * @tparam SUBFIELD   Subfield (finite-field type)
 * @tparam SUPERFIELD Superfield (finite-field type, must satisfy `SubfieldOf<SUPERFIELD, SUBFIELD>`)
 *
 * The embedding is determined by mapping a generator: with factor = (|SUPERFIELD| − 1) /
 * (|SUBFIELD| − 1), it sends g_sub^k ↦ g_super^{k · factor}. Identities φ(0) = 0, φ(1) = 1 are
 * fixed. The full table is computed once per template instantiation and cached.
 *
 * Forward map (`operator()`) is an O(1) array lookup; @ref extract reverses it via linear search
 * through the cached map. When `SUPERFIELD` is an @ref CECCO::Iso, @ref extract walks the MAIN
 * representation first, then each of OTHERS, until it finds one that contains `SUBFIELD`.
 *
 * @warning Mathematical containment is necessary but not sufficient — `SUBFIELD` must appear in
 * the construction tower of `SUPERFIELD` (or in one of an `Iso`'s components). Use an `Iso` to
 * merge towers when needed.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F2  = Fp<2>;
 * using F4  = Ext<F2, {1, 1, 1}>;
 * using F16 = Ext<F4, {2, 1, 1}>;
 *
 * Embedding<F4, F16> phi;
 * F4 a(2);
 * F16 b = phi(a);             // upcast — always succeeds
 * F4  c = phi.extract(b);     // throws std::invalid_argument if b ∉ φ(F4)
 * @endcode
 */
template <FiniteFieldType SUBFIELD, FiniteFieldType SUPERFIELD>
    requires SubfieldOf<SUPERFIELD, SUBFIELD>
class Embedding {
   public:
    /// @brief Constructs the embedding (cached on first instantiation per template arguments)
    Embedding();

    /// @brief Apply φ: SUBFIELD → SUPERFIELD
    constexpr SUPERFIELD operator()(const SUBFIELD& sub) const {
#ifdef CECCO_ERASURE_SUPPORT
        if (sub.is_erased()) return SUPERFIELD().erase();
#endif
        return SUPERFIELD(embedding_map[sub.get_label()]);
    }

    /**
     * @brief Reverse φ: find the unique `s ∈ SUBFIELD` with `φ(s) == super`
     *
     * @throws std::invalid_argument if @p super is not in the image of φ
     *
     * O(|SUBFIELD|) for regular fields; for `Iso` superfields, O(k · |SUBFIELD|) where k is
     * the number of inspected components.
     */
    constexpr SUBFIELD extract(const SUPERFIELD& super) const;

   private:
    std::span<const size_t> embedding_map;

    /// @brief Compute the table `embedding_map[i] = φ(SUBFIELD(i))`
    static std::vector<size_t> compute_embedding();
};

/*
Embedding member functions
*/

template <FiniteFieldType SUBFIELD, FiniteFieldType SUPERFIELD>
    requires SubfieldOf<SUPERFIELD, SUBFIELD>
Embedding<SUBFIELD, SUPERFIELD>::Embedding() {
    static std::once_flag computed_flag;
    static std::vector<size_t> cached_embedding;
    std::call_once(computed_flag, []() { cached_embedding = compute_embedding(); });
    embedding_map = std::span<const size_t>(cached_embedding);
}

template <FiniteFieldType SUBFIELD, FiniteFieldType SUPERFIELD>
    requires SubfieldOf<SUPERFIELD, SUBFIELD>
constexpr SUBFIELD Embedding<SUBFIELD, SUPERFIELD>::extract(const SUPERFIELD& super) const {
#ifdef CECCO_ERASURE_SUPPORT
    if (super.is_erased()) return SUBFIELD().erase();
#endif
    if constexpr (details::iso_info<SUPERFIELD>::is_iso) {
        // SUPERFIELD is an Iso type - find the correct component containing SUBFIELD

        // Try to extract from MAIN first...
        using MainType = typename details::iso_info<SUPERFIELD>::main_type;
        if constexpr (SubfieldOf<MainType, SUBFIELD>) {
            auto embedding = Embedding<SUBFIELD, MainType>();
            return embedding.extract(super.main());
        } else {
            // ... then extract OTHERS from SUPERFIELD
            using OthersTuple = typename details::iso_info<SUPERFIELD>::others_tuple;

            SUBFIELD result{};
            bool extraction_done = false;
            auto try_extracting = [&]<typename OtherType>() {
                if constexpr (SubfieldOf<OtherType, SUBFIELD>) {
                    if (!extraction_done) {
                        auto embedding = Embedding<SUBFIELD, OtherType>();
                        OtherType other_repr(super);
                        result = embedding.extract(other_repr);
                        extraction_done = true;
                    }
                }
            };

            // Apply the lambda to each type in the tuple
            std::apply([&]<typename... Types>(Types...) { (try_extracting.template operator()<Types>(), ...); },
                       OthersTuple{});

            if (!extraction_done) {
                throw std::invalid_argument("subfield not found in any Iso component");
            }
            return result;
        }
    } else {
        // SUPERFIELD is a regular field type
        auto it = std::ranges::find(embedding_map, super.get_label());
        if (it == embedding_map.end()) throw std::invalid_argument("superfield element is not in subfield");
        return SUBFIELD(std::distance(embedding_map.begin(), it));
    }
}

template <FiniteFieldType SUBFIELD, FiniteFieldType SUPERFIELD>
    requires SubfieldOf<SUPERFIELD, SUBFIELD>
std::vector<size_t> Embedding<SUBFIELD, SUPERFIELD>::compute_embedding() {
    constexpr size_t sub_size = SUBFIELD::get_size();
    constexpr size_t super_size = SUPERFIELD::get_size();
    std::vector<size_t> embedding(sub_size);

    // Zero maps to zero
    embedding[0] = 0;
    // Identity maps to identity
    embedding[1] = 1;

    // Embedding prime field: canonical embedding, no power factor needed
    if (sub_size == SUPERFIELD::get_characteristic()) {
        for (size_t i = 2; i < sub_size; ++i) embedding[i] = i;
        return embedding;
    }

    // Get generators and power factor
    const auto sub_gen = SUBFIELD::get_generator();
    const auto super_gen = SUPERFIELD::get_generator();
    constexpr size_t power_factor = (super_size - 1) / (sub_size - 1);

    // Compute super_gen^power_factor
    auto super_gen_to_power_factor = SUPERFIELD(1);
    for (size_t i = 0; i < power_factor; ++i) {
        super_gen_to_power_factor *= super_gen;
    }

    auto sub_elem = sub_gen;
    auto sup_elem = super_gen_to_power_factor;
    for (size_t i = 1; i < sub_size - 1; ++i) {
        embedding[sub_elem.get_label()] = sup_elem.get_label();
        sub_elem *= sub_gen;
        sup_elem *= super_gen_to_power_factor;
    }

    return embedding;
}

namespace details {

/**
 * @brief Shared static storage of the forward and reverse maps for the isomorphism A ↔ B
 *
 * @tparam A First finite field
 * @tparam B Second finite field (`Isomorphic<A, B>`)
 *
 * Lets both `Isomorphism<A, B>` and `Isomorphism<B, A>` share one cached pair of maps,
 * computed exactly once per template instantiation.
 */
template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
struct IsomorphismPair {
    static std::once_flag computed_flag;
    static std::vector<size_t> forward_iso;  // A -> B
    static std::vector<size_t> reverse_iso;  // B -> A

    /// @brief Compute and cache both maps on first call (no-op afterwards)
    static void compute_if_needed() {
        std::call_once(computed_flag, []() {
            constexpr size_t size = A::get_size();
            forward_iso.resize(size);
            reverse_iso.resize(size);

            // Compute forward isomorphism A -> B using existing algorithm
            const size_t m = A(0).template as_vector<Fp<A::get_p()>>().get_n();

            A alpha;
            B beta;

            const auto h = ConwayPolynomial<A::get_characteristic(), details::degree_over_prime_v<A>>();

            // Find root of h in A - is always a generator (property of Conway polynomials)
            Polynomial<A> h_A(h);
            for (size_t i = 1; i < size; ++i) {
                A a(i);
                if (h_A(a) == A(0)) {
                    alpha = a;
                    break;
                }
            }

            // Find root of h in B - is always a generator (property of Conway polynomials)
            Polynomial<B> h_B(h);
            for (size_t i = 1; i < size; ++i) {
                B b(i);
                if (h_B(b) == B(0)) {
                    beta = b;
                    break;
                }
            }

            // Construct new basis from generator
            std::vector<A> basis(m);
            A a(1);
            for (size_t i = 0; i < m; ++i) {
                basis[i] = a;
                a *= alpha;
            }

            // Calculate change-of-basis matrix
            Matrix<Fp<A::get_p()>> Mi(m, m);
            for (size_t i = 0; i < m; ++i) {
                Mi.set_submatrix(0, i,
                                 transpose(Matrix<Fp<A::get_p()>>(basis[i].template as_vector<Fp<A::get_p()>>())));
            }
            Mi.invert();

            // Compute forward mapping A -> B
            for (size_t i = 0; i < size; ++i) {
                A a(i);
                auto v = a.template as_vector<Fp<A::get_p()>>();
                Vector<B> w = v * transpose(Mi);
                auto p = Polynomial<B>(w);
                auto b = p(beta);
                forward_iso[i] = b.get_label();
            }

            // Compute reverse mapping B -> A (inverse of forward mapping)
            for (size_t i = 0; i < size; ++i) reverse_iso[forward_iso[i]] = i;
        });
    }
};

// Static member definitions
template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
std::once_flag details::IsomorphismPair<A, B>::computed_flag;

template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
std::vector<size_t> details::IsomorphismPair<A, B>::forward_iso;

template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
std::vector<size_t> details::IsomorphismPair<A, B>::reverse_iso;

}  // namespace details

/**
 * @brief Functor representing the field isomorphism φ: A → B between two same-size finite fields
 *
 * @tparam A Source finite field
 * @tparam B Target finite field (`Isomorphic<A, B>`)
 *
 * The isomorphism is built deterministically: a Conway polynomial of the prime field gives
 * generators α ∈ A and β ∈ B as common roots, and φ is extended linearly via a change-of-basis
 * matrix over the prime subfield. The resulting table is stored in a single
 * @ref details::IsomorphismPair shared between `Isomorphism<A, B>` and `Isomorphism<B, A>`.
 * It is a field homomorphism: φ(a + b) = φ(a) + φ(b), φ(a · b) = φ(a) · φ(b), φ(0) = 0,
 * φ(1) = 1.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F2    = Fp<2>;
 * using F4    = Ext<F2, {1, 1, 1}>;
 * using F16_a = Ext<F4, {2, 1, 1}>;
 * using F16_b = Ext<F4, {1, 2, 1}>;
 *
 * Isomorphism<F16_a, F16_b> phi;
 * F16_a a(4);
 * F16_b b = phi(a);
 * F16_a c = phi.inverse()(b);
 * assert(a == c);
 * @endcode
 */
template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
class Isomorphism {
    using PrimeField = Fp<A::get_p()>;

   public:
    /// @brief Construct (or retrieve from cache) the isomorphism map A → B
    Isomorphism();

    /**
     * @brief Construct from a precomputed mapping table (`iso[i] = φ(A(i))`)
     *
     * @warning No validation — incorrect input gives undefined behaviour. Internal use.
     */
    constexpr Isomorphism(const std::vector<size_t>& iso) : iso(iso) {}

    /// @brief Apply φ to @p a
    constexpr B operator()(const A& a) const {
#ifdef CECCO_ERASURE_SUPPORT
        if (a.is_erased()) return B().erase();
#endif
        return B(iso[a.get_label()]);
    }

    /// @brief Inverse isomorphism φ⁻¹: B → A
    Isomorphism<B, A> inverse() const;

   private:
    std::vector<size_t> iso;
};

/*
Isomorphism member functions
*/

template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
Isomorphism<A, B>::Isomorphism() : iso(A::get_size()) {
    // Use runtime comparison of get_info() strings to determine canonical template parameter ordering
    // This ensures both Isomorphism<A,B> and Isomorphism<B,A> use the same details::IsomorphismPair
    if constexpr (std::is_same_v<A, B>) {
        // Same field - trivial identity mapping
        std::iota(iso.begin(), iso.end(), 0);
    } else {
        // Use get_info() strings to determine canonical ordering
        if (A::get_info() < B::get_info()) {
            // A is "smaller" - use details::IsomorphismPair<A,B> forward mapping
            details::IsomorphismPair<A, B>::compute_if_needed();
            std::copy(details::IsomorphismPair<A, B>::forward_iso.begin(),
                      details::IsomorphismPair<A, B>::forward_iso.end(), iso.begin());
        } else {
            // B is "smaller" - use details::IsomorphismPair<B,A> reverse mapping
            details::IsomorphismPair<B, A>::compute_if_needed();
            std::copy(details::IsomorphismPair<B, A>::reverse_iso.begin(),
                      details::IsomorphismPair<B, A>::reverse_iso.end(), iso.begin());
        }
    }
}

template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
Isomorphism<B, A> Isomorphism<A, B>::inverse() const {
    std::vector<size_t> iso_inv(A::get_size());
    for (size_t i = 0; i < A::get_size(); ++i) iso_inv[iso[i]] = i;
    return Isomorphism<B, A>(std::move(iso_inv));
}

/**
 * @brief Prime field 𝔽_p ≅ ℤ/pℤ
 *
 * @tparam p Prime modulus, 2 ≤ p ≤ 65521 (largest prime fitting in `uint16_t`)
 *
 * Elements are stored as `label_t` integers in {0, 1, …, p − 1} (width is the smallest unsigned
 * type that fits p). Arithmetic is direct mod-p by default; define @ref CECCO_USE_LUTS_FOR_FP
 * to switch to LUTs (rarely advisable). The primality of @p p is enforced by `static_assert`.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F5 = Fp<5>;
 * F5 a(3), b(4);
 * auto c = a + b;                               // 2 (7 mod 5)
 * auto d = a / b;                               // 3 · 4⁻¹ = 3 · 4 = 2 in 𝔽₅
 * size_t ord = a.get_multiplicative_order();    // order of 3 in 𝔽₅*
 * @endcode
 */
template <uint16_t p>
class Fp : public details::Field<Fp<p>> {
    static_assert(is_prime(p), "p is not a prime");

   public:
    using label_t = ::CECCO::label_t<p>;

    /// @brief Default constructor: 0
    constexpr Fp() noexcept : label(0) {}

    /// @brief Construct from `int`, reducing modulo p (handles negative values)
    constexpr Fp(int l);

    constexpr Fp(const Fp& other) noexcept = default;
    constexpr Fp(Fp&& other) noexcept = default;

    /**
     * @brief Extract a prime-field element from an extension field (downcast)
     *
     * @tparam S Base of the source extension field
     * @tparam ext_modulus Modulus of the source extension field
     * @throws std::invalid_argument if @p other is not in the prime subfield
     *
     * Uses a cached @ref CECCO::Embedding from `Fp<p>` to `Ext<S, ext_modulus, mode>`; since
     * 𝔽_p is the minimal subfield of any extension over it, this is the canonical reverse
     * lookup. Source and target must share the characteristic (enforced by `static_assert`).
     */
    template <FiniteFieldType S, MOD ext_modulus, LutMode mode>
    Fp(const Ext<S, ext_modulus, mode>& other);

    /**
     * @brief Extract a prime-field element from an `Iso` containing this prime subfield
     *
     * @throws std::invalid_argument if no `Iso` component contains an element in 𝔽_p
     *
     * Tries the MAIN representation first, then each of OTHERS, until one yields a successful
     * downcast to `Fp<p>`.
     */
    template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
    Fp(const Iso<MAIN, OTHERS...>& other)
        requires SubfieldOf<Iso<MAIN, OTHERS...>, Fp<p>>;

    ~Fp() noexcept = default;

    /// @brief Assign `int`, reducing modulo p
    constexpr Fp& operator=(int l) noexcept;

    constexpr Fp& operator=(const Fp& rhs) noexcept = default;
    constexpr Fp& operator=(Fp&& rhs) noexcept = default;

    /// @brief Project an extension-field element to 𝔽_p (copy-and-swap; same semantics as the constructor)
    template <FiniteFieldType S, MOD ext_modulus, LutMode mode>
    Fp& operator=(const Ext<S, ext_modulus, mode>& other);

    /// @brief Project an `Iso` element to 𝔽_p (copy-and-swap; same semantics as the constructor)
    template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
        requires(MAIN::get_characteristic() == p)
    Fp& operator=(const Iso<MAIN, OTHERS...>& other);

    constexpr bool operator==(const Fp& rhs) const noexcept { return label == rhs.get_label(); }

    /// @brief Additive inverse (lvalue): returns a new element with `-label mod p`
    constexpr Fp operator-() const& noexcept;
    /// @brief Additive inverse (rvalue): in place
    constexpr Fp& operator-() && noexcept;

    /// @brief `*this = (label + rhs.label) mod p`
    constexpr Fp& operator+=(const Fp& rhs) noexcept;
    /// @brief `*this = (label − rhs.label) mod p`
    constexpr Fp& operator-=(const Fp& rhs) noexcept;
    /// @brief `*this = (label · rhs.label) mod p`
    constexpr Fp& operator*=(const Fp& rhs) noexcept;
    /// @brief `*this = (label · s) mod p` (repeated addition)
    constexpr Fp& operator*=(int s) noexcept;
    /// @brief `*this = (label · rhs.label⁻¹) mod p`; throws `std::invalid_argument` if rhs is zero
    Fp& operator/=(const Fp& rhs);

    /// @brief Uniform random element in {0, …, p − 1}
    Fp& randomize();
    /// @brief Like @ref randomize but guaranteed to differ from the current value
    Fp& randomize_force_change();

    /**
     * @brief Multiplicative order in 𝔽_p*
     *
     * @throws std::invalid_argument if `*this` is zero
     */
    size_t get_multiplicative_order() const;

    /// @brief Additive order: 1 for zero, p otherwise
    size_t get_additive_order() const;

    /// @brief Human-readable description: `"prime field with p elements"`
    static std::string get_info() {
        static const std::string info = "prime field with " + std::to_string(p) + " elements";
        return info;
    }

    static constexpr size_t get_characteristic() noexcept { return p; }

    /// @brief Underlying integer label in {0, …, p − 1}
    constexpr size_t get_label() const noexcept { return label; }

    /// @brief Generator (primitive root) of 𝔽_p*; precomputed label is @ref Gen
    static constexpr Fp get_generator() noexcept { return Fp(Gen); }

    static constexpr size_t get_p() noexcept { return p; }
    static constexpr size_t get_m() noexcept { return 1; }
    static constexpr size_t get_q() noexcept { return p; }
    static constexpr size_t get_size() noexcept { return p; }

    /**
     * @brief Always true for prime fields
     *
     * Prime fields' arithmetic interface is constexpr regardless of @ref CECCO_USE_LUTS_FOR_FP,
     * so a `CompileTime` @ref CECCO::Ext can always be built on top.
     */
    static constexpr bool is_constexpr_ready() noexcept { return true; }

    /// @brief Print all lookup tables to `std::cout` (debugging aid)
    static void show_tables();

    /// @brief Always true (finite fields are unordered)
    constexpr bool has_positive_sign() const noexcept { return true; }
    /// @brief True iff this is the additive identity
    constexpr bool is_zero() const noexcept { return label == 0; }

#ifdef CECCO_ERASURE_SUPPORT
    /**
     * @brief Mark this element as erased (encoded as `label == max(label_t)`)
     *
     * @warning Erased elements must not participate in field arithmetic — see
     * @ref CECCO_ERASURE_SUPPORT.
     */
    constexpr Fp& erase() noexcept;
    /// @brief Clear the erasure flag, resetting to the additive identity
    constexpr Fp& unerase() noexcept;
    /// @brief Test whether this element is currently erased
    constexpr bool is_erased() const noexcept { return label == std::numeric_limits<label_t>::max(); }
#endif

    /**
     * @brief Compile-time signal that all LUTs are constructed
     *
     * Used by extension fields built on top of 𝔽_p to defer their own LUT computation until
     * the base field is fully instantiated, avoiding compiler recursion-depth issues.
     */
    static constexpr bool ready() {
#ifdef CECCO_USE_LUTS_FOR_FP
        return luts_ready;
#else
        return true;  // Always ready when not using LUTs
#endif
    }

    // LUT-compatible interface for (default) non-LUT mode. This allows Fp to be used as base field B in Ext without
    // enabling CECCO_USE_LUTS_FOR_FP. These functions delegate to the optimized operator implementations to avoid
    // duplication.
#ifndef CECCO_USE_LUTS_FOR_FP
    /// @brief Addition function with LUT-compatible interface: lut_add(a,b) = (a + b) mod p
    static constexpr label_t lut_add(label_t a, label_t b) noexcept {
        Fp temp_a(a), temp_b(b);
        temp_a += temp_b;
        return temp_a.get_label();
    }

    /// @brief Multiplication function with LUT-compatible interface: lut_mul(a,b) = (a * b) mod p
    static constexpr label_t lut_mul(label_t a, label_t b) noexcept {
        Fp temp_a(a), temp_b(b);
        temp_a *= temp_b;
        return temp_a.get_label();
    }

    /// @brief Negation function with LUT-compatible interface: lut_neg(a) = (-a) mod p
    static constexpr label_t lut_neg(label_t a) noexcept {
        Fp temp(a);
        temp = -std::move(temp);
        return temp.get_label();
    }
#endif

    /// @brief Primitive element (generator) of F_p*
    ///
    /// Smallest primitive root mod p, computed at compile time via constexpr power enumeration
    /// using Fp's own multiplication.
    static constexpr label_t Gen = details::compute_primitive_root<Fp>();

   private:
    label_t label;  ///< Element value in {0, 1, ..., p-1}

#ifdef CECCO_USE_LUTS_FOR_FP
    /// @brief Type alias for 1D lookup tables
    using Lut1D = details::Lut1D<label_t, p>;
    /// @brief Type alias for 2D lookup tables
    using Lut2D = details::Lut2D<label_t, p>;

   public:
    /**
     * @name Precomputed Lookup Tables
     * @brief Compile-time generated tables for field operations
     * @{
     */

    /// @brief Addition lookup table: lut_add(a,b) = (a + b) mod p
    static constexpr Lut2D lut_add = details::compute_modular_addition_table<label_t, p>();

    /// @brief Multiplication lookup table: lut_mul(a,b) = (a * b) mod p
    static constexpr Lut2D lut_mul = details::compute_modular_multiplication_table<label_t, p>();

    /// @brief Additive inverse lookup table: lut_neg[a] = (-a) mod p
    static constexpr Lut1D lut_neg = details::compute_additive_inverses_direct<label_t, p>();

    /// @brief Multiplicative inverse lookup table: lut_inv[a] = a^(-1) mod p
    static constexpr Lut1D lut_inv = details::compute_multiplicative_inverses_direct<label_t, p>();

    /// @brief Multiplicative order lookup table: lut_mul_ord[a] = multiplicative order of a in 𝔽_p\c\{0}
    static constexpr Lut1D lut_mul_ord = details::compute_multiplicative_orders<label_t, p>(lut_mul);

    static constexpr bool luts_ready = []() constexpr {
        static_assert(lut_add(0, 0) == 0);  // Forces immediate calculation of lut_add
        static_assert(lut_neg(0) == 0);     // Forces immediate calculation lut_neg
        static_assert(lut_mul(0, 1) == 0);  // Forces immediate calculation lut_mul
        return true;
    }();

    /** @} */

#endif
};

/* member functions for Fp */

template <uint16_t p>
constexpr Fp<p>::Fp(int l) {
    if (l <= -(int)p || l >= (int)p) l %= (int)p;
    if (l < 0) l += (int)p;
    label = l;
}

template <uint16_t p>
template <FiniteFieldType S, MOD ext_modulus, LutMode mode>
Fp<p>::Fp(const Ext<S, ext_modulus, mode>& other) {
    // Ensure same characteristic
    static_assert(Fp<p>::get_characteristic() == Ext<S, ext_modulus, mode>::get_characteristic(),
                  "trying to convert between fields with different characteristic");

#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif

    // Extract Fp<p> element from extension field (downcast via largest common subfield)
    // Since Fp<p> is minimal, details::largest_common_subfield_t<Fp<p>, Ext<S, ...>> is always Fp<p>
    auto embedding = Embedding<Fp<p>, Ext<S, ext_modulus, mode>>();
    Fp<p> result = embedding.extract(other);
    label = result.get_label();
}

template <uint16_t p>
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
Fp<p>::Fp(const Iso<MAIN, OTHERS...>& other)
    requires SubfieldOf<Iso<MAIN, OTHERS...>, Fp<p>>
{
#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif

    // Try to extract from MAIN first
    if constexpr (SubfieldOf<MAIN, Fp<p>>) {
        *this = Fp(other.main());
    } else {
        // Try to extract from each OTHER component
        bool converted = false;
        (([&]() {
             if constexpr (SubfieldOf<OTHERS, Fp<p>>) {
                 if (!converted) {
                     *this = Fp(other.template as<OTHERS>());
                     converted = true;
                 }
             }
         }()),
         ...);

        if (!converted) {
            throw std::invalid_argument("no conversion path found from Iso to prime field");
        }
    }
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator=(int l) noexcept {
    if (l <= -(int)p || l >= (int)p) l %= (int)p;
    if (l < 0) l += (int)p;
    label = l;
    return *this;
}

template <uint16_t p>
template <FiniteFieldType S, MOD ext_modulus, LutMode mode>
Fp<p>& Fp<p>::operator=(const Ext<S, ext_modulus, mode>& rhs) {
    Fp temp(rhs);
    std::swap(*this, temp);
    return *this;
}

template <uint16_t p>
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
    requires(MAIN::get_characteristic() == p)
Fp<p>& Fp<p>::operator=(const Iso<MAIN, OTHERS...>& rhs) {
    Fp temp(rhs);
    std::swap(*this, temp);
    return *this;
}

template <uint16_t p>
constexpr Fp<p> Fp<p>::operator-() const& noexcept {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return Fp().erase();
#endif
    Fp res(*this);
    if (res.label != 0) {
#ifndef CECCO_USE_LUTS_FOR_FP
        int temp = -(int)res.label + (int)p;
        res.label = temp;
#else
        res.label = lut_neg(res.label);
#endif
    }
    return res;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator-() && noexcept {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return this->erase();
#endif
    if (label != 0) {
#ifndef CECCO_USE_LUTS_FOR_FP
        label = -(int)label + (int)p;
#else
        label = lut_neg(label);
#endif
    }
    return *this;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator+=(const Fp& rhs) noexcept {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
#ifndef CECCO_USE_LUTS_FOR_FP
    int temp = label + rhs.get_label();
    if (temp < p)
        label = temp;
    else
        label = temp - p;
#else
    label = lut_add(label, rhs.get_label());
#endif
    return *this;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator-=(const Fp& rhs) noexcept {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
#ifndef CECCO_USE_LUTS_FOR_FP
    int temp = (int)label - (int)rhs.get_label();
    if (temp >= 0)
        label = temp;
    else
        label = temp + p;
#else
    label = lut_add(label, lut_neg(rhs.get_label()));
#endif
    return *this;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator*=(const Fp& rhs) noexcept {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
#ifndef CECCO_USE_LUTS_FOR_FP
    // uint32_t intermediate avoids signed-int overflow for primes near 2^16
    // (max product (p-1)^2 ≈ 4.29e9 fits in uint32_t but not in signed int).
    uint32_t temp = static_cast<uint32_t>(label) * rhs.get_label();
    if (temp < p)
        label = static_cast<label_t>(temp);
    else
        label = static_cast<label_t>(temp % p);
#else
    label = lut_mul(label, rhs.get_label());
#endif
    return *this;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator*=(int s) noexcept {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return *this;
#endif
    s %= p;
    Fp res = daa<Fp>(*this, s);
    *this = std::move(res);
    return *this;
}

template <uint16_t p>
Fp<p>& Fp<p>::operator/=(const Fp& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    if (rhs.label == 0) throw std::invalid_argument("trying to divide by zero");
#ifndef CECCO_USE_LUTS_FOR_FP
    *this *= Fp(modinv<p, int64_t>(rhs.get_label()));
#else
    label = lut_mul(label, lut_inv(rhs.get_label()));
#endif
    return *this;
}

template <uint16_t p>
Fp<p>& Fp<p>::randomize() {
#ifdef CECCO_ERASURE_SUPPORT
    this->unerase();
#endif
    thread_local std::uniform_int_distribution<int> dist(0, p - 1);
#ifndef CECCO_USE_LUTS_FOR_FP
    int temp = label + dist(gen());
    if (temp < p)
        label = temp;
    else
        label = temp - p;
#else
    label = lut_add(label, dist(gen()));
#endif
    return *this;
}

template <uint16_t p>
Fp<p>& Fp<p>::randomize_force_change() {
#ifdef CECCO_ERASURE_SUPPORT
    this->unerase();
#endif
    thread_local std::uniform_int_distribution<int> dist(1, p - 1);
#ifndef CECCO_USE_LUTS_FOR_FP
    int temp = label + dist(gen());
    if (temp < p)
        label = temp;
    else
        label = temp - p;
#else
    label = lut_add(label, dist(gen()));
#endif
    return *this;
}

template <uint16_t p>
size_t Fp<p>::get_multiplicative_order() const {
#ifdef CECCO_ERASURE_SUPPORT
    if (is_erased()) throw std::invalid_argument("trying to calculate multiplicative order of erased element");
#endif
    if (label == 0) throw std::invalid_argument("trying to calculate multiplicative order of additive neutral element");
#ifndef CECCO_USE_LUTS_FOR_FP
    return details::calculate_multiplicative_order(*this);
#else
    return lut_mul_ord(label);
#endif
}

template <uint16_t p>
size_t Fp<p>::get_additive_order() const {
#ifdef CECCO_ERASURE_SUPPORT
    if (is_erased()) throw std::invalid_argument("trying to calculate additive order of erased element");
#endif
    if (label == 0) return 1;
    return p;
}

template <uint16_t p>
void Fp<p>::show_tables() {
    std::cout << "addition table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < p; ++i) {
        for (label_t j = 0; j < p; ++j) std::cout << (int)lut_add(i, j) << ", ";
        std::cout << std::endl;
    }

#ifdef CECCO_USE_LUTS_FOR_FP
    std::cout << "additive inverse table (row headers omitted)" << std::endl;
    for (label_t i = 0; i < p; ++i) std::cout << (int)lut_neg(i) << std::endl;
#endif

    std::cout << "multiplication table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < p; ++i) {
        for (label_t j = 0; j < p; ++j) std::cout << (int)lut_mul(i, j) << ", ";
        std::cout << std::endl;
    }

#ifdef CECCO_USE_LUTS_FOR_FP
    std::cout << "multiplicative inverse table (row headers omitted)" << std::endl;
    for (label_t i = 0; i < p; ++i) std::cout << (int)lut_inv(i) << std::endl;
#endif
}

#ifdef CECCO_ERASURE_SUPPORT
template <uint16_t p>
constexpr Fp<p>& Fp<p>::erase() noexcept {
    label = std::numeric_limits<label_t>::max();
    return *this;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::unerase() noexcept {
    if (is_erased()) (*this) = 0;
    return *this;
}
#endif

/// @brief Print as the integer label (or @ref ERASURE_MARKER if erased)
template <uint16_t p>
std::ostream& operator<<(std::ostream& os, const Fp<p>& e) {
#ifdef CECCO_ERASURE_SUPPORT
    if (e.is_erased()) {
        os << ERASURE_MARKER;
        return os;
    }
#endif
    os << (int)e.get_label();
    return os;
}

/**
 * @brief Extension field 𝔽_{q^m} ≅ B[x]/(f(x)), constructed from a base field and an
 *        irreducible monic modulus polynomial
 *
 * @tparam B Base field, either @ref CECCO::Fp, another @ref CECCO::Ext, or @ref CECCO::Iso
 * @tparam modulus Coefficients of f(x) low-to-high (constant term first); leading coefficient
 *                 must be 1, degree m ≥ 2; f(x) must be irreducible over B
 * @tparam mode @ref CECCO::LutMode::RunTime (default) or @ref CECCO::LutMode::CompileTime
 *
 * The result has q = |B| and Q = q^m elements; elements are stored as `label_t` integers in
 * {0, …, Q − 1}. Towers are built by re-using `Ext` as the base. Pick `CompileTime` for small
 * fields (zero startup, larger binary) or `RunTime` for large fields (faster compilation,
 * lazy initialisation on first use). LUT modes mix freely across a tower; a `CompileTime`
 * extension can only be built when its base satisfies @ref CECCO::Ext::is_constexpr_ready.
 *
 * @warning A non-irreducible @p modulus is detected during LUT construction and surfaces as
 * `std::invalid_argument` at runtime, or as a constexpr-evaluation error at compile time when
 * `mode == LutMode::CompileTime`. Use @ref CECCO::find_irreducible to obtain a valid one.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F3  = Fp<3>;
 * using F9  = Ext<F3, {2, 2, 1}>;            // 𝔽₃[x]/(2 + 2x + x²)
 * using F27 = Ext<F9, {1, 2, 1}>;            // 3-level tower 𝔽₃ ⊂ 𝔽₉ ⊂ 𝔽₂₇
 *
 * F9 a(5), b(7);
 * auto c = a * b + F9(1);                    // arithmetic
 * Vector<F3> coeffs = a.as_vector<F3>();     // coefficient vector over the prime subfield
 * size_t ord = a.get_multiplicative_order();
 *
 * F27 x(100);
 * Vector<F3> v = x.as_vector<F3>();          // descend straight to the prime subfield
 * @endcode
 */
template <FiniteFieldType B, MOD modulus, LutMode mode = LutMode::RunTime>
class Ext : public details::Field<Ext<B, modulus, mode>> {
    static_assert(modulus.back() == 1, "provided polynomial is not monic");
    static_assert(modulus.size() > 2, "provided polynomial has degree less than 2");
    static_assert(B::ready(), "base field must be fully instantiated");
    static_assert(mode == LutMode::RunTime || B::is_constexpr_ready(),
                  "CompileTime extension fields require all base fields to have constexpr-ready interfaces");

   private:
    static constexpr uint8_t m = modulus.size() - 1;
    static constexpr size_t q = B::get_size();
    static constexpr size_t Q = sqm(q, m);

   public:
    using label_t = ::CECCO::label_t<Q>;
    using BASE_FIELD = B;

    /// @brief Default constructor: 0
    constexpr Ext() noexcept : label{0} {}

    /// @brief Construct from an integer label `l ∈ {0, …, Q − 1}`; throws `std::invalid_argument` otherwise
    Ext(int l);

    constexpr Ext(const Ext& other) noexcept = default;
    constexpr Ext(Ext&& other) noexcept = default;

    /// @brief Embed a base-field element via the natural embedding B → Ext<B, modulus, mode>
    Ext(const B& other);

    /**
     * @brief Cross-field conversion from another extension field of the same characteristic
     *
     * @tparam S Base of the source extension field
     * @tparam ext_modulus Modulus of the source extension field
     * @throws std::invalid_argument on a downcast whose source value lies outside the target
     *
     * Picks the cheapest available path: direct copy if the type matches; cached @ref Isomorphism
     * for isomorphic fields; cached @ref Embedding for tower relationships (upcast cannot fail,
     * downcast may); two-step conversion via @ref details::largest_common_subfield_t for
     * unrelated towers.
     */
    template <FiniteFieldType S, MOD ext_modulus, LutMode ext_mode>
        requires(!std::is_same_v<Ext<B, modulus, mode>, Ext<S, ext_modulus, ext_mode>>)
    Ext(const Ext<S, ext_modulus, ext_mode>& other);

    /**
     * @brief Construct from a coefficient vector over a subfield
     *
     * @tparam T Subfield type
     * @throws std::invalid_argument if `v.length()` does not match the extension degree of this field over T
     *
     * Reads @p v as base-|T| coefficients (low-to-high). With @ref CECCO_ERASURE_SUPPORT, an
     * erased component in @p v produces an erased element here.
     */
    template <FiniteFieldType T>
    Ext(const Vector<T>& v);

    /**
     * @brief Cross-field conversion from an `Iso` of the same characteristic
     *
     * Delegates to the `Ext(other.main())` overload, letting the Ext-to-Ext logic choose the
     * conversion path; works equally for downcasts, upcasts, and cross-tower bridges.
     *
     * @throws std::invalid_argument if no conversion path exists
     */
    template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
    Ext(const Iso<MAIN, OTHERS...>& other);

    /**
     * @brief Embed a prime-field element when 𝔽_p is a (possibly indirect) subfield
     *
     * Uses the cached @ref Embedding from `Fp<p>` to this `Ext`; cannot fail for any value
     * permitted by `SubfieldOf<Ext, Fp<p>>`.
     */
    template <uint16_t p>
    constexpr Ext(const Fp<p>& other)
        requires SubfieldOf<Ext<B, modulus, mode>, Fp<p>> && (!std::is_same_v<B, Fp<p>>);

    /// @brief Assign integer label `l ∈ {0, …, Q − 1}`; throws `std::invalid_argument` otherwise
    constexpr Ext& operator=(int l);

    constexpr Ext& operator=(const Ext& rhs) noexcept = default;
    Ext& operator=(Ext&& rhs) noexcept = default;

    /// @brief Cross-field assignment from another extension (copy-and-swap; same semantics as the constructor)
    template <FiniteFieldType S, MOD ext_modulus, LutMode ext_mode>
        requires(S::get_characteristic() == B::get_characteristic())
    Ext& operator=(const Ext<S, ext_modulus, ext_mode>& other);

    /// @brief Embed an `Fp` element of matching characteristic (copy-and-swap)
    template <uint16_t p>
        requires(p == B::get_characteristic())
    Ext& operator=(const Fp<p>& other);

    /// @brief Cross-field assignment from an `Iso` of matching characteristic (copy-and-swap)
    template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
        requires(MAIN::get_characteristic() == B::get_characteristic())
    Ext& operator=(const Iso<MAIN, OTHERS...>& other);

    constexpr bool operator==(const Ext& rhs) const noexcept { return label == rhs.get_label(); }

    /// @brief Additive inverse (lvalue): returns a new element
    constexpr Ext operator-() const&;
    /// @brief Additive inverse (rvalue): in place
    constexpr Ext& operator-() &&;

    /// @brief `*this += rhs` via the addition LUT
    constexpr Ext& operator+=(const Ext& rhs);
    /// @brief `*this -= rhs` via the addition LUT applied to `−rhs`
    constexpr Ext& operator-=(const Ext& rhs);
    /// @brief `*this *= rhs` via the multiplication LUT
    constexpr Ext& operator*=(const Ext& rhs);
    /// @brief Scalar multiplication by an `int` (repeated addition, reduced mod characteristic)
    constexpr Ext& operator*=(int s);
    /// @brief `*this /= rhs`; throws `std::invalid_argument` if rhs is zero
    Ext& operator/=(const Ext& rhs);

    /// @brief Uniform random element in {0, …, Q − 1}
    Ext& randomize();
    /// @brief Like @ref randomize but guaranteed to differ from the current value
    Ext& randomize_force_change();

    /**
     * @brief Multiplicative order in the field's multiplicative group
     *
     * @throws std::invalid_argument if `*this` is zero
     */
    size_t get_multiplicative_order() const;

    /// @brief Additive order: 1 for zero, characteristic p otherwise
    size_t get_additive_order() const;

    /**
     * @brief Minimal polynomial of this element over a subfield S (defaults to immediate base field B)
     *
     * @tparam S Subfield (`SubfieldOf<Ext, S>`)
     *
     * Computed from the S-conjugacy orbit { α, α^{|S|}, α^{|S|²}, … } as the polynomial whose
     * roots are exactly those conjugates. Useful for working with polynomials over an
     * intermediate or the prime subfield in a tower.
     *
     * @code{.cpp}
     * F16 alpha = F16::get_generator();
     * auto over_F4 = alpha.get_minimal_polynomial<F4>();
     * auto over_F2 = alpha.get_minimal_polynomial<F2>();   // absolute
     * @endcode
     */
    template <FiniteFieldType S = B>
    Polynomial<S> get_minimal_polynomial() const
        requires SubfieldOf<Ext<B, modulus, mode>, S>;

    /// @brief Human-readable description (size, base field, modulus)
    static std::string get_info();

    static constexpr size_t get_characteristic() noexcept { return B::get_p(); }
    /// @brief Underlying integer label in {0, …, Q − 1}
    constexpr size_t get_label() const noexcept { return label; }

    /**
     * @brief True for `LutMode::CompileTime`, false for `LutMode::RunTime`
     *
     * Used by extensions further up in a tower to gate their own `CompileTime` instantiation
     * (a `CompileTime` field cannot be built on a `RunTime` base).
     */
    static constexpr bool is_constexpr_ready() noexcept { return mode == LutMode::CompileTime; }

    /// @brief Modulus polynomial f(x) — the irreducible used to construct this field
    static constexpr Polynomial<B> get_modulus();

    /**
     * @brief Generator (primitive element) of the multiplicative group
     *
     * Smallest label with multiplicative order |field| − 1; cached statically.
     */
    static Ext get_generator();

    static constexpr size_t get_p() noexcept { return B::get_p(); }
    static constexpr size_t get_m() noexcept { return m; }
    static constexpr size_t get_q() noexcept { return Q; }
    static constexpr size_t get_size() noexcept { return Q; }

    /// @brief Isomorphism mapping `Ext → T` for any `Isomorphic<Ext, T>` target
    template <FiniteFieldType T>
    static Isomorphism<Ext, T> isomorphism_to();

    /// @brief Print all lookup tables to `std::cout` (debugging aid)
    static void show_tables();

    /// @brief Always true (finite fields are unordered)
    constexpr bool has_positive_sign() const noexcept { return true; }
    /// @brief True iff this is the additive identity
    constexpr bool is_zero() const noexcept { return label == 0; }

#ifdef CECCO_ERASURE_SUPPORT
    /**
     * @brief Mark this element as erased (encoded as `label == max(label_t)`)
     *
     * @warning Erased elements must not participate in field arithmetic — see
     * @ref CECCO_ERASURE_SUPPORT.
     */
    constexpr Ext& erase() noexcept;
    /// @brief Clear the erasure flag, resetting to the additive identity
    constexpr Ext& unerase() noexcept;
    /// @brief Test whether this element is currently erased
    constexpr bool is_erased() const noexcept { return label == std::numeric_limits<label_t>::max(); }
#endif

    /**
     * @brief Coordinate vector over a proper subfield T (defaults to the base field B)
     *
     * @tparam T Subfield (`SubfieldOf<Ext, T>` and `T ≠ Ext`)
     * @return Vector of length [Ext : T]; round-trip through the `Ext(Vector<T>)` constructor
     *
     * @code{.cpp}
     * F16 x(10);
     * Vector<F4> v4 = x.as_vector<F4>();   // length 2
     * Vector<F2> v2 = x.as_vector<F2>();   // length 4
     * F16 y = F16(v2);                      // round trip
     * @endcode
     */
    template <FiniteFieldType T = B>
        requires(SubfieldOf<Ext<B, modulus, mode>, T>) && (!std::is_same_v<Ext<B, modulus, mode>, T>)
    Vector<T> as_vector() const;

    /**
     * @brief Compile-time signal that all LUTs are constructed
     *
     * Used by extensions further up in a tower to defer their own LUT computation until this
     * one is fully instantiated, preventing compiler recursion-depth issues. The implementation
     * is the constexpr `luts_ready` flag, which forces immediate evaluation of every LUT on
     * a `CompileTime` instantiation.
     *
     * @warning Only well-defined after all LUT declarations in the class body have been seen.
     */
    static constexpr bool ready() { return luts_ready; }

   private:
    label_t label;  ///< Element label in {0, 1, ..., Q - 1}

    /// @brief Generator element storage
    struct Gen {
        label_t value{};  ///< Label of primitive element
    };

    /// @brief Type alias for 1D lookup tables
    using Lut1D = details::Lut1D<label_t, Q>;
    /// @brief Type alias for 2D lookup tables
    using Lut2D = details::Lut2D<label_t, Q>;
    /// @brief Type alias for coefficient lookup tables
    using Lut2Dcoeff = details::Lut2Dcoeff<typename B::label_t, m, Q>;

   public:
    /**
     * @name Precomputed Lookup Tables
     * @brief Compile-time generated tables for field operations
     * @{
     */

    /// @brief Element coefficients: lut_coeff[i] = polynomial coefficients of element i
    static constexpr auto lambda = []() constexpr -> Lut2Dcoeff {
        Lut2Dcoeff lut_coeff;

        // calculate base-q representation of all labels (MSB left)
        for (label_t i = 0; i < Q; ++i) {
            lut_coeff.values[i][m - 1] = i % q;
            label_t t = q;
            for (uint8_t s = 1; s < m; ++s) {
                lut_coeff.values[i][m - 1 - s] = (i / t) % q;
                t *= q;
            }
        }

        return lut_coeff;
    };

    using LUT_COEFF = details::LutHolderNoProvider<Lut2Dcoeff, lambda, mode>;
    static constexpr auto& lut_coeff() { return LUT_COEFF::get_lut(); }

    /// @brief Addition table: lut_add(a,b) = (polynomial_a + polynomial_b) mod f(X)
    static constexpr Lut2D compute_add_lut_wrapper(const Lut2Dcoeff& (*provider)()) {
        const Lut2Dcoeff& coeffs = provider();
        return details::compute_polynomial_addition_table<label_t, Q, Lut2Dcoeff, m, B>(coeffs);
    }
    using LUT_ADD = details::LutHolder<Lut2D, Lut2Dcoeff, &lut_coeff, &compute_add_lut_wrapper, mode>;
    static constexpr auto& lut_add() { return LUT_ADD::get_lut(); }

    /// @brief Multiplication table: lut_mul(a,b) = (polynomial_a * polynomial_b) mod f(X)
    static constexpr Lut2D compute_mul_lut_wrapper(const Lut2Dcoeff& (*provider)()) {
        const Lut2Dcoeff& coeffs = provider();
        return details::compute_polynomial_multiplication_table<label_t, Q, Lut2Dcoeff, m, B, modulus>(coeffs);
    }
    using LUT_MUL = details::LutHolder<Lut2D, Lut2Dcoeff, &lut_coeff, &compute_mul_lut_wrapper, mode>;
    static constexpr auto& lut_mul() { return LUT_MUL::get_lut(); }

    /// @brief Additive inverse table: lut_neg[a] = -a
    static constexpr Lut1D compute_neg_lut_wrapper(const Lut2D& (*provider)()) {
        const Lut2D& add = provider();
        return details::compute_additive_inverses_search<label_t, Q>(add);
    }
    using LUT_NEG = details::LutHolder<Lut1D, Lut2D, &lut_add, &compute_neg_lut_wrapper, mode>;
    static constexpr auto& lut_neg() { return LUT_NEG::get_lut(); }

    /// @brief Multiplicative inverse table: lut_inv[a] = a^(-1)
    static constexpr Lut1D compute_inv_lut_wrapper(const Lut2D& (*provider)()) {
        const Lut2D& mul = provider();
        return details::compute_multiplicative_inverses_search<label_t, Q>(mul);
    }
    using LUT_INV = details::LutHolder<Lut1D, Lut2D, &lut_mul, &compute_inv_lut_wrapper, mode>;
    static constexpr auto& lut_inv() { return LUT_INV::get_lut(); }

    /// @brief Multiplicative order table: lut_mul_ord[a] = order of a in multiplicative group
    static constexpr Lut1D compute_mul_ord_lut_wrapper(const Lut2D& (*provider)()) {
        const Lut2D& mul = provider();
        return details::compute_multiplicative_orders<label_t, Q>(mul);
    }
    using LUT_MUL_ORD = details::LutHolder<Lut1D, Lut2D, &lut_mul, &compute_mul_ord_lut_wrapper, mode>;
    static constexpr auto& lut_mul_ord() { return LUT_MUL_ORD::get_lut(); }

    /// @brief Primitive element (generator) of the multiplicative group
    static constexpr Gen compute_generator_wrapper(const Lut1D& (*provider)()) {
        const Lut1D& mul_ord = provider();
        return Gen{details::find_generator<label_t, Q>(mul_ord)};
    }
    using LUT_GEN = details::LutHolder<Gen, Lut1D, &lut_mul_ord, &compute_generator_wrapper, mode>;
    static constexpr auto& g() { return LUT_GEN::get_lut(); }

    // LUT-compatible interface for use as BaseFieldType
    // These allow Ext fields to be used as base fields for higher-order extensions
    static constexpr label_t lut_add(label_t a, label_t b) { return lut_add()(a, b); }
    static constexpr label_t lut_mul(label_t a, label_t b) { return lut_mul()(a, b); }
    static constexpr label_t lut_neg(label_t a) { return lut_neg()(a); }
    static constexpr label_t lut_inv(label_t a) { return lut_inv()(a); }

    static constexpr bool luts_ready = []() constexpr {
        // Ensure base field LUTs are ready first
        static_assert(B::ready());  // Forces base field LUT computation

        if constexpr (mode == LutMode::CompileTime) {
            // For CompileTime mode, force LUT computation during compilation
            static_assert(lut_coeff().values[0][0] == 0);  // Forces immediate calculation lut_coeff
            static_assert(lut_add()(0, 0) == 0);           // Forces immediate calculation of lut_add
            static_assert(lut_neg()(0) == 0);              // Forces immediate calculation of lut_neg
            static_assert(lut_mul()(0, 1) == 0);           // Forces immediate calculation of lut_mul
            static_assert(lut_mul_ord()(0) == 0);          // Forces immediate calculation of lut_mul_ord
            static_assert(g().value != 1);                 // Forces immediate calculation of generator g
        }
        // For Runtime mode, LUTs will be computed when first accessed
        return true;
    }();

    /** @} */
};

/* member functions for Ext */

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>::Ext(int l) {
    if (l < 0 || l >= Q) throw std::invalid_argument("l must be positive and no larger than Q-1");
    label = l;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>::Ext(const B& other) {
#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif
    // Use cached subfield embedding for mathematically correct embedding
    auto embedding = Embedding<B, Ext>();
    Ext result = embedding(other);
    label = result.get_label();
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType T>
Isomorphism<Ext<B, modulus, mode>, T> Ext<B, modulus, mode>::isomorphism_to() {
    return Isomorphism<Ext<B, modulus, mode>, T>();
}

#ifdef CECCO_ERASURE_SUPPORT
template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::erase() noexcept {
    label = std::numeric_limits<label_t>::max();
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::unerase() noexcept {
    if (is_erased()) (*this) = 0;
    return *this;
}
#endif

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType S, MOD ext_modulus, LutMode ext_mode>
    requires(!std::is_same_v<Ext<B, modulus, mode>, Ext<S, ext_modulus, ext_mode>>)
Ext<B, modulus, mode>::Ext(const Ext<S, ext_modulus, ext_mode>& other) {
    // Ensure same characteristic
    static_assert(Ext<B, modulus, mode>::get_characteristic() == Ext<S, ext_modulus, ext_mode>::get_characteristic(),
                  "trying to convert between fields with different characteristic");

#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif

    using IN = Ext<S, ext_modulus, ext_mode>;
    using OUT = Ext;

    // Note: same-field case handled by simple copy constructor
    if constexpr (Isomorphic<OUT, IN>) {
        // Isomorphic fields - use isomorphism for conversion
        auto iso = Isomorphism<IN, OUT>();
        OUT result = iso(other);
        label = result.get_label();
    } else if constexpr (SubfieldOf<OUT, IN>) {
        // Upcast: Source ⊆ Target (ExtensionOf<Source, Target>) - cannot throw
        // Use cached subfield embedding for mathematically correct embedding
        auto embedding = Embedding<IN, OUT>();
        OUT result = embedding(other);
        label = result.get_label();
    } else if constexpr (SubfieldOf<IN, OUT>) {
        // Downcast: Target ⊆ Source (SubfieldOf<Target, Source>) - may throw
        // Use cached subfield embedding to find if superfield element is in subfield
        auto embedding = Embedding<OUT, IN>();
        OUT result = embedding.extract(other);
        label = result.get_label();
    } else {
        // Fields with same characteristic but not directly related - convert via largest common subfield in order
        // to maximize
        using CommonField = details::largest_common_subfield_t<OUT, IN>;

        CommonField intermediate(other);  // Extract to largest common subfield, throws if not possible
        *this = Ext(intermediate);        // Embed from largest common subfield, works always
    }
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType T>
Ext<B, modulus, mode>::Ext(const Vector<T>& v) {
    if constexpr (std::is_same_v<B, T>) {
        if (v.get_n() != m)
            throw std::invalid_argument(
                "trying to construct extension field element using base field vector of wrong length");

#ifdef CECCO_ERASURE_SUPPORT
        bool erased = false;
        for (size_t i = 0; i < v.get_n(); ++i)
            if (v[i].is_erased()) {
                erased = true;
                break;
            }
        if (erased) {
            this->erase();
            return;
        }
#endif
        label = v.as_integer();

    } else {
        static_assert(SubfieldOf<Ext<B, modulus, mode>, T>,
                      "extension field elements can only be constructed from vectors over subfields");

        if (v.get_n() != get_m() * B::get_m())
            throw std::invalid_argument(
                "trying to construct extension field element using subfield vector of wrong length");

#ifdef CECCO_ERASURE_SUPPORT
        bool erased = false;
        for (size_t i = 0; i < v.get_n(); ++i)
            if (v[i].is_erased()) {
                erased = true;
                break;
            }
        if (erased) {
            this->erase();
            return;
        }
#endif
        {
            Vector<B> intermediate(get_m());
            for (uint8_t i = 0; i < get_m(); ++i) {
                Vector<T> sub(B::get_m());
                for (uint8_t j = 0; j < B::get_m(); ++j) sub.set_component(j, v[i * B::get_m() + j]);
                intermediate.set_component(i, B(sub));
            }

            *this = Ext(intermediate);
        }
    }
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
Ext<B, modulus, mode>::Ext(const Iso<MAIN, OTHERS...>& other) {
    static_assert(Ext<B, modulus, mode>::get_characteristic() == Iso<MAIN, OTHERS...>::get_characteristic(),
                  "trying to convert between fields with different characteristic");

#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif

    using IN = Iso<MAIN, OTHERS...>;
    using OUT = Ext;

    // Branch 1: If IN is isomorphic to out -> implies IN is isomorphic to MAIN
    if constexpr (Isomorphic<IN, OUT>) {
        auto isomorphism = Isomorphism<MAIN, OUT>();
        OUT result = isomorphism(other.main());
        label = result.get_label();

        // Branch 2 (upcast): If IN is subfield of OUT
    } else if constexpr (SubfieldOf<OUT, IN>) {
        if constexpr (SubfieldOf<OUT, MAIN>) {
            auto embedding = Embedding<MAIN, OUT>();
            OUT result = embedding(other.main());
            label = result.get_label();
        } else if constexpr ((SubfieldOf<OUT, OTHERS> || ...)) {
            auto try_embedding = [&]<typename OtherType>() {
                if constexpr (SubfieldOf<OUT, OtherType>) {
                    auto embedding = Embedding<OtherType, OUT>();
                    OtherType other_repr(other);
                    OUT result = embedding(other_repr);
                    label = result.get_label();
                }
            };
            (try_embedding.template operator()<OTHERS>(), ...);
        }

        // Branch 3 (downcast): If IN is superfield of OUT
    } else if constexpr (SubfieldOf<IN, OUT>) {
        if constexpr (SubfieldOf<MAIN, OUT>) {
            auto embedding = Embedding<OUT, MAIN>();
            OUT result = embedding.extract(other.main());
            label = result.get_label();
        } else if constexpr ((SubfieldOf<OTHERS, OUT> || ...)) {
            auto try_downcast = [&]<typename OtherType>() {
                if constexpr (SubfieldOf<OtherType, OUT>) {
                    auto embedding = Embedding<OUT, OtherType>();
                    OtherType other_repr(other);
                    OUT result = embedding.extract(other_repr);
                    label = result.get_label();
                }
            };
            (try_downcast.template operator()<OTHERS>(), ...);
        }

        // Branch 4 (cross-cast through largest common subfield): This is the else case
    } else {
        using CommonField = details::largest_common_subfield_t<OUT, IN>;

        if constexpr (details::iso_info<CommonField>::is_iso) {
            // CommonField is an Iso, continue with its MAIN
            using CommonMainField = typename details::iso_info<CommonField>::main_type;
            CommonMainField intermediate(other);  // Downcast other to CommonField's MAIN
            *this = OUT(intermediate);            // Use existing cross-field Ext->Ext constructor
        } else {
            // CommonField is an Ext, continue with CommonField
            CommonField intermediate(other);  // Downcast other to CommonField
            *this = OUT(intermediate);        // Use existing cross-field Ext->Ext constructor
        }
    }
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <uint16_t p>
constexpr Ext<B, modulus, mode>::Ext(const Fp<p>& other)
    requires SubfieldOf<Ext<B, modulus, mode>, Fp<p>> && (!std::is_same_v<B, Fp<p>>)
{
    static_assert(Ext<B, modulus, mode>::get_characteristic() == p,
                  "Prime field characteristic must match extension field characteristic");

#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif

    // Use the cached embedding for mathematically correct embedding
    auto embedding = Embedding<Fp<p>, Ext<B, modulus, mode>>();
    Ext<B, modulus, mode> result = embedding(other);
    label = result.get_label();
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator=(int l) {
    if (l < 0 || l >= Q) throw std::invalid_argument("l must be positive and no larger than Q-1");
    label = l;
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType S, MOD ext_modulus, LutMode ext_mode>
    requires(S::get_characteristic() == B::get_characteristic())
Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator=(const Ext<S, ext_modulus, ext_mode>& rhs) {
    Ext temp(rhs);
    std::swap(*this, temp);
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <uint16_t p>
    requires(p == B::get_characteristic())
Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator=(const Fp<p>& rhs) {
    Ext temp(rhs);
    std::swap(*this, temp);
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
    requires(MAIN::get_characteristic() == B::get_characteristic())
Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator=(const Iso<MAIN, OTHERS...>& rhs) {
    Ext temp(rhs);
    std::swap(*this, temp);
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode> Ext<B, modulus, mode>::operator-() const& {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return Ext().erase();
#endif
    Ext res(*this);
    if (res.label != 0) res.label = lut_neg()(res.label);
    return res;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator-() && {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return this->erase();
#endif
    if (label != 0) label = lut_neg()(label);
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator+=(const Ext& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    label = lut_add()(label, rhs.get_label());
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator-=(const Ext& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    label = lut_add()(label, lut_neg()(rhs.get_label()));
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator*=(const Ext& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    label = lut_mul()(label, rhs.get_label());
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator*=(int s) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return *this;
#endif
    if constexpr (get_characteristic() != 0) s %= static_cast<int>(get_characteristic());
    Ext res = daa<Ext>(*this, s);
    *this = std::move(res);
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator/=(const Ext& rhs) {
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#endif
    if (rhs.label == 0) throw std::invalid_argument("trying to divide by zero");
    label = lut_mul()(label, lut_inv()(rhs.get_label()));
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>& Ext<B, modulus, mode>::randomize() {
#ifdef CECCO_ERASURE_SUPPORT
    this->unerase();
#endif
    thread_local std::uniform_int_distribution<label_t> dist(0, Q - 1);
    label = dist(gen());
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>& Ext<B, modulus, mode>::randomize_force_change() {
#ifdef CECCO_ERASURE_SUPPORT
    this->unerase();
#endif
    thread_local std::uniform_int_distribution<label_t> dist(1, Q - 1);
    label = lut_add()(label, dist(gen()));
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
size_t Ext<B, modulus, mode>::get_multiplicative_order() const {
#ifdef CECCO_ERASURE_SUPPORT
    if (is_erased()) throw std::invalid_argument("trying to calculate multiplicative order of erased element");
#endif
    if (label == 0) throw std::invalid_argument("trying to calculate multiplicative order of additive neutral element");
    return lut_mul_ord()(label);
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
size_t Ext<B, modulus, mode>::get_additive_order() const {
#ifdef CECCO_ERASURE_SUPPORT
    if (is_erased()) throw std::invalid_argument("trying to calculate additive order of erased element");
#endif
    if (label == 0) return 1;
    return get_characteristic();
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType S>
Polynomial<S> Ext<B, modulus, mode>::get_minimal_polynomial() const
    requires SubfieldOf<Ext<B, modulus, mode>, S>
{
#ifdef CECCO_ERASURE_SUPPORT
    if (is_erased()) throw std::invalid_argument("trying to compute minimal polynomial of erased element");
#endif

    Polynomial<Ext> res = {1};
    size_t i = 0;
    do {
        const Ext beta = sqm<Ext>(*this, sqm<size_t>(S::get_size(), i));
        if (i > 0 && beta == *this) break;
        res *= Polynomial<Ext>({-beta, 1});
        ++i;
    } while (true);

    return Polynomial<S>(res);
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
std::string Ext<B, modulus, mode>::get_info() {
    std::stringstream ss;
    ss << "finite field with " + std::to_string(Q) + " elements, specified as degree " + std::to_string(m) +
              " extension of (" + B::get_info() + "), irreducible polynomial ";
    ss << get_modulus();
    return ss.str();
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Polynomial<B> Ext<B, modulus, mode>::get_modulus() {
    Polynomial<B> rho;
    uint8_t i = 0;
    for (auto it = modulus.cbegin(); it != modulus.cend(); ++it) {
        rho.set_coefficient(i, *it);
        ++i;
    }
    return rho;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode> Ext<B, modulus, mode>::get_generator() {
    // Use local static cache for each specific field type
    static std::once_flag computed_flag;
    static Ext cached_generator{0};

    std::call_once(computed_flag, []() { cached_generator = Ext(g().value); });

    return cached_generator;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType T>
    requires(SubfieldOf<Ext<B, modulus, mode>, T>) && (!std::is_same_v<Ext<B, modulus, mode>, T>)
Vector<T> Ext<B, modulus, mode>::as_vector() const {
    if constexpr (std::is_same_v<B, T>) {
        Vector<T> res(m);
#ifdef CECCO_ERASURE_SUPPORT
        if (is_erased()) {
            std::vector<size_t> indices(m);
            std::iota(indices.begin(), indices.end(), 0);
            res.erase_components(indices);
            return res;
        }
#endif
        const auto coeffs = lut_coeff().values[label];
        for (uint8_t i = 0; i < m; ++i) res.set_component(i, coeffs[i]);
        return res;
    } else {
        static_assert(SubfieldOf<Ext<B, modulus, mode>, T>,
                      "extension field elements can only be converted into vectors over subfields");
        const auto intermediate = as_vector<B>();  // Explicitly call with B
        Vector<T> res(get_m() * B::get_m());
        for (uint8_t i = 0; i < get_m(); ++i) {
            auto sub = intermediate[i].template as_vector<T>();
            // Copy sub_vector elements into result at appropriate positions
            for (uint8_t j = 0; j < B::get_m(); ++j) {
                res.set_component(i * B::get_m() + j, sub[j]);
            }
        }
        return res;
    }
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
void Ext<B, modulus, mode>::show_tables() {
    std::cout << "addition table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) {
        for (label_t j = 0; j < Q; ++j) std::cout << (int)(lut_add()(i, j)) << ", ";
        std::cout << std::endl;
    }

    std::cout << "additive inverse table (row headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) std::cout << (int)(lut_neg()(i)) << std::endl;

    std::cout << "multiplication table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) {
        for (label_t j = 0; j < Q; ++j) std::cout << (int)(lut_mul()(i, j)) << ", ";
        std::cout << std::endl;
    }

    std::cout << "multiplicative inverse table (row headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) std::cout << (int)(lut_inv()(i)) << std::endl;

    std::cout << "multiplicative order table (row headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) std::cout << (int)(lut_mul_ord()(i)) << std::endl;

    std::cout << "element coefficients table (column headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) {
        std::cout << (int)i << ": ";
        for (uint8_t j = 0; j < m; ++j) std::cout << (int)(lut_coeff().values[i][j]) << ", ";
        std::cout << std::endl;
    }

    std::cout << "generator (with mult. order)" << std::endl;
    std::cout << get_generator() << " (" << get_generator().get_multiplicative_order() << ")" << std::endl;
}

/// @brief Print as the integer label (or @ref ERASURE_MARKER if erased)
template <FiniteFieldType B, MOD modulus, LutMode mode>
std::ostream& operator<<(std::ostream& os, const Ext<B, modulus, mode>& e) {
#ifdef CECCO_ERASURE_SUPPORT
    if (e.is_erased()) {
        os << ERASURE_MARKER;
        return os;
    }
#endif
    os << (int)e.get_label();
    return os;
}

/**
 * @brief Single logical field unifying several pairwise-isomorphic representations
 *
 * @tparam MAIN Primary representation; satisfies @ref CECCO::FiniteFieldType. All operations
 *              are forwarded to it, so the `Iso` inherits its @ref CECCO::LutMode and
 *              performance characteristics.
 * @tparam OTHERS Alternative representations of the same abstract field (each `Isomorphic<MAIN, OTHER>`).
 *                Pairwise distinctness is enforced at instantiation.
 *
 * Useful for merging two construction towers that meet at the same mathematical field — e.g.
 * 𝔽₁₆ built once from 𝔽₂ and once from 𝔽₄. With those two `Ext` types wrapped in an `Iso`,
 * @ref CECCO::SubfieldOf can recognise both as containing 𝔽₂ and 𝔽₄, which makes the
 * cross-field constructors of @ref CECCO::Ext and @ref CECCO::Iso pick optimal paths. The
 * `OTHERS` representations may use different LUT modes from `MAIN`; isomorphism conversions
 * across representations happen at runtime via cached maps.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F2   = Fp<2>;
 * using F4_a = Ext<F2, {1, 1, 1}>;
 * using F4_b = Ext<F2, {1, 0, 1}>;
 * using F4   = Iso<F4_a, F4_b>;
 *
 * F4_a a(2);
 * F4_b b(3);
 * F4 c(a);                   // wrap a into the unified field
 * c += b;                    // OTHERS-overload converts b into MAIN behind the scenes
 * F4_b d = c.as<F4_b>();     // explicit projection back to F4_b
 * @endcode
 */
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
class Iso : public details::Base {
    // Type safety: validate isomorphism at template instantiation
    static_assert((Isomorphic<MAIN, OTHERS> && ...), "All OTHERS must be isomorphic to MAIN");
    // Ensure we have multiple representations to unify, otherwise Iso doesn't make sense
    static_assert(sizeof...(OTHERS) > 0, "Iso requires at least two field representations");
    // Ensure all fields in the  union of MAIN and OTHERS... are pairwise distinct
    static_assert(details::pairwise_distinct<MAIN, OTHERS...>(),
                  "All field representations in Iso must be pairwise distinct");
    // Comment: the last two assert that prime fields cannot occur in Isos

   public:
    using label_t = typename MAIN::label_t;
    using BASE_FIELD = typename MAIN::BASE_FIELD;

   private:
    MAIN main_;

   public:
    /// @brief Default constructor: 0 (consistent across all representations)
    constexpr Iso() noexcept : main_() {}

    /// @brief Construct from `int` via the MAIN representation
    constexpr Iso(int l) : main_(l) {}

    /// @brief Wrap a MAIN-representation element
    constexpr Iso(const MAIN& other) noexcept : main_(other) {}
    /// @brief Wrap a MAIN-representation rvalue
    constexpr Iso(MAIN&& other) noexcept : main_(std::move(other)) {}

    /// @brief Wrap an `OTHERS` element by converting to MAIN
    template <BelongsTo<OTHERS...> OTHER>
    constexpr Iso(const OTHER& other) : main_(MAIN(other)) {}
    /// @brief Wrap an `OTHERS` rvalue by converting to MAIN
    template <BelongsTo<OTHERS...> OTHER>
    constexpr Iso(OTHER&& other) : main_(MAIN(std::move(other))) {}

    constexpr Iso(const Iso& other) noexcept = default;
    constexpr Iso(Iso&& other) noexcept = default;

    /**
     * @brief Construct from a coefficient vector over a subfield T
     *
     * @tparam T Subfield type
     * @throws std::invalid_argument if no representation in the `Iso` can be constructed from @p v
     *
     * Tries MAIN first, then each of OTHERS until one succeeds. Round-trip correct with
     * @ref as_vector.
     */
    template <FiniteFieldType T>
    Iso(const Vector<T>& v);

    /**
     * @brief Cross-field conversion from an extension field
     *
     * Delegates to `MAIN(Ext(...))` (or to an `OTHERS` representation when MAIN cannot reach
     * the source directly), then stores the MAIN result. All paths supported by the
     * `Ext`-from-`Ext` constructor are available — direct copy, isomorphism, upcast, downcast,
     * cross-tower bridge.
     *
     * @throws std::invalid_argument if no conversion path exists (typically a downcast where
     * the source value lies outside the target)
     */
    template <FiniteFieldType B, MOD modulus, LutMode mode>
    Iso(const Ext<B, modulus, mode>& other);

    /**
     * @brief Embed a prime-field element when 𝔽_p is a (possibly indirect) subfield of the `Iso`
     *
     * Picks the first of MAIN / OTHERS that contains 𝔽_p and embeds via the corresponding
     * @ref Embedding. The constraints rule out the trivial cases where 𝔽_p already appears
     * literally as MAIN or one of OTHERS (those use the wrapping constructors above).
     */
    template <uint16_t p>
    constexpr Iso(const Fp<p>& other)
        requires SubfieldOf<Iso<MAIN, OTHERS...>, Fp<p>> && (!std::is_same_v<MAIN, Fp<p>>) &&
                 (!BelongsTo<Fp<p>, OTHERS...>);

    /**
     * @brief Cross-field conversion from another `Iso` of the same characteristic
     *
     * Same four-way decision as the @ref Ext cross-field constructor — direct isomorphism,
     * upcast, downcast, or bridge via @ref details::largest_common_subfield_t — but extended
     * to handle every (MAIN, OTHERS...) pairing on both sides. Each side's representation
     * tower is searched for the cheapest viable path.
     *
     * @throws std::invalid_argument on a downcast whose source value lies outside the target
     */
    template <FiniteFieldType OTHER_MAIN, FiniteFieldType... OTHER_OTHERS>
    Iso(const Iso<OTHER_MAIN, OTHER_OTHERS...>& other);

    /// @brief Additive inverse (delegates to MAIN)
    constexpr Iso operator-() const { return Iso{-main_}; }

    /// @brief `*this += other` (delegates to MAIN)
    constexpr Iso& operator+=(const Iso& other);
    /// @brief `*this += other` after converting an `OTHERS` operand to MAIN
    template <typename OTHER>
    constexpr Iso& operator+=(const OTHER& other)
        requires BelongsTo<OTHER, OTHERS...>;

    /// @brief `*this -= other` (delegates to MAIN)
    constexpr Iso& operator-=(const Iso& other);
    /// @brief `*this -= other` after converting an `OTHERS` operand to MAIN
    template <typename OTHER>
    constexpr Iso& operator-=(const OTHER& other)
        requires BelongsTo<OTHER, OTHERS...>;

    /// @brief `*this *= other` (delegates to MAIN)
    constexpr Iso& operator*=(const Iso& other);
    /// @brief `*this *= other` after converting an `OTHERS` operand to MAIN
    template <typename OTHER>
    constexpr Iso& operator*=(const OTHER& other)
        requires BelongsTo<OTHER, OTHERS...>;

    /// @brief Scalar multiplication by an `int` (delegates to MAIN)
    constexpr Iso& operator*=(int s);

    /// @brief `*this /= other`; throws `std::invalid_argument` if other is zero
    Iso& operator/=(const Iso& other);
    /// @brief `*this /= other` after converting an `OTHERS` operand to MAIN; same exception
    template <typename OTHER>
    Iso& operator/=(const OTHER& other)
        requires BelongsTo<OTHER, OTHERS...>;

    /**
     * @brief Project to a specific representation by applying the cached @ref Isomorphism
     *
     * @tparam TO Target representation, one of the `OTHERS`
     */
    template <typename TO>
    constexpr TO as() const
        requires BelongsTo<TO, OTHERS...>;

    /// @brief Read-only access to the underlying MAIN-representation element
    constexpr const MAIN& main() const noexcept { return main_; }

    constexpr bool is_zero() const noexcept { return main_.is_zero(); }

    constexpr bool has_positive_sign() const noexcept { return main_.has_positive_sign(); }

    constexpr Iso& randomize() {
        main_.randomize();
        return *this;
    }

    constexpr Iso& randomize_force_change() {
        main_.randomize_force_change();
        return *this;
    }

    size_t get_multiplicative_order() const { return main_.get_multiplicative_order(); }

    size_t get_additive_order() const { return main_.get_additive_order(); }

    constexpr auto get_label() const noexcept { return main_.get_label(); }

#ifdef CECCO_ERASURE_SUPPORT
    /**
     * @brief Mark this element as erased (delegates to MAIN)
     *
     * @warning Erased elements must not participate in field arithmetic — see
     * @ref CECCO_ERASURE_SUPPORT.
     */
    constexpr Iso& erase() noexcept;
    /// @brief Clear the erasure flag, resetting MAIN to its additive identity
    constexpr Iso& unerase() noexcept;
    /// @brief Test whether this element is currently erased
    constexpr bool is_erased() const noexcept { return main_.is_erased(); }
#endif

    constexpr Iso& operator=(const Iso& other);
    constexpr Iso& operator=(Iso&& other) noexcept = default;

    /// @brief Assign a MAIN-representation element directly
    constexpr Iso& operator=(const MAIN& other);
    /// @brief Assign an `int` via MAIN
    constexpr Iso& operator=(int other);

    /// @brief Assign from an `OTHERS` representation (copy-and-swap; same semantics as the constructor)
    template <typename OTHER>
    Iso& operator=(const OTHER& other)
        requires BelongsTo<OTHER, OTHERS...>;

    /// @brief Assign from an `Fp` of matching characteristic (copy-and-swap)
    template <uint16_t p>
        requires(p == MAIN::get_characteristic())
    Iso& operator=(const Fp<p>& other);

    /// @brief Assign from an `Ext` of matching characteristic (copy-and-swap)
    template <FiniteFieldType B, MOD ext_modulus, LutMode mode>
        requires(B::get_characteristic() == MAIN::get_characteristic())
    Iso& operator=(const Ext<B, ext_modulus, mode>& other);

    /// @brief Assign from another `Iso` of matching characteristic (copy-and-swap)
    template <FiniteFieldType OTHER_MAIN, FiniteFieldType... OTHER_OTHERS>
    Iso& operator=(const Iso<OTHER_MAIN, OTHER_OTHERS...>& other)
        requires(OTHER_MAIN::get_characteristic() == MAIN::get_characteristic()) &&
                (!std::is_same_v<Iso<OTHER_MAIN, OTHER_OTHERS...>, Iso<MAIN, OTHERS...>>);

    // Equality operators
    constexpr bool operator==(const Iso& other) const noexcept { return main_ == other.main_; }

    constexpr bool operator==(const MAIN& other) const noexcept { return main_ == other; }

    constexpr bool operator!=(const Iso& other) const { return main_ != other.main_; }

    constexpr bool operator!=(const MAIN& other) const { return main_ != other; }

    // Binary arithmetic operators handled by global template functions (like Fp and Ext)

    // Stream operator - delegate to underlying value
    friend std::ostream& operator<<(std::ostream& os, const Iso& iso) { return os << iso.main_; }

    static const std::string get_info();

    // Static methods required by FiniteFieldType concept
    static constexpr size_t get_characteristic() noexcept { return MAIN::get_characteristic(); }

    static constexpr size_t get_p() noexcept { return MAIN::get_p(); }

    static constexpr size_t get_q() noexcept { return MAIN::get_q(); }

    static constexpr size_t get_m() noexcept { return MAIN::get_m(); }

    static constexpr size_t get_size() noexcept { return MAIN::get_size(); }

    static constexpr Polynomial<BASE_FIELD> get_modulus() { return MAIN::get_modulus(); }

    static constexpr Iso get_generator() { return Iso{MAIN::get_generator()}; }

    /// @brief Inherits constexpr-readiness from MAIN (`OTHERS` may differ; isomorphisms run at runtime)
    static constexpr bool is_constexpr_ready() noexcept { return MAIN::is_constexpr_ready(); }

    // Required for Ext to use Iso as base field
    static constexpr bool ready() { return MAIN::ready() && (OTHERS::ready() && ...); }

    // Simple reference forwarding to MAIN's LUTs - no copying
    // The ready() mechanism ensures proper ordering
    static constexpr auto& lut_add() { return MAIN::lut_add(); }

    static constexpr auto& lut_neg() { return MAIN::lut_neg(); }

    static constexpr auto& lut_mul() { return MAIN::lut_mul(); }

    static constexpr auto& lut_inv() { return MAIN::lut_inv(); }

    // LUT-compatible interface for use as BaseFieldType
    // These allow Iso fields to be used as base fields for higher-order extensions
    static constexpr label_t lut_add(label_t a, label_t b) { return lut_add()(a, b); }

    static constexpr label_t lut_mul(label_t a, label_t b) { return lut_mul()(a, b); }

    static constexpr label_t lut_neg(label_t a) { return lut_neg()(a); }

    static constexpr label_t lut_inv(label_t a) { return lut_inv()(a); }

    /**
     * @brief Coordinate vector over a subfield T of MAIN or any `OTHERS` representation
     *
     * @tparam T Subfield reachable from MAIN or one of `OTHERS`
     */
    template <FiniteFieldType T>
    Vector<T> as_vector() const
        requires((SubfieldOf<MAIN, T> || ((SubfieldOf<OTHERS, T>) || ...))) &&
                (!std::is_same_v<Iso<MAIN, OTHERS...>, T>);
};

/* member functions for Iso */

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(const Iso& other) {
    main_ = other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(const MAIN& other) {
    main_ = other;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(int other) {
    main_ = other;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <typename OTHER>
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(const OTHER& other)
    requires BelongsTo<OTHER, OTHERS...>
{
    Iso temp(other);
    std::swap(*this, temp);
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <uint16_t p>
    requires(p == MAIN::get_characteristic())
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(const Fp<p>& other) {
    Iso temp(other);
    std::swap(*this, temp);
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <FiniteFieldType B, MOD ext_modulus, LutMode mode>
    requires(B::get_characteristic() == MAIN::get_characteristic())
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(const Ext<B, ext_modulus, mode>& other) {
    Iso temp(other);
    std::swap(*this, temp);
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <FiniteFieldType OTHER_MAIN, FiniteFieldType... OTHER_OTHERS>
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(const Iso<OTHER_MAIN, OTHER_OTHERS...>& other)
    requires(OTHER_MAIN::get_characteristic() == MAIN::get_characteristic()) &&
            (!std::is_same_v<Iso<OTHER_MAIN, OTHER_OTHERS...>, Iso<MAIN, OTHERS...>>)
{
    Iso temp(other);
    std::swap(*this, temp);
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <FiniteFieldType B, MOD modulus, LutMode mode>
Iso<MAIN, OTHERS...>::Iso(const Ext<B, modulus, mode>& other) {
    static_assert(MAIN::get_characteristic() == Ext<B, modulus, mode>::get_characteristic(),
                  "trying to convert between fields with different characteristic");

#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif

    using IN = Ext<B, modulus, mode>;
    using OUT = Iso;

    // Branch 1: If MAIN or any of the OTHERS is isomorphic to Ext then use the correct isomorphism
    if constexpr (Isomorphic<MAIN, IN>) {
        auto isomorphism = Isomorphism<IN, MAIN>();
        main_ = isomorphism(other);

        // Branch 2 (upcast): If IN is a subfield to OUT
    } else if constexpr (SubfieldOf<OUT, IN>) {
        if constexpr (SubfieldOf<MAIN, IN>) {
            // Ext ⊆ MAIN - use cached embedding from Ext to MAIN
            auto embedding = Embedding<IN, MAIN>();
            main_ = embedding(other);
        } else if constexpr ((SubfieldOf<OTHERS, IN> || ...)) {
            // Check which OTHERS type contains ExtType as subfield
            auto try_embedding = [&]<typename OtherType>() {
                if constexpr (SubfieldOf<OtherType, IN>) {
                    auto embedding = Embedding<IN, OtherType>();
                    OtherType other_repr = embedding(other);
                    OUT temp_iso(other_repr);
                    main_ = temp_iso.main_;
                }
            };
            (try_embedding.template operator()<OTHERS>(), ...);
        }
        // Branch 3 (downcast): If Iso or MAIN or any of the OTHERS is a subfield of Ext
    } else if constexpr (SubfieldOf<IN, OUT>) {
        if constexpr (SubfieldOf<IN, MAIN>) {
            auto embedding = Embedding<MAIN, IN>();
            main_ = embedding.extract(other);
        } else if constexpr ((SubfieldOf<OTHERS, IN> || ...)) {
            auto try_downcast = [&]<typename OtherType>() {
                if constexpr (SubfieldOf<OtherType, IN>) {
                    auto embedding = Embedding<IN, OtherType>();
                    OtherType other_repr = embedding.extract(other);
                    OUT temp_iso(other_repr);
                    main_ = temp_iso.main_;
                }
            };
            (try_downcast.template operator()<OTHERS>(), ...);
        }
    } else {
        // Branch 4 (cross-cast through largest common subfield): This is the else case
        using CommonField = details::largest_common_subfield_t<OUT, IN>;

        if constexpr (details::iso_info<CommonField>::is_iso) {
            // CommonField is an Iso, continue with its MAIN
            using CommonMainField = typename details::iso_info<CommonField>::main_type;
            CommonMainField intermediate(other);  // Downcast other to CommonField's MAIN
            main_ = MAIN(intermediate);           // Use existing cross-field Ext->Ext constructor to convert to MAIN
        } else {
            // CommonField is an Ext, continue with CommonField
            CommonField intermediate(other);  // Downcast other to CommonField
            main_ = MAIN(intermediate);       // Use existing cross-field Ext->Ext constructor to convert to MAIN
        }
    }
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <uint16_t p>
constexpr Iso<MAIN, OTHERS...>::Iso(const Fp<p>& other)
    requires SubfieldOf<Iso<MAIN, OTHERS...>, Fp<p>> && (!std::is_same_v<MAIN, Fp<p>>) && (!BelongsTo<Fp<p>, OTHERS...>)
{
    static_assert(MAIN::get_characteristic() == p, "Prime field characteristic must match Iso field characteristic");

#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif

    // Try direct embedding into MAIN first
    if constexpr (SubfieldOf<MAIN, Fp<p>>) {
        main_ = MAIN(other);
    } else {
        // Try embedding via any OTHERS representation that contains Fp<p>
        bool conversion_done = false;
        auto try_others_embedding = [&]<typename OtherType>() {
            if constexpr (SubfieldOf<OtherType, Fp<p>>) {
                if (!conversion_done) {
                    // Embed Fp<p> -> OtherType, then convert OtherType -> MAIN via isomorphism
                    OtherType other_elem(other);
                    auto isomorphism = Isomorphism<OtherType, MAIN>();
                    main_ = isomorphism(other_elem);
                    conversion_done = true;
                }
            }
        };
        (try_others_embedding.template operator()<OTHERS>(), ...);
    }
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <FiniteFieldType OTHER_MAIN, FiniteFieldType... OTHER_OTHERS>
Iso<MAIN, OTHERS...>::Iso(const Iso<OTHER_MAIN, OTHER_OTHERS...>& other) {
    static_assert(MAIN::get_characteristic() == OTHER_MAIN::get_characteristic(),
                  "trying to convert between fields with different characteristic");

#ifdef CECCO_ERASURE_SUPPORT
    if (other.is_erased()) {
        this->erase();
        return;
    }
#endif

    using IN = Iso<OTHER_MAIN, OTHER_OTHERS...>;
    using OUT = Iso;

    // Branch 1: IN and OUT are isomorphic (implies: MAIN and OTHER_MAIN are isomorphic)
    if constexpr (Isomorphic<OTHER_MAIN, MAIN>) {
        auto isomorphism = Isomorphism<OTHER_MAIN, MAIN>();
        main_ = isomorphism(other.main());

        // Branch 2 (upcast): IN is subfield of OUT (but not the same since this would be caught by copy constr.)
    } else if constexpr (SubfieldOf<OUT, IN>) {
        // Sub-branch 2a: OTHER_MAIN is subfield of MAIN
        if constexpr (SubfieldOf<MAIN, OTHER_MAIN>) {
            main_ = MAIN(other.main());  // Use existing Ext-from-Ext cross-field constructor

            // Sub-branch 2b: OTHER_MAIN is subfield of one of OTHERS
        } else if constexpr ((SubfieldOf<OTHERS, OTHER_MAIN> || ...)) {
            bool conversion_done = false;
            auto try_main_to_others = [&]<typename OutputOtherType>() {
                if constexpr (SubfieldOf<OutputOtherType, OTHER_MAIN>) {
                    if (!conversion_done) {
                        OutputOtherType intermediate(
                            other.main());  // Use existing Ext-from-Ext cross-field constructor
                        auto isomorphism = Isomorphism<OutputOtherType, MAIN>();
                        main_ = isomorphism(intermediate);
                        conversion_done = true;
                    }
                }
            };
            (try_main_to_others.template operator()<OTHERS>(), ...);

            // Sub-branch 2c: One of OTHER_OTHERS is subfield of MAIN
        } else if constexpr ((SubfieldOf<MAIN, OTHER_OTHERS> || ...)) {
            bool conversion_done = false;
            auto try_others_to_main = [&]<typename InputOtherType>() {
                if constexpr (SubfieldOf<MAIN, InputOtherType>) {
                    if (!conversion_done) {
                        InputOtherType input_other(other);  // Convert Iso to InputOtherType
                        main_ = MAIN(input_other);          // Use existing Ext-from-Ext cross-field constructor
                        conversion_done = true;
                    }
                }
            };
            (try_others_to_main.template operator()<OTHER_OTHERS>(), ...);

            // Sub-branch 2d: One of OTHER_OTHERS is subfield of one of OTHERS
        } else {
            bool conversion_done = false;
            auto try_others_to_others = [&]<typename OutputOtherType>() {
                if (conversion_done) return;
                auto try_input_others = [&]<typename InputOtherType>() {
                    if constexpr (SubfieldOf<OutputOtherType, InputOtherType>) {
                        if (!conversion_done) {
                            InputOtherType input_other(other);  // Convert Iso to InputOtherType
                            OutputOtherType intermediate(
                                input_other);  // Use existing Ext-from-Ext cross-field constructor
                            auto isomorphism = Isomorphism<OutputOtherType, MAIN>();
                            main_ = isomorphism(intermediate);
                            conversion_done = true;
                        }
                    }
                };
                (try_input_others.template operator()<OTHER_OTHERS>(), ...);  // This calls the inner lambda
            };
            (try_others_to_others.template operator()<OTHERS>(), ...);  // This calls the outer lambda
        }

        // Branch 3 (downcast): OUT is subfield of IN (but not the same since this would be caught by copy constr.)
    } else if constexpr (SubfieldOf<IN, OUT>) {
        // Sub-branch 3a: OTHER_MAIN is superfield of MAIN
        if constexpr (SubfieldOf<OTHER_MAIN, MAIN>) {
            main_ = MAIN(other.main());  // Use existing Ext-from-Ext cross-field constructor

            // Sub-branch 3b: OTHER_MAIN is superfield of one of OTHERS
        } else if constexpr ((SubfieldOf<OTHER_MAIN, OTHERS> || ...)) {
            bool conversion_done = false;
            auto try_main_to_others = [&]<typename OutputOtherType>() {
                if constexpr (SubfieldOf<OTHER_MAIN, OutputOtherType>) {
                    if (!conversion_done) {
                        OutputOtherType intermediate(
                            other.main());  // Use existing Ext-from-Ext cross-field constructor
                        auto isomorphism = Isomorphism<OutputOtherType, MAIN>();
                        main_ = isomorphism(intermediate);
                        conversion_done = true;
                    }
                }
            };
            (try_main_to_others.template operator()<OTHERS>(), ...);

            // Sub-branch 3c: One of OTHER_OTHERS is superfield of MAIN
        } else if constexpr ((SubfieldOf<OTHER_OTHERS, MAIN> || ...)) {
            bool conversion_done = false;
            auto try_others_to_main = [&]<typename InputOtherType>() {
                if constexpr (SubfieldOf<InputOtherType, MAIN>) {
                    if (!conversion_done) {
                        InputOtherType input_other(other);  // Convert Iso to InputOtherType
                        main_ = MAIN(input_other);          // Use existing Ext-from-Ext cross-field constructor
                        conversion_done = true;
                    }
                }
            };
            (try_others_to_main.template operator()<OTHER_OTHERS>(), ...);

            // Sub-branch 3d: One of OTHER_OTHERS is superfield of one of OTHERS
        } else {
            bool conversion_done = false;
            auto try_others_to_others = [&]<typename OutputOtherType>() {
                if (conversion_done) return;
                auto try_input_others = [&]<typename InputOtherType>() {
                    if constexpr (SubfieldOf<InputOtherType, OutputOtherType>) {
                        if (!conversion_done) {
                            InputOtherType input_other(other);          // Convert Iso to InputOtherType
                            OutputOtherType intermediate(input_other);  // Use existing Ext-from-Ext cross-field
                            auto isomorphism = Isomorphism<OutputOtherType, MAIN>();
                            main_ = isomorphism(intermediate);
                            conversion_done = true;
                        }
                    }
                };
                (try_input_others.template operator()<OTHER_OTHERS>(), ...);  // This calls the inner lambda
            };
            (try_others_to_others.template operator()<OTHERS>(), ...);  // This calls the outer lambda
        }

    } else {
        // Branch 4 (cross-cast through largest common subfield): Consistent with Ext-from-Iso constructor
        using CommonField = details::largest_common_subfield_t<OUT, IN>;

        main_ = MAIN(CommonField(other));
    }
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <FiniteFieldType T>
Iso<MAIN, OTHERS...>::Iso(const Vector<T>& v) {
    bool conversion_done = false;

    // Branch 1: Try MAIN directly (compile-time check)
    if constexpr (SubfieldOf<MAIN, T>) {
        main_ = MAIN(v);
        conversion_done = true;
    }

    // Branch 2: Try compatible OTHERS
    if (!conversion_done && sizeof...(OTHERS) > 0) {
        auto try_others = [&]<typename OtherType>() {
            if constexpr (SubfieldOf<OtherType, T>) {
                if (!conversion_done) {
                    OtherType other_elem(v);
                    auto isomorphism = Isomorphism<OtherType, MAIN>();
                    main_ = isomorphism(other_elem);
                    conversion_done = true;
                }
            }
        };
        (try_others.template operator()<OTHERS>(), ...);
    }

    // Branch 3: Error case
    if (!conversion_done) throw std::invalid_argument("Vector incompatible with Iso stack");
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator+=(const Iso& other) {
    main_ += other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <typename OTHER>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator+=(const OTHER& other)
    requires BelongsTo<OTHER, OTHERS...>
{
    main_ += Iso(other).main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator-=(const Iso& other) {
    main_ -= other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <typename OTHER>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator-=(const OTHER& other)
    requires BelongsTo<OTHER, OTHERS...>
{
    main_ -= Iso(other).main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator*=(const Iso& other) {
    main_ *= other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <typename OTHER>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator*=(const OTHER& other)
    requires BelongsTo<OTHER, OTHERS...>
{
    main_ *= Iso(other).main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator*=(int s) {
    main_ *= s;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator/=(const Iso& other) {
    main_ /= other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <typename OTHER>
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator/=(const OTHER& other)
    requires BelongsTo<OTHER, OTHERS...>
{
    main_ /= Iso(other).main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <typename TO>
constexpr TO Iso<MAIN, OTHERS...>::as() const
    requires BelongsTo<TO, OTHERS...>
{
#ifdef CECCO_ERASURE_SUPPORT
    if (this->is_erased()) return TO().erase();
#endif

    auto phi = Isomorphism<MAIN, TO>();
    return phi(main_);
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
const std::string Iso<MAIN, OTHERS...>::get_info() {
    std::stringstream ss;
    ss << "stack of isomorphic fields, main field: ";
    ss << MAIN::get_info();
    return ss.str();
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <FiniteFieldType T>
Vector<T> Iso<MAIN, OTHERS...>::as_vector() const
    requires((SubfieldOf<MAIN, T> || ((SubfieldOf<OTHERS, T>) || ...))) && (!std::is_same_v<Iso<MAIN, OTHERS...>, T>)
{
    if constexpr (std::is_same_v<T, MAIN>) {
        // T is MAIN - direct conversion
        return main_.template as_vector<T>();
    } else if constexpr (((std::is_same_v<T, OTHERS>) || ...)) {
        // T is one of OTHERS - convert main_ to T first
        return as<T>().template as_vector<T>();
    } else if constexpr (SubfieldOf<MAIN, T>) {
        // T is subfield of MAIN - use main_ directly
        return main_.template as_vector<T>();
    } else {
        // T might be subfield of one of OTHERS - try each one
        bool converted = false;
        Vector<T> result;
        (([&]() {
             if constexpr (SubfieldOf<OTHERS, T>) {
                 if (!converted) {
                     result = as<OTHERS>().template as_vector<T>();
                     converted = true;
                 }
             }
         }()),
         ...);

        return result;
    }
}

#ifdef CECCO_ERASURE_SUPPORT
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::erase() noexcept {
    main_.erase();
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::unerase() noexcept {
    main_.unerase();
    return *this;
}
#endif

}  // namespace CECCO

#endif
