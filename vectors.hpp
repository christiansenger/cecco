/**
 * @file vectors.hpp
 * @brief Vector arithmetic library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.1.10
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
 * Vector arithmetic for error-control coding over any @ref CECCO::ComponentType (finite fields,
 * floating-point, complex, signed integers including `InfInt`). Supports safe cross-field
 * conversions via @ref CECCO::SubfieldOf / @ref CECCO::ExtensionOf / @ref CECCO::largest_common_subfield_t,
 * bidirectional Vector ↔ Matrix integration, and lazy O(1) caching of Hamming weight / burst length.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * // Basic vector operations
 * Vector<int> u = {1, 2, 3, 4};
 * Vector<int> v(4, 5);            // Vector of length 4, all components = 5
 * auto w = u + v;                 // Element-wise addition
 * int dot = inner_product(u, v);  // Inner product
 *
 * // Finite field vectors
 * using F4 = Ext<Fp<2>, MOD{1, 1, 1}>;
 * Vector<F4> x = {0, 1, 2, 3};
 * size_t weight = x.wH();                  // Hamming weight
 * size_t distance = dH(x, Vector<F4>(4));  // Hamming distance to zero vector
 *
 * // Cross-field upcast (𝔽_2 ⊆ 𝔽_4 via construction tower)
 * Vector<Fp<2>> y = {1, 0, 1, 1};
 * Vector<F4> z(y);
 * @endcode
 *
 * @see @ref fields.hpp, @ref matrices.hpp, @ref field_concepts_traits.hpp
 */

#ifndef VECTORS_HPP
#define VECTORS_HPP

#include <initializer_list>
#include <numeric>

#include "matrices.hpp"

/*
//transitive
#include <algorithm>
#include <complex>
#include <concepts>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "InfInt.hpp"
#include "field_concepts_traits.hpp"
#include "helpers.hpp"
*/

namespace CECCO {

template <ComponentType T>
class Vector;
template <CoefficientType T>
class Polynomial;
template <ComponentType T>
class Matrix;

template <ComponentType T>
T inner_product(const Vector<T>& lhs, const Vector<T>& rhs);
template <ComponentType T>
Vector<T> Schur_product(const Vector<T>& lhs, const Vector<T>& rhs);
template <ComponentType T>
Vector<T> unit_vector(size_t length, size_t i);
template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& rhs);
double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs);

/**
 * @brief Vector v = (v₀, v₁, …, vₙ₋₁) over a @ref CECCO::ComponentType
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType (finite field, `double`,
 *           `std::complex<double>`, or signed integer including `InfInt`)
 *
 * Components are stored contiguously; length is fixed except via the explicit resizers
 * (@ref pad_front, @ref pad_back, @ref append, @ref prepend, @ref delete_components, …).
 * Dimension mismatches in arithmetic throw `std::invalid_argument`. Hamming weight is
 * computed lazily and cached; the cache is invalidated on any mutation. Methods that
 * compare against zero (@ref wH, @ref supp, @ref burst_length, …) are gated by
 * `requires ReliablyComparableType<T>`.
 *
 * Cross-field constructors and assignment operators between two finite fields of the same
 * characteristic route through @ref CECCO::details::largest_common_subfield_t, so vectors
 * over fields from disjoint construction towers can interoperate.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F4 = Ext<Fp<2>, MOD{1, 1, 1}>;
 * Vector<F4> v = {0, 1, 2, 3};
 * size_t weight = v.wH();              // Hamming weight (cached)
 * size_t burst  = v.burst_length();
 * auto M = v.as_matrix<Fp<2>>();       // 2 × 4 matrix over 𝔽₂ (𝔽₂ ⊆ 𝔽₄)
 * @endcode
 */
template <ComponentType T>
class Vector {
    template <ReliablyComparableType U>
    friend constexpr bool operator==(const Vector<U>& lhs, const Vector<U>& rhs);
    friend T inner_product<>(const Vector<T>& lhs, const Vector<T>& rhs);
    friend Vector<T> Schur_product<>(const Vector<T>& lhs, const Vector<T>& rhs);
    friend Vector unit_vector<>(size_t length, size_t i);
    friend std::ostream& operator<< <>(std::ostream& os, const Vector& rhs);
    friend double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs);
    friend class Matrix<T>;
    template <CoefficientType U>
    friend class Polynomial;

   public:
    // Cache configuration for this class
    enum CacheIds { Weight = 0 };

    /** @name Constructors
     * @{
     */

    /// @brief Default constructor: empty vector (length 0)
    constexpr Vector() noexcept = default;

    /// @brief Length-`n` vector with zero-initialised components
    explicit Vector(size_t n) : data(n, T(0)) {
        if constexpr (ReliablyComparableType<T>) cache.template set<Weight>(0);
    }

    /// @brief Length-`n` vector with every component equal to `l`
    Vector(size_t n, const T& l);

    /// @brief From an initializer list of components, in order
    Vector(const std::initializer_list<T>& l) : data(l) {}

    Vector(const Vector& other);
    constexpr Vector(Vector&& other) noexcept;

    /**
     * @brief From a matrix over a subfield: each column becomes one T component
     *
     * @tparam S Subfield of T (in the construction tower)
     * @throws std::invalid_argument if `mat.get_m()` does not match `[T:S]` (the extension
     *         degree of T over S)
     */
    template <FiniteFieldType S>
    Vector(const Matrix<S>& mat)
        requires FiniteFieldType<T> && ExtensionOf<S, T>;

    /**
     * @brief Cross-field conversion between two finite fields of the same characteristic
     *
     * @tparam S Source field type (`Vector<S>`); must share characteristic with T
     *
     * Converts component by component via T's cross-field constructor, which routes through
     * @ref CECCO::details::largest_common_subfield_t and so handles disjoint construction towers.
     * Propagates `std::invalid_argument` if any component is not representable in T.
     */
    template <FiniteFieldType S>
        requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
    Vector(const Vector<S>& other);

    /**
     * @brief From polynomial coefficients: component `i` becomes `poly[i]`
     *
     * Resulting length is `poly.degree() + 1`. For cross-field conversion, convert the
     * polynomial first.
     *
     * @note Member template with U fixed to T, so that instantiating a vector over polynomial
     * components does not form the unsupported type `Polynomial<Polynomial<...>>`.
     */
    template <CoefficientType U>
        requires std::same_as<U, T>
    Vector(const Polynomial<U>& poly);

    /** @} */

    /** @name Assignment Operators
     * @{
     */

    constexpr Vector& operator=(const Vector& rhs);
    constexpr Vector& operator=(Vector&& rhs) noexcept;

    /// @brief Set every component to `rhs`
    constexpr Vector& operator=(const T& rhs);

    /// @brief Cross-field assignment (same semantics as the cross-field constructor)
    template <FiniteFieldType S>
        requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
    Vector& operator=(const Vector<S>& other);

    /** @} */

    /** @name Unary Arithmetic Operations
     * @{
     */

    /// @brief Unary `+` (lvalue): returns a copy
    constexpr Vector operator+() const& { return *this; }
    /// @brief Unary `+` (rvalue): returns the rvalue itself
    constexpr Vector operator+() && noexcept { return std::move(*this); }

    /// @brief Unary `−` (lvalue): returns a new vector with each component negated
    constexpr Vector operator-() const&;
    /// @brief Unary `−` (rvalue): negates components in place
    constexpr Vector operator-() &&;

    /** @} */

    /** @name Compound Assignment Operations
     * @{
     */

    /**
     * @brief Component-wise addition `this[i] += rhs[i]`
     *
     * @throws std::invalid_argument if `this->get_n() != rhs.get_n()`
     */
    Vector& operator+=(const Vector& rhs);

    /**
     * @brief Component-wise subtraction `this[i] -= rhs[i]`
     *
     * @throws std::invalid_argument if `this->get_n() != rhs.get_n()`
     */
    Vector& operator-=(const Vector& rhs);

    /// @brief Multiply every component by the scalar `s`
    constexpr Vector& operator*=(const T& s);

    /**
     * @brief Divide every component by the scalar `s`
     *
     * @throws std::invalid_argument if `s == T(0)`
     * @note Round-trip `(v / s) * s == v` is only guaranteed when T satisfies @ref CECCO::FieldType
     * (otherwise integer rounding may corrupt components).
     */
    Vector& operator/=(const T& s)
        requires CoefficientType<T>;

    /** @} */

    /** @name Randomization
     * @{
     */

    /**
     * @brief Replace components with random values appropriate for T
     *
     * Distribution per component: finite-field types draw uniformly from the field; signed
     * integers from [−100, 100]; `double` and the parts of `std::complex<double>` from [−1, 1].
     *
     * @note Unavailable for polynomial components: randomizing a polynomial needs a degree,
     * see @ref CECCO::Polynomial::randomize.
     */
    Vector& randomize()
        requires CoefficientType<T>;

    /// @brief Like @ref randomize but every component is guaranteed non-zero
    Vector& randomize_nonzero()
        requires FieldType<T>;

    /**
     * @brief Fill with pairwise distinct field elements (shuffled labels 0, …, q − 1)
     *
     * @throws std::invalid_argument if the vector length exceeds the field cardinality q
     */
    Vector& randomize_pairwise_distinct()
        requires FiniteFieldType<T>;

    /** @} */

    /** @name Information and Properties
     * @{
     */

    /// @brief Vector length (number of components)
    constexpr size_t get_n() const noexcept { return data.size(); }

    /// @brief True iff length is 0 (distinct from a non-empty all-zero vector)
    constexpr bool is_empty() const noexcept { return data.empty(); }

    /// @brief True iff every component equals T(0); also true for the empty vector
    constexpr bool is_zero() const
        requires ReliablyComparableType<T>;

    /// @brief True iff no two components are equal
    constexpr bool is_pairwise_distinct() const
        requires ReliablyComparableType<T>;

    /**
     * @brief Sorted indices of non-zero, non-erased components (empty vector for an input with only zeros and erasures)
     *
     * @throws std::invalid_argument if `*this` is empty (length 0)
     */
    std::vector<size_t> supp() const
        requires ReliablyComparableType<T>
    {
        if (data.empty()) throw std::invalid_argument("Support of an empty (length zero) vector is undefined!");
        const T zero(0);
        std::vector<size_t> supp;
        for (size_t i = 0; i < data.size(); ++i) {
#ifdef CECCO_ERASURE_SUPPORT
            if constexpr (FieldType<T>)
                if (data[i].is_erased()) continue;
#endif
            if (data[i] != zero) supp.push_back(i);
        }
        return supp;
    }

    /// @brief Hamming weight: number of non-zero, non-erased components; cached on first call
    size_t wH() const
        requires ReliablyComparableType<T>
    {
        return cache.template get_or_compute<Weight>([this] { return calculate_weight(); });
    }

    /// @brief Burst length R − L + 1, where L, R are the first and last non-zero indices; 0 for an all-zero or empty
    /// vector
    constexpr size_t burst_length() const
        requires ReliablyComparableType<T>;

    /// @brief Cyclic burst length: shortest circular arc covering all non-zero positions; 0 for an all-zero or empty
    /// vector
    constexpr size_t cyclic_burst_length() const
        requires ReliablyComparableType<T>;

    /** @} */

    /** @name Component Access and Manipulation
     * @{
     */

    /**
     * @brief Set component `i` by perfect-forwarding `c`
     *
     * @throws std::invalid_argument if `i` is out of bounds
     */
    template <typename U>
    Vector& set_component(size_t i, U&& c)
        requires std::convertible_to<std::decay_t<U>, T>;

    /**
     * @brief Read access to component `i`
     *
     * @throws std::invalid_argument if `i` is out of bounds
     */
    const T& operator[](size_t i) const;

    /**
     * @brief Extract the contiguous subvector `[i, i + w)`
     *
     * @throws std::invalid_argument if `[i, i + w)` exceeds the vector
     */
    Vector get_subvector(size_t i, size_t w) const&;

    /// @brief In-place rvalue overload of @ref get_subvector (truncates to `[i, i + w)`)
    Vector& get_subvector(size_t i, size_t w) &&;

    /**
     * @brief Overwrite components `[i, i + v.get_n())` with `v`
     *
     * @throws std::invalid_argument if the replacement region exceeds the vector
     */
    Vector& set_subvector(const Vector& v, size_t i);

    /// @brief Rvalue overload of @ref set_subvector (move-assigns components from `v`)
    Vector& set_subvector(Vector&& v, size_t i);

    /// @brief Append `rhs`: result becomes `[*this | rhs]`
    Vector& append(const Vector& rhs);
    /// @brief Rvalue overload of @ref append (moves components from `rhs`)
    Vector& append(Vector&& rhs);

    /// @brief Append after converting `rhs` to `Vector<T>`
    template <typename U>
    Vector& append(U&& rhs)
        requires std::constructible_from<Vector<T>, U>;

    /// @brief Prepend `rhs`: result becomes `[rhs | *this]`
    Vector& prepend(const Vector& rhs);
    /// @brief Rvalue overload of @ref prepend (moves components from `rhs`)
    Vector& prepend(Vector&& rhs);

    /// @brief Prepend after converting `lhs` to `Vector<T>`
    template <typename U>
    Vector& prepend(U&& lhs)
        requires std::constructible_from<Vector<T>, U>;

    /**
     * @brief Delete the components whose indices appear in `v` and compact
     *
     * @param v Indices (deduplicated internally)
     * @throws std::invalid_argument if any index in `v` is out of bounds
     */
    Vector& delete_components(const std::vector<size_t>& v);

    /**
     * @brief Delete component `i` (single-index convenience for @ref delete_components)
     *
     * @throws std::invalid_argument if `i` is out of bounds
     */
    Vector& delete_component(size_t i) { return delete_components({i}); }

#ifdef CECCO_ERASURE_SUPPORT

    /**
     * @brief Mark every component whose index appears in `v` as erased
     *
     * @param v Indices (deduplicated internally)
     * @throws std::invalid_argument if any index in `v` is out of bounds
     *
     * @warning Erased components must not participate in field arithmetic, see
     * @ref CECCO_ERASURE_SUPPORT.
     */
    Vector& erase_components(const std::vector<size_t>& v)
        requires FieldType<T>;

    /// @brief Erase component `i` (single-index convenience for @ref erase_components)
    Vector& erase_component(size_t i)
        requires FieldType<T>
    {
        return erase_components({i});
    }

    /**
     * @brief Clear the erasure flag on every component whose index appears in `v` (resets to T(0))
     *
     * @param v Indices (deduplicated internally)
     * @throws std::invalid_argument if any index in `v` is out of bounds
     */
    Vector& unerase_components(const std::vector<size_t>& v)
        requires FieldType<T>;

    /// @brief Un-erase component `i` (single-index convenience for @ref unerase_components)
    Vector& unerase_component(size_t i)
        requires FieldType<T>
    {
        return unerase_components({i});
    }

#endif

    /// @brief Prepend zeros so the vector has length at least `n` (no-op if already ≥ `n`)
    Vector& pad_front(size_t n);

    /// @brief Append zeros so the vector has length at least `n` (no-op if already ≥ `n`)
    Vector& pad_back(size_t n);

    /** @} */

    /** @name Transformations
     * @{
     */

    /** @brief Circular left shift by `i` positions: `j ↦ (j − i) mod n`. */
    constexpr Vector& rotate_left(size_t i);

    /** @brief Circular right shift by `i` positions: `j ↦ (j + i) mod n`. */
    constexpr Vector& rotate_right(size_t i);

    /** @brief Reverse component order: `i ↦ n − 1 − i`. */
    constexpr Vector& reverse();

    /** @brief Set every component to `value`. */
    constexpr Vector& fill(const T& value);

    /** @} */

    /** @name Finite Field Specific Operations
     * @{
     */

    /**
     * @brief Interpret as a base-q integer, where q = T::get_size()
     *
     * Component `data[n − 1 − i]` is the i-th digit (least-significant first), so the value
     * is Σ data[n − 1 − i] · qⁱ.
     *
     * @note May overflow `size_t` for large vectors over large fields.
     * @throws std::invalid_argument if any component is erased.
     */
    size_t as_integer() const
        requires FiniteFieldType<T>;

    /**
     * @brief Replace `*this` with the length-`n` base-q encoding of `value`
     *
     * Left inverse of @ref as_integer: `v.from_integer(v.as_integer(), v.get_n()) == v`.
     *
     * @throws std::out_of_range if `value` does not fit into n base-q digits
     */
    Vector& from_integer(size_t value, size_t n)
        requires FiniteFieldType<T>;

    /**
     * @brief Expand each component into its S-coefficient column
     *
     * @tparam S Subfield of T
     * @return [T:S] × `get_n()` matrix whose column j is `data[j].as_vector<S>()`
     *
     * @note Under @ref CECCO_ERASURE_SUPPORT, an erased component j produces an entirely
     * erased column j.
     */
    template <FiniteFieldType S>
    Matrix<S> as_matrix() const
        requires FiniteFieldType<T> && SubfieldOf<T, S> && (!std::is_same_v<T, S>);

    /**
     * @brief Reshape into an `m × (n/m)` matrix, row-major
     *
     * Inverse of @ref Matrix::to_vector: index k of this vector lands at `(k/cols, k%cols)`
     * with `cols = n/m`.
     *
     * @throws std::invalid_argument if `m` does not divide the vector length
     */
    Matrix<T> to_matrix(size_t m) const;

    /** @} */

   private:
    std::vector<T> data;

    /// @brief Cache for Hamming weight (invalidated by mutating operations)
    mutable details::Cache<details::CacheEntry<Weight, size_t>> cache;

    constexpr size_t calculate_weight() const
        requires ReliablyComparableType<T>;
};

/// @brief Deduction guide: a vector constructed from a polynomial deduces the coefficient type
template <CoefficientType T>
Vector(const Polynomial<T>&) -> Vector<T>;

template <ComponentType T>
Vector<T>::Vector(size_t n, const T& l) : data(n) {
    fill(l);
}

template <ComponentType T>
Vector<T>::Vector(const Vector<T>& other) : data(other.data), cache(other.cache) {}

template <ComponentType T>
constexpr Vector<T>::Vector(Vector<T>&& other) noexcept : data(std::move(other.data)), cache(std::move(other.cache)) {
    other.cache.invalidate();
}

template <ComponentType T>
template <FiniteFieldType S>
Vector<T>::Vector(const Matrix<S>& mat)
    requires FiniteFieldType<T> && ExtensionOf<S, T>
{
    const auto m = T().template as_vector<S>().get_n();
    if (m != mat.get_m())
        throw std::invalid_argument("trying to construct base field vector from subfield matrix of incompatible size");
    data.resize(mat.get_n());
    for (size_t i = 0; i < mat.get_n(); ++i) {
        data[i] = T(mat.get_col(i));
    }
}

template <ComponentType T>
template <FiniteFieldType S>
    requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
Vector<T>::Vector(const Vector<S>& other) {
    data.resize(other.get_n());
    for (size_t i = 0; i < other.get_n(); ++i) {
        data[i] = T(other[i]);
    }
}

template <ComponentType T>
template <CoefficientType U>
    requires std::same_as<U, T>
Vector<T>::Vector(const Polynomial<U>& poly) {
    data.resize(poly.degree() + 1);
    for (size_t i = 0; i <= poly.degree(); ++i) {
        data[i] = poly[i];
    }
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::operator=(const Vector<T>& rhs) {
    if (this == &rhs) return *this;
    data = rhs.data;
    cache = rhs.cache;
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::operator=(Vector<T>&& rhs) noexcept {
    if (this == &rhs) return *this;
    data = std::move(rhs.data);
    cache = std::move(rhs.cache);
    rhs.cache.invalidate();
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::operator=(const T& rhs) {
    return fill(rhs);
}

template <ComponentType T>
template <FiniteFieldType S>
    requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
Vector<T>& Vector<T>::operator=(const Vector<S>& other) {
    cache.invalidate();
    data.resize(other.get_n());
    for (size_t i = 0; i < other.get_n(); ++i) {
        data[i] = T(other[i]);
    }
    return *this;
}

template <ComponentType T>
constexpr Vector<T> Vector<T>::operator-() const& {
    Vector res(*this);
    std::ranges::for_each(res.data, [](T& v) { v = -v; });
    return res;  // move elision
}

template <ComponentType T>
constexpr Vector<T> Vector<T>::operator-() && {
    std::ranges::for_each(data, [](T& v) { v = -v; });
    return std::move(*this);
}

template <ComponentType T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& rhs) {
    if (data.size() != rhs.data.size()) throw std::invalid_argument("trying to add vectors of different lengths");
    cache.invalidate();
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<T>{});
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& rhs) {
    if (data.size() != rhs.data.size())
        throw std::invalid_argument(
            "trying to subtract vectors of different "
            "lengths");
    cache.invalidate();
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::minus<T>{});
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::operator*=(const T& s) {
    if (s == T(0)) {
        fill(T(0));
    } else {
        std::ranges::for_each(data, [&s](T& v) { v *= s; });
    }

    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::operator/=(const T& s)
    requires CoefficientType<T>
{
    if (s == T(0)) throw std::invalid_argument("trying to divide components of vector by zero");
    if constexpr (FieldType<T>) {
        operator*=(T(1) / s);
    } else {
        cache.invalidate();
        std::ranges::for_each(data, [&s](T& v) { v /= s; });
    }
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::append(const Vector& rhs) {
    if (this == &rhs) {
        Vector tmp(rhs);
        return append(std::move(tmp));
    }

    data.reserve(data.size() + rhs.data.size());
    data.insert(data.end(), rhs.data.begin(), rhs.data.end());
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::append(Vector&& rhs) {
    if (this == &rhs) return *this;

    data.reserve(data.size() + rhs.data.size());
    data.insert(data.end(), std::make_move_iterator(rhs.data.begin()), std::make_move_iterator(rhs.data.end()));
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::prepend(const Vector& lhs) {
    if (this == &lhs) {
        Vector tmp(lhs);
        return prepend(std::move(tmp));
    }

    data.reserve(data.size() + lhs.data.size());
    data.insert(data.begin(), lhs.data.begin(), lhs.data.end());
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::prepend(Vector&& lhs) {
    if (this == &lhs) return *this;

    data.reserve(data.size() + lhs.data.size());
    data.insert(data.begin(), std::make_move_iterator(lhs.data.begin()), std::make_move_iterator(lhs.data.end()));
    cache.invalidate();
    return *this;
}

template <ComponentType T>
template <typename U>
Vector<T>& Vector<T>::append(U&& rhs)
    requires std::constructible_from<Vector<T>, U>
{
    if constexpr (std::same_as<std::remove_cvref_t<U>, Vector<T>>) {
        if constexpr (std::is_lvalue_reference_v<U&&>)
            return append(static_cast<const Vector&>(rhs));
        else
            return append(std::move(rhs));
    } else {
        Vector<T> tmp(std::forward<U>(rhs));
        return append(std::move(tmp));
    }
}

template <ComponentType T>
template <typename U>
Vector<T>& Vector<T>::prepend(U&& lhs)
    requires std::constructible_from<Vector<T>, U>
{
    if constexpr (std::same_as<std::remove_cvref_t<U>, Vector<T>>) {
        if constexpr (std::is_lvalue_reference_v<U&&>)
            return prepend(static_cast<const Vector&>(lhs));
        else
            return prepend(std::move(lhs));
    } else {
        Vector<T> tmp(std::forward<U>(lhs));
        return prepend(std::move(tmp));
    }
}

template <ComponentType T>
Vector<T>& Vector<T>::delete_components(const std::vector<size_t>& v) {
    if (v.empty()) return *this;

    std::set<size_t> indices(v.begin(), v.end());
    for (const size_t i : indices) {
        if (i >= data.size()) throw std::invalid_argument("trying to delete non-existent component");
    }

    std::vector<T> new_data;
    new_data.reserve(data.size() - indices.size());

    for (size_t i = 0; i < data.size(); ++i) {
        if (!indices.contains(i)) new_data.push_back(std::move(data[i]));
    }

    data = std::move(new_data);
    cache.invalidate();
    return *this;
}

#ifdef CECCO_ERASURE_SUPPORT

template <ComponentType T>
Vector<T>& Vector<T>::erase_components(const std::vector<size_t>& v)
    requires FieldType<T>
{
    if (v.empty()) return *this;

    std::set<size_t> indices(v.begin(), v.end());
    for (auto it = indices.cbegin(); it != indices.cend(); ++it) {
        if (*it >= data.size()) throw std::invalid_argument("trying to erase non-existent component");
    }

    cache.invalidate();
    for (auto it = indices.cbegin(); it != indices.cend(); ++it) data[*it].erase();

    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::unerase_components(const std::vector<size_t>& v)
    requires FieldType<T>
{
    if (v.empty()) return *this;

    std::set<size_t> indices(v.begin(), v.end());
    for (auto it = indices.cbegin(); it != indices.cend(); ++it) {
        if (*it >= data.size()) throw std::invalid_argument("trying to un-erase non-existent component");
    }

    cache.invalidate();
    for (auto it = indices.cbegin(); it != indices.cend(); ++it) data[*it].unerase();

    return *this;
}

#endif

template <ComponentType T>
Vector<T> delete_component(const Vector<T>& lhs, size_t i) {
    Vector<T> res(lhs);
    res.delete_component(i);
    return res;
}

template <ComponentType T>
Vector<T> delete_component(Vector<T>&& lhs, size_t i) {
    Vector<T> res(std::move(lhs));
    res.delete_component(i);
    return res;
}

template <ComponentType T>
Vector<T>& Vector<T>::pad_front(size_t n) {
    if (n <= data.size()) return *this;

    std::vector<T> new_data(n, T(0));
    std::copy(data.begin(), data.end(), new_data.begin() + (n - data.size()));
    data = std::move(new_data);
    return *this;  // padding with zeros leaves the Hamming weight unchanged
}

template <ComponentType T>
Vector<T>& Vector<T>::pad_back(size_t n) {
    if (n <= data.size()) return *this;

    data.resize(n, T(0));
    return *this;  // padding with zeros leaves the Hamming weight unchanged
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::fill(const T& value) {
    cache.invalidate();
    std::fill(data.begin(), data.end(), value);
    if constexpr (ReliablyComparableType<T>) {
        if (value == T(0))
            cache.template set<Weight>(0);
        else
            cache.template set<Weight>(data.size());
    }

    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::rotate_left(size_t i) {
    if (data.size() > 1) std::rotate(data.begin(), data.begin() + i % data.size(), data.end());
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::rotate_right(size_t i) {
    if (data.size() > 1) std::rotate(data.rbegin(), data.rbegin() + i % data.size(), data.rend());
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::reverse() {
    std::reverse(data.begin(), data.end());
    return *this;
}

template <ComponentType T>
template <typename U>
Vector<T>& Vector<T>::set_component(size_t i, U&& c)
    requires std::convertible_to<std::decay_t<U>, T>
{
    if (i >= data.size()) throw std::invalid_argument("trying to access non-existent element");

    T& old_value = data[i];

    T new_value(std::forward<U>(c));
    if constexpr (ReliablyComparableType<T>) {
        if (old_value == new_value) return *this;
    }
    old_value = std::move(new_value);

    cache.invalidate();

    return *this;
}

template <ComponentType T>
const T& Vector<T>::operator[](size_t i) const {
    if (i >= data.size()) throw std::invalid_argument("trying to access non-existent element");
    return data[i];
}

template <ComponentType T>
Vector<T> Vector<T>::get_subvector(size_t i, size_t w) const& {
    if (i + w > data.size())
        throw std::invalid_argument(
            "trying to extract a subvector with incompatible "
            "length");
    Vector res;
    res.data.assign(data.begin() + i, data.begin() + i + w);
    return res;
}

template <ComponentType T>
Vector<T>& Vector<T>::get_subvector(size_t i, size_t w) && {
    if (i + w > data.size())
        throw std::invalid_argument(
            "trying to extract a subvector with incompatible "
            "length");
    data.resize(i + w);
    data.erase(data.begin(), data.begin() + i);
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::set_subvector(const Vector& v, size_t i) {
    if (i + v.get_n() > data.size())
        throw std::invalid_argument(
            "trying to replace subvector with "
            "vector of incompatible length");
    if (this == &v) return *this;  // only reachable with i == 0: overwriting the vector with itself
    cache.invalidate();
    std::copy(v.data.begin(), v.data.end(), data.begin() + i);
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::set_subvector(Vector&& v, size_t i) {
    if (i + v.get_n() > data.size())
        throw std::invalid_argument(
            "trying to replace subvector with "
            "vector of incompatible length");
    if (this == &v) return *this;  // only reachable with i == 0: overwriting the vector with itself
    cache.invalidate();
    std::move(v.data.begin(), v.data.end(), data.begin() + i);
    return *this;
}

template <ComponentType T>
constexpr bool Vector<T>::is_zero() const
    requires ReliablyComparableType<T>
{
    const T zero(0);
    return std::all_of(data.cbegin(), data.cend(), [&zero](const T& x) { return x == zero; });
}

template <ComponentType T>
constexpr bool Vector<T>::is_pairwise_distinct() const
    requires ReliablyComparableType<T>
{
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = i + 1; j < data.size(); ++j) {
            if (data[i] == data[j]) return false;
        }
    }

    return true;
}

template <ComponentType T>
constexpr size_t Vector<T>::calculate_weight() const
    requires ReliablyComparableType<T>
{
    size_t res = data.size() - std::count(data.cbegin(), data.cend(), T(0));

#ifdef CECCO_ERASURE_SUPPORT
    if constexpr (FieldType<T>) res -= std::count_if(data.cbegin(), data.cend(), [](T x) { return x.is_erased(); });
#endif

    return res;
}

template <ComponentType T>
constexpr size_t Vector<T>::burst_length() const
    requires ReliablyComparableType<T>
{
    const T zero(0);

    auto first_nonzero = std::find_if(data.begin(), data.end(), [&zero](const T& x) { return x != zero; });

    if (first_nonzero == data.end()) return 0;  // All zeros

    auto last_nonzero = std::find_if(data.rbegin(), data.rend(), [&zero](const T& x) { return x != zero; });

    size_t L = std::distance(data.begin(), first_nonzero);
    size_t R = data.size() - 1 - std::distance(data.rbegin(), last_nonzero);

    return R - L + 1;
}

template <ComponentType T>
constexpr size_t Vector<T>::cyclic_burst_length() const
    requires ReliablyComparableType<T>
{
    if (data.empty()) return 0;

    // Handle all-zero vector
    if (burst_length() == 0) return 0;

    const T zero(0);
    size_t n = data.size();
    size_t max_zero_run = 0;
    size_t current_zero_run = 0;

    for (size_t i = 0; i < n; ++i) {
        if (data[i] == zero) {
            current_zero_run++;
            max_zero_run = std::max(max_zero_run, current_zero_run);
        } else {
            current_zero_run = 0;
        }
    }

    // a zero run may wrap around from the end of the vector to its start
    if (data[0] == zero && data[n - 1] == zero) {
        size_t zeros_from_start = 0;
        for (size_t i = 0; i < n && data[i] == zero; ++i) {
            zeros_from_start++;
        }

        size_t zeros_from_end = 0;
        for (size_t i = n; i > 0 && data[i - 1] == zero; --i) {
            zeros_from_end++;
        }

        if (zeros_from_start + zeros_from_end < n) {  // Avoid double counting all-zero case
            max_zero_run = std::max(max_zero_run, zeros_from_start + zeros_from_end);
        }
    }

    return n - max_zero_run;
}

template <ComponentType T>
Vector<T>& Vector<T>::randomize()
    requires CoefficientType<T>
{
    cache.invalidate();
    if constexpr (FieldType<T>) {
        std::ranges::for_each(data, std::mem_fn(&T::randomize));
    } else if constexpr (std::same_as<T, double>) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::ranges::for_each(data, [&](double& val) { val = dist(gen()); });
    } else if constexpr (std::same_as<T, std::complex<double>>) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::ranges::for_each(data,
                              [&](std::complex<double>& val) { val = std::complex<double>(dist(gen()), dist(gen())); });
    } else if constexpr (SignedIntType<T>) {
        std::uniform_int_distribution<long long> dist(-100, 100);
        std::ranges::for_each(data, [&](T& val) { val = T(dist(gen())); });
    }
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::randomize_nonzero()
    requires FieldType<T>
{
    cache.invalidate();
    const T zero(0);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = zero;
        data[i].randomize_force_change();
    }
    if constexpr (ReliablyComparableType<T>) cache.template set<Weight>(data.size());
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::randomize_pairwise_distinct()
    requires FiniteFieldType<T>
{
    constexpr size_t q = T::get_size();
    const size_t n = data.size();
    if (n > q)
        throw std::invalid_argument("Cannot generate pairwise distinct vector: length " + std::to_string(n) +
                                    " exceeds field size " + std::to_string(q));
    cache.invalidate();
    // partial Fisher-Yates over the label range, tracking only displaced entries: O(n) instead of O(q)
    std::unordered_map<size_t, size_t> displaced;
    const auto label_at = [&](size_t idx) {
        const auto it = displaced.find(idx);
        return it != displaced.end() ? it->second : idx;
    };
    bool zero_drawn = false;
    for (size_t i = 0; i < n; ++i) {
        const size_t j = std::uniform_int_distribution<size_t>(i, q - 1)(gen());
        const size_t label = label_at(j);
        displaced[j] = label_at(i);
        if (label == 0) zero_drawn = true;
        data[i] = T(label);
    }
    // labels are pairwise distinct, so the zero element occurs at most once
    cache.template set<Weight>(zero_drawn ? n - 1 : n);
    return *this;
}

template <ComponentType T>
size_t Vector<T>::as_integer() const
    requires FiniteFieldType<T>
{
    constexpr size_t q = T::get_size();
    size_t result = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        const auto& digit = data[i];
#ifdef CECCO_ERASURE_SUPPORT
        if (digit.is_erased()) throw std::invalid_argument("Cannot convert vector with erased component to integer");
#endif
        if (result > (std::numeric_limits<size_t>::max() - digit.get_label()) / q)
            throw std::overflow_error("Cannot convert vector to integer, value exceeds the size_t range!");
        result = result * q + digit.get_label();
    }
    return result;
}

template <ComponentType T>
Vector<T>& Vector<T>::from_integer(size_t value, size_t n)
    requires FiniteFieldType<T>
{
    constexpr size_t q = T::get_size();

    Vector<T> v(n);
    for (size_t i = 0; i < n; ++i) {
        v.data[n - 1 - i] = T(value % q);
        value /= q;
    }
    v.cache.invalidate();  // raw writes bypass set_component's cache handling
    if (value != 0)
        throw std::out_of_range("Cannot convert integer value to base " + std::to_string(T::get_size()) +
                                " vector of length " + std::to_string(n) + "!");
    *this = std::move(v);
    return *this;
}

template <ComponentType T>
template <FiniteFieldType S>
Matrix<S> Vector<T>::as_matrix() const
    requires FiniteFieldType<T> && SubfieldOf<T, S> && (!std::is_same_v<T, S>)
{
    const auto m = T().template as_vector<S>().get_n();
    Matrix<S> res(m, data.size());

    for (size_t i = 0; i < data.size(); ++i) {
#ifdef CECCO_ERASURE_SUPPORT
        if (data[i].is_erased()) {
            for (size_t j = 0; j < m; ++j) {
                res.erase_component(j, i);
            }
        } else {
            Matrix<S> temp(data[i].template as_vector<S>());
            temp.transpose();
            res.set_submatrix(0, i, temp);
        }
#else
        Matrix<S> temp(data[i].template as_vector<S>());
        temp.transpose();
        res.set_submatrix(0, i, temp);
#endif
    }

    return res;
}

template <ComponentType T>
Matrix<T> Vector<T>::to_matrix(size_t m) const {
    if (m == 0) throw std::invalid_argument("Trying to convert vector into a matrix with zero rows");
    if (get_n() % m != 0)
        throw std::invalid_argument(std::string("Cannot convert vector into a matrix with ") + std::to_string(m) +
                                    std::string(" rows, number of rows m (") + std::to_string(m) +
                                    std::string(") is not a divisor of vector length n (") + std::to_string(get_n()) +
                                    ")!");

    Matrix<T> M(m, get_n() / m);
    std::copy(data.begin(), data.end(), M.data.begin());
    M.type = details::Generic;  // raw writes bypass set_component's tag tracking
    M.determine_type_tag();
    return M;
}

/* free functions wrt. Vector */

/*
 * vector + vector
 */

template <ComponentType T>
constexpr Vector<T> operator+(const Vector<T>& lhs, const Vector<T>& rhs) {
    Vector res(lhs);
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator+(Vector<T>&& lhs, const Vector<T>& rhs) {
    Vector res(std::move(lhs));
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator+(const Vector<T>& lhs, Vector<T>&& rhs) {
    Vector res(std::move(rhs));
    res += lhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator+(Vector<T>&& lhs, Vector<T>&& rhs) {
    Vector res(std::move(lhs));
    res += rhs;
    return res;
}

/*
 * vector - vector
 */

template <ComponentType T>
constexpr Vector<T> operator-(const Vector<T>& lhs, const Vector<T>& rhs) {
    Vector res(lhs);
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator-(Vector<T>&& lhs, const Vector<T>& rhs) {
    Vector res(std::move(lhs));
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator-(const Vector<T>& lhs, Vector<T>&& rhs) {
    Vector res(-std::move(rhs));
    res += lhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator-(Vector<T>&& lhs, Vector<T>&& rhs) {
    Vector res(std::move(lhs));
    res -= rhs;
    return res;
}

/*
 * vector * T
 */

template <ComponentType T>
constexpr Vector<T> operator*(const Vector<T>& lhs, const T& rhs) {
    Vector res(lhs);
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator*(Vector<T>&& lhs, const T& rhs) {
    Vector res(std::move(lhs));
    res *= rhs;
    return res;
}

/*
 * T * vector
 */

template <ComponentType T>
constexpr Vector<T> operator*(const T& lhs, const Vector<T>& rhs) {
    Vector res(rhs);
    res *= lhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator*(const T& lhs, Vector<T>&& rhs) {
    Vector res(std::move(rhs));
    res *= lhs;
    return res;
}

/*
 * vector / T
 */

template <CoefficientType T>
constexpr Vector<T> operator/(const Vector<T>& lhs, const T& rhs) {
    Vector res(lhs);
    res /= rhs;
    return res;
}

template <CoefficientType T>
constexpr Vector<T> operator/(Vector<T>&& lhs, const T& rhs) {
    Vector res(std::move(lhs));
    res /= rhs;
    return res;
}

/// @brief Discrete linear convolution (= coefficients of `Polynomial(v) * Polynomial(w)`); the
/// result always has length `v.get_n() + w.get_n() - 1`
/// @throws std::invalid_argument if either input is empty
template <CoefficientType T>
Vector<T> convolve(const Vector<T>& v, const Vector<T>& w) {
    if (v.get_n() == 0 || w.get_n() == 0) throw std::invalid_argument("Cannot convolve empty vector(s)!");
    Vector<T> res(Polynomial(v) * Polynomial(w));
    res.pad_back(v.get_n() + w.get_n() - 1);
    return res;
}

template <CoefficientType T>
Vector<T> randomize(const Vector<T>& v) {
    Vector<T> result = v;
    result.randomize();
    return result;
}

template <CoefficientType T>
Vector<T> randomize(Vector<T>&& v) {
    Vector<T> result = std::move(v);
    result.randomize();
    return result;
}

template <ComponentType T>
Vector<T> set_component(const Vector<T>& v, size_t i, const T& c) {
    Vector<T> res(v);
    res.set_component(i, c);
    return res;
}

template <ComponentType T>
Vector<T> set_component(Vector<T>&& v, size_t i, const T& c) {
    Vector<T> res(std::move(v));
    res.set_component(i, c);
    return res;
}

template <ComponentType T>
constexpr Vector<T> get_subvector(const Vector<T>& v, size_t start, size_t end) {
    return v.get_subvector(start, end);
}

template <ComponentType T>
constexpr Vector<T> get_subvector(Vector<T>&& v, size_t start, size_t end) {
    return std::move(v).get_subvector(start, end);
}

template <ComponentType T>
constexpr Vector<T> set_subvector(Vector<T> v, size_t start, const Vector<T>& w) {
    return v.set_subvector(w, start);
}

template <ComponentType T>
constexpr Vector<T> set_subvector(Vector<T>&& v, size_t start, const Vector<T>& w) {
    return std::move(v).set_subvector(w, start);
}

template <ComponentType T>
constexpr Vector<T> concatenate(const Vector<T>& lhs, const Vector<T>& rhs) {
    Vector res(lhs);
    res.append(rhs);
    return res;
}

template <ComponentType T>
constexpr Vector<T> concatenate(Vector<T>&& lhs, const Vector<T>& rhs) {
    Vector res(std::move(lhs));
    res.append(rhs);
    return res;
}

template <ComponentType T>
constexpr Vector<T> concatenate(const Vector<T>& lhs, Vector<T>&& rhs) {
    Vector res(std::move(rhs));
    res.prepend(lhs);
    return res;
}

template <ComponentType T>
constexpr Vector<T> concatenate(Vector<T>&& lhs, Vector<T>&& rhs) {
    Vector res(std::move(lhs));
    res.append(rhs);
    return res;
}

template <ComponentType T>
Vector<T> delete_components(const Vector<T>& lhs, const std::vector<size_t>& v) {
    Vector res(lhs);
    res.delete_components(v);
    return res;
}

template <ComponentType T>
Vector<T> delete_components(Vector<T>&& lhs, const std::vector<size_t>& v) {
    Vector res(std::move(lhs));
    res.delete_components(v);
    return res;
}

#ifdef CECCO_ERASURE_SUPPORT

template <ComponentType T>
Vector<T> erase_components(const Vector<T>& lhs, const std::vector<size_t>& v) {
    Vector res(lhs);
    res.erase_components(v);
    return res;
}

template <ComponentType T>
Vector<T> erase_components(Vector<T>&& lhs, const std::vector<size_t>& v) {
    Vector res(std::move(lhs));
    res.erase_components(v);
    return res;
}

template <ComponentType T>
Vector<T> erase_component(const Vector<T>& lhs, size_t i) {
    Vector res(lhs);
    res.erase_component(i);
    return res;
}

template <ComponentType T>
Vector<T> erase_component(Vector<T>&& lhs, size_t i) {
    Vector res(std::move(lhs));
    res.erase_component(i);
    return res;
}

template <ComponentType T>
Vector<T> unerase_components(const Vector<T>& lhs, const std::vector<size_t>& v) {
    Vector res(lhs);
    res.unerase_components(v);
    return res;
}

template <ComponentType T>
Vector<T> unerase_components(Vector<T>&& lhs, const std::vector<size_t>& v) {
    Vector res(std::move(lhs));
    res.unerase_components(v);
    return res;
}

template <ComponentType T>
Vector<T> unerase_component(const Vector<T>& lhs, size_t i) {
    Vector res(lhs);
    res.unerase_component(i);
    return res;
}

template <ComponentType T>
Vector<T> unerase_component(Vector<T>&& lhs, size_t i) {
    Vector res(std::move(lhs));
    res.unerase_component(i);
    return res;
}

#endif

template <ComponentType T>
constexpr Vector<T> pad_front(const Vector<T>& v, size_t n) {
    Vector res(v);
    res.pad_front(n);
    return res;
}

template <ComponentType T>
constexpr Vector<T> pad_front(Vector<T>&& v, size_t n) {
    Vector res(std::move(v));
    res.pad_front(n);
    return res;
}

template <ComponentType T>
constexpr Vector<T> pad_back(const Vector<T>& v, size_t n) {
    Vector res(v);
    res.pad_back(n);
    return res;
}

template <ComponentType T>
constexpr Vector<T> pad_back(Vector<T>&& v, size_t n) {
    Vector res(std::move(v));
    res.pad_back(n);
    return res;
}

template <ComponentType T>
constexpr Vector<T> rotate_left(const Vector<T>& v, size_t i) {
    Vector res(v);
    res.rotate_left(i);
    return res;
}

template <ComponentType T>
constexpr Vector<T> rotate_left(Vector<T>&& v, size_t i) {
    Vector res(std::move(v));
    res.rotate_left(i);
    return res;
}

template <ComponentType T>
constexpr Vector<T> rotate_right(const Vector<T>& v, size_t i) {
    Vector res(v);
    res.rotate_right(i);
    return res;
}

template <ComponentType T>
constexpr Vector<T> rotate_right(Vector<T>&& v, size_t i) {
    Vector res(std::move(v));
    res.rotate_right(i);
    return res;
}

template <ComponentType T>
constexpr Vector<T> reverse(const Vector<T>& v) {
    Vector res(v);
    res.reverse();
    return res;
}

template <ComponentType T>
constexpr Vector<T> reverse(Vector<T>&& v) {
    Vector res(std::move(v));
    res.reverse();
    return res;
}

template <ComponentType T>
constexpr Vector<T> fill(const Vector<T>& v, const T& value) {
    Vector<T> res(v);
    res.fill(value);
    return res;
}

template <ComponentType T>
constexpr Vector<T> fill(Vector<T>&& v, const T& value) {
    Vector<T> res(std::move(v));
    res.fill(value);
    return res;
}

/**
 * @brief Inner product ⟨lhs, rhs⟩ = Σᵢ lhs[i] · rhs[i]
 *
 * For complex inputs this is the standard product (not the conjugate inner product).
 *
 * @throws std::invalid_argument if lengths differ
 */
template <ComponentType T>
T inner_product(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate inner product of "
            "vectors of different lengths");
    return std::inner_product(lhs.data.cbegin(), lhs.data.cend(), rhs.data.begin(), T(0));
}

/**
 * @brief Schur (component-wise) product
 * @throws std::invalid_argument if lengths differ
 */
template <ComponentType T>
Vector<T> Schur_product(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Schur product of "
            "vectors of different lengths");
    Vector<T> res(lhs.get_n());
    for (size_t i = 0; i < res.get_n(); ++i) res.data[i] = lhs.data[i] * rhs.data[i];
    res.cache.invalidate();
    return res;
}

template <ReliablyComparableType T>
constexpr bool operator==(const Vector<T>& lhs, const Vector<T>& rhs) {
    return lhs.data == rhs.data;
}

template <ReliablyComparableType T>
constexpr bool operator!=(const Vector<T>& lhs, const Vector<T>& rhs) {
    return !(lhs == rhs);
}

/**
 * @brief Length-`length` vector with `T(1)` at index `i` and zeros elsewhere
 * @throws std::invalid_argument if `i >= length`
 */
template <ComponentType T>
Vector<T> unit_vector(size_t length, size_t i) {
    if (i >= length) throw std::invalid_argument("trying to create invalid unit vector");
    Vector<T> res(length);
    res.set_component(i, T(1));
    if constexpr (ReliablyComparableType<T>) res.cache.template set<Vector<T>::Weight>(1);
    return res;
}

/** @brief Stream output as `( c₀, c₁, …, cₙ₋₁ )`. */
template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& rhs) {
    os << "( ";
    const char* sep = "";
    for (size_t i = 0; i < rhs.data.size(); ++i) {
        os << sep << rhs.data[i];
        sep = ", ";
    }
    os << " )";
    return os;
}

/** @name Error control coding-related functions
 * @{
 */

/** @brief Hamming weight; see @ref Vector::wH for semantics. */
template <ReliablyComparableType T>
constexpr size_t wH(const Vector<T>& v) {
    return v.wH();
}

/** @brief Support; see @ref Vector::supp for semantics. */
template <ReliablyComparableType T>
std::vector<size_t> supp(const Vector<T>& v) {
    return v.supp();
}

/**
 * @brief Hamming distance dₕ(lhs, rhs) = wₕ(lhs − rhs)
 *
 * Number of positions in which `lhs` and `rhs` differ. Under @ref CECCO_ERASURE_SUPPORT,
 * positions where either side is erased are not counted (subtraction propagates the
 * erasure marker, and @ref wH skips erased components).
 *
 * @throws std::invalid_argument if lengths differ
 */
template <ReliablyComparableType T>
size_t dH(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between vectors of different "
            "lengths");
    return (lhs - rhs).wH();
}

// rvalue overloads of dH share semantics with the const&,const& version above.
template <ReliablyComparableType T>
size_t dH(Vector<T>&& lhs, const Vector<T>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between vectors of different "
            "lengths");
    return (std::move(lhs) - rhs).wH();
}

template <ReliablyComparableType T>
size_t dH(const Vector<T>& lhs, Vector<T>&& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between vectors of different "
            "lengths");
    return (lhs - std::move(rhs)).wH();
}
template <ReliablyComparableType T>
size_t dH(Vector<T>&& lhs, Vector<T>&& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between vectors of different "
            "lengths");
    return (std::move(lhs) - std::move(rhs)).wH();
}

/** @brief Burst length; see @ref Vector::burst_length for semantics. */
template <ReliablyComparableType T>
constexpr size_t burst_length(const Vector<T>& v) {
    return v.burst_length();
}

/** @brief Cyclic burst length; see @ref Vector::cyclic_burst_length for semantics. */
template <ReliablyComparableType T>
constexpr size_t cyclic_burst_length(const Vector<T>& v) {
    return v.cyclic_burst_length();
}

/**
 * @brief Euclidean distance ‖lhs − rhs‖₂ = √(Σᵢ |lhs[i] − rhs[i]|²) for complex vectors
 * @throws std::invalid_argument if lengths differ
 */
inline double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate euclidean distance between vectors of different "
            "lengths");

    double sum_of_squares = std::transform_reduce(
        lhs.data.begin(), lhs.data.end(), rhs.data.begin(), 0.0, std::plus<double>{},
        [](const std::complex<double>& l, const std::complex<double>& r) { return std::norm(l - r); });

    return std::sqrt(sum_of_squares);
}

/** @} */

}  // namespace CECCO

#endif
