/**
 * @file polynomials.hpp
 * @brief Polynomial arithmetic library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.2.3
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
 * This header file provides an implementation of univariate polynomial arithmetic. It supports:
 *
 * - **Generic polynomial operations**: Over any @ref CECCO::ComponentType including finite fields,
 *   floating-point numbers, complex numbers, and signed integers
 * - **Field-specific algorithms**: Polynomial long division, GCD computation, and factorization
 *   support for @ref CECCO::FieldType coefficients
 * - **Cross-field compatibility**: Safe conversions between polynomials over related fields using
 *   @ref CECCO::SubfieldOf, @ref CECCO::ExtensionOf, and @ref CECCO::largest_common_subfield_t
 * - **Performance optimizations**: High-performance O(1) caching for expensive operations, move semantics, and
 *   Horner's method for efficient evaluation
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * // Basic polynomial operations
 * using F7 = Fp<7>;
 * Polynomial<F7> p = {1, 2, 3};  // 1 + 2x + 3x²
 * Polynomial<F7> q = {4, 5};     // 4 + 5x
 * auto r = p * q;                // Polynomial multiplication
 * F7 result = p(F7(2));          // Evaluate p at x=2 using Horner's method
 *
 * // Advanced polynomial algorithms
 * auto [quotient, remainder] = p.poly_long_div(q);  // Polynomial division
 * auto gcd = GCD(p, q);                            // Greatest common divisor
 * auto derivative_p = p.differentiate(1);          // First derivative
 *
 * // Cross-field operations (field tower compatibility)
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, MOD{1, 1, 1}>;
 * Polynomial<F2> a = {1, 0, 1};       // x² + 1 over F₂
 * Polynomial<F4> b(a);                // Safe upcast: F₂ ⊆ F₄
 * Vector<F2> coeffs = Vector<F2>(b);  // Convert to coefficient vector
 * @endcode
 *
 * @section Performance_Features
 *
 * - **Lazy evaluation**: Hamming weight computed on-demand with compile-time optimized caching
 * - **Move semantics**: Optimal performance for temporary polynomial operations and chained computations
 * - **Horner's method**: O(n) polynomial evaluation algorithm for maximum efficiency
 * - **Automatic pruning**: Maintains canonical form by removing leading zero coefficients
 * - **STL integration**: Uses standard algorithms for optimal compiler optimization
 * - **Type safety**: C++20 concepts prevent invalid operations:
 *   - @ref CECCO::ComponentType Ensures valid coefficient types
 *   - @ref CECCO::FieldType Required for division and advanced algorithms
 *   - @ref CECCO::FiniteFieldType Enables specialized finite field operations
 *   - @ref CECCO::largest_common_subfield_t Enables generalized cross-field conversions
 *
 * @see @ref fields.hpp for finite field arithmetic and extension field operations
 * @see @ref vectors.hpp for vector representations and linear algebra integration
 * @see @ref matrices.hpp for matrix representations and matrix operations
 * @see @ref field_concepts_traits.hpp for type constraints and field relationships (C++20 concepts)
 */

#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP

#include <initializer_list>
// #include <ranges> // transitive through matrices.hpp
// #include <utility> // transitive through field_concepts_traits.hpp
// #include <vector> // transitive through field_concepts_traits.hpp

#include "matrices.hpp"
// #include <algorithm> // transitive through matrices.hpp
// #include "field_concepts_traits.hpp" // transitive through matrices.hpp
// #include "helpers.hpp" // transitive through field_concepts_traits.hpp

namespace CECCO {

template <ComponentType T>
class Vector;
template <ComponentType T>
class Polynomial;

template <ComponentType T>
constexpr bool operator==(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept;
template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Polynomial<T>& rhs) noexcept;

template <ComponentType T>
Polynomial<T> ZeroPolynomial();

/**
 * @class Polynomial
 * @brief Generic univariate polynomial class for error control coding (CECCO) and finite field applications
 *
 * @tparam T Coefficient type satisfying @ref CECCO::ComponentType concept. Supported types include:
 *   - **Finite field types**: @ref CECCO::Fp, @ref CECCO::Ext, also satisfying concept @ref CECCO::FiniteFieldType
 *   - **Floating-point types**: `double`
 *   - **Complex types**: `std::complex<double>`
 *   - **Signed integer types**: Signed integer types including `InfInt` satisfying concept @ref CECCO::SignedIntType
 *
 * This class provides comprehensive polynomial operations optimized for error-correcting codes (CECCO),
 * with special support for finite field arithmetic and cross-field conversions.
 *
 * @section Implementation_Notes
 *
 * - **Coefficient storage**: Uses `std::vector<T>` with index i representing coefficient of x^i
 * - **Canonical form**: Automatically removes leading zero coefficients via pruning
 * - **Cross-field compatibility**: Safe conversions between related field types using concepts
 * - **Performance optimization**: Lazy evaluation with compile-time optimized O(1) caching for expensive operations
 * - **Type safety**: Compile-time validation of coefficient types and field relationships
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * // Create polynomials over finite fields
 * using F7 = Fp<7>;
 * Polynomial<F7> p = {1, 2, 3};  // 1 + 2x + 3x²
 * Polynomial<F7> q({4, 5});      // 4 + 5x
 *
 * // Basic polynomial operations
 * auto sum = p + q;      // Addition
 * auto product = p * q;  // Multiplication
 * F7 value = p(F7(3));   // Evaluation at x=3
 *
 * // Advanced operations (field types only)
 * auto [quotient, remainder] = p.poly_long_div(q);  // Polynomial division
 * auto derivative = p.differentiate(1);             // First derivative
 * p.normalize();                                    // Make monic
 *
 * // Applications (e.g., error control coding)
 * size_t weight = p.wH();     // Hamming weight (non-zero coefficients)
 * auto gcd_poly = GCD(p, q);  // Greatest common divisor
 *
 * // Cross-field operations (F₂ ⊆ F₄)
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, MOD{1, 1, 1}>;
 * Polynomial<F2> p_f2 = {1, 0, 1};  // x² + 1 over F₂
 * Polynomial<F4> p_f4(p_f2);        // Safe upcast: F₂ ⊆ F₄
 * @endcode
 *
 * @note Polynomial operations require compatible types. Division operations require
 *          coefficient types satisfying @ref CECCO::FieldType concept.
 *
 * @note Additional methods available via concept constraints:
 *       - **Field types only**: poly_long_div(), normalize(), differentiate()
 *       - **Finite fields and signed integers**: degree(), wH() (Hamming weight)
 *
 * @see Concept @ref CECCO::ComponentType for supported coefficient types
 * @see Concepts @ref CECCO::FieldType, @ref CECCO::FiniteFieldType for advanced operations
 * @see Concepts @ref CECCO::SubfieldOf, @ref CECCO::largest_common_subfield_t for cross-field operation constraints
 * @see @ref CECCO::Vector for coefficient vector representations
 */
template <ComponentType T>
class Polynomial {
    friend bool operator== <>(const Polynomial& lhs, const Polynomial& rhs) noexcept;
    friend std::ostream& operator<< <>(std::ostream& os, const Polynomial& rhs) noexcept;

   public:
    // Cache configuration for this class
    enum CacheIds { Weight = 0 };

    /** @name Constructors
     * @{
     */

    /**
     * @brief Default constructor creating the zero polynomial
     *
     * Creates an empty polynomial representing the zero polynomial.
     */
    constexpr Polynomial() noexcept : data(0) {}

    /**
     * @brief Constructs a constant polynomial from an integer
     *
     * @param e Integer value to convert to coefficient type T
     *
     * Creates a constant polynomial p(x) = T(e).
     */
    constexpr Polynomial(int e) noexcept : data(1) { data.back() = T(e); }

    /**
     * @brief Constructs a constant polynomial from a coefficient
     *
     * @param e Coefficient value of type T
     *
     * Creates a constant polynomial p(x) = e.
     */
    constexpr Polynomial(const T& e) : data(1) { data.back() = e; }

    /**
     * @brief Constructs a polynomial from an initializer list of coefficients
     *
     * @param l Initializer list containing coefficients in ascending degree order
     *
     * Creates a polynomial where l[i] is the coefficient of x^i.
     * Enables convenient polynomial initialization syntax:
     * @code{.cpp}
     * Polynomial<int> p{1, 2, 3};       // 1 + 2x + 3x²
     * Polynomial<Fp<7>> q{0, 1, 0, 1};  // x + x³
     * @endcode
     *
     * Automatically removes leading zero coefficients from list in order to maintain canonical form.
     */
    constexpr Polynomial(const std::initializer_list<T>& l) : data(l) { prune(); }

    /**
     * @brief Copy constructor
     *
     * @param other Polynomial to copy from
     */
    constexpr Polynomial(const Polynomial& other) noexcept : data(other.data), cache(other.cache) {}

    /**
     * @brief Move constructor
     *
     * @param other Polynomial to move from (left in valid but unspecified state)
     */
    constexpr Polynomial(Polynomial&& other) noexcept : data(std::move(other.data)), cache(std::move(other.cache)) {}

    /**
     * @brief Cross-field constructor between finite fields with the same characteristic
     *
     * @tparam S Source field type that must have the same characteristic as T
     * @param other Polynomial over field S to convert
     *
     * Safely converts polynomials between any finite fields with the same characteristic using
     * @ref largest_common_subfield_t as the conversion bridge. Supports conversions across
     * different field towers, not just within the same construction hierarchy.
     *
     * @throws std::invalid_argument if a coefficient cannot be represented in target field (downcasting not
     * possible)
     * @throws std::bad_alloc if memory allocation fails
     */
    template <FiniteFieldType S>
        requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
    Polynomial(const Polynomial<S>& other);

    /**
     * @brief Constructs polynomial from coefficient vector
     *
     * @param v Vector whose components become polynomial coefficients
     *
     * Creates a polynomial where v[i] is the coefficient of x^i.
     * The polynomial degree equals v.get_n() - 1.
     *
     * @note If you need cross-field polynomial <-> vector conversion then first convert v into the other field.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Polynomial(const Vector<T>& v);

    /** @} */

    /** @name Assignment Operators
     * @{
     */

    /**
     * @brief Scalar assignment operator
     *
     * @param rhs Coefficient value to assign
     * @return Reference to this polynomial after assignment
     */
    constexpr Polynomial& operator=(const T& rhs);

    /**
     * @brief Copy assignment operator
     *
     * @param rhs Polynomial to copy from
     * @return Reference to this polynomial after assignment
     */
    constexpr Polynomial& operator=(const Polynomial<T>& rhs);

    /**
     * @brief Move assignment operator
     *
     * @param rhs Polynomial to move from (left in valid but unspecified state)
     * @return Reference to this polynomial after assignment
     */
    constexpr Polynomial& operator=(Polynomial&& rhs) noexcept;

    /**
     * @brief Cross-field assignment operator between finite fields with the same characteristic
     *
     * @tparam S Source field type that must have the same characteristic as T
     * @param other Polynomial over field S to convert
     * @return Reference to this polynomial after assignment
     *
     * Safely converts polynomials between any finite fields with the same characteristic using
     * @ref largest_common_subfield_t as the conversion bridge. Supports conversions across
     * different field towers, not just within the same construction hierarchy.
     *
     * @throws std::invalid_argument if a coefficient cannot be represented in target field (downcasting not
     * possible)
     * @throws std::bad_alloc if memory allocation fails
     */
    template <FiniteFieldType S>
        requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
    Polynomial& operator=(const Polynomial<S>& other);

    /** @} */

    /** @name Unary Arithmetic Operations
     * @{
     */

    /**
     * @brief Unary plus operator for lvalue references (identity)
     *
     * @return Copy of this polynomial (mathematical identity operation)
     */
    constexpr Polynomial operator+() const& noexcept { return *this; }

    /**
     * @brief Unary plus operator for rvalue references (move optimization)
     *
     * @return This polynomial moved (mathematical identity operation)
     */
    constexpr Polynomial operator+() && noexcept { return std::move(*this); }

    /**
     * @brief Unary minus operator for lvalue references
     *
     * @return New polynomial with all coefficients negated
     *
     * Creates a new polynomial -p(x) where each coefficient is negated.
     * Uses copy constructor for lvalue references.
     */
    constexpr Polynomial operator-() & noexcept;

    /**
     * @brief Unary minus operator for rvalue references (move optimization)
     *
     * @return This polynomial with all coefficients negated in-place
     *
     * Efficiently negates coefficients in-place for temporary polynomials.
     */
    constexpr Polynomial operator-() && noexcept;

    /**
     * @brief Polynomial evaluation using Horner's method
     *
     * @param s Value at which to evaluate the polynomial
     * @return Result of p(s) = a₀ + a₁s + a₂s² + ... + aₙsⁿ
     *
     * Efficiently evaluates the polynomial at the given value using Horner's method
     * for O(n) complexity. Essential for Reed-Solomon encoding/decoding and root finding.
     *
     * @throws std::invalid_argument if attempting to evaluate empty polynomial
     */
    T operator()(const T& s) const;

    /** @} */

    /** @name Compound Assignment Operations
     * @{
     */

    /**
     * @brief Polynomial addition assignment
     *
     * @param rhs Polynomial to add to this polynomial
     * @return Reference to this polynomial after addition
     *
     * Performs coefficient-wise addition with automatic degree handling.
     * Automatically expands coefficient storage if rhs has higher degree.
     * Maintains canonical form through automatic pruning.
     */
    constexpr Polynomial& operator+=(const Polynomial& rhs) noexcept;

    /**
     * @brief Polynomial subtraction assignment
     *
     * @param rhs Polynomial to subtract from this polynomial
     * @return Reference to this polynomial after subtraction
     *
     * Performs coefficient-wise subtraction with automatic degree handling.
     * Automatically expands coefficient storage if rhs has higher degree.
     * Maintains canonical form through automatic pruning.
     */
    constexpr Polynomial& operator-=(const Polynomial& rhs) noexcept;

    /**
     * @brief Polynomial multiplication assignment
     *
     * @param rhs Polynomial to multiply with this polynomial
     * @return Reference to this polynomial after multiplication
     *
     * Performs polynomial multiplication using convolution algorithm.
     * Result degree is deg(this) + deg(rhs). Uses O(n²) naive algorithm.
     */
    constexpr Polynomial& operator*=(const Polynomial& rhs);

    /**
     * @brief Polynomial division assignment (field coefficients only)
     *
     * @param rhs Polynomial to divide this polynomial by
     * @return Reference to this polynomial after division (quotient)
     *
     * Performs polynomial long division, storing the quotient in this polynomial.
     * Requires field coefficient type for division operations.
     *
     * @throws std::invalid_argument if attempting to divide by zero polynomial
     *
     * @note Only available for coefficient types satisfying @ref CECCO::FieldType
     */
    Polynomial& operator/=(const Polynomial& rhs)
        requires FieldType<T>;

    /**
     * @brief Polynomial modulo assignment (field coefficients only)
     *
     * @param rhs Polynomial to compute remainder against
     * @return Reference to this polynomial after modulo operation (remainder)
     *
     * Computes polynomial remainder: this = this mod rhs.
     * Uses optimized algorithm for equal degree case, full division otherwise.
     * Critical for modular polynomial arithmetic in coding theory.
     *
     * @throws std::invalid_argument if attempting modulo by zero polynomial
     *
     * @note Only available for coefficient types satisfying @ref CECCO::FieldType
     */
    Polynomial& operator%=(const Polynomial& rhs)
        requires FieldType<T>;

    /**
     * @brief Scalar multiplication assignment
     *
     * @param s Scalar value to multiply with
     * @return Reference to this polynomial after multiplication
     *
     * Multiplies each coefficient by the scalar: p(x) *= s.
     */
    constexpr Polynomial& operator*=(const T& s) noexcept;

    /**
     * @brief Scalar division assignment
     *
     * @param s Scalar value to divide by
     * @return Reference to this polynomial after division
     *
     * Divides each coefficient by the scalar: p(x) /= s.
     *
     * @throws std::invalid_argument if attempting to divide by zero scalar
     *
     * @warning Reliable results ((p / s) *  s == p) for a polynomial p and nonzero scalar s are only guaranteed in case
     * T fulfills concept FieldType<T>
     */
    Polynomial& operator/=(const T& s);

    /**
     * @brief Integer multiplication assignment (field coefficients only)
     *
     * @param n Integer value to multiply with (handles characteristic)
     * @return Reference to this polynomial after multiplication
     *
     * Multiplies each coefficient by integer n, accounting for field characteristic.
     * For finite fields, reduces n modulo characteristic before multiplication.
     *
     * @note Only available for coefficient types satisfying @ref CECCO::FieldType
     */
    constexpr Polynomial& operator*=(size_t n) noexcept
        requires FieldType<T>;

    /**
     * @brief Polynomial long division
     *
     * @param rhs Divisor polynomial
     * @return Pair containing quotient and remainder polynomials
     *
     * Performs complete polynomial long division: this = quotient * rhs + remainder.
     * Essential for polynomial GCD computation and modular arithmetic.
     * Handles special cases (constant divisor, degree conditions) efficiently.
     *
     * @throws std::invalid_argument if attempting division by zero polynomial
     *
     * @note Only available for coefficient types satisfying @ref CECCO::FieldType
     *
     * @code{.cpp}
     * using F7 = Fp<7>;
     * Polynomial<F7> p = {1, 2, 3};  // 1 + 2x + 3x²
     * Polynomial<F7> q = {4, 5};     // 4 + 5x
     * auto [quotient, remainder] = p.poly_long_div(q);
     * @endcode
     */
    std::pair<Polynomial<T>, Polynomial<T>> poly_long_div(const Polynomial<T>& rhs) const
        requires FieldType<T>;

    /** @} */

    /** @name Randomization
     * @{
     */

    /**
     * @brief Randomize polynomial coefficients
     *
     * @param d Desired degree of the randomized polynomial
     * @return Reference to this polynomial after randomization
     *
     * Creates a random polynomial of degree exactly d with random coefficients.
     * Ensures the leading coefficient is non-zero to maintain the exact degree.
     * Uses the coefficient type's randomize() method for field elements.
     */
    Polynomial& randomize(size_t d) noexcept;

    /** @} */

    /** @name Differentiation
     * @{
     */

    /**
     * @brief Compute s-th classical derivative (field coefficients only)
     *
     * @param s Order of differentiation (0 = identity, 1 = first derivative, etc.)
     * @return Reference to this polynomial after differentiation
     *
     * Computes the s-th formal derivative using the standard formula:
     * d^s/dx^s [∑ aᵢxᵢ] = ∑ (i!/(i-s)!) aᵢx^(i-s) for i ≥ s
     *
     * For s=0, returns the original polynomial unchanged.
     *
     * @throws std::invalid_argument if attempting to differentiate empty polynomial
     *
     * @note Only available for coefficient types satisfying @ref CECCO::FieldType
     */
    Polynomial& differentiate(size_t s)
        requires FieldType<T>;

    /**
     * @brief Compute s-th Hasse derivative (field coefficients only)
     *
     * @param s Order of Hasse differentiation
     * @return Reference to this polynomial after Hasse differentiation
     *
     * Computes the s-th Hasse derivative using binomial coefficients:
     * D^(s)[∑ aᵢxᵢ] = ∑ C(i,s) aᵢx^(i-s) for i ≥ s
     *
     * @throws std::invalid_argument if attempting to Hasse differentiate empty polynomial
     *
     * @note Only available for coefficient types satisfying @ref CECCO::FieldType
     */
    Polynomial& Hasse_differentiate(size_t s)
        requires FieldType<T>;

    /** @} */

    /** @name Information and Properties
     * @{
     */

    /**
     * @brief Get the degree of the polynomial
     *
     * @return Degree of the polynomial (highest power of x with non-zero coefficient)
     *
     * Returns deg(p) = max{i : aᵢ ≠ 0} for polynomial p(x) = ∑ aᵢxᵢ.
     *
     * @throws std::invalid_argument if called on empty polynomial
     *
     * @note Only available for types satisfying @ref ReliablyComparableType
     */
    size_t degree() const
        requires ReliablyComparableType<T>;

    /**
     * @brief Get the trailing degree (lowest power with non-zero coefficient)
     *
     * @return Trailing degree of the polynomial
     *
     * Returns the trailing degree: min{i : aᵢ ≠ 0} for polynomial p(x) = ∑ aᵢxᵢ.
     *
     * @throws std::invalid_argument if called on empty polynomial
     *
     * @note Only available for types satisfying @ref ReliablyComparableType
     */
    size_t trailing_degree() const
        requires ReliablyComparableType<T>;

    /**
     * @brief Get the trailing coefficient (coefficient of lowest power term)
     *
     * @return Reference to trailing coefficient
     *
     * Returns the coefficient of x^k where k is the trailing degree.
     *
     * @throws std::invalid_argument if called on empty polynomial
     *
     * @note Only available for types satisfying @ref ReliablyComparableType
     */
    const T& trailing_coefficient() const
        requires ReliablyComparableType<T>;

    /**
     * @brief Get the leading coefficient (coefficient of highest power term)
     *
     * @return Reference to leading coefficient
     *
     * Returns the coefficient of x^n where n is the degree.
     *
     * @throws std::invalid_argument if called on empty polynomial
     */
    const T& leading_coefficient() const;

    /**
     * @brief Check if polynomial is empty
     *
     * @return true if polynomial has no coefficients, false otherwise
     *
     * An empty polynomial represents an uninitialized state, distinct from
     * the zero polynomial which has degree 0.
     */
    constexpr bool is_empty() const noexcept { return data.size() == 0; }

    /**
     * @brief Check if polynomial is the zero polynomial
     *
     * @return true if polynomial equals 0, false otherwise
     *
     * Tests whether the polynomial is identically zero: p(x) = 0.
     * Different from empty polynomial.
     *
     * @throws std::invalid_argument if called on empty polynomial
     *
     * @note Only available for types satisfying @ref ReliablyComparableType
     */
    bool is_zero() const
        requires ReliablyComparableType<T>
    {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is zero");
        return degree() == 0 && trailing_coefficient() == T(0);
    }

    /**
     * @brief Check if polynomial is the constant polynomial 1
     *
     * @return true if polynomial equals 1, false otherwise
     *
     * Tests whether the polynomial is the multiplicative identity: p(x) = 1.
     *
     * @throws std::invalid_argument if called on empty polynomial
     *
     * @note Only available for types satisfying @ref ReliablyComparableType
     */
    bool is_one() const
        requires ReliablyComparableType<T>
    {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is one");
        return degree() == 0 && trailing_coefficient() == T(1);
    }

    /**
     * @brief Check if polynomial is monic (leading coefficient equals 1)
     *
     * @return true if leading coefficient is 1, false otherwise
     *
     * A monic polynomial has leading coefficient 1.
     *
     * @throws std::invalid_argument if called on empty polynomial
     *
     * @note Only available for types satisfying @ref ReliablyComparableType
     */
    bool is_monic() const
        requires ReliablyComparableType<T>
    {
        if (is_empty()) throw std::invalid_argument("trying to check whether empty polynomial is monic");
        return leading_coefficient() == T(1);
    }

    /**
     * @brief Check whether the polynomial is irreducible over its coefficient field
     *
     * @return True if the polynomial is irreducible, false otherwise
     *
     * Tests irreducibility by trial division: checks all monic polynomials of degree
     * up to deg(p)/2 for divisibility. A constant polynomial (degree 0) is not
     * considered irreducible; a linear polynomial (degree 1) is always irreducible.
     *
     * @throws std::invalid_argument if called on empty polynomial (via degree())
     *
     * @note Only available for coefficient types satisfying @ref CECCO::FieldType
     */
    constexpr bool is_irreducible() const
        requires FieldType<T>
    {
        const size_t d = degree();
        if (d == 0) return false;
        if (d == 1) return true;

        // find "smaller half" of factorization (if factor of deg > d/2 there must be a factor of deg < d/2 and we would
        // already have found it once i > d/2)
        for (size_t i = 1; i <= d / 2; ++i) {
            if (i == 1) {
                for (size_t j = 0; j < T::get_size(); ++j) {
                    auto p = Polynomial({T(j), 1});
                    if ((*this) % p == ZeroPolynomial<T>()) return false;
                }
            } else {
                auto I = IdentityMatrix<T>(i - 1);
                auto v = I.rowspace();
                for (auto it = v.begin(); it != v.end(); ++it) {
                    auto p = Polynomial((*it).append(Vector<T>({T(1)})));
                    if ((*this) % p == ZeroPolynomial<T>()) return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Compute Hamming weight (number of non-zero coefficients)
     *
     * @return Number of non-zero coefficients
     *
     * Uses lazy evaluation with O(1) compile-time optimized caching for optimal performance on repeated calls.
     *
     * @note Only available for types satisfying @ref ReliablyComparableType
     */
    size_t wH() const noexcept
        requires ReliablyComparableType<T>
    {
        return cache.template get_or_compute<Weight>([this] { return calculate_weight(); });
    }

    /** @} */

    /** @name Coefficient Access and Manipulation
     * @{
     */

    /**
     * @brief Add value to coefficient at specified position using perfect forwarding
     *
     * @tparam U Type that can be converted to T
     * @param i Index of coefficient to modify (power of x)
     * @param c Value to add to the coefficient
     * @return Reference to this polynomial after modification
     *
     * Adds c to the coefficient of x^i. Automatically grows polynomial
     * if i exceeds current degree. Maintains canonical form through pruning.
     */
    template <typename U>
    constexpr Polynomial& add_to_coefficient(size_t i, U&& c)
        requires std::convertible_to<std::decay_t<U>, T>;

    /**
     * @brief Set coefficient at specified position using perfect forwarding
     *
     * @tparam U Type that can be converted to T
     * @param i Index of coefficient to set (power of x)
     * @param c New value for the coefficient
     * @return Reference to this polynomial after modification
     *
     * Sets the coefficient of x^i to c. Automatically grows polynomial
     * if i exceeds current degree. Maintains canonical form through pruning.
     */
    template <typename U>
    constexpr Polynomial& set_coefficient(size_t i, U&& c)
        requires std::convertible_to<std::decay_t<U>, T>;

    /**
     * @brief Reverse coefficient order (reciprocal polynomial)
     *
     * @return Reference to this polynomial after reciprocal operation
     *
     * Transforms p(x) = a₀ + a₁x + ... + aₙx^n into x^n·p(1/x) = aₙ + aₙ₋₁x + ... + a₀x^n.
     */
    constexpr Polynomial& reciprocal() noexcept;

    /**
     * @brief Normalize polynomial to monic form (field coefficients only)
     *
     * @return Reference to this polynomial after normalization
     *
     * Divides all coefficients by the leading coefficient to make the polynomial monic.
     *
     * @note Only available for coefficient types satisfying @ref CECCO::FieldType
     */
    constexpr Polynomial& normalize()
        requires FieldType<T>;

    /**
     * @brief Access coefficient by index (const version)
     *
     * @param i Index of coefficient to access (power of x)
     * @return Coefficient of x^i (returns T(0) if i exceeds degree)
     *
     * Provides safe, bounds-checked access to polynomial coefficients.
     * Returns zero for indices beyond the polynomial degree.
     */
    constexpr T operator[](size_t i) const noexcept;

    /** @} */

   private:
    std::vector<T> data;

    /// High-performance O(1) cache for expensive operations (Hamming weight, etc.) - uses compile-time optimized array
    /// storage
    mutable details::Cache<details::CacheEntry<Weight, size_t>> cache;

    constexpr size_t calculate_weight() const noexcept
        requires ReliablyComparableType<T>;

    constexpr Polynomial& prune() noexcept;
};

/* member functions for Polynomial */

template <ComponentType T>
template <FiniteFieldType S>
    requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
Polynomial<T>::Polynomial(const Polynomial<S>& other) {
    const size_t deg = other.degree();
    auto indices = std::views::iota(size_t{0}, deg + 1);
    std::ranges::for_each(
        indices, [&](size_t i) { this->set_coefficient(i, T(other[i])); });  // Uses enhanced cross-field constructors
}

template <ComponentType T>
Polynomial<T>::Polynomial(const Vector<T>& v) : data(v.get_n()) {
    auto indices = std::views::iota(size_t{0}, data.size());
    std::ranges::transform(indices, data.begin(), [&v](size_t i) { return v[i]; });
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
    data = rhs.data;
    cache = rhs.cache;
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator=(Polynomial<T>&& rhs) noexcept {
    if (this == &rhs) return *this;
    data = std::move(rhs.data);
    cache = std::move(rhs.cache);
    return *this;
}

template <ComponentType T>
template <FiniteFieldType S>
    requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
Polynomial<T>& Polynomial<T>::operator=(const Polynomial<S>& rhs) {
    data.resize(0);
    for (size_t i = 0; i <= rhs.degree(); ++i)
        this->set_coefficient(i, T(rhs[i]));  // Uses enhanced cross-field constructors
    cache.invalidate();
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T> Polynomial<T>::operator-() & noexcept {
    Polynomial res(*this);
    std::ranges::for_each(res.data, [](T& c) { c = -c; });
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> Polynomial<T>::operator-() && noexcept {
    std::ranges::for_each(data, [](T& c) { c = -c; });
    cache.invalidate();
    return std::move(*this);
}

template <ComponentType T>
T Polynomial<T>::operator()(const T& s) const {
    if (data.size() == 0) throw std::invalid_argument("trying to evaluate empty polynomial");
    if (data.size() == 1) return data.front();

    // Use std::accumulate with Horner's method for polynomial evaluation
    return std::accumulate(data.crbegin() + 1, data.crend(), data.back(),
                           [&s](const T& acc, const T& coeff) { return acc * s + coeff; });
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator+=(const Polynomial<T>& rhs) noexcept {
    if (data.size() < rhs.data.size()) data.resize(rhs.data.size());
    std::transform(rhs.data.begin(), rhs.data.end(), data.begin(), data.begin(), std::plus<T>{});
    cache.invalidate();
    prune();
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator-=(const Polynomial<T>& rhs) noexcept {
    if (data.size() < rhs.data.size()) data.resize(rhs.data.size());
    std::transform(rhs.data.begin(), rhs.data.end(), data.begin(), data.begin(), std::minus<T>{});
    cache.invalidate();
    prune();
    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator*=(const Polynomial<T>& rhs) {
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
    } else {  // full-blown polynomial long division
        auto res = this->poly_long_div(rhs);
        *this = res.second;
        cache.invalidate();
    }

    return *this;
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::operator*=(const T& s) noexcept {
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
constexpr Polynomial<T>& Polynomial<T>::operator*=(size_t n) noexcept
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
Polynomial<T>& Polynomial<T>::randomize(size_t d) noexcept {
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
    if (data.size() == 0) throw std::invalid_argument("trying to differentiate empty polynomial");

    if (s == 0) return *this;
    const size_t d = data.size() - 1;
    if (d == 0 || s > d) {
        data.resize(1);
        data[0] = T(0);
        return *this;
    }
    auto indices = std::views::iota(size_t{0}, d - s + 1);
    std::ranges::for_each(indices, [&](size_t i) { data[i] = fac<size_t>(i + s) / fac<size_t>(i) * data[i + s]; });
    data.resize(data.size() - s);
    prune();
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Polynomial<T>& Polynomial<T>::Hasse_differentiate(size_t s)
    requires FieldType<T>
{
    if (data.size() == 0) throw std::invalid_argument("trying to Hasse differentiate empty polynomial");

    if (s == 0) return *this;
    const size_t d = data.size() - 1;
    if (d == 0 || s > d) {
        data.resize(1);
        data[0] = T(0);
        return *this;
    }
    auto indices = std::views::iota(size_t{0}, d - s + 1);
    std::ranges::for_each(indices, [&](size_t i) { data[i] = bin<size_t>(i + s, s) * data[i + s]; });
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
    if (data.size() == 0 || i >= data.size()) {
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
    if (data.size() == 0 || i >= data.size()) {
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
constexpr Polynomial<T>& Polynomial<T>::reciprocal() noexcept {
    std::reverse(data.begin(), data.end());
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
constexpr T Polynomial<T>::operator[](size_t i) const noexcept {
    if (i >= data.size()) return T(0);
    return data[i];
}

template <ComponentType T>
constexpr size_t Polynomial<T>::calculate_weight() const noexcept
    requires ReliablyComparableType<T>
{
    return data.size() - std::count(data.cbegin(), data.cend(), T(0));
}

template <ComponentType T>
constexpr Polynomial<T>& Polynomial<T>::prune() noexcept {
    if (data.size() == 0) return *this;

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
constexpr Polynomial<T> operator+(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(lhs);
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator+(Polynomial<T>&& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator+(const Polynomial<T>& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(rhs));
    res += lhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator+(Polynomial<T>&& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator-(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(lhs);
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator-(Polynomial<T>&& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator-(const Polynomial<T>& lhs, Polynomial<T>&& rhs) noexcept {
    Polynomial<T> res(std::move(rhs));
    res = -res;
    res += lhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator-(Polynomial<T>&& lhs, Polynomial<T>&& rhs) noexcept {
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
constexpr Polynomial<T> operator*(const Polynomial<T>& lhs, const T& rhs) noexcept {
    Polynomial<T> res(lhs);
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(Polynomial<T>&& lhs, const T& rhs) noexcept {
    Polynomial<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(const T& lhs, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(rhs);
    res *= lhs;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(const T& lhs, Polynomial<T>&& rhs) noexcept {
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
constexpr Polynomial<T> operator*(const Polynomial<T>& lhs, size_t n) noexcept {
    Polynomial<T> res(lhs);
    res *= n;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(Polynomial<T>&& lhs, size_t n) noexcept {
    Polynomial<T> res(std::move(lhs));
    res *= n;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(size_t n, const Polynomial<T>& rhs) noexcept {
    Polynomial<T> res(rhs);
    res *= n;
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> operator*(size_t n, Polynomial<T>&& rhs) noexcept {
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
constexpr Polynomial<T> reciprocal(const Polynomial<T>& poly) noexcept {
    Polynomial<T> res(poly);
    res.reciprocal();
    return res;
}

template <ComponentType T>
constexpr Polynomial<T> reciprocal(Polynomial<T>&& poly) noexcept {
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
constexpr bool operator==(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept {
    return lhs.data == rhs.data;
}

template <ComponentType T>
constexpr bool operator!=(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept {
    return !(lhs == rhs);
}

template <ComponentType T>
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
                bool positive_sign;
                if constexpr (FieldType<T>) {
                    positive_sign = rhs.data[i + 1].has_positive_sign();
                } else if constexpr (std::is_same<T, std::complex<double>>()) {
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
 * @brief Create a monomial polynomial ax^i
 *
 * @tparam T Coefficient type satisfying @ref CECCO::ComponentType
 * @param i Degree of the monomial (power of x)
 * @param a Coefficient of the monomial (defaults to 1)
 * @return Polynomial representing ax^i
 *
 * Creates a monomial polynomial with single non-zero term ax^i.
 */
template <ComponentType T>
constexpr Polynomial<T> Monomial(size_t i, auto&& a = T(1)) noexcept
    requires std::convertible_to<std::decay_t<decltype(a)>, T>
{
    Polynomial<T> res;
    res.set_coefficient(i, std::forward<decltype(a)>(a));
    return res;
}

/**
 * @brief Create the zero polynomial
 *
 * @tparam T Coefficient type satisfying @ref CECCO::ComponentType
 * @return Polynomial representing the constant 0
 *
 * Creates the polynomial p(x) = 0.
 */
template <ComponentType T>
Polynomial<T> ZeroPolynomial() {
    static const Polynomial<T> zero(0);
    return zero;
}

/**
 * @brief Create the one polynomial
 *
 * @tparam T Coefficient type satisfying @ref CECCO::ComponentType
 * @return Polynomial representing the constant 1
 *
 * Creates the polynomial p(x) = 1.
 */
template <ComponentType T>
Polynomial<T> OnePolynomial() {
    static const Polynomial<T> one(1);
    return one;
}

/**
 * @brief Compute greatest common divisor of two polynomials using Extended Euclidean Algorithm
 *
 * @tparam T Coefficient type satisfying @ref CECCO::FieldType
 * @param a First polynomial
 * @param b Second polynomial
 * @param s Optional pointer to store Bézout coefficient for polynomial a
 * @param t Optional pointer to store Bézout coefficient for polynomial b
 * @return Greatest common divisor polynomial gcd(a,b)
 *
 * Computes gcd(a,b) using the Extended Euclidean Algorithm. If s and t pointers are provided,
 * also computes Bézout coefficients such that: gcd(a,b) = s·a + t·b
 *
 * @note Only available for coefficient types satisfying @ref CECCO::FieldType
 * @note Automatically handles degree ordering (larger degree polynomial processed first)
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
 * @brief Compute greatest common divisor of multiple polynomials
 *
 * @tparam T Coefficient type satisfying @ref CECCO::FieldType
 * @param polys std::vector of polynomials to find GCD of
 * @return Greatest common divisor of all polynomials in the std::vector
 *
 * Computes gcd(p₁, p₂, ..., pₙ) by iteratively applying GCD.
 * Uses the associative property: gcd(a,b,c) = gcd(gcd(a,b), c).
 *
 * @throws std::invalid_argument if polys is empty
 *
 * @note Only available for coefficient types satisfying @ref CECCO::FieldType
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

/**
 * @brief Compute least common multiple of two polynomials
 *
 * @tparam T Coefficient type satisfying @ref CECCO::FieldType
 * @param a First polynomial
 * @param b Second polynomial
 * @return Least common multiple lcm(a,b)
 *
 * Computes lcm(a,b) using the identity: lcm(a,b) = (a·b)/gcd(a,b).
 * The LCM is the polynomial of smallest degree divisible by both a and b.
 *
 * Important for polynomial ideal operations and code construction algorithms.
 *
 * @note Only available for coefficient types satisfying @ref CECCO::FieldType
 */
template <ComponentType T>
Polynomial<T> LCM(const Polynomial<T>& a, const Polynomial<T>& b)
    requires FieldType<T>
{
    return (a * b) / GCD(a, b);
}

/**
 * @brief Compute least common multiple of multiple polynomials
 *
 * @tparam T Coefficient type satisfying @ref CECCO::FieldType
 * @param polys Vector of polynomials to find LCM of
 * @return Least common multiple of all polynomials in the vector
 *
 * Computes lcm(p₁, p₂, ..., pₙ) by iteratively applying binary LCM.
 * Uses the associative property: lcm(a,b,c) = lcm(lcm(a,b), c).
 *
 * @throws std::invalid_argument if polys is empty
 *
 * @note Only available for coefficient types satisfying @ref CECCO::FieldType
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
 * @brief Polynomial exponentiation operator
 *
 * @tparam T Coefficient type satisfying @ref CECCO::ComponentType
 * @param base Polynomial to raise to exponent
 * @param exponent Integer exponent
 * @return Polynomial base^exponent
 *
 * Computes polynomial exponentiation using square-and-multiply algorithm.
 *
 * @warning **DANGEROUS OPERATOR**: This operator violates usual precedence rules!
 * Expression `b*a^p` is evaluated as `(b*a)^p` instead of expected `b*(a^p)`.
 * Use explicit parentheses: `b*(a^p)` to ensure correct evaluation order.
 *
 * @note Consider using explicit function calls for clarity in complex expressions
 */
template <ComponentType T>
constexpr Polynomial<T> operator^(const Polynomial<T>& base, int exponent) noexcept {
    return sqm<Polynomial<T>>(base, exponent);
}

/**
 * @brief Find a random irreducible polynomial of given degree
 *
 * @tparam T Field type for coefficients (must satisfy @ref CECCO::FieldType)
 * @param degree Degree of the irreducible polynomial to find
 * @return A monic irreducible polynomial of the specified degree
 *
 * Generates random monic polynomials of the given degree and tests them for
 * irreducibility until one is found.
 *
 * @note Relies on randomization; runtime depends on the density of irreducible
 *       polynomials at the given degree
 */
template <FieldType T>
Polynomial<T> find_irreducible(size_t degree) {
    Polynomial<T> res;
    do {
        res = Polynomial<T>().randomize(degree);
    } while (!res.is_irreducible());

    res.normalize();
    return res;
}

/**
 * @brief Get the coefficient vector of the Conway polynomial for 𝔽_{p^m}
 *
 * @tparam p Prime characteristic
 * @tparam m Extension degree
 * @return Coefficient vector [a₀, a₁, ..., aₘ] of the Conway polynomial
 *
 * Returns the standardized Conway polynomial coefficients for the finite field
 * with p^m elements.
 *
 * @note Returns an empty vector if the requested (p, m) pair is not in the table
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
 * @brief Get the Conway polynomial for 𝔽_{p^m}
 *
 * @tparam p Prime characteristic
 * @tparam m Extension degree
 * @return The Conway polynomial for 𝔽_{p^m} as a Polynomial<Fp<p>>
 *
 * Convenience wrapper that constructs a polynomial from @ref ConwayCoefficients.
 *
 * @note Returns an empty polynomial if the requested (p, m) pair is not in the table
 * @see ConwayCoefficients
 */
template <uint16_t p, size_t m>
constexpr Polynomial<Fp<p>> ConwayPolynomial() {
    return Polynomial<Fp<p>>(ConwayCoefficients<p, m>());
}

}  // namespace CECCO

#endif
