/**
 * @file fields.hpp
 * @brief Finite field arithmetic library
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.0
 * @date 2025
 *
 * @copyright
 * Copyright (c) 2025, Christian Senger <senger@inue.uni-stuttgart.de>
 *
 * Licensed for noncommercial use only, including academic teaching, research, and personal non-profit purposes.
 * Commercial use is prohibited without a separate commercial license. See the [LICENSE](../../LICENSE) file in the
 * repository root for full terms and how to request a commercial license.
 *
 *
 * @section Description
 *
 * This header file provides a complete implementation of field arithmetic. It supports:
 *
 * - **Prime fields**: ℤ_p for prime p (using Fp&lt;p&gt;)
 * - **Finite extension fields**: 𝔽_{p^m} constructed via irreducible modulus polynomials (using Ext<B, modulus>)
 *   for any prime p with configurable lookup table generation modes (runtime or compile-time generated via optional
 * template parameter mode)
 * - **Field towers/graphs**: Nested extensions like 𝔽_2 ⊂ 𝔽_4 ⊂ 𝔽_16 ⊂ 𝔽_256 (also with "gaps"), more complicated
 * graphs consisting of multiple field towers with intersections are supported
 * - **Isomorphic field representations**: Multiple equivalent representations via Iso<MAIN, OTHERS...>
 * - **Cross-field constructors**: Safe conversions (isomorphic casts, up- and downcasts) using @ref
 * details::largest_common_subfield_t for optimal compatibility
 * - **Rational numbers**: ℚ with arbitrary precision arithmetic (determined by template parameter T) using Rationals<T>
 *
 * @warning A field tower in the sense of this library is a sequence of finite fields that are constructed from each
 * other as a sequence of extensions. If for example 𝔽_16 is constructed directly from 𝔽_2, then 𝔽_4 is not an element
 * of the tower (while it certainly is in the mathematical sense). If however we also create 𝔽_16 as an extension of 𝔽_4
 and 𝔽_4 as an extension of 𝔽_2 then we can merge the two towers into one by creating an isomorphic 𝔽_4 (Iso) out of the
 two constructed 𝔽_4. The isomorphic 𝔽_4 then acts as the intersection between the two field towers.
 *
 * A **finite field** 𝔽_q with q = p^m elements (p prime, m ≥ 1) is constructed as:
 * - **Prime field**: 𝔽_p ≅ ℤ_p = {0, 1, ..., p-1} with arithmetic mod p (using Fp&lt;p&gt;)
 * - **Extension field**: 𝔽_{q^m} ≅ 𝔽_q[x]/(f(x)) where q=p^m and f(x) is the irreducible (over 𝔽_q) and monic modulus
 * polynomial of degree m (using Ext<B, modulus>, where B = Fp&lt;p&gt;)
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * // Prime fields
 * using F2 = Fp<2>;
 * using F3 = Fp<3>;
 *
 * // Finite extensions of prime fields (using irreducible polynomials)
 * using F4 = Ext<F2, {1, 1, 1}, LutMode::CompileTime>;  // F2[x]/(1 + x + x²) - CompileTime LUTs
 * using F8 = Ext<F2, {1, 1, 0, 1}>;
 * using F16_a = Ext<F2, {1, 0, 0, 1, 1}>;
 * using F256_a = Ext<F2, {1, 1, 0, 1, 0, 0, 0, 1, 1}>;  // Runtime LUTs for large fields (default LUT mode)
 * using F9 = Ext<F3, {2, 2, 1}, LutMode::CompileTime>;  // F3[x]/(2 + 2x + x²)
 * using F27 = Ext<F3, {1, 0, 2, 1}, LutMode::Runtime>;  // F3[x]/(1 + 2x + x³) - Runtime LUTs (explicit)
 *
 * // Finite extensions of extension field (using irreducible polynomials)
 * using F16_b = Ext<F4, {2, 1, 1}>;         // Alternative representation of F16 as extension of F4
 * using F256_b = Ext<F4, {2, 2, 2, 0, 1}>;  // F256 as extension of F4
 * using F256_c = Ext<F16, {6, 13, 1}>;      // F256 as extension of F16
 *
 * // Isomorphic field representations
 * using F16 = Iso<F16_a, F16_b>;             // Isomorphic field with multiple representations (merges...
 *                                            // ... the field towers F2 ⊂ F16_a and F2 ⊂ F4 ⊂ F16_b)...
 * using F256 = Iso<F256_a, F256_b, F256_c>;  // ... directly using F16_a, ... F256_a, ... etc. discouraged,...
                                              // ... use F16 and F256 instead for maximal compatibility
 *
 * // Cross-field conversions
 * F9 a(5), b(7);                                // Element generation
 * auto c = a * b + F9(1);                       // Field arithmetic, automatic type of result variable
 * size_t order = a.get_multiplicative_order();  // Element order
 * Vector<F3> v = a.as_vector<F3>();             // Vector representation over F3
 *
 * F16 d(1);   // Element generation
 * F256 e(d);  // Safe upcast: F16 ⊆ F256
 * F8 f(e);    // Cross-tower cast via largest common subfield, here: F2 (can throw)
 * @endcode
 *
 * @section Performance_Features
 *
 * - **Lookup tables (LUTs)**: Used for all finite extension fields with configurable generation modes:
 *      + `LutMode::Runtime` (default): LUTs computed on first access with lazy initialization for faster compilation
 (use for large fields, with more than 150 or so elements)
 *      + `LutMode::CompileTime`: All LUTs computed at compile-time via constexpr for zero initialization overhead (use
 for small fields or zero "startup time")
 * - **Compression**: For extension fields with q ≥ COMPRESS_LUTS_FROM_Q (default: 65), + and * operations use
 compressed LUTs
 * reducing memory by ~50%
 * - **Move semantics**: Temporaries are moved for optimal performance
 * - **Type Safety**: C++20 concepts prevent invalid operations
 *      + `ECC::FieldType`: Basic field interface requirements
 *      + `ECC::FiniteFieldType`: Additional finite field interface requirements (size, characteristic ≠ 0)
 * - **Type-based Dispatch**: Compile-time polymorphism using CRTP (Curiously Recurring Template Pattern), no virtual
 function tables
 *
 * @subsection LUT_Mode_Selection
    *
 * **RunTime Mode** (default):
 * + Faster compilation, especially for large fields (more than 150 or so elements)
 * - Lazy initialization overhead on first field operations
 * - Potential runtime memory allocation
 *
 * **CompileTime Mode** :
 * + Zero runtime initialization cost
 * + Immediate field operations after construction, LUTs are already in binary
 * - (Significantly) increased compilation time and compilation memory requirements for large fields
 *
 * @section Irreducible_Polynomial_Construction
 *
 * Irreducible polynomials for extension field construction can be found using the library itself or using computational
 * algebra systems. This library requires **monic** irreducible modulus polynomials with most significant coefficient on
 * the right.
 *
 * Using the library:
 * @code{.cpp}
 * using B = Fp<3>;                  // Specify base field (any Fp, Ext or Iso)
 * size_t m = 4;                     // Specify extension degree
 * auto p = find_irreducible<B>(m);  // Find (random) monic polynomial of degree m with coefficients from B and...
 *                                   // ... irreducible over B
 * std::cout << p << std::endl;      // Output polynomial form
 * auto v = Vector(p);
 * std::cout << v << std::endl;      // Output vector form (to be used as modulus in the Ext constructor, replate (->{
 *                                    // and )->})
 * @endcode
 *
 * Using computer algebra system (here: Magma ,http://magma.maths.usyd.edu.au/calc/)
 * @code{.cpp}
 * p:=2; m:=6; F:=GaloisField(p);    // get deterministic Conway polynomial
 * P<x>:=PolynomialRing(F);
 * px:=IrreduciblePolynomial(F, m);
 * Reverse(Coefficients(px));
 * @endcode
 *
 * @code
 * q:=9; m:=2; F:=GaloisField(q); P<x>:=PolynomialRing(F); // get random polynomial
 * repeat
 *     px:=elt<P|1>;
 *     for i in [0..m] do
 *         px+:=Random(F)*x^i;
 *     end for;
 * until Degree(px) eq m and IsIrreducible(px);
 * px:=Normalize(px);
 * Reverse(Coefficients(px));
 * @endcode
 *
 * @see @ref vectors.hpp for vectors and associated operations
 * @see @ref matrices.hpp for matrices and linear algebra
 * @see @ref polynomials.hpp for polynomial arithmetic and operations
 * @see @ref field_concepts_traits.hpp for type constraints and field relationships (C++20 concepts)
 */

#ifndef FIELDS_HPP
#define FIELDS_HPP

/**
 * @def USE_LUTS_FOR_FP
 * @brief When defined, prime fields use lookup tables instead of modular arithmetic. This is normally not a good idea
 * and leads to bad performance, so don't define the macro.
 */
#ifdef DOXYGEN
#define USE_LUTS_FOR_FP
#endif

/// @def COMPRESS_LUTS_FROM_Q
/// @brief For fields with  COMPRESS_LUTS_FROM_Q or more elements, compress 2D lookup tables to save ~50% memory by
/// exploiting commutativity (store only upper triangle). Optimal value depends on cache size.
#define COMPRESS_LUTS_FROM_Q 65

#include <any>
#include <memory>
#include <mutex>
#include <optional>
#include <typeindex>
// #include <random> // transitive through field_concepts_traits.hpp

#include "polynomials.hpp"
// #include "field_concepts_traits.hpp" // transitive through polynomials.hpp
// #include "matrices.hpp"// transitive through polynomials.hpp
// #include <string> // transitive through polynomials.hpp
// #include <vector> // transitive through polynomials.hpp
// #include <array> // transitive through polynomials.hpp
// #include <ranges> // transitive through polynomials.hpp
// #include <map> // transitive through polynomials.hpp
// #include "InfInt.hpp" // transitive through polynomials.hpp
// #include "helpers.hpp" // transitive through polynomials.hpp

namespace ECC {

template <ComponentType T>
class Vector;
template <ComponentType T>
class Polynomial;
template <ComponentType T>
class Matrix;
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
class Iso;
}  // namespace ECC

namespace ECC {

namespace details {
/**
 * @class Base
 * @brief CRTP protection base class to prevent operator overload conflicts
 *
 * This class serves as a "tag" base to protect the Curiously Recurring Template Pattern (CRTP)
 * operator overloads from being applied to unintended types. For example: If x and y are two ints
 * then x + y is not supposed to call the field addition operator. Base cannot be instantiated directly,
 * ensuring that only derived field types can use the protected constructor.
 *
 * @note This follows the "CRTP protection idiom" to prevent template operators from matching
 *       unrelated types that might accidentally satisfy template requirements.
 */
class Base {
   protected:
    /**
     * @brief Protected constructor prevents direct instantiation
     *
     * Only derived field classes can construct Base, ensuring proper CRTP usage.
     */
    Base() = default;
};

/**
 * @class Field
 * @brief CRTP base class defining the field interface and documenting the mathematical interface
 *
 * @tparam T The derived field type (CRTP parameter)
 *
 * This class serves as documentation and organizational structure for field types using the
 * Curiously Recurring Template Pattern (CRTP). It defines the complete mathematical interface
 * that field types should implement, but uses **compile-time enforcement** via concepts rather
 * than virtual functions.
 *
 * This design uses **pure CRTP**:
 * @code{.cpp}
 * template<FieldType T>
 * T operator+(T lhs, const T& rhs) {
 *     lhs += rhs;  // Direct call to T::operator+= (no virtual dispatch)
 *     return lhs;
 * }
 * @endcode
 *
 * @note This provides **zero-cost abstraction** - no virtual function overhead while
 *       maintaining full type safety and mathematical correctness.
 * The field interface is enforced at **compile-time** through:
 * - **@ref FieldType concept**: Ensures required methods exist with correct signatures
 * - **Static assertions**: Verify mathematical properties and constraints
 * - **Template instantiation**: Fails if required operations are missing
 */
template <class T>
class Field : public details::Base {
   protected:
    /**
     * @brief Protected destructor (CRTP design - no polymorphic destruction needed)
     *
     * The destructor is protected to prevent direct deletion through Field<T>*,
     * which aligns with the CRTP design where objects are always of the concrete
     * derived type T.
     */
    ~Field() noexcept = default;

   public:
    /**
     * @name Assignment Operators
     * @{
     */

    /**
     * @brief Assign from integer
     * @param l Integer value to assign
     * @return Reference to this object after assignment
     *
     * @note Implementation required: Must be provided by derived field types
     */
    T& operator=(int l) = delete;

    /**
     * @brief Copy assignment operator
     * @param rhs Right-hand side field element to copy
     * @return Reference to this object after assignment
     *
     * @note Implementation required: Must be provided by derived field types
     */
    T& operator=(const T& rhs) noexcept = delete;

    /**
     * @brief Move assignment operator
     * @param rhs Right-hand side field element to move from
     * @return Reference to this object after assignment
     *
     * @note Implementation required: Must be provided by derived field types
     */
    T& operator=(T&& rhs) noexcept = delete;

    /** @} */

    /**
     * @name Comparison Operators
     * @{
     */

    /**
     * @brief Inequality comparison
     * @param rhs Right-hand side field element
     * @return true if elements are not equal, false otherwise
     *
     * @note Implementation provided
     */
    constexpr bool operator!=(const T& rhs) const noexcept { return !(static_cast<const T&>(*this) == rhs); }

    /** @} */

    /**
     * @name Unary Operations
     * @{
     */

    /**
     * @brief Unary plus (identity operation) for lvalue references
     * @return Copy of this element (mathematically: +a = a)
     *
     * @note Implementation provided: Identity operation returns copy of this element
     */
    constexpr T operator+() const& noexcept { return static_cast<const T&>(*this); }

    /**
     * @brief Unary plus (identity operation) for rvalue references
     * @return Rvalue reference to this element (avoids unnecessary copy)
     *
     * @note Implementation provided: Optimized version for temporary objects
     */
    constexpr T&& operator+() && noexcept { return static_cast<T&&>(*this); }

    /**
     * @brief Additive inverse for lvalue references
     * @return Additive inverse of this element (mathematically: -a where a + (-a) = 0)
     *
     * @note Implementation required: Must be provided by derived field types
     */
    T operator-() const& noexcept = delete;

    /**
     * @brief Additive inverse for rvalue references (in-place)
     * @return Reference to this element after negation
     *
     * @note Implementation required: Must be provided by derived field types
     */
    T& operator-() && noexcept = delete;

    /** @} */

    /**
     * @name Compound Assignment Operators
     * @{
     */

    /**
     * @brief Addition assignment (field addition)
     * @param rhs Right-hand side addend
     * @return Reference to this element after addition
     * @note Implements field addition: this = this + rhs
     * @note Implementation required: Must be provided by derived field types
     */
    T& operator+=(const T& rhs) noexcept = delete;

    /**
     * @brief Subtraction assignment (field subtraction)
     * @param rhs Right-hand side subtrahend
     * @return Reference to this element after subtraction
     * @note Implements field subtraction: this = this - rhs = this + (-rhs)
     * @note Implementation required: Must be provided by derived field types
     */
    T& operator-=(const T& rhs) noexcept = delete;

    /**
     * @brief Multiplication assignment (field multiplication)
     * @param rhs Right-hand side multiplier
     * @return Reference to this element after multiplication
     * @note Implements field multiplication: this = this * rhs
     * @note Implementation required: Must be provided by derived field types
     */
    T& operator*=(const T& rhs) noexcept = delete;

    /**
     * @brief Division assignment (field division)
     * @param rhs Right-hand side divisor
     * @return Reference to this element after division
     * @throws std::invalid_argument if rhs is zero (division by zero)
     * @note Implements field division: this = this / rhs = this * rhs⁻¹
     * @note Implementation required: Must be provided by derived field types
     */
    T& operator/=(const T& rhs) = delete;

    /** @} */

    /**
     * @name Randomization
     * @{
     */

    /**
     * @brief Set this element to a random value from the field
     * @return Reference to this element after randomization
     *
     * Uses the global random number generator to select a uniformly random
     * element from the field. May result in the same value.
     *
     * @note Implementation required: Must be provided by derived field types
     */
    Field& randomize() = delete;

    /**
     * @brief Set this element to a random value different from current
     * @return Reference to this element after randomization
     *
     * Like @ref randomize, but guarantees the new value differs from the current value.
     * Useful for generating distinct random elements.
     *
     * @note Implementation required: Must be provided by derived field types
     */
    Field& randomize_force_change() = delete;

    /** @} */

    /**
     * @name Properties
     * @{
     */

    /**
     * @brief Get the multiplicative order of this element
     * @return Smallest positive integer k such that this^k = 1
     * @throws std::invalid_argument if this element is zero (has no multiplicative order)
     *
     * The multiplicative order is the order of this element in the multiplicative
     * group (𝔽\c\{0}, *). For finite fields, this divides |𝔽\c\{0}| = |𝔽| - 1.
     *
     * @note Implementation required: Must be provided by derived field types
     */
    size_t get_multiplicative_order() const = delete;

    /**
     * @brief Get the additive order of this element
     * @return Smallest positive integer k such that k*this = 0
     *
     * The additive order is the order of this element in the additive group (𝔽, +).
     * For finite fields of characteristic p, this is p for non-zero elements and 1 for zero.
     * For ℚ, this is 1 for zero and 0 (infinite) for non-zero elements.
     *
     * @note Implementation required: Must be provided by derived field types
     */
    size_t get_additive_order() const = delete;

    /**
     * @brief Check if this element has a positive sign
     * @return true if element is considered "positive", false otherwise
     *
     * @note For finite fields, this always returns true (no ordering).
     *       For ℚ, this returns true if the rational number is positive.
     * @note Implementation required: Must be provided by derived field types
     */
    bool has_positive_sign() const noexcept = delete;

    /**
     * @brief Check if this element is the additive identity (zero)
     * @return true if this element equals the zero element of the field
     *
     * Tests whether this element is the additive identity (0) of the field,
     * satisfying: ∀a ∈ 𝔽: a + 0 = 0 + a = a
     *
     * @note Implementation required: Must be provided by derived field types
     */
    bool is_zero() const noexcept = delete;

    /**
     * @brief Erases this element, i.e., sets it to an "outside of field" marker
     * @return Reference to this element after erasing
     *
     * This is mainly used in error control coding, where an erasure means total ambiguity about the actual value of a
     * field element.
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     *
     * @note Implementation required: Must be provided by derived field types
     */
    Field& erase() noexcept = delete;

    /**
     * @brief Un-erases this element, i.e., sets it to an actual field element (the additive neutral 0)
     * @return Reference to this element after un-erasing
     *
     * This is mainly used in error control coding, where an erasure means total ambiguity about the actual value of a
     * field element.
     *
     * @warning After a field element is un-erased it can be used normally.
     *
     * @note Implementation required: Must be provided by derived field types
     */
    Field& unerase() noexcept = delete;

    /**
     * @brief Checks whether this element is erased
     * @return true if this element is erased, false otherwise (meaning it actually is a field element)
     *
     * This is mainly used in error control coding, where an erasure means total ambiguity about the actual value of a
     * field element.
     *
     * @note Implementation required: Must be provided by derived field types
     */
    bool is_erased() const noexcept = delete;
    /** @} */
};

}  // namespace details

/**
 * @name Field Arithmetic Operators
 * @brief CRTP-based operators providing field arithmetic for all field types
 *
 * These template operators use CRTP to provide consistent arithmetic operations
 * across all field types. They delegate to the compound assignment operators
 * implemented by each field class. The FieldType concept ensures they only apply
 * to valid field types.
 * @{
 */

/**
 * @brief Field addition operator
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (const reference)
 * @param rhs Right operand (const reference)
 * @return Sum lhs + rhs in the field
 */
template <FieldType T>
constexpr T operator+(const T& lhs, const T& rhs) noexcept {
    T res(lhs);
    res += rhs;
    return res;
}

/**
 * @brief Field addition operator (lvalue + rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (rvalue reference, moved)
 * @param rhs Right operand (const reference)
 * @return Sum lhs + rhs in the field
 */
template <FieldType T>
constexpr T operator+(T&& lhs, const T& rhs) noexcept {
    T res(std::move(lhs));
    res += rhs;
    return res;
}

/**
 * @brief Field addition operator (lvalue + rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (const reference)
 * @param rhs Right operand (rvalue reference, moved)
 * @return Sum lhs + rhs in the field
 */
template <FieldType T>
constexpr T operator+(const T& lhs, T&& rhs) noexcept {
    T res(std::move(rhs));
    res += lhs;
    return res;
}

/**
 * @brief Field addition operator (rvalue + rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (rvalue reference, moved)
 * @param rhs Right operand (rvalue reference)
 * @return Sum lhs + rhs in the field
 */
template <FieldType T>
constexpr T operator+(T&& lhs, T&& rhs) noexcept {
    T res(std::move(lhs));
    res += rhs;
    return res;
}

/**
 * @brief Field subtraction operator
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (const reference)
 * @param rhs Right operand (const reference)
 * @return Difference lhs - rhs in the field
 */
template <FieldType T>
constexpr T operator-(const T& lhs, const T& rhs) noexcept {
    T res(lhs);
    res -= rhs;
    return res;
}

/**
 * @brief Field subtraction operator (lvalue - rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (rvalue reference, moved)
 * @param rhs Right operand (const reference)
 * @return Difference lhs - rhs in the field
 */
template <FieldType T>
constexpr T operator-(T&& lhs, const T& rhs) noexcept {
    T res(std::move(lhs));
    res -= rhs;
    return res;
}

/**
 * @brief Field subtraction operator (lvalue - rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (const reference)
 * @param rhs Right operand (rvalue reference, moved)
 * @return Difference lhs - rhs in the field
 */
template <FieldType T>
constexpr T operator-(const T& lhs, T&& rhs) noexcept {
    T res(-std::move(rhs));
    res += lhs;
    return res;
}

/**
 * @brief Field subtraction operator (rvalue - rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (rvalue reference, moved)
 * @param rhs Right operand (rvalue reference)
 * @return Difference lhs - rhs in the field.
 */
template <FieldType T>
constexpr T operator-(T&& lhs, T&& rhs) noexcept {
    T res(std::move(lhs));
    res -= rhs;
    return res;
}

/**
 * @brief Field multiplication operator
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (const reference)
 * @param rhs Right operand (const reference)
 * @return Product lhs * rhs in the field
 */
template <FieldType T>
constexpr T operator*(const T& lhs, const T& rhs) noexcept {
    T res(lhs);
    res *= rhs;
    return res;
}

/**
 * @brief Field multiplication operator (lvalue * rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (rvalue reference, moved)
 * @param rhs Right operand (const reference)
 * @return Product lhs * rhs in the field
 */
template <FieldType T>
constexpr T operator*(T&& lhs, const T& rhs) noexcept {
    T res(std::move(lhs));
    res *= rhs;
    return res;
}

/**
 * @brief Field multiplication operator (lvalue * rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (const)
 * @param rhs Right operand (rvalue reference, moved)
 * @return Product lhs * rhs in the field
 */
template <FieldType T>
constexpr T operator*(const T& lhs, T&& rhs) noexcept {
    T res(std::move(rhs));
    res *= lhs;
    return res;
}

/**
 * @brief Field multiplication operator (rvalue * rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Left operand (rvalue reference, moved)
 * @param rhs Right operand (rvalue reference)
 * @return Product lhs * rhs in the field
 */
template <FieldType T>
constexpr T operator*(T&& lhs, T&& rhs) noexcept {
    T res(std::move(lhs));
    res *= rhs;
    return res;
}

/**
 * @brief Field multiplication by scalar (field * integer)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Field element
 * @param rhs Integer scalar
 * @return Product lhs * rhs (repeated addition)
 */
template <FieldType T>
constexpr T operator*(const T& lhs, uint16_t rhs) noexcept {
    T res(lhs);
    res *= rhs;
    return res;
}

/**
 * @brief Field multiplication by scalar (integer * field)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Integer scalar
 * @param rhs Field element
 * @return Product lhs * rhs (repeated addition)
 */
template <FieldType T>
constexpr T operator*(uint16_t lhs, const T& rhs) noexcept {
    T res(rhs);
    res *= lhs;
    return res;
}

/**
 * @brief Field multiplication by scalar (field * integer, rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Field element (rvalue reference, moved)
 * @param rhs Integer scalar
 * @return Product lhs * rhs (repeated addition)
 */
template <FieldType T>
constexpr T operator*(T&& lhs, int rhs) noexcept {
    T res(std::move(lhs));
    res *= rhs;
    return res;
}

/**
 * @brief Field multiplication by scalar (integer * field, rvalue)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Integer scalar
 * @param rhs Field element (rvalue reference, moved)
 * @return Product lhs * rhs (repeated addition)
 */
template <FieldType T>
constexpr T operator*(int lhs, T&& rhs) noexcept {
    T res(std::move(rhs));
    res *= lhs;
    return res;
}

/**
 * @brief Field division operator
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Dividend (const reference)
 * @param rhs Divisor (must be non-zero)
 * @return Quotient lhs / rhs = lhs * rhs⁻¹
 * @throws std::invalid_argument if rhs is zero
 */
template <FieldType T>
T operator/(const T& lhs, const T& rhs) {
    T res(lhs);
    res /= rhs;
    return res;
}

/**
 * @brief Field division operator
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param lhs Dividend (rvalue reference, moved)
 * @param rhs Divisor (must be non-zero)
 * @return Quotient lhs / rhs = lhs * rhs⁻¹
 * @throws std::invalid_argument if rhs is zero
 */
template <FieldType T>
T operator/(T&& lhs, const T& rhs) {
    T res(std::move(lhs));
    res /= rhs;
    return res;
}

/**
 * @brief Field exponentiation operator (USE WITH CAUTION)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param base Base element
 * @param exponent Integer exponent
 * @return base^exponent computed via square-and-multiply
 *
 * @warning This operator violates usual precedence rules!
 *          Expression `b*a^p` evaluates as `(b*a)^p` instead of `b*(a^p)`.
 *          Use explicit parentheses: `b*(a^p)` for clarity.
 *
 * @note Prefer explicit sqm<T>(base, exponent) to avoid precedence confusion.
 */
template <FieldType T>
constexpr T operator^(const T& base, int exponent) noexcept {
    return sqm<T>(base, exponent);
}

/**
 * @brief Field exponentiation operator for rvalue base (USE WITH CAUTION)
 * @tparam T Field type (must satisfy @ref ECC::FieldType concept)
 * @param base Base element (rvalue reference, moved)
 * @param exponent Integer exponent
 * @return base^exponent computed via square-and-multiply
 *
 * @warning This operator violates usual precedence rules!
 *          Expression `b*a^p` evaluates as `(b*a)^p` instead of `b*(a^p)`.
 *          Use explicit parentheses: `b*(a^p)` for clarity.
 *
 * @note Prefer explicit sqm<T>(base, exponent) to avoid precedence confusion.
 */
template <FieldType T>
constexpr T operator^(T&& base, int exponent) noexcept {
    return sqm<T>(std::move(base), exponent);
}

/** @} */

const std::string ERASURE_MARKER = "X";

/**
 * @class Rationals
 * @brief The field of rational numbers ℚ with selectable precision arithmetic for numerator and denominator
 *
 * @tparam T Signed integer type for numerator and denominator (default: InfInt)
 *
 * Implements the infinite field of rational numbers ℚ = {p/q : p,q ∈ ℤ, q ≠ 0}.
 * This is the prototypical field of **characteristic 0**, meaning no positive integer n
 * satisfies n * 1 = 0.
 *
 * @section Implementation_Notes
 *
 * - **Automatic Simplification**: All rationals always stored in lowest terms (gcd(p,q) = 1)
 * - **Sign Normalization**: Negative sign always in numerator (denominator > 0)
 * - **Arbitrary Precision**: Use T = InfInt for arbitrary precision at (significant) performance cost
 * - **Division by Zero**: Throws std::invalid_argument for zero denominators
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * Rationals<> a(3, 4);         // 3/4
 * Rationals<> b(5, 6);         // 5/6
 * Rationals<> c = a + b;       // 19/12 (automatically simplified)
 * Rationals<> d = a / b;       // 9/10
 *
 * std::cout << c;                       // Prints "19/12"
 * std::cout << c.get_characteristic();  // Prints 0
 * @endcode
 *
 * @warning For applications requiring exact arithmetic, use Rationals<InfInt>
 *          to avoid integer overflow in numerator/denominator calculations. Only in that case Rationals actually
 * realizes ℚ!
 */
template <SignedIntType T = InfInt>  // use InfInt for infinite precision... and bad performance
class Rationals : public details::Field<Rationals<T>> {
   public:
    /**
     * @brief Construct rational number from numerator and (nonzero) denominator
     * @param n Numerator (default: 0)
     * @param d Denominator (default: 1)
     * @throws std::invalid_argument if d == 0
     *
     * Constructs the rational number n/d and automatically simplifies to lowest terms.
     * The sign is normalized so that the denominator is always positive.
     */
    Rationals(int n = 0, int d = 1);

    /**
     * @brief Copy constructor
     */
    Rationals(const Rationals& other) noexcept = default;

    /**
     * @brief Move constructor
     */
    Rationals(Rationals&& other) noexcept = default;

    /* assignment operators */

    /**
     * @brief Assign integer value as rational number
     * @param l Integer value to assign
     * @return Reference to this rational number
     *
     * Sets the rational to l/1 (numerator = l, denominator = 1).
     * No simplification needed as denominator is always 1
     */
    constexpr Rationals& operator=(int l) noexcept;

    /**
     * @brief Copy assignment operator
     */
    constexpr Rationals& operator=(const Rationals& rhs) noexcept = default;

    /**
     * @brief Move assignment operator
     */
    Rationals& operator=(Rationals&& rhs) noexcept = default;

    /* comparison */
    constexpr bool operator==(const Rationals<T>& rhs) const noexcept {
        return numerator * rhs.get_denominator() == rhs.get_numerator() * denominator;
    }

    /* operations */

    /**
     * @brief Additive inverse for lvalue references
     * @return Additive inverse of this rational (copy with negated numerator)
     *
     * Returns -this, the additive inverse such that this + (-this) = 0.
     * Creates a new rational with negated numerator.
     */
    constexpr Rationals operator-() const& noexcept;

    /**
     * @brief Additive inverse for rvalue references
     * @return Reference to this rational after in-place negation
     *
     * Optimized version for temporary objects: negates numerator in-place
     * and returns reference to avoid unnecessary copy.
     */
    constexpr Rationals& operator-() && noexcept;

    /* operational assignments */

    /**
     * @brief Add rational number to this rational
     * @param rhs Right-hand side rational to add
     * @return Reference to this rational after addition
     *
     * Performs in-place addition: this = this + rhs
     * Result is automatically simplified to lowest terms.
     */
    constexpr Rationals& operator+=(const Rationals& rhs) noexcept;

    /**
     * @brief Subtract rational number from this rational
     * @param rhs Right-hand side rational to subtract
     * @return Reference to this rational after subtraction
     *
     * Performs in-place subtraction: this = this - rhs
     * Result is automatically simplified to lowest terms.
     */
    constexpr Rationals& operator-=(const Rationals& rhs) noexcept;

    /**
     * @brief Multiply this rational by another rational
     * @param rhs Right-hand side rational to multiply by
     * @return Reference to this rational after multiplication
     *
     * Performs in-place multiplication: this = this * rhs
     * Result is automatically simplified to lowest terms.
     */
    constexpr Rationals& operator*=(const Rationals& rhs) noexcept;

    /**
     * @brief Divide this rational by another rational
     * @param rhs Right-hand side rational to divide by
     * @return Reference to this rational after division
     * @throws std::invalid_argument if rhs is zero (division by zero)
     *
     * Performs in-place division: this = this / rhs
     * Result is automatically simplified to lowest terms.
     */
    Rationals& operator/=(const Rationals& rhs);

    /* randomization */

    /**
     * @brief Set this rational to a random value
     * @return Reference to this element after randomization
     *
     * Generates a random rational number by setting random numerator and denominator
     * within appropriate bounds, then simplifies to lowest terms.
     */
    Rationals& randomize() noexcept;

    /**
     * @brief Set this rational to a random value different from current
     * @return Reference to this element after randomization
     *
     * Generates a random rational number that is guaranteed to be different
     * from the current value. Useful for testing and sampling algorithms.
     */
    Rationals& randomize_force_change() noexcept;

#if 0
    /**
     * @brief Get multiplicative order of this rational
     * @return 1 if this rational equals 1, 0 (infinite) otherwise
     * @throws std::invalid_argument if this rational is zero
     *
     * In ℚ, only 1 and -1 have finite multiplicative order (both order 1 and 2 respectively).
     * All other non-zero rationals have infinite multiplicative order.
     */
    // size_t get_multiplicative_order() const;
#endif

#if 0
    /**
     * @brief Get additive order of this rational
     * @return 1 if this rational is zero, 0 (infinite) otherwise
     *
     * In ℚ, only 0 has finite additive order (order 1). All non-zero rationals
     * have infinite additive order since ℚ has characteristic 0.
     */
    // size_t get_additive_order() const noexcept;
#endif

    /**
     * @brief Get human-readable field description
     * @return String "rational number"
     */
    static const std::string& get_info() noexcept {
        static const std::string info = "rational number";
        return info;
    }

    /**
     * @brief Get field characteristic
     * @return 0 (characteristic of ℚ)
     *
     * The characteristic of ℚ is 0, meaning no positive integer n satisfies n * 1 = 0.
     */
    static constexpr size_t get_characteristic() noexcept { return 0; }

    /* properties */

    /**
     * @brief Check if element has positive sign
     * @return True if numerator and denominator have the same sign
     */
    constexpr bool has_positive_sign() const noexcept {
        return (numerator >= 0 && denominator > 0) || (numerator <= 0 && denominator < 0);
    }

    /**
     * @brief Check if this element is zero
     * @return true if this is the additive identity (0/1)
     */
    constexpr bool is_zero() const noexcept { return numerator == 0; }

    /**
     * @brief Erases this element, i.e., sets it to an "outside of field" marker
     * @return Reference to this element after erasing
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     */
    constexpr Rationals& erase() noexcept;

    /**
     * @brief Un-erases this element, i.e., sets it to an actual field element (the additive neutral 0/1)
     * @return Reference to this element after un-erasing
     */
    constexpr Rationals& unerase() noexcept;

    /**
     * @brief Checks whether this element is erased
     * @return true if this element is erased, false otherwise (meaning it actually is a field element)
     */
    constexpr bool is_erased() const noexcept { return denominator == 0; }

    /**
     * @brief Get the numerator of this rational
     * @return Numerator value
     */
    constexpr auto get_numerator() const noexcept { return numerator; }

    /**
     * @brief Get the denominator of this rational
     * @return Denominator value (always positive)
     */
    constexpr auto get_denominator() const noexcept { return denominator; }

   private:
    T numerator;    ///< Numerator (can be negative)
    T denominator;  ///< Denominator (always positive after simplification)

    /**
     * @brief Reduce this rational to lowest terms
     *
     * If numerator is zero then divides by the denominator in order to obtain 0/1 (additive neutral).
     *
     * Otherwise divides both numerator and denominator by their greatest common divisor
     * and ensures the denominator is positive. Called automatically by constructors
     * and arithmetic operations (so any rational is in lowest terms at all times).
     */
    constexpr void simplify() noexcept;
};

/* member functions for Rationals */

template <SignedIntType T>
Rationals<T>::Rationals(int n, int d) : numerator(n), denominator(d) {
    if (d == 0) throw std::invalid_argument("denominator must not be zero");
    simplify();
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator=(int l) noexcept {
    numerator = l;
    denominator = 1;
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T> Rationals<T>::operator-() const& noexcept {
    if (this->is_erased()) return Rationals().erase();
    Rationals res(*this);
    res.numerator = -res.numerator;
    return res;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator-() && noexcept {
    if (this->is_erased()) return this->erase();
    numerator = -numerator;
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator+=(const Rationals& rhs) noexcept {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    auto tn = numerator * rhs.get_denominator() + denominator * rhs.get_numerator();
    auto td = denominator * rhs.get_denominator();
    numerator = tn;
    denominator = td;
    simplify();
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator-=(const Rationals& rhs) noexcept {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    auto tn = numerator * rhs.get_denominator() - denominator * rhs.get_numerator();
    auto td = denominator * rhs.get_denominator();
    numerator = tn;
    denominator = td;
    simplify();
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::operator*=(const Rationals& rhs) noexcept {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    auto tn = numerator * rhs.get_numerator();
    auto td = denominator * rhs.get_denominator();
    numerator = tn;
    denominator = td;
    simplify();
    return *this;
}

template <SignedIntType T>
Rationals<T>& Rationals<T>::operator/=(const Rationals& rhs) {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    if (rhs.numerator == 0) throw std::invalid_argument("division by zero");
    auto tn = numerator * rhs.get_denominator();
    auto td = denominator * rhs.get_numerator();
    numerator = tn;
    denominator = td;
    simplify();
    return *this;
}

template <SignedIntType T>
Rationals<T>& Rationals<T>::randomize() noexcept {
    this->unerase();
    static std::uniform_int_distribution<int> dist(-100, 100);
    numerator = dist(gen());
    do {
        denominator = dist(gen());
    } while (denominator == 0);
    simplify();
    return *this;
}

template <SignedIntType T>
Rationals<T>& Rationals<T>::randomize_force_change() noexcept {
    this->unerase();
    static std::uniform_int_distribution<int> dist(-100, 100);
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
    return *this;
}

/*
template <SignedIntType T>
 size_t Rationals<T>::get_multiplicative_order() const {
    if (numerator == 0)
        throw std::invalid_argument(
            "trying to calculate multiplicative order "
            "of additive neutral element");
    if (numerator == denominator) {
        return 1;
    }
    return 0;
}
*/

/*
template <SignedIntType T>
 size_t Rationals<T>::get_additive_order() const noexcept {
    if (numerator == 0) {
        return 1;
    }
    return 0;
}
*/

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::erase() noexcept {
    denominator = 0;
    return *this;
}

template <SignedIntType T>
constexpr Rationals<T>& Rationals<T>::unerase() noexcept {
    if (is_erased()) (*this) = 0;
    return *this;
}

template <SignedIntType T>
constexpr void Rationals<T>::simplify() noexcept {
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
 * @brief Output stream operator for nicely printing rational numbers
 * @tparam T Integer type for numerator/denominator
 * @param os Output stream
 * @param e Rational number to output
 * @return Reference to the output stream
 *
 * Formats rational numbers as "numerator/denominator" or just "numerator"
 * when denominator is 1. Uses single stream insertion for std::setw() compatibility.
 */
template <SignedIntType T>
std::ostream& operator<<(std::ostream& os, const Rationals<T>& e) noexcept {
    if (e.is_erased()) {
        os << ERASURE_MARKER;
    } else {
        std::string temp = std::to_string(e.get_numerator());
        if (e.get_denominator() != 1) temp += "/" + std::to_string(e.get_denominator());
        os << temp;
    }
    return os;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
class Ext;

// The <255 (instead of <256) is important since we're using std::numeric_limits<>::max() as erasure flag
template <uint16_t p, uint8_t m = 1>
using label_t = typename std::conditional_t<sqm(p, m) < 255, uint8_t, uint16_t>;

namespace details {

/**
 * @brief Convert polynomial coefficients to integer label
 * @tparam q Base field size, coefficient field
 * @tparam m Extension degree, degree of polynomial
 * @tparam SIZE Array size (must be ≥ m)
 * @param coeffs Polynomial coefficients (constant term first)
 * @return Integer label representing the polynomial
 *
 * Converts polynomial coefficients to a unique integer label using
 * positional notation in base q. Used for mapping between polynomial
 * and label representations in extension fields.
 */
template <uint16_t q, uint8_t m, size_t SIZE>
static size_t constexpr integer_from_coeffs(const std::array<size_t, SIZE>& coeffs) noexcept {
    static_assert(SIZE >= static_cast<size_t>(m), "Do not specify a custom Policy");
    size_t res = coeffs[0];
    size_t t = q;
    for (uint8_t i = 1; i < m; ++i) {
        res += coeffs[i] * t;
        t *= q;
    }
    return res;
}

/**
 * @brief One-dimensional lookup table for field operations
 * @tparam LabelType Integer type for field element labels
 * @tparam FieldSize Number of elements in the field
 *
 * Stores unary operation results, such as additive/multiplicative inverses
 * and element orders.
 */
template <typename LabelType, size_t FieldSize>
struct Lut1D {
    constexpr LabelType operator()(size_t i) const noexcept { return values[i]; }
    std::array<LabelType, FieldSize> values{};
};

/**
 * @brief Two-dimensional lookup table with compression (depending on COMPRESS_LUTS_FROM_Q) for commutative operations
 * @tparam LabelType Integer type for field element labels
 * @tparam FieldSize Number of elements in the field
 *
 * Stores binary operation results with memory compression for large fields (depending on COMPRESS_LUTS_FROM_Q): For
 * compression it exploits commutativity (op(a,b) = op(b,a)) to reduce memory usage by ~50% when FieldSize ≥
 * COMPRESS_LUTS_FROM_Q.
 *
 * @section Compression_Strategy
 *
 * For large fields, only stores the upper triangle of the operation table:
 * - Small fields (< COMPRESS_LUTS_FROM_Q): Full FieldSize * FieldSize table
 * - Large fields (≥ COMPRESS_LUTS_FROM_Q): Compressed table with smart indexing
 */
template <typename LabelType, size_t FieldSize>
struct Lut2D {
    /**
     * @brief Access operator result with automatic compression handling, used to set up LUT
     * @param i First operand label
     * @param j Second operand label
     * @return Mutable reference to operation result
     */
    constexpr LabelType& operator()(size_t i, size_t j) {
        if (i > j) return operator()(j, i);
        if constexpr (FieldSize < COMPRESS_LUTS_FROM_Q) {
            return values[i][j];
        } else {
            if (i > floor_constexpr(FieldSize / 2.0)) {
                return values[FieldSize - i][j - i];
            } else {
                return values[i][j];
            }
        }
    }

    /**
     * @brief Const access operator result with automatic compression handling, used to access LUT
     * @param i First operand label
     * @param j Second operand label
     * @return Operation result
     */
    constexpr LabelType operator()(size_t i, size_t j) const noexcept {
        if (i > j) return operator()(j, i);
        if constexpr (FieldSize < COMPRESS_LUTS_FROM_Q) {
            return values[i][j];
        } else {
            if (i > floor_constexpr(FieldSize / 2.0)) {
                return values[FieldSize - i][j - i];
            } else {
                return values[i][j];
            }
        }
    }

    /// @brief Compressed storage array (size depends on field size and compression threshold COMPRESS_LUTS_FROM_Q)
    std::array < std::array<LabelType, FieldSize>,
        FieldSize<COMPRESS_LUTS_FROM_Q ? FieldSize : static_cast<size_t>(floor_constexpr(FieldSize / 2.0) + 1)>
            values{};
};

/**
 * @brief Coefficient lookup table template
 * @tparam LabelType Integer type for coefficient field element labels
 * @tparam ExtensionDegree Degree of the field extension
 * @tparam FieldSize Number of elements in the field
 *
 * Maps extension field element labels to their polynomial coefficient representations
 * over the base field.
 */
template <typename LabelType, size_t ExtensionDegree, size_t FieldSize>
struct Lut2Dcoeff {
    std::array<std::array<LabelType, ExtensionDegree>, FieldSize> values{};
};

/// @brief Computes multiplicative orders for all field elements
///
/// The multiplicative order of an element a in a finite field is the smallest positive integer k
/// such that a^k = 1. This function generates a 1D lookup table where lut_mul_ord[a] contains the
/// multiplicative order of element a in the multiplicative group 𝔽_Q \ {0}.
///
/// @tparam LabelType The field element label type
/// @tparam FieldSize The size of the finite field Q = q^m
/// @param mul_lut Reference to the 2D multiplication lookup table
/// @return A Lut1D lookup table where index i maps to the multiplicative order of field element i
///
/// @note Used internally by Fp and Ext field classes to precompute multiplicative orders
/// @note The multiplicative order divides Q-1 for finite field 𝔽_Q
/// @note Element 0 has multiplicative order 0 by convention (since 0 ∉ 𝔽_Q \ {0})
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

/// @brief Computes multiplicative inverses using direct modular arithmetic
///
/// This function computes multiplicative inverses for all non-zero field elements using the
/// extended Euclidean algorithm (via modinv function). For element a ≠ 0, it finds a^(-1)
/// such that a * a^(-1) ≡ 1 (mod p) for prime fields 𝔽_p.
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the finite field (must be prime for this method)
/// @return A Lut1D lookup table where lut_inv[a] = a^(-1) for a ≠ 0, and lut_inv[0] = 0
///
/// @note Used by Fp (prime field) classes for efficient multiplicative inverse computation
/// @note More efficient than search-based method for prime fields where direct computation is possible
/// @note For extension fields, use compute_multiplicative_inverses_search instead
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_multiplicative_inverses_direct()
    requires(is_prime<LabelType>(FieldSize))
{
    Lut1D<LabelType, FieldSize> lut_inv;
    lut_inv.values[0] = 0;
    for (LabelType i = 1; i < FieldSize; ++i) {
        int s = modinv<FieldSize, int>(i);
        if (s <= -(int)FieldSize || s >= (int)FieldSize) s %= (int)FieldSize;
        if (s < 0) s += (int)FieldSize;
        lut_inv.values[i] = s;
    }
    return lut_inv;
}

/// @brief Computes multiplicative inverses using exhaustive search
///
/// This function computes multiplicative inverses by searching through the multiplication table
/// to find pairs (a,b) such that a * b = 1. For each element a ≠ 0, it finds the unique element
/// b such that a * b ≡ 1 in the finite field.
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the finite field Q = q^m
/// @param mul_lut Reference to the 2D multiplication lookup table
/// @return A Lut1D lookup table where lut_inv[a] = a^(-1) for a ≠ 0, and lut_inv[0] = 0
///
/// @note Used by Ext (extension field) classes where direct modular arithmetic is not applicable
/// @note More general than direct method - works for any finite field structure
/// @note Exploits symmetry: if a * b = 1, then both lut_inv[a] = b and lut_inv[b] = a
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

/// @brief Computes additive inverses using direct arithmetic
///
/// This function computes additive inverses (negation) for all field elements using direct
/// modular arithmetic. For element a, it finds -a such that a + (-a) ≡ 0 (mod p) in prime
/// fields 𝔽_p, where -a = p - a for a ≠ 0.
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the finite field (must be prime for this method)
/// @return A Lut1D lookup table where lut_neg[a] = -a (additive inverse of a)
///
/// @note Used by Fp (prime field) classes for efficient additive inverse computation
/// @note In 𝔽_p: -0 = 0 and -a = p - a for a ∈ {1, 2, ..., p-1}
/// @note More efficient than search-based method for prime fields
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_additive_inverses_direct()
    requires(is_prime<LabelType>(FieldSize))
{
    Lut1D<LabelType, FieldSize> lut_neg;
    lut_neg.values[0] = 0;
    for (LabelType i = 1; i < FieldSize; ++i) lut_neg.values[i] = FieldSize - i;
    return lut_neg;
}

/// @brief Computes additive inverses using exhaustive search
///
/// This function computes additive inverses by searching through the addition table to find
/// pairs (a,b) such that a + b = 0. For each element a, it finds the unique element b such
/// that a + b ≡ 0 in the finite field.
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the finite field Q = q^m
/// @param add_lut Reference to the 2D addition lookup table
/// @return A Lut1D lookup table where lut_neg[a] = -a (additive inverse of a)
///
/// @note Used by Ext (extension field) classes where direct modular arithmetic is not applicable
/// @note More general than direct method - works for any finite field structure
/// @note Exploits symmetry: if a + b = 0, then both lut_neg[a] = b and lut_neg[b] = a
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

/// @brief Computes addition table for prime fields using modular arithmetic
///
/// This function generates a 2D lookup table for addition in prime fields 𝔽_p, where each entry
/// lut_add(a,b) = (a + b) mod p. The table exploits commutativity (a + b = b + a) for efficiency.
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the prime field p (must be prime)
/// @return A Lut2D lookup table where lut_add(a,b) = (a + b) mod p
///
/// @note Used by Fp (prime field) classes for precomputing addition operations
/// @note Only works for prime field sizes - enforced by requires clause
/// @note For extension fields, use compute_polynomial_addition_table instead
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

/// @brief Computes addition table for extension fields using polynomial arithmetic
///
/// This function generates a 2D lookup table for addition in extension fields 𝔽_{q^m}, where
/// elements are represented as polynomials over the base field 𝔽_q. Addition is performed
/// coefficient-wise: (a₀ + a₁x + ... + aₘ₋₁x^{m-1}) + (b₀ + b₁x + ... + bₘ₋₁x^{m-1}) =
/// (a₀+b₀) + (a₁+b₁)x + ... + (aₘ₋₁+bₘ₋₁)x^{m-1}.
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the extension field q^m
/// @tparam LutCoeffType The coefficient lookup table type
/// @tparam ExtensionDegree The extension degree m
/// @tparam BaseFieldType The base field type 𝔽_q
/// @param lut_coeff Coefficient lookup table mapping field elements to polynomial coefficients
/// @return A Lut2D lookup table where lut_add(a,b) represents polynomial addition a + b
///
/// @note Used by Ext (extension field) classes for precomputing addition operations
/// @note Polynomial addition is performed coefficient-wise in the base field
/// @note Works for any extension field structure
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

/// @brief Calculates the multiplicative order of a field element
///
/// The multiplicative order of an element a in a finite field is the smallest positive integer k
/// such that a^k = 1. This function computes the order by repeated multiplication until reaching
/// the multiplicative identity element.
///
/// @tparam FieldType The finite field type (must support multiplication and equality)
/// @param element The field element whose multiplicative order to compute (must be non-zero)
/// @return The multiplicative order of the given element
/// @throws std::invalid_argument if element is zero (no multiplicative order exists)
///
/// @note Used for runtime computation of individual element orders
/// @note More efficient batch (pre-)computation is provided by compute_multiplicative_orders
/// @note The multiplicative order always divides Q-1 for finite field 𝔽_Q
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
    return i;
}

/// @brief Finds a generator (primitive element) of the multiplicative group of 𝔽_Q \ {0}
///
/// A generator g of the multiplicative group has multiplicative order Q-1, meaning
/// {g⁰, g¹, g², ..., g^{Q-2}} = 𝔽_Q \ {0}. This function searches the multiplicative order table
/// to find the first element with maximum order Q-1.
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the finite field Q
/// @param lut_mul_ord 1D lookup table containing multiplicative orders of all field elements
/// @return A generator element with multiplicative order Q-1, or 0 if none found (impossible)
///
/// @note Used by finite field classes to identify primitive elements for various algorithms
template <typename LabelType, LabelType FieldSize>
constexpr LabelType find_generator(const Lut1D<LabelType, FieldSize>& lut_mul_ord) {
    for (LabelType i = 1; i < FieldSize; ++i)
        if (lut_mul_ord(i) == FieldSize - 1) return i;

    return LabelType{0};  // cannot happen
}

/// @brief Computes multiplication table for prime fields using modular arithmetic
///
/// This function generates a 2D lookup table for multiplication in prime fields 𝔽_p, where each
/// entry lut_mul(a,b) = (a * b) mod p. The table exploits commutativity (a * b = b * a) and
/// stores only the upper triangular portion for memory efficiency.
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the prime field p (must be prime)
/// @return A Lut2D lookup table where lut_mul(a,b) = (a * b) mod p
///
/// @note Used by Fp (prime field) classes for precomputing multiplication operations
/// @note Only works for prime field sizes - enforced by requires clause
/// @note For extension fields, use compute_polynomial_multiplication_table instead
template <typename LabelType, LabelType FieldSize>
constexpr auto compute_modular_multiplication_table()
    requires(is_prime<LabelType>(FieldSize))
{
    Lut2D<LabelType, FieldSize> lut_mul;

    // Main computation loop with symmetry optimization
    for (LabelType i = 0; i < FieldSize; ++i) {
        for (LabelType j = i; j < FieldSize; ++j) lut_mul(i, j) = (i * j) % FieldSize;
    }

    return lut_mul;
}

/// @brief Computes multiplication table for extension fields using base field polynomial arithmetic modulo a monic
/// irreducible Modulus
///
/// This function generates a 2D lookup table for multiplication in extension fields 𝔽_{q^m},
/// where elements are polynomials over the base field 𝔽_q. Multiplication is performed as
/// polynomial multiplication followed by reduction modulo the irreducible polynomial Modulus.
/// The process: (a₀ + a₁x + ... + aₘ₋₁x^{m-1}) * (b₀ + b₁x + ... + bₘ₋₁x^{m-1}) mod f(x).
///
/// @tparam LabelType The field element label type (e.g., uint8_t, uint16_t)
/// @tparam FieldSize The size of the extension field q^m
/// @tparam LutCoeffType The coefficient lookup table type
/// @tparam ExtensionDegree The extension degree m
/// @tparam BaseFieldType The base field type 𝔽_q
/// @tparam Modulus The irreducible polynomial used for field construction
/// @param lut_coeff Coefficient lookup table mapping field elements to polynomial coefficients
/// @return A Lut2D lookup table where lut_mul(a,b) represents polynomial multiplication a * b mod f(x)
///
/// @note Used by Ext (extension field) classes for precomputing multiplication operations
/// @note The modulus must be monic and irreducible over the base field
/// @throws std::invalid_argument if the modulus is not monic and irreducible (detected during computation)
template <typename LabelType, LabelType FieldSize, typename LutCoeffType, uint8_t ExtensionDegree,
          typename BaseFieldType, auto& Modulus>
constexpr auto compute_polynomial_multiplication_table(const LutCoeffType& lut_coeff) {
    Lut2D<LabelType, FieldSize> lut_mul;
    constexpr auto q = BaseFieldType::get_size();
    constexpr auto m = ExtensionDegree;

    // Handle special cases: multiplication by 0 and 1
    for (LabelType i = 0; i < FieldSize; ++i) {
        lut_mul(0, i) = 0;
        lut_mul(i, 0) = 0;
        lut_mul(1, i) = i;
        lut_mul(i, 1) = i;
    }

    // Main computation for elements >= 2
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
                throw std::invalid_argument("* ERROR: Extension field construction requires _irreducible_ modulus! *");
            }
        }
    }

    return lut_mul;
}

/**
 * @brief LutHolder wrapper for functions that require no provider tables
 *
 * @tparam LutType Type of the lookup table to be generated
 * @tparam F Function pointer to the table generation function
 * @tparam mode Compilation mode (CompileTime or RunTime)
 *
 * LutHolderNoProvider manages lookup table generation for functions that compute
 * tables independently without requiring other lookup tables as inputs (providers). This is
 * used for fundamental tables like modular arithmetic operations in prime fields.
 */
template <typename LutType, LutType (*F)(), LutMode mode>
struct LutHolderNoProvider;

/**
 * @brief Compile-time specialization for independent (no provider) table generation
 *
 * Generates the lookup table at compile time and stores it as a constexpr
 * static member. This provides zero-overhead table access but increases
 * compilation time and binary size for large tables.
 */
template <typename LutType, LutType (*F)()>
struct LutHolderNoProvider<LutType, F, LutMode::CompileTime> {
    static constexpr LutType lut = F();
    static constexpr const LutType& get_lut() { return lut; }
};
template <typename LutType, LutType (*F)()>
constexpr LutType LutHolderNoProvider<LutType, F, LutMode::CompileTime>::lut;

/**
 * @brief Runtime specialization for independent (no provider) table generation
 *
 * Generates the lookup table at runtime using thread-safe lazy initialization.
 * The table is computed only once when first accessed and stored in a static
 * variable. This reduces compilation time at the cost of a
 * one-time runtime computation overhead.
 */
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
 * @brief LutHolder wrapper for functions that require dependency tables (providers)
 *
 * @tparam LutType Type of the lookup table to be generated
 * @tparam ProviderLutType Type of the dependency table required as input
 * @tparam P Function pointer to obtain the dependency table (provider)
 * @tparam F Function pointer to the table generation function
 * @tparam mode Compilation mode (CompileTime or RunTime)
 *
 * LutHolder manages lookup table generation for functions that depend on other
 * lookup tables as inputs (providers). This is commonly used for extension field operations
 * that require coefficient tables from their base fields. The holder ensures
 * proper dependency resolution and initialization order.
 */
template <typename LutType, typename ProviderLutType, const ProviderLutType& (*P)(),
          LutType (*F)(const ProviderLutType& (*)()), LutMode mode>
struct LutHolder;

/**
 * @brief Compile-time specialization for dependent table generation
 *
 * Generates the target table at compile time using a dependency table (provider).
 * The dependency table is accessed via P() and passed to F() to generate
 * the final table. All computation occurs at compile time.
 */
template <typename LutType, typename ProviderLutType, const ProviderLutType& (*P)(),
          LutType (*F)(const ProviderLutType& (*)())>
struct LutHolder<LutType, ProviderLutType, P, F, LutMode::CompileTime> {
    static constexpr LutType lut = F(P);
    static constexpr const LutType& get_lut() { return lut; }
};
template <typename LutType, typename ProviderLutType, const ProviderLutType& (*P)(),
          LutType (*F)(const ProviderLutType& (*)())>
constexpr LutType LutHolder<LutType, ProviderLutType, P, F, LutMode::CompileTime>::lut;

/**
 * @brief Runtime specialization for dependent table generation
 *
 * Generates the target table at runtime using thread-safe lazy initialization.
 * The dependency table (provider) is accessed via P() and passed to F() when the target
 * table is first accessed. The target table is cached in a static variable.
 */
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
 * @class Embedding
 * @brief Explicit field embedding (functor) from subfield to superfield with reverse lookup
 *
 * @tparam SUBFIELD Source field type (must satisfy FiniteFieldType concept)
 * @tparam SUPERFIELD Target field type (must satisfy FiniteFieldType concept and SubfieldOf<SUPERFIELD, SUBFIELD>)
 *
 * Constructs and represents an explicit field embedding φ: SUBFIELD → SUPERFIELD between
 * a subfield and its containing superfield. The embedding preserves field operations and
 * provides efficient forward mapping and reverse lookup capabilities.
 *
 * @warning A mathematical SUBFIELD/SUPERFIELD relationship is only necessary, not sufficient. The two fields have to be
 * in the same construction tower, that is, SUBFIELD is used in some way in the construction of SUPERFIELD!
 *
 * @section Mathematical_Foundation
 *
 * The embedding is constructed using generator power mapping:
 * - Maps the additive and multiplicative identities: φ(0) = 0 and φ(1) = 1
 * - Computes power factor: factor = (|SUPERFIELD| - 1) / (|SUBFIELD| - 1)
 * - Maps subfield generator powers: φ(g_sub^k) = g_super^(k * factor)
 * - Ensures proper subfield inclusion while preserving structure
 *   - φ(a + b) = φ(a) + φ(b) (additivity)
 *   - φ(a * b) = φ(a) * φ(b) (multiplicativity)
 *   - φ(0_sub) = 0_super and φ(1_sub) = 1_super (identity preservation)
 *
 * @section Performance_Features
 *
 * - **Static caching**: Computed once per template instantiation using std::once_flag
 * - **O(1) forward mapping**: Direct array lookup for embedding elements
 * - **O(n) reverse lookup**: Linear search for downcast operations (optimized with std::ranges::find)
 * - **Type safety**: Template constraints ensure valid subfield relationships
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;
 * using F16 = Ext<F4, {2, 1, 1}>;
 *
 * // Forward embedding (upcast, always succeeds)
 * Embedding<F4, F16> embed;
 * F4 a(2);
 * F16 b = embed(a);  // Embed F4 element into F16
 *
 * // Reverse lookup (downcast validation)
 * F16 c(5);
 * if (auto result = embed.try_extract(c)) {
 *     F4 d = *result;  // Safe extraction succeeded
 * } else {
 *     // c is not in F4 subfield
 * }
 *
 * // Throwing extraction
 * try {
 *     F4 e = embed.extract(c);  // May throw if c not in subfield
 * } catch (const std::invalid_argument&) {
 *     // Handle extraction failure
 * }
 * @endcode
 *
 * @see @ref ECC::SubfieldOf for the concept constraining valid field relationships
 * @see @ref ECC::FiniteFieldType for underlying field type requirements
 * @see @ref ECC::Isomorphism for mappings between same-size fields
 */
template <FiniteFieldType SUBFIELD, FiniteFieldType SUPERFIELD>
    requires SubfieldOf<SUPERFIELD, SUBFIELD>
class Embedding {
   public:
    /**
     * @brief Default constructor - computes embedding using static cache
     *
     * Constructs a deterministic embedding φ: SUBFIELD → SUPERFIELD using generator-based mapping.
     * The computation is performed once per template instantiation and cached statically
     * for performance using std::once_flag.
     *
     * The embedding is constructed by:
     * 1. Setting identity mappings: φ(0) = 0 and φ(1) = 1 (required for field homomorphism)
     * 2. Computing power factor based on field sizes: factor = (|SUPERFIELD| - 1) / (|SUBFIELD| - 1)
     * 3. Mapping generator powers: φ(g_sub^k) = g_super^(k * factor) for all elements
     */
    Embedding();

    /**
     * @brief Applies the embedding φ: SUBFIELD → SUPERFIELD to transform field element
     * @param sub Source field element from SUBFIELD to be embedded
     * @return Corresponding field element in SUPERFIELD such that φ(sub) ∈ SUPERFIELD
     *
     * Transforms a subfield element to its representation in the superfield while
     * preserving structure.
     *
     * @note This operation cannot fail as every subfield element has a unique representation in the superfield
     */
    constexpr SUPERFIELD operator()(const SUBFIELD& sub) const noexcept {
        return SUPERFIELD(embedding_map[sub.get_label()]);
    }

    /**
     * @brief Extracts a subfield element from superfield
     * @param super Superfield element to extract from
     * @return Corresponding SUBFIELD element
     * @throws std::invalid_argument if the superfield element is not in the embedded subfield
     *
     * Performs reverse lookup with exception on failure.
     *
     * @note Time complexity: O(|SUBFIELD|) due to linear search through embedding map
     */
    constexpr SUBFIELD extract(const SUPERFIELD& super) const {
        auto it = std::ranges::find(embedding_map, super.get_label());
        if (it == embedding_map.end()) throw std::invalid_argument("superfield element is not in subfield");
        return SUBFIELD(std::distance(embedding_map.begin(), it));
    }

   private:
    std::vector<size_t> embedding_map;

    /**
     * @brief Computes the embedding mapping vector
     * @return Vector where embedding_map[i] = φ(SUBFIELD(i)) for i ∈ [0, |SUBFIELD|)
     *
     * Internal method that performs the actual embedding computation using the same
     * algorithm as the original compute_subfield_embedding function.
     */
    static std::vector<size_t> compute_embedding();
};

/*
Embedding member functions
*/

template <FiniteFieldType SUBFIELD, FiniteFieldType SUPERFIELD>
    requires SubfieldOf<SUPERFIELD, SUBFIELD>
Embedding<SUBFIELD, SUPERFIELD>::Embedding() : embedding_map(SUBFIELD::get_size()) {
    // Use local static cache for each specific SUBFIELD,SUPERFIELD combination
    static std::once_flag computed_flag;
    static std::vector<size_t> cached_embedding(SUBFIELD::get_size());
    std::call_once(computed_flag, []() { cached_embedding = compute_embedding(); });
    // Copy cached result to this instance
    std::copy(cached_embedding.begin(), cached_embedding.end(), embedding_map.begin());
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

    // for SUBFIELD = F2: that's it
    if (sub_size <= 2) return embedding;

    // Get generators and power factor
    const auto sub_gen = SUBFIELD::get_generator();
    const auto super_gen = SUPERFIELD::get_generator();
    constexpr size_t power_factor = (super_size - 1) / (sub_size - 1);

    // Compute super_gen^power_factor once
    auto super_gen_to_power_factor = SUPERFIELD(1);
    for (size_t i = 0; i < power_factor; ++i) {
        super_gen_to_power_factor *= super_gen;
    }

    // Map subfield generator powers to superfield generator powers
    auto sub_elem = sub_gen;                     // Start with subfield generator
    auto sup_elem = super_gen_to_power_factor;   // Start with super_gen^power_factor
    for (size_t i = 1; i < sub_size - 1; ++i) {  // Skip 0 and 1, already handled
        embedding[sub_elem.get_label()] = sup_elem.get_label();
        // Move to next powers
        sub_elem *= sub_gen;
        sup_elem *= super_gen_to_power_factor;
    }

    return embedding;
}

/**
 * @class Isomorphism
 * @brief Explicit field isomorphism (functor) between isomorphic finite fields
 *
 * @tparam A Source field type (must satisfy FiniteFieldType concept)
 * @tparam B Target field type (must satisfy FiniteFieldType concept and Isomorphic<A, B>)
 *
 * Constructs and represents an explicit field isomorphism φ: A → B between two isomorphic
 * finite fields. The isomorphism preserves field operations: φ(a + b) = φ(a) + φ(b) and
 * φ(ab) = φ(a)φ(b) for all a, b ∈ A. This enables seamless conversion of field elements
 * between different representations of the same abstract finite field (for example the fields stacked together in an
 * Iso).
 *
 * @section Mathematical_Foundation
 *
 * The isomorphism is constructed using a deterministic generator-based approach:
 * - Maps the additive and multiplicative identities: φ(0) = 0 and φ(1) = 1
 * - Maps field generators deterministically: φ(g_A) = g_B where g_A, g_B are generators of A, B
 * - Extends consistently via power mapping: φ(g_A^k) = g_B^k for all valid powers k
 * - Ensures field homomorphism properties: φ(a + b) = φ(a) + φ(b) and φ(ab) = φ(a)φ(b)
 *
 * @section Performance_Features
 *
 * - **Compile-time construction**: Marked constexpr for compile-time isomorphism computation
 * - **Cached computation**: Map is computed once during construction and reused
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;
 * using F8 = Ext<F2, {1, 1, 0, 1}>;
 * using F16_a = Ext<F4, {2, 1, 1}>;
 * using F16_b = Ext<F4, {1, 2, 1}>;
 * using F64_a = Ext<F8, {7, 1, 1}>;
 * using F64_b = Ext<F4, {1, 2, 0, 1}>;
 *
 * Isomorphism<F16_a, F16_b> phi_16;
 * auto phi_16_inv = phi_16.inverse();
 * F16_a a(4);
 * F16_b b = phi_16(a);
 * F16_a c = phi_16_inv(b);
 * assert(a == c);
 * assert(F16_a(5) * a == phi_16_inv(phi_16(F16_a(5)) * phi_16(a)));
 *
 * Isomorphism<F64_a, F64_b> phi_64;
 * auto x = F64_a().randomize();
 * F64_b y = phi_64(x);
 * assert(x == phi_64.inverse()(y));
 * @endcode
 *
 * @see @ref ECC::Isomorphic for the concept constraining valid field pairs
 * @see @ref ECC::FiniteFieldType for underlying field type requirements
 * @see @ref ECC::Ext for extension field construction
 * @see @ref ECC::Fp for prime field implementation
 */
template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
class Isomorphism {
    using PrimeField = Fp<A::get_p()>;

   public:
    /**
     * @brief Default constructor - computes isomorphism using local static cache
     *
     * Constructs a deterministic isomorphism φ: A → B using generator-based mapping.
     * The computation is performed once per template instantiation and cached locally
     * for performance using std::once_flag/lazy initialization.
     *
     * The isomorphism is constructed by:
     * 1. Setting identity mappings: φ(0) = 0 and φ(1) = 1 (required for field homomorphism)
     * 2. Mapping generators: φ(g_A) = g_B where g_A, g_B are field generators
     * 3. Extending via powers: φ(g_A^k) = g_B^k for all multiplicative group elements
     * 4. Ensuring consistent mapping for all field elements
     */
    Isomorphism();

    /**
     * @brief Direct constructor from precomputed mapping vector
     * @param iso Precomputed mapping vector where iso[i] = φ(A(i))
     *
     * Constructs isomorphism directly from a mapping vector. This constructor
     * is primarily used internally and assumes the provided mapping is a valid
     * field isomorphism (preserves addition and multiplication).
     *
     * @warning No validation is performed on the mapping - incorrect mappings
     *          will result in undefined behavior
     */
    constexpr Isomorphism(const std::vector<size_t>& iso) : iso(iso) {}

    /**
     * @brief Applies the isomorphism φ: A → B to transform field element
     * @param a Source field element from A to be transformed
     * @return Corresponding field element in B such that φ(a) ∈ B
     *
     * Transforms a field element from source field A to target field B while
     * preserving all field operations. The mapping satisfies:
     * - φ(a + b) = φ(a) + φ(b) (additivity)
     * - φ(a * b) = φ(a) * φ(b) (multiplicativity)
     * - φ(0_A) = 0_B and φ(1_A) = 1_B (identity preservation)
     */
    constexpr B operator()(const A& a) const noexcept { return B(iso[a.get_label()]); }

    /**
     * @brief Computes the inverse isomorphism φ⁻¹: B → A
     *
     * @return Isomorphism<B, A> The inverse isomorphism mapping elements from B back to A
     *
     * Constructs the inverse isomorphism by reversing the mapping table. For every element
     * a ∈ A with φ(a) = b, the inverse satisfies φ⁻¹(b) = a. This ensures that
     * φ⁻¹(φ(a)) = a and φ(φ⁻¹(b)) = b for all valid elements.
     *
     * @section Usage_Example
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4 = Ext<F2, {1, 1, 1}>;
     * using F16_a = Ext<F4, {2, 1, 1}>;
     * using F16_b = Ext<F4, {1, 2, 1}>;
     *
     * Isomorphism<F16_a, F16_b> phi;
     * auto phi_inv = phi.inverse();
     *
     * F16_1 a(5);
     * F16_2 b = phi(a);
     * F16_1 c = phi_inv(b);
     * assert(a == c);  // Round-trip preservation
     * @endcode
     */
    constexpr Isomorphism<B, A> inverse() const;

   private:
    std::vector<size_t> iso;
};

/*
Isomorphism member functions
*/

template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
Isomorphism<A, B>::Isomorphism() : iso(A::get_size()) {
    // Use local static cache for each specific A,B combination
    static std::once_flag computed_flag;
    static std::vector<size_t> cached_iso(A::get_size());

    std::call_once(computed_flag, []() {
        // Use deterministic generator-based mapping for same-size fields

        // Map 0 → 0 and 1 → 1 (required for any field homomorphism)
        cached_iso[0] = 0;
        cached_iso[1] = 1;

        A a_gen = A::get_generator();
        B b_gen = B::get_generator();

        // Map generator to generator
        cached_iso[a_gen.get_label()] = b_gen.get_label();

        // Generate the rest by mapping powers consistently
        A a_power(1);
        B b_power(1);

        for (size_t power = 0; power < A::get_size() - 1; ++power) {
            if (cached_iso[a_power.get_label()] == 0 && a_power.get_label() != 0)
                cached_iso[a_power.get_label()] = b_power.get_label();
            a_power *= a_gen;
            b_power *= b_gen;
        }
    });

    // Copy cached result to this instance
    std::copy(cached_iso.begin(), cached_iso.end(), iso.begin());
}

template <FiniteFieldType A, FiniteFieldType B>
    requires Isomorphic<A, B>
constexpr Isomorphism<B, A> Isomorphism<A, B>::inverse() const {
    std::vector<size_t> iso_inv(A::get_size());
    auto indices = std::views::iota(size_t{0}, A::get_size());
    std::ranges::for_each(indices, [&](size_t i) { iso_inv[iso[i]] = i; });
    return Isomorphism<B, A>(std::move(iso_inv));
}

/**
 * @class Fp
 * @brief Prime field 𝔽_p ≅ ℤ/pℤ
 *
 * @tparam p Prime modulus (must be prime and ≥ 2)
 *
 * Implements finite fields of prime order p, consisting of integers {0, 1, 2, ..., p-1}
 * with arithmetic performed modulo p. These are the building blocks for all finite fields.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * // Define common prime fields
 * using F2 = Fp<2>;  // Binary field {0, 1}
 * using F3 = Fp<3>;  // Ternary field {0, 1, 2}
 * using F5 = Fp<5>;  // Quinary field {0, 1, 2, 3, 4}
 *
 * // Basic arithmetic
 * F5 a(3), b(4);
 * auto c = a + b;  // c = 2 (since 3 + 4 = 7 ≡ 2 (mod 5))
 * auto d = a * b;  // d = 2 (since 3 * 4 = 12 ≡ 2 (mod 5))
 * auto e = a / b;  // e = 2 (since 3 / 4 = 3 * 4⁻¹ = 3 * 4 = 2)
 * F5 f = 3;
 * auto h = f * e;  // e = 1 (since 3 * 2 = 6 = 1)
 *
 * // Element properties
 * size_t order = a.get_multiplicative_order();  // Order of 3 in F5 \ {0}
 * @endcode
 *
 * @section Implementation_Details
 *
 * - **Label Storage**: Elements stored as integers of type label_t<p> (automatically chosen based on field size)
 * - **Type Safety**: Template parameter ensures compile-time prime checking of template parameter p
 */
template <uint16_t p>
class Fp : public details::Field<Fp<p>> {
    static_assert(is_prime(p), "p is not a prime");

   public:
    using label_t = ::ECC::label_t<p>;

    /* constructors */

    /**
     * @brief Default constructor - creates zero element
     *
     * Initializes this prime field element to 0 (additive identity).
     */
    constexpr Fp() noexcept : label(0) {}

    /**
     * @brief Construct prime field element from integer
     * @param l Integer value to convert to prime field element
     *
     * Creates prime field element by reducing l modulo p.
     * Automatically handles negative values and values ≥ p.
     */
    constexpr Fp(int l);

    /**
     * @brief Copy constructor
     */
    constexpr Fp(const Fp& other) noexcept = default;

    /**
     * @brief Move constructor
     */
    constexpr Fp(Fp&& other) noexcept = default;

    /**
     * @brief Cross-field constructor for extracting (target) prime field elements from (source) extension fields
     * @tparam S Base field type of the source extension field
     * @tparam ext_modulus Modulus polynomial of the source extension field
     * @param other Source extension field element to extract from
     * @throws std::invalid_argument if extension field element is not in prime field
     *
     * This constructor extracts prime field elements from any extension field over the same
     * characteristic using the @ref details::largest_common_subfield_t algorithm. Since prime
     * fields are minimal, the largest common subfield is always the prime field itself.
     *
     * The algorithm uses cached subfield embeddings for mathematically correct extraction:
     * 1. **Same characteristic check**: Verifies source and target have same characteristic
     * 2. **Embedding lookup**: Uses cached embedding map Fp&lt;p&gt; → Ext&lt;S, ext_modulus&gt;
     * 3. **Reverse search**: Finds extension field element in embedding (throws if not found)
     * 4. **Index extraction**: Returns prime field element corresponding to found position
     *
     * This approach enables extractions from any extension field path over the same prime field.
     *
     * @section Usage_Examples
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4 = Ext<F2, {1, 1, 1}>;   // F2 → F4
     * using F16 = Ext<F4, {2, 2, 1}>;  // F2 → F4 → F16
     *
     * // Successful cases
     * F2 a(1);
     * F4 b(a);
     * F16 c(b);
     * auto d = F2(b);
     * auto e = F2(c);
     *
     * // Error cases
     * F4 f(2);        // F4 element not in F2
     * auto g = F2(f); // throws
     * @endcode
     *
     * @note This constructor uses the same algorithm as @ref Ext and @ref Iso cross-field
     * constructors, but simplified since prime fields are always the largest common subfield
     * @see @ref details::largest_common_subfield_t
     * @see Enhanced cross-field constructors in @ref Ext and @ref Iso
     */
    template <FiniteFieldType S, MOD ext_modulus, LutMode mode>
    Fp(const Ext<S, ext_modulus, mode>& other);

    /**
     * @brief Construct prime field element from Iso (stack of isomorphic fields)
     * @tparam MAIN Main field type of the stack of isomorphic fields
     * @tparam OTHERS Alternative field representations in the stack of isomorphic fields
     * @param other Iso object to convert from
     * @throws std::invalid_argument if element is not in prime subfield
     *
     * Extracts prime field element from any component of the Iso that contains this prime field.
     */
    template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
    Fp(const Iso<MAIN, OTHERS...>& other)
        requires SubfieldOf<Iso<MAIN, OTHERS...>, Fp<p>>;

    ~Fp() noexcept = default;

    /* assignment operators */

    /**
     * @brief Assign integer value to prime field element
     * @param l Integer value to assign
     * @return Reference to this element after assignment
     *
     * Assigns the value l (mod p) to this prime field element.
     * Automatically reduces l modulo p.
     */
    constexpr Fp& operator=(int l) noexcept;

    /**
     * @brief Copy assignment operator
     */
    constexpr Fp& operator=(const Fp& rhs) noexcept = default;

    /**
     * @brief Move assignment operator
     */
    constexpr Fp& operator=(Fp&& rhs) noexcept = default;

    /**
     * @brief Assign prime field element by projection from superfield
     * @tparam S Base field type of the superfield
     * @tparam ext_modulus Irreducible modulus polynomial defining the superfield
     * @param other Superfield element to project to prime field
     * @return Reference to this prime field element
     * @throws std::invalid_argument if the two fields are not in the same tower of fields (tower extension)
     *
     * Performs field projection/restriction assignment from an extension field
     * Ext<S, ext_modulus> to the prime field 𝔽_p. This operator assigns the prime
     * field component extracted from the extension field element.
     *
     * The operation is identical to the corresponding constructor, handling:
     * - **Direct extension**: When S ≡ 𝔽_p, extracts element from 𝔽_p → Ext<𝔽_p, modulus>
     *   by verifying the element lies in the prime subfield
     * - **Tower extension**: When S ≢ 𝔽_p, handles field towers like
     *   𝔽_p → Ext<𝔽_p, mod₁> → Ext<Ext<𝔽_p, mod₁>, mod₂> by converting to vector
     *   representation over Fp and extracting the constant term
     *
     * @note Implementation uses copy-and-swap idiom for exception safety,
     *       ensuring the object remains unchanged if projection fails
     *
     * @section Usage_Example
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4 = Ext<F2, {1, 1, 1}>;
     * using F16 = Ext<F4, {2, 1, 1}>;
     *
     * // Successful cases
     * F16 a = F4(2);
     * F16 b = 1;
     * F4 c = a;  // OK, b generated from F4 element
     * F4 d = b;  // OK, 1 is in every field of the tower (even F2)
     *
     * // Error cases
     * F16 e = 13;
     * F4 g = e;  // Not OK (throws), e is not in F4...
     * F2 h(e);   // ... and certainly not in F2 (throws as well)
     * @endcode
     */
    template <FiniteFieldType S, MOD ext_modulus, LutMode mode>
    Fp& operator=(const Ext<S, ext_modulus, mode>& other);

    /**
     * @brief Cross-field assignment from Iso type
     *
     * Performs field conversion assignment from an Iso field with the same characteristic.
     * Uses copy-and-swap idiom leveraging enhanced cross-field constructors for exception safety.
     *
     * @tparam MAIN Main field type of the Iso (must have same characteristic as this Fp)
     * @tparam OTHERS Additional isomorphic field types in the Iso group
     * @param other Source Iso element to assign from
     * @return Reference to this Fp element after assignment
     * @throws std::invalid_argument if conversion not possible
     */
    template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
        requires(MAIN::get_characteristic() == p)
    Fp& operator=(const Iso<MAIN, OTHERS...>& other);

    /* comparison */
    constexpr bool operator==(const Fp& rhs) const noexcept { return label == rhs.get_label(); }

    /* operations */

    /**
     * @brief Additive inverse for lvalue references
     * @return Additive inverse of this element (-this mod p)
     *
     * Returns the additive inverse such that this + (-this) ≡ 0 (mod p).
     * Creates a new element with the negated value.
     */
    constexpr Fp operator-() const& noexcept;

    /**
     * @brief Additive inverse for rvalue references
     * @return Reference to this element after in-place negation
     *
     * Optimized version for temporary objects: negates value in-place
     * and returns reference to avoid unnecessary copy.
     */
    constexpr Fp& operator-() && noexcept;

    /* operational assignments */

    /**
     * @brief Add prime field element to this element
     * @param rhs Right-hand side element to add
     * @return Reference to this element after addition
     *
     * Performs in-place addition: this = (this + rhs) mod p
     */
    constexpr Fp& operator+=(const Fp& rhs) noexcept;

    /**
     * @brief Subtract prime field element from this element
     * @param rhs Right-hand side element to subtract
     * @return Reference to this element after subtraction
     *
     * Performs in-place subtraction: this = (this - rhs) mod p
     */
    constexpr Fp& operator-=(const Fp& rhs) noexcept;

    /**
     * @brief Multiply this element by another prime field element
     * @param rhs Right-hand side element to multiply by
     * @return Reference to this element after multiplication
     *
     * Performs in-place multiplication: this = (this * rhs) mod p
     */
    constexpr Fp& operator*=(const Fp& rhs) noexcept;

    /**
     * @brief Multiply this element by a scalar
     * @param s Scalar value to multiply by
     * @return Reference to this element after multiplication
     *
     * Performs in-place scalar multiplication: this = (this * s) mod p
     */
    constexpr Fp& operator*=(int s) noexcept;

    /**
     * @brief Divide this element by another prime field element
     * @param rhs Right-hand side element to divide by
     * @return Reference to this element after division
     * @throws std::invalid_argument if rhs is zero (division by zero)
     *
     * Performs in-place division: this = (this * rhs^(-1)) mod p
     * Uses modular multiplicative inverse of rhs.
     */
    Fp& operator/=(const Fp& rhs);

    /* randomization */

    /**
     * @brief Set this element to a random value
     * @return Reference to this element after randomization
     *
     * Generates a random element from 𝔽_p by setting a uniformly
     * random value in the range [0, p-1].
     */
    Fp& randomize() noexcept;

    /**
     * @brief Set this element to a random value different from current
     * @return Reference to this element after randomization
     *
     * Generates a random element from 𝔽_p that is guaranteed to be
     * different from the current value. Useful for testing algorithms.
     */
    Fp& randomize_force_change() noexcept;

    /* getters */

    /**
     * @brief Get multiplicative order of this field element
     * @return Smallest positive integer k such that this^k = 1
     * @throws std::invalid_argument if this element is zero
     *
     * Computes the multiplicative order in the group 𝔽_p*.
     * For the zero element, multiplicative order is undefined.
     */
    size_t get_multiplicative_order() const;

    /**
     * @brief Get additive order of this field element
     * @return Characteristic p (order of additive group)
     *
     * In prime fields, every non-zero element has additive order p,
     * and zero has additive order 1.
     */
    size_t get_additive_order() const;

    /**
     * @brief Get human-readable information about this prime field
     * @return String describing the field
     *
     * Returns formatted string with field size.
     */
    static const std::string& get_info() noexcept {
        static const std::string info = "prime field with " + std::to_string(p) + " elements";
        return info;
    }

    static constexpr size_t get_characteristic() noexcept { return p; }
    constexpr size_t get_label() const noexcept { return label; }

    static constexpr Fp get_generator() noexcept { return p == 2 ? Fp{1} : Fp{2}; }
    static constexpr size_t get_p() noexcept { return p; }
    static constexpr size_t get_m() noexcept { return 1; }
    static constexpr size_t get_q() noexcept { return p; }
    static constexpr size_t get_size() noexcept { return p; }

    /**
     * @brief Check if this field interface is constexpr-ready for compile-time usage
     * @return Always true for prime fields
     *
     * Prime fields always provide a constexpr-compatible interface to extension fields,
     * regardless of internal implementation details. Whether Fp uses direct modular arithmetic
     * (default) or actual lookup tables (USE_LUTS_FOR_FP), the interface functions like
     * lut_add(), lut_mul(), lut_neg() are always constexpr and can be used by CompileTime
     * extension fields during their LUT generation.
     *
     * @see LutMode for extension field LUT generation modes
     * @see Ext::is_constexpr_ready() for extension field constexpr-readiness checking
     */
    static constexpr bool is_constexpr_ready() noexcept { return true; }

    /**
     * @brief Display lookup tables for debugging
     *
     * Prints the internal lookup tables (addition, multiplication, etc.)
     * to standard output for debugging and verification purposes.
     */
    static void show_tables() noexcept;

    /**
     * @brief Check if element has positive sign
     * @return Always true (finite fields have no natural ordering)
     */
    constexpr bool has_positive_sign() const noexcept { return true; }

    /**
     * @brief Check if this element is zero
     * @return true if this is the additive identity (0)
     */
    constexpr bool is_zero() const noexcept { return label == 0; }

    /**
     * @brief Erases this element, i.e., sets it to an "outside of field" marker
     * @return Reference to this element after erasing
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     */
    constexpr Fp& erase();

    /**
     * @brief Un-erases this element, i.e., sets it to an actual field element (the additive neutral 0)
     * @return Reference to this element after un-erasing
     */
    constexpr Fp& unerase();

    /**
     * @brief Checks whether this element is erased
     * @return true if this element is erased, false otherwise (meaning it actually is a field element)
     */
    constexpr bool is_erased() const noexcept { return label == std::numeric_limits<label_t>::max(); }

    /**
     * @brief Compile-time synchronization point for staged template instantiation
     * @return true when all LUTs for this field are computed
     *
     * This function serves as a compile-time synchronization mechanism for staged template
     * instantiation in field towers. It returns true when all lookup tables (LUTs) have been computed and are ready for
     * use.
     *
     * @section Purpose
     *
     * The ready() function is designed to prevent compiler recursion depth issues that
     * can occur when constructing complex field towers. By forcing completion of base
     * field LUT calculations before extension fields begin their own computations, it
     * ensures a controlled, staged instantiation process.
     *
     * @section Usage_in_Field_Towers
     *
     * Extension fields should check their base field's readiness before starting their
     * own LUT computations:
     *
     * @see luts_ready
     * @see Ext::ready()
     */
    static constexpr bool ready() {
#ifdef USE_LUTS_FOR_FP
        return luts_ready;
#else
        return true;  // Always ready when not using LUTs
#endif
    }

    // LUT-compatible interface for (default) non-LUT mode. This allows Fp to be used as base field B in Ext without
    // enabling USE_LUTS_FOR_FP. These functions delegate to the optimized operator implementations to avoid
    // duplication.
#ifndef USE_LUTS_FOR_FP
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

   private:
    label_t label;  ///< Element value in {0, 1, ..., p-1}

#ifdef USE_LUTS_FOR_FP
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
#ifdef USE_PRECOMPILED_LUTS
    static const Lut2D lut_add;
#else
    static constexpr Lut2D lut_add = details::compute_modular_addition_table<label_t, p>();
#endif

    /// @brief Multiplication lookup table: lut_mul(a,b) = (a * b) mod p
#ifdef USE_PRECOMPILED_LUTS
    static const Lut2D lut_mul;
#else
    static constexpr Lut2D lut_mul = details::compute_modular_multiplication_table<label_t, p>();
#endif

    /// @brief Additive inverse lookup table: lut_neg[a] = (-a) mod p
#ifdef USE_PRECOMPILED_LUTS
    static const Lut1D lut_neg;
#else
    static constexpr Lut1D lut_neg = details::compute_additive_inverses_direct<label_t, p>();
#endif

    /// @brief Multiplicative inverse lookup table: lut_inv[a] = a^(-1) mod p
#ifdef USE_PRECOMPILED_LUTS
    static const Lut1D lut_inv;
#else
    static constexpr Lut1D lut_inv = details::compute_multiplicative_inverses_direct<label_t, p>();
#endif

    /// @brief Multiplicative order lookup table: lut_mul_ord[a] = multiplicative order of a in 𝔽_p\c\{0}
#ifdef USE_PRECOMPILED_LUTS
    static const Lut1D lut_mul_ord;
#else
    static constexpr Lut1D lut_mul_ord = details::compute_multiplicative_orders<label_t, p>(lut_mul);
#endif

    static constexpr bool luts_ready = []() constexpr {
#ifndef USE_PRECOMPILED_LUTS
        static_assert(lut_add(0, 0) == 0);  // Forces immediate calculation of lut_add
        static_assert(lut_neg(0) == 0);     // Forces immediate calculation lut_neg
        static_assert(lut_mul(0, 1) == 0);  // Forces immediate calculation lut_mul
#endif
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
            throw std::invalid_argument("No conversion path found from Iso to prime field");
        }
    }
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator=(int l) noexcept {
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
    if (this->is_erased()) return Fp().erase();
    Fp res(*this);
    if (res.label != 0) {
#ifndef USE_LUTS_FOR_FP
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
    if (this->is_erased()) return this->erase();
    if (label != 0) {
#ifndef USE_LUTS_FOR_FP
        label = -(int)label + (int)p;
#else
        label = lut_neg(label);
#endif
    }
    return *this;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator+=(const Fp& rhs) noexcept {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#ifndef USE_LUTS_FOR_FP
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
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#ifndef USE_LUTS_FOR_FP
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
    if (this->is_erased() || rhs.is_erased()) return this->erase();
#ifndef USE_LUTS_FOR_FP
    int temp = label * rhs.get_label();
    if (temp < p)
        label = temp;
    else
        label = temp % p;
#else
    label = lut_mul(label, rhs.get_label());
#endif
    return *this;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::operator*=(int s) noexcept {
    if (this->is_erased()) return *this;
    if (get_characteristic() != 0) s %= static_cast<int>(get_characteristic());
    Fp res = daa<Fp>(*this, s);
    *this = std::move(res);
    return *this;
}

template <uint16_t p>
Fp<p>& Fp<p>::operator/=(const Fp& rhs) {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    if (rhs.label == 0) throw std::invalid_argument("trying to divide by zero");
#ifndef USE_LUTS_FOR_FP
    *this *= Fp(modinv<p, int>(rhs.get_label()));
#else
    label = lut_mul(label, lut_inv(rhs.get_label()));
#endif
    return *this;
}

template <uint16_t p>
Fp<p>& Fp<p>::randomize() noexcept {
    this->unerase();
    static std::uniform_int_distribution<int> dist(0, p - 1);
#ifndef USE_LUTS_FOR_FP
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
Fp<p>& Fp<p>::randomize_force_change() noexcept {
    this->unerase();
    static std::uniform_int_distribution<int> dist(1, p - 1);
#ifndef USE_LUTS_FOR_FP
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
    if (is_erased()) throw std::invalid_argument("trying to calculate multiplicative order of erased element");
    if (label == 0) throw std::invalid_argument("trying to calculate multiplicative order of additive neutral element");
#ifndef USE_LUTS_FOR_FP
    return details::calculate_multiplicative_order(*this);
#else
    return lut_mul_ord(label);
#endif
}

template <uint16_t p>
size_t Fp<p>::get_additive_order() const {
    if (is_erased()) throw std::invalid_argument("trying to calculate additive order of erased element");
    if (label == 0) return 1;
    return p;
}

template <uint16_t p>
void Fp<p>::show_tables() noexcept {
    std::cout << "addition table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < p; ++i) {
        for (label_t j = 0; j < p; ++j) std::cout << (int)lut_add(i, j) << ", ";
        std::cout << std::endl;
    }

#ifdef USE_LUTS_FOR_FP
    std::cout << "additive inverse table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < p; ++i) std::cout << (int)lut_neg(i) << std::endl;
#endif

    std::cout << "multiplication table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < p; ++i) {
        for (label_t j = 0; j < p; ++j) std::cout << (int)lut_mul(i, j) << ", ";
        std::cout << std::endl;
    }

#ifdef USE_LUTS_FOR_FP
    std::cout << "multiplicative inverse table (row and column headers "
                 "omitted)"
              << std::endl;
    for (label_t i = 0; i < p; ++i) std::cout << (int)lut_inv(i) << std::endl;
#endif
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::erase() {
    label = std::numeric_limits<label_t>::max();
    return *this;
}

template <uint16_t p>
constexpr Fp<p>& Fp<p>::unerase() {
    if (is_erased()) (*this) = 0;
    return *this;
}

/**
 * @brief Output stream operator for prime field elements
 * @tparam p Prime modulus
 * @param os Output stream
 * @param e Prime field element to output
 * @return Reference to the output stream
 *
 * Outputs the integer label of the field element.
 */
template <uint16_t p>
std::ostream& operator<<(std::ostream& os, const Fp<p>& e) noexcept {
    if (e.is_erased())
        os << ERASURE_MARKER;
    else
        os << (int)e.get_label();
    return os;
}

/**
 * @class Ext
 * @brief Extension field 𝔽_{q^m} constructed via irreducible polynomial
 *
 * @tparam B Base field type (must satisfy FiniteFieldType concept, either Fp, Ext or Iso)
 * @tparam modulus Irreducible monic polynomial (of degree at least two) coefficients as initializer list, constant
 * coefficient first
 * @tparam mode Lookup table generation mode (LutMode::Runtime or LutMode::CompileTime, defaults to Runtime)
 *
 * Constructs finite extension fields of the form 𝔽_{q^m} ≅ B[x]/(f(s)) where:
 * - B is the base field with q = |B| elements (q either a prime p or a prime power p^m)
 * - The resulting field has Q = q^m elements
 *
 * @section Irreducible_Polynomial_Format
 *
 * The modulus template parameter specifies f(x) = a_0 + a_1 x + ... + a_m x^m:
 * - **Coefficients**: Listed from a_0 to a_m (constant coefficient first)
 * - **Degree**: deg(f(x)) = m >= 2
 * - **Monic Requirement**: a_m = 1 (leading coefficient must be 1)
 * - **Irreducibility**: Must be irreducible over the base field B
 *
 * @warning The modulus polynomial must be monic (leading coefficient = 1) and irreducible over the base field.
 * Non-irreducible polynomials will cause errors during table generation with a message indicating that a
 * reducible polynomial was chosen. Irreducible polynomials can be found using @ref find_irreducible().
 *
 * @section Implementation_Details
 *
 * - **Label Storage**: Elements stored as integers of type label_t<p> (automatically chosen based on field size)
 * - **Type Safety**: Template parameter ensures compile-time checking that B is in fact a finite field (prime or
 * extended)
 * - **Vector Interface**: Seamless conversion to/from vector representations (vectors over subfields)
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F3 = Fp<3>;
 * using F9 = Ext<F3, {2, 2, 1}>;
 *
 * F9 a(5), b(7);                                // Elements with labels 5 and 7
 * F9 c = a * b + F9(1);                         // Field arithmetic
 * Vector<F3> vec = a.as_vector<F3>();           // Polynomial coefficients
 * size_t order = a.get_multiplicative_order();  // Element order
 *
 * // Tower construction
 * using F27 = Ext<F9, {1, 2, 1}>;            // F9[y]/(1 + 2y + y²)
 * F27 x(100);                                // Element in 3-level tower
 * Vector<F9> base_vec = x.as_vector<F9>();   // Extract to vector over intermediate extension field
 * Vector<F3> prime_vec = x.as_vector<F3>();  // Extract to vector over prime field
 * @endcode
 *
 * @section Field_Towers_and_LUT_modes
 *
 * This implementation supports field towers through recursive construction with configurable LUT modes:
 *
 * **Runtime Mode**: LUTs are computed on first access. This is the default.
 *
 * **CompileTime Mode**: LUTs are computed during compilation and embedded in the executable.
 * No initialization required - field operations are immediately available.
 *
 * @warning Field constructions with CompileTime LUT calculation are only possible from base fields with CompileTime LUT
 * calculation.
 *
 * @code{.cpp}
 * // Tower: F2 ⊂ F4 ⊂ F16 ⊂ F256 with mixed LUT modes
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}, LutMode::CompileTime>;  // F2[X]/(X² + X + 1) - CompileTime
 * using F16 = Ext<F4, {2, 1, 1}, LutMode::RunTime>;     // F4[Y]/(Y² + Y + 2) - RunTime (explicit)
 * using F256 = Ext<F16, {2, 5, 1}>;                     // F16[Z]/(Z² + 5Z + 2) - Runtime (default)
 *
 * @section Performance considerations for field towers:
 * - Small fields (F2, F4, F8, F16, F64, F127): CompileTime mode recommended
 * - Large fields (F256, F512, ...): Runtime mode for faster compilation
 * - Mixed modes work seamlessly together in the same tower
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
    using label_t = ::ECC::label_t<Q>;

   public:
    /* constructors */

    /**
     * @brief Default constructor - creates zero element
     *
     * Initializes this extension field element to 0 (additive identity).
     */
    constexpr Ext() noexcept : label{0} {}

    /**
     * @brief Construct extension field element from integer label
     * @param l Integer label to convert to extension field element
     * @throws std::invalid_argument l is not in {0, ..., Q-1}
     *
     * Creates extension field element, throws in case of l not in {0, ..., Q-1}.
     */
    Ext(int l);

    /**
     * @brief Copy constructor
     */
    constexpr Ext(const Ext& other) noexcept = default;

    /**
     * @brief Move constructor
     */
    constexpr Ext(Ext&& other) noexcept = default;

    /**
     * @brief Construct extension field element from base field element
     * @param other Base field element to embed in extension field
     *
     * Embeds base field element into extension field as constant polynomial.
     * This creates the natural field embedding B → Ext<B, modulus>.
     */
    Ext(const B& other) noexcept;

    /**
     * @brief Cross-field constructor for compatible extension fields
     * @tparam S Base field type of the source extension field
     * @tparam ext_modulus Modulus polynomial of the source extension field
     * @param other Extension field element to convert
     * @throws std::invalid_argument if conversion is not valid
     *
     * This constructor handles all cross-field conversions between extension fields:
     * - **Isomorphic fields**: Direct conversion using cached isomorphisms
     * - **Subfield embedding (upcast)**: Source ⊆ Target via cached embedding maps (never throws)
     * - **Superfield extraction (downcast)**: Target ⊆ Source via reverse lookup (may throw)
     * - **Different towers**: Conversion via @ref details::largest_common_subfield_t
     *
     * The algorithm automatically selects the most appropriate conversion method:
     * 1. **Same field**: Handled by copy constructor
     * 2. **Isomorphic**: Uses deterministic cached isomorphism
     * 3. **Tower relationship**: Direct embedding/extraction
     * 4. **Cross-tower**: Two-step conversion via largest common subfield
     *
     * @section Usage_Examples
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4 = Ext<F2, {1, 1, 1}>;
     * using F16_a = Ext<F2, {1, 0, 0, 1, 1}>;
     * using F16_b = Ext<F4, {2, 2, 1}>;
     *
     * F2 a(1);
     * F4 b(a);     // Upcast: F2 → F4 (always succeeds)
     * F16_a c(b);  // Cross-tower: F4 → F16_a via F2 (largest common subfield)
     * F16_b d(c);  // Cross-tower: F16_a → F16_b via F2
     * F4 e(d);     // Downcast: F16_b → F4 (may throw if d not in F4)
     * @endcode
     */
    template <FiniteFieldType S, MOD ext_modulus, LutMode ext_mode>
        requires(!std::is_same_v<Ext<B, modulus, mode>, Ext<S, ext_modulus, ext_mode>>)
    Ext(const Ext<S, ext_modulus, ext_mode>& other);

    /**
     * @brief Construct extension field element from coefficient vector over subfield
     * @tparam T Coefficient field type
     * @param v Vector of coefficients over the subfield
     * @throws std::invalid_argument if vector dimension is incompatible
     *
     * Vector v represents polynomial starting with constant coefficient.
     *
     * @note If at least one of the components of v has the erasure flag then the resulting element of Ext will have the
     * erasure flag as well.
     */
    template <FiniteFieldType T>
    Ext(const Vector<T>& v);

    /**
     * @brief Cross-field constructor from Iso using largest common subfield
     * @tparam MAIN Main field type of the Iso stack
     * @tparam OTHERS Alternative field representations in the Iso stack
     * @param iso Iso object to convert from
     * @throws std::invalid_argument if conversion is not valid
     *
     * This constructor handles all cross-field conversions from Iso to Ext using the
     * @ref details::largest_common_subfield_t algorithm.
     *
     * **Simple Delegation Approach:**
     *
     * **Delegation Process:**
     * 1. **Extract MAIN**: Get `iso.main()` from the source Iso
     * 2. **Delegate**: Use `Ext(iso.main())` constructor
     * 3. **Ext constructor logic decides**: The Ext constructor determines optimal conversion path
     *
     * @section Usage_Examples
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4_a = Ext<F2, {1, 1, 1}>;        // F2 → F4 (path 1)
     * using F4_b = Ext<F2, {1, 0, 1}>;        // F2 → F4 (path 2)
     * using F16 = Ext<F4_a, {2, 2, 1}>;       // F2 → F4_a → F16
     * using F8 = Ext<F2, {1, 1, 0, 1}>;       // F2 → F8 (different branch)
     * using F4_iso = Iso<F4_a, F4_b>;
     *
     * F4_iso iso_elem(F2(1));
     * F4_a result_a(iso_elem);     // Direct: MAIN matches Ext
     * F4_b result_b(iso_elem);     // Isomorphic: F4_a → F4_b via enhanced logic
     * F16 result_super(iso_elem);  // Upcast: F4_a embedded in F16 via enhanced logic
     * F8 result_cross(iso_elem);   // Cross-field: F4_a → F8 via F2 common subfield
     * @endcode
     *
     * @see @ref details::largest_common_subfield_t
     * @see Cross-field constructors in @ref Ext and @ref Iso
     */
    template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
    Ext(const Iso<MAIN, OTHERS...>& other);

    /* assignment operators */

    /**
     * @brief Assign integer value to extension field element
     * @param l Integer label to convert to extension field element
     * @return Reference to this element after assignment
     * @throws std::invalid_argument l is not in {0, ..., Q-1}
     *
     * Creates extension field element, throws if l is not in {0, ..., Q-1}
     */
    constexpr Ext& operator=(int l);

    /**
     * @brief Copy assignment operator
     */
    constexpr Ext& operator=(const Ext& rhs) noexcept = default;

    /**
     * @brief Move assignment operator
     */
    Ext& operator=(Ext&& rhs) noexcept = default;

    /**
     * @brief Assign extension field element from other extension field
     * @tparam S Base field type of the source extension field
     * @tparam ext_modulus Modulus polynomial of the source extension field
     * @param other Extension field element to assign
     * @return Reference to this element after assignment
     * @throws std::invalid_argument if conversion is not valid
     *
     * Performs field conversion assignment between compatible extension fields.
     * Uses copy-and-swap idiom for exception safety, supports both subfield extraction and field tower operations.
     */
    template <FiniteFieldType S, MOD ext_modulus, LutMode ext_mode>
        requires(S::get_characteristic() == B::get_characteristic())
    Ext& operator=(const Ext<S, ext_modulus, ext_mode>& other);

    /**
     * @brief Cross-field assignment from Fp type
     *
     * Performs field conversion assignment from an Fp field with the same characteristic.
     * Uses copy-and-swap idiom leveraging enhanced cross-field constructors for exception safety.
     *
     * @tparam p Prime of the source Fp field (must match base field characteristic)
     * @param other Source Fp element to assign from
     * @return Reference to this Ext element after assignment
     * @throws std::invalid_argument if conversion not possible
     */
    template <uint16_t p>
        requires(p == B::get_characteristic())
    Ext& operator=(const Fp<p>& other);

    /**
     * @brief Cross-field assignment from Iso type
     *
     * Performs field conversion assignment from an Iso field with the same characteristic.
     * Uses copy-and-swap idiom leveraging enhanced cross-field constructors for exception safety.
     *
     * @tparam MAIN Main field type of the Iso (must have same characteristic as this Ext)
     * @tparam OTHERS Additional isomorphic field types in the Iso group
     * @param other Source Iso element to assign from
     * @return Reference to this Ext element after assignment
     * @throws std::invalid_argument if conversion not possible
     */
    template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
        requires(MAIN::get_characteristic() == B::get_characteristic())
    Ext& operator=(const Iso<MAIN, OTHERS...>& other);

    /* comparison */
    constexpr bool operator==(const Ext& rhs) const noexcept { return label == rhs.get_label(); }

    /* operations */

    /**
     * @brief Additive inverse for lvalue references
     * @return Additive inverse of this element (-this mod modulus)
     *
     * Returns the additive inverse such that this + (-this) ≡ 0 (mod modulus).
     * Creates a new element with the negated value.
     */
    constexpr Ext operator-() const& noexcept;

    /**
     * @brief Additive inverse for rvalue references
     * @return Reference to this element after in-place negation
     *
     * Optimized version for temporary objects: negates value in-place
     * and returns reference to avoid unnecessary copy.
     */
    constexpr Ext& operator-() && noexcept;

    /* operational assignments */

    /**
     * @brief Add extension field element to this element
     * @param rhs Right-hand side element to add
     * @return Reference to this element after addition
     *
     * Performs in-place addition: this = (this + rhs) mod modulus
     * Uses efficient lookup table operations when available.
     */
    constexpr Ext& operator+=(const Ext& rhs) noexcept;

    /**
     * @brief Subtract extension field element from this element
     * @param rhs Right-hand side element to subtract
     * @return Reference to this element after subtraction
     *
     * Performs in-place subtraction: this = (this - rhs) mod modulus
     * Uses efficient lookup table operations when available.
     */
    constexpr Ext& operator-=(const Ext& rhs) noexcept;

    /**
     * @brief Multiply this element by another extension field element
     * @param rhs Right-hand side element to multiply by
     * @return Reference to this element after multiplication
     *
     * Performs in-place multiplication: this = (this * rhs) mod modulus
     * Uses efficient lookup table operations when available.
     */
    constexpr Ext& operator*=(const Ext& rhs) noexcept;

    /**
     * @brief Multiply this element by an integer scalar
     * @param s Integer scalar to multiply with
     * @return Reference to this element after scalar multiplication
     *
     * Performs in-place scalar multiplication: this = this * s
     * Uses repeated addition in the extension field.
     */
    constexpr Ext& operator*=(int s) noexcept;

    /**
     * @brief Divide this element by another extension field element
     * @param rhs Right-hand side element to divide by
     * @return Reference to this element after division
     * @throws std::invalid_argument if rhs is zero (division by zero)
     *
     * Performs in-place division: this = (this * rhs^(-1)) mod modulus
     * Uses multiplicative inverse of rhs in the extension field.
     */
    Ext& operator/=(const Ext& rhs);

    /* randomization */

    /**
     * @brief Set this element to a random value
     * @return Reference to this element after randomization
     *
     * Generates a random element from the extension field by setting
     * a uniformly random value in the range [0, Q-1].
     */
    Ext& randomize() noexcept;

    /**
     * @brief Set this element to a random value different from current
     * @return Reference to this element after randomization
     *
     * Generates a random element from the extension field that is guaranteed
     * to be different from the current value. Useful for testing algorithms.
     */
    Ext& randomize_force_change() noexcept;

    /* getters */

    /**
     * @brief Get multiplicative order of this field element
     * @return Smallest positive integer k such that this^k = 1
     * @throws std::invalid_argument if this element is zero
     */
    size_t get_multiplicative_order() const;

    /**
     * @brief Get additive order of this field element
     * @return Either one or characteristic p (order of additive group)
     *
     * In extension fields, every non-zero element has additive order p (the characteristic),
     * and zero has additive order one.
     */
    size_t get_additive_order() const;

    /**
     * @brief Get minimal polynomial of this field element over arbitrary subfield
     * @tparam S Subfield type (must be subfield of this extension field)
     * @return Minimal polynomial over the specified subfield S
     *
     * Computes the minimal polynomial of this element with coefficients from subfield S.
     * This is particularly useful for:
     * - Computing absolute minimal polynomials over the prime field
     * - Finding polynomials over intermediate subfields in field towers
     * - Constructing field isomorphisms between different representations
     *
     * The algorithm computes the S-conjugacy orbit {α, α^|S|, α^|S|², ...} and forms
     * the polynomial having these conjugates as roots.
     *
     * @section Usage_Example
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4 = Ext<F2, MOD{1, 1, 1}>;
     * using F16 = Ext<F4, MOD{2, 1, 1}>;
     *
     * F16 alpha = F16::get_generator();
     *
     * auto poly_F4 = alpha.get_minimal_polynomial<F4>();   // Over F4
     * auto poly_F2 = alpha.get_minimal_polynomial<F2>();   // Over F2 (absolute)
     * auto poly_base = alpha.get_minimal_polynomial();     // Over F4 (default)
     * @endcode
     */
    template <FiniteFieldType S>
    Polynomial<S> get_minimal_polynomial() const
        requires SubfieldOf<Ext<B, modulus, mode>, S>;

    /**
     * @brief Get human-readable information about this extension field
     * @return String describing the field structure
     *
     * Returns formatted string with field size, base field, and modulus.
     */
    static std::string get_info() noexcept;

    static constexpr size_t get_characteristic() noexcept { return B::get_p(); }
    constexpr size_t get_label() const noexcept { return label; }

    /**
     * @brief Check if this field interface is constexpr-ready for compile-time usage
     * @return true if mode == LutMode::CompileTime, false if mode == LutMode::Runtime
     *
     * Extension fields using CompileTime mode compute all lookup tables at compile-time
     * via constexpr evaluation, making their interface constexpr-ready with zero
     * initialization overhead but potentially increasing compilation time and executable size.
     *
     * Extension fields using Runtime mode compute lookup tables on first access using
     * lazy initialization. Their LUT access functions return runtime data, making them
     * non-constexpr and ruling out constexpr-readiness.
     *
     * @note CompileTime extension fields require that all base fields also have
     *       constexpr-ready interfaces to ensure constexpr compatibility throughout the field tower.
     *
     * @see LutMode for detailed mode descriptions
     * @see static_assert constraint that enforces constexpr-readiness requirement for base fields
     */
    static constexpr bool is_constexpr_ready() noexcept { return mode == LutMode::CompileTime; }

    /**
     * @brief Get the irreducible modulus polynomial
     * @return Modulus polynomial defining this extension field
     *
     * Returns the monic irreducible polynomial used to construct
     * this extension field Ext<B, modulus> ≅ B[x]/(modulus).
     */
    static constexpr Polynomial<B> get_modulus() noexcept;

    static constexpr Ext get_generator() noexcept {
#ifdef USE_PRECOMPILED_LUTS
        return Ext(g.value);
#else
        return Ext(g().value);
#endif
    }
    static constexpr size_t get_p() noexcept { return B::get_p(); }
    static constexpr size_t get_m() noexcept { return m; }
    static constexpr size_t get_q() noexcept { return Q; }
    static constexpr size_t get_size() noexcept { return Q; }

    /**
     * @brief Get isomorphism to isomorphic field T
     * @return Isomorphism, mapping from this extension field to T
     */
    template <FiniteFieldType T>
    static Isomorphism<Ext, T> isomorphism_to();

    /**
     * @brief Display lookup tables for debugging
     *
     * Prints the internal lookup tables (addition, multiplication, etc.)
     * to standard output for debugging and verification purposes.
     */
    static void show_tables() noexcept;

    /**
     * @brief Check if element has positive sign
     * @return Always true (finite fields have no natural ordering)
     */
    constexpr bool has_positive_sign() const noexcept { return true; }

    /**
     * @brief Check if this element is zero
     * @return true if this is the additive identity
     */
    constexpr bool is_zero() const noexcept { return label == 0; }

    /**
     * @brief Erases this element, i.e., sets it to an "outside of field" marker
     * @return Reference to this element after erasing
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     */
    constexpr Ext& erase() noexcept;

    /**
     * @brief Un-erases this element, i.e., sets it to an actual field element (the additive neutral 0)
     * @return Reference to this element after un-erasing
     */
    constexpr Ext& unerase() noexcept;

    /**
     * @brief Checks whether this element is erased
     * @return true if this element is erased, false otherwise (meaning it actually is a field element)
     */
    constexpr bool is_erased() const noexcept { return label == std::numeric_limits<label_t>::max(); }

    /**
     * @brief Converts extension field element to vector representation over proper subfield
     * @tparam T Subfield type (must be a subfield in tower), default: base field B
     * @return Vector of coefficients of appropriate length representing this element over T
     *
     * Converts this extension field element to its vector representation
     * over the specified proper subfield T.
     *
     * For field tower T ⊂ B ⊂ Ext<B, modulus>, this method provides
     * the T-linear representation of elements. The vector length is
     * [Ext : T] = (extension degree of Ext over T).
     *
     * The resulting vector can be converted back into an element of the extension field using a constructor.
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4 = Ext<C, MOD{1, 1, 1}>;
     * using F16 = Ext<B, MOD{2, 1, 1}>;
     *
     * F16 x(10);
     * Vector<F4> v4 = x.as_vector<F4>();  // Length 2 (F16 = F4^2)
     * Vector<F2> v2 = x.as_vector<F2>();  // Length 4 (F16 = F2^4)
     * F16 y = F16(v2);                    // transform vector back to superfield element
     * @endcode
     */
    template <FiniteFieldType T = B>
        requires(SubfieldOf<Ext<B, modulus, mode>, T>) && (!std::is_same_v<Ext<B, modulus, mode>, T>)
    Vector<T> as_vector() const noexcept;

    /**
     * @brief Compile-time synchronization point for staged template instantiation
     * @return true when all LUTs for this field are computed
     *
     * This function serves as a compile-time synchronization mechanism for staged template
     * instantiation in field towers. It returns true when all lookup tables (LUTs) for
     * this extension field have been computed and are ready for use.
     *
     * @section Purpose
     *
     * The ready() function guarantees that further extension fields can safely begin their computations.
     *
     * This prevents compiler recursion depth issues that can occur when multiple
     * extension field layers attempt to compute their LUTs simultaneously.
     *
     * @section Usage_in_Field_Towers
     *
     * Higher-level extension fields should check this field's readiness before
     * starting their own LUT computations:
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4 = Ext<F2, MOD{1, 1, 1}>;
     * using F16 = Ext<F4, MOD{2, 1, 1}>;
     *
     * // F16 construction should check:
     * static_assert(F4::ready(), "F4 LUTs must be ready before F16 construction");
     * @endcode
     *
     * @section Implementation_Details
     *
     * The function returns the value of the `luts_ready` constexpr variable, which is
     * computed by a lambda that forces evaluation of all field LUTs through static_assert
     * statements. This ensures that:
     * - All LUTs are computed at compile time
     * - The computation order is deterministic
     * - Extension fields wait for base field completion
     * - Compiler recursion depth limits are respected
     *
     * @warning This function must only be called after all LUT declarations in the class.
     *          The base field B must be ready before this extension field's LUTs are computed.
     *
     * @see luts_ready
     * @see Fp::ready()
     * @see B::ready()
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
#ifdef USE_PRECOMPILED_LUTS
    static const Lut2Dcoeff& lut_coeff();
#else
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
#endif

    /// @brief Addition table: lut_add(a,b) = (polynomial_a + polynomial_b) mod f(X)
#ifdef USE_PRECOMPILED_LUTS
    static const Lut2D& lut_add();
#else
    static constexpr Lut2D compute_add_lut_wrapper(const Lut2Dcoeff& (*provider)()) {
        const Lut2Dcoeff& coeffs = provider();
        return details::compute_polynomial_addition_table<label_t, Q, Lut2Dcoeff, m, B>(coeffs);
    }
    using LUT_ADD = details::LutHolder<Lut2D, Lut2Dcoeff, &lut_coeff, &compute_add_lut_wrapper, mode>;
    static constexpr auto& lut_add() { return LUT_ADD::get_lut(); }
#endif

    /// @brief Multiplication table: lut_mul(a,b) = (polynomial_a * polynomial_b) mod f(X)
#ifdef USE_PRECOMPILED_LUTS
    static const Lut2D& lut_mul();
#else
    static constexpr Lut2D compute_mul_lut_wrapper(const Lut2Dcoeff& (*provider)()) {
        const Lut2Dcoeff& coeffs = provider();
        return details::compute_polynomial_multiplication_table<label_t, Q, Lut2Dcoeff, m, B, modulus>(coeffs);
    }
    using LUT_MUL = details::LutHolder<Lut2D, Lut2Dcoeff, &lut_coeff, &compute_mul_lut_wrapper, mode>;
    static constexpr auto& lut_mul() { return LUT_MUL::get_lut(); }
#endif

    /// @brief Additive inverse table: lut_neg[a] = -a
#ifdef USE_PRECOMPILED_LUTS
    static const Lut1D& lut_neg();
#else
    static constexpr Lut1D compute_neg_lut_wrapper(const Lut2D& (*provider)()) {
        const Lut2D& add = provider();
        return details::compute_additive_inverses_search<label_t, Q>(add);
    }
    using LUT_NEG = details::LutHolder<Lut1D, Lut2D, &lut_add, &compute_neg_lut_wrapper, mode>;
    static constexpr auto& lut_neg() { return LUT_NEG::get_lut(); }
#endif

    /// @brief Multiplicative inverse table: lut_inv[a] = a^(-1)
#ifdef USE_PRECOMPILED_LUTS
    static const Lut1D& lut_inv();
#else
    static constexpr Lut1D compute_inv_lut_wrapper(const Lut2D& (*provider)()) {
        const Lut2D& mul = provider();
        return details::compute_multiplicative_inverses_search<label_t, Q>(mul);
    }
    using LUT_INV = details::LutHolder<Lut1D, Lut2D, &lut_mul, &compute_inv_lut_wrapper, mode>;
    static constexpr auto& lut_inv() { return LUT_INV::get_lut(); }
#endif

    /// @brief Multiplicative order table: lut_mul_ord[a] = order of a in multiplicative group
#ifdef USE_PRECOMPILED_LUTS
    static const Lut1D& lut_mul_ord();
#else
    static constexpr Lut1D compute_mul_ord_lut_wrapper(const Lut2D& (*provider)()) {
        const Lut2D& mul = provider();
        return details::compute_multiplicative_orders<label_t, Q>(mul);
    }
    using LUT_MUL_ORD = details::LutHolder<Lut1D, Lut2D, &lut_mul, &compute_mul_ord_lut_wrapper, mode>;
    static constexpr auto& lut_mul_ord() { return LUT_MUL_ORD::get_lut(); }
#endif

    /// @brief Primitive element (generator) of the multiplicative group
#ifdef USE_PRECOMPILED_LUTS
    static const Gen& g();
#else
    static constexpr Gen compute_generator_wrapper(const Lut1D& (*provider)()) {
        const Lut1D& mul_ord = provider();
        return Gen{details::find_generator<label_t, Q>(mul_ord)};
    }
    using LUT_GEN = details::LutHolder<Gen, Lut1D, &lut_mul_ord, &compute_generator_wrapper, mode>;
    static constexpr auto& g() { return LUT_GEN::get_lut(); }
#endif

    // LUT-compatible interface for use as BaseFieldType
    // These allow Ext fields to be used as base fields for higher-order extensions
    static constexpr label_t lut_add(label_t a, label_t b) noexcept { return lut_add()(a, b); }
    static constexpr label_t lut_mul(label_t a, label_t b) noexcept { return lut_mul()(a, b); }
    static constexpr label_t lut_neg(label_t a) noexcept { return lut_neg()(a); }
    static constexpr label_t lut_inv(label_t a) noexcept { return lut_inv()(a); }

    static constexpr bool luts_ready = []() constexpr {
#ifndef USE_PRECOMPILED_LUTS
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
#endif
        return true;
    }();

    /** @} */
};

/* member functions for Ext */

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>::Ext(int l) {
    if (l < 0 || l >= Q) throw std::invalid_argument("l must be positve and no larger than Q-1");
    label = l;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>::Ext(const B& other) noexcept {
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

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType S, MOD ext_modulus, LutMode ext_mode>
    requires(!std::is_same_v<Ext<B, modulus, mode>, Ext<S, ext_modulus, ext_mode>>)
Ext<B, modulus, mode>::Ext(const Ext<S, ext_modulus, ext_mode>& other) {
    // Ensure same characteristic
    static_assert(Ext<B, modulus, mode>::get_characteristic() == Ext<S, ext_modulus, ext_mode>::get_characteristic(),
                  "trying to convert between fields with different characteristic");

    // Note: same-field case handled by simple copy constructor
    if constexpr (Isomorphic<Ext, Ext<S, ext_modulus, ext_mode>>) {
        // Isomorphic fields - use isomorphism for conversion
        auto iso = Isomorphism<Ext<S, ext_modulus, ext_mode>, Ext>();
        *this = iso(other);
    } else if constexpr (SubfieldOf<Ext<B, modulus, mode>, Ext<S, ext_modulus, ext_mode>>) {
        // Upcast: Source ⊆ Target (ExtensionOf<Source, Target>) - cannot throw
        // Use cached subfield embedding for mathematically correct embedding
        auto embedding = Embedding<Ext<S, ext_modulus, ext_mode>, Ext>();
        Ext result = embedding(other);
        label = result.get_label();
    } else if constexpr (SubfieldOf<Ext<S, ext_modulus, ext_mode>, Ext<B, modulus, mode>>) {
        // Downcast: Target ⊆ Source (SubfieldOf<Target, Source>) - may throw
        // Use cached subfield embedding to find if superfield element is in subfield
        auto embedding = Embedding<Ext, Ext<S, ext_modulus, ext_mode>>();
        Ext result = embedding.extract(other);
        label = result.get_label();
    } else {
        // Fields with same characteristic but not directly related - convert via largest common subfield in order to
        // maximize
        std::cout << "DEBUG: First field: " << Ext<B, modulus, mode>::get_info() << ")\n";
        std::cout << "DEBUG: Second field: " << Ext<S, ext_modulus, ext_mode>::get_info() << ")\n";
        using CommonField = details::largest_common_subfield_t<Ext<B, modulus, mode>, Ext<S, ext_modulus, ext_mode>>;
        std::cout << "DEBUG: Using common field: " << CommonField::get_info() << " (size " << CommonField::get_size()
                  << ")\n";
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

        bool erased =
            std::ranges::any_of(std::views::iota(size_t{0}, v.get_n()), [&](size_t i) { return v[i].is_erased(); });

        if (erased) {
            this->erase();
        } else {
            label = 0;
            for (uint8_t i = 0; i < m; ++i) *this += Ext(v[i]) * Ext(sqm(B::get_size(), i));  // monomial basis
        }
    } else {
        static_assert(SubfieldOf<Ext<B, modulus, mode>, T>,
                      "extension field elements can only be constructed from vectors over subfields");

        if (v.get_n() != get_m() * B::get_m())
            throw std::invalid_argument(
                "trying to construct extension field element using subfield vector of wrong length");

        bool erased =
            std::ranges::any_of(std::views::iota(size_t{0}, v.get_n()), [&](size_t i) { return v[i].is_erased(); });

        if (erased) {
            this->erase();
        } else {
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

/**
 * @brief Cross-field constructor from Iso field using enhanced four-branch conversion logic
 * @tparam MAIN Main field type of the source Iso
 * @tparam OTHERS Alternative representation types in the source Iso
 * @param iso Source Iso field element to convert
 *
 * Advanced constructor that handles conversion from any Iso field representation using optimal
 * conversion paths. Employs a four-branch decision tree to determine the most efficient conversion:
 *
 * **Branch 1 (Isomorphic)**: Direct isomorphism if Ext ≅ MAIN or Ext ≅ any OTHERS
 * **Branch 2 (Upcast)**: Cached embedding if Iso/MAIN/OTHERS ⊆ Ext
 * **Branch 3 (Downcast)**: Reverse embedding with validation if Ext ⊆ Iso/MAIN/OTHERS
 * **Branch 4 (Cross-cast)**: Bridge through @ref details::largest_common_subfield_t when no direct relationship exists
 *
 * @throws std::invalid_argument if downcast validation fails (element not in subfield)
 * @throws std::bad_alloc if memory allocation fails during conversion
 *
 * @section Conversion_Examples
 * @code{.cpp}
 * using F16_v1 = Ext<F2, {1, 0, 0, 1, 1}>;
 * using F16_v2 = Ext<F4, {2, 2, 1}>;
 * using F16_Iso = Iso<F16_v1, F16_v2>;
 * using F4 = Ext<F2, {1, 1, 1}>;
 *
 * F16_Iso iso_elem(7);
 * F4 result(iso_elem);  // Branch 2: F4 ⊆ F16_v2 via cached embedding
 * @endcode
 *
 * @see @ref details::largest_common_subfield_t, @ref SubfieldOf, @ref Isomorphic
 */
template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
Ext<B, modulus, mode>::Ext(const Iso<MAIN, OTHERS...>& other) {
    static_assert(Ext<B, modulus, mode>::get_characteristic() == Iso<MAIN, OTHERS...>::get_characteristic(),
                  "trying to convert between fields with different characteristic");

    using ExtType = Ext<B, modulus, mode>;
    using IsoType = Iso<MAIN, OTHERS...>;

    // Branch 1: If Ext is isomorphic to any of MAIN or the OTHERS then use the corresponding isomorphism
    if constexpr (Isomorphic<ExtType, MAIN>) {
        auto isomorphism = Isomorphism<MAIN, ExtType>();
        *this = isomorphism(other.main());
    } else if constexpr (sizeof...(OTHERS) > 0 && (Isomorphic<ExtType, OTHERS> || ...)) {
        // Check which OTHERS type is isomorphic to ExtType
        bool conversion_done = false;
        auto try_isomorphism = [&]<typename OtherType>() {
            if constexpr (Isomorphic<ExtType, OtherType>) {
                auto isomorphism = Isomorphism<OtherType, ExtType>();
                *this = isomorphism(OtherType(other));
                conversion_done = true;
            }
        };
        (try_isomorphism.template operator()<OTHERS>(), ...);

        // Branch 2 (upcast): If Iso or any of MAIN or the OTHERS is a subfield of Ext
    } else if constexpr (SubfieldOf<ExtType, IsoType>) {
        // Iso ⊆ Ext - use cached embedding from Iso to Ext
        auto embedding = Embedding<IsoType, ExtType>();
        ExtType result = embedding(other);
        label = result.get_label();
    } else if constexpr (SubfieldOf<ExtType, MAIN>) {
        // MAIN ⊆ Ext - use cached embedding from MAIN to Ext
        auto embedding = Embedding<MAIN, ExtType>();
        ExtType result = embedding(other.main());
        label = result.get_label();
    } else if constexpr (sizeof...(OTHERS) > 0 && (SubfieldOf<ExtType, OTHERS> || ...)) {
        // Check which OTHERS type is a subfield of ExtType
        bool conversion_done = false;
        auto try_embedding = [&]<typename OtherType>() {
            if constexpr (SubfieldOf<ExtType, OtherType>) {
                auto embedding = Embedding<OtherType, ExtType>();
                OtherType other_repr(other);
                ExtType result = embedding(other_repr);
                label = result.get_label();
                conversion_done = true;
            }
        };
        (try_embedding.template operator()<OTHERS>(), ...);

        // Branch 3 (downcast): If Ext is a subfield of Iso or any of MAIN or the OTHERS
    } else if constexpr (SubfieldOf<IsoType, ExtType>) {
        // Ext ⊆ Iso - use cached embedding to check if downcast is possible
        auto embedding = Embedding<ExtType, IsoType>();
        ExtType result = embedding.extract(other);
        label = result.get_label();
    } else if constexpr (SubfieldOf<MAIN, ExtType>) {
        // Ext ⊆ MAIN - use cached embedding to check if downcast is possible
        auto embedding = Embedding<ExtType, MAIN>();
        ExtType result = embedding.extract(other.main());
        label = result.get_label();
    } else if constexpr (sizeof...(OTHERS) > 0 && (SubfieldOf<OTHERS, ExtType> || ...)) {
        // Check which OTHERS type contains ExtType as subfield
        bool conversion_done = false;
        auto try_downcast = [&]<typename OtherType>() {
            if constexpr (SubfieldOf<OtherType, ExtType>) {
                auto embedding = Embedding<ExtType, OtherType>();
                OtherType other_repr(other);
                ExtType result = embedding.extract(other_repr);
                label = result.get_label();
                conversion_done = true;
            }
        };
        (try_downcast.template operator()<OTHERS>(), ...);

    } else {
        // Branch 4 (cross-cast through largest common subfield): This is the else case
        using CommonField = details::largest_common_subfield_t<ExtType, IsoType>;

        if constexpr (details::iso_info<CommonField>::is_iso) {
            // CommonField is an Iso, continue with its MAIN
            using CommonMainField = details::iso_info<CommonField>::main_type;
            CommonMainField intermediate(other);  // Downcast other to CommonField's MAIN
            *this = ExtType(intermediate);        // Use existing cross-field Ext->Ext constructor
        } else {
            // CommonField is an Ext, continue with CommonField
            CommonField intermediate(other);  // Downcast other to CommonField
            *this = ExtType(intermediate);    // Use existing cross-field Ext->Ext constructor
        }
    }
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator=(int l) {
    if (l < 0 || l >= Q) throw std::invalid_argument("l must be positve and no larger than Q-1");
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
constexpr Ext<B, modulus, mode> Ext<B, modulus, mode>::operator-() const& noexcept {
    if (this->is_erased()) return Ext().erase();
    Ext res(*this);
    if (res.label != 0) res.label = lut_neg()(res.label);
    return res;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator-() && noexcept {
    if (this->is_erased()) return this->erase();
    if (label != 0) label = lut_neg()(label);
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator+=(const Ext& rhs) noexcept {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    label = lut_add()(label, rhs.get_label());
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator-=(const Ext& rhs) noexcept {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    label = lut_add()(label, lut_neg()(rhs.get_label()));
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator*=(const Ext& rhs) noexcept {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    label = lut_mul()(label, rhs.get_label());
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator*=(int s) noexcept {
    if (this->is_erased()) return *this;
    // if (get_characteristic() != 0) s%= static_cast<int>(get_characteristic());
    Ext res = daa<Ext>(*this, s);
    *this = std::move(res);
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>& Ext<B, modulus, mode>::operator/=(const Ext& rhs) {
    if (this->is_erased() || rhs.is_erased()) return this->erase();
    if (rhs.label == 0) throw std::invalid_argument("trying to divide by zero");
    label = lut_mul()(label, lut_inv()(rhs.get_label()));
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>& Ext<B, modulus, mode>::randomize() noexcept {
    this->unerase();
    static std::uniform_int_distribution<label_t> dist(0, Q - 1);
    label = dist(gen());
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
Ext<B, modulus, mode>& Ext<B, modulus, mode>::randomize_force_change() noexcept {
    this->unerase();
    static std::uniform_int_distribution<label_t> dist(1, Q - 1);
    label = lut_add()(label, dist(gen()));
    return *this;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
size_t Ext<B, modulus, mode>::get_multiplicative_order() const {
    if (is_erased()) throw std::invalid_argument("trying to calculate multiplicative order of erased element");
    if (label == 0) throw std::invalid_argument("calculation of multiplicative order of additive neutral element");
    return lut_mul_ord()(label);
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
size_t Ext<B, modulus, mode>::get_additive_order() const {
    if (is_erased()) throw std::invalid_argument("trying to calculate additive order of erased element");
    if (label == 0) return 1;
    return get_characteristic();
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType S>
Polynomial<S> Ext<B, modulus, mode>::get_minimal_polynomial() const
    requires SubfieldOf<Ext<B, modulus, mode>, S>
{
    if (is_erased()) {
        throw std::invalid_argument("trying to compute minimal polynomial of erased element");
    }

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
std::string Ext<B, modulus, mode>::get_info() noexcept {
    std::stringstream ss;
    ss << "finite field with " + std::to_string(Q) + " elements, specified as degree " + std::to_string(m) +
              " extension of (" + B::get_info() + "), irreducible polynomial ";
    ss << get_modulus();
    std::string s = ss.str();
    return ss.str();
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
constexpr Polynomial<B> Ext<B, modulus, mode>::get_modulus() noexcept {
    Polynomial<B> rho;
    uint8_t i = 0;
    for (auto it = modulus.cbegin(); it != modulus.cend(); ++it) {
        rho.set_coefficient(i, *it);
        ++i;
    }
    return rho;
}

template <FiniteFieldType B, MOD modulus, LutMode mode>
template <FiniteFieldType T>
    requires(SubfieldOf<Ext<B, modulus, mode>, T>) && (!std::is_same_v<Ext<B, modulus, mode>, T>)
Vector<T> Ext<B, modulus, mode>::as_vector() const noexcept {
    if constexpr (std::is_same_v<B, T>) {
        Vector<T> res(m);
        if (is_erased()) {
            std::vector<size_t> indices(m);
            std::iota(indices.begin(), indices.end(), 0);
            res.erase_components(indices);
        } else {
            const auto coeffs = lut_coeff().values[label];
            for (uint8_t i = 0; i < m; ++i) res.set_component(i, coeffs[m - 1 - i]);
        }
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
void Ext<B, modulus, mode>::show_tables() noexcept {
    std::cout << "addition table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) {
        for (label_t j = 0; j < Q; ++j) std::cout << (int)(lut_add()(i, j)) << ", ";
        std::cout << std::endl;
    }

    std::cout << "additive inverse table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) std::cout << (int)(lut_neg()(i)) << std::endl;

    std::cout << "multiplication table (row and column headers omitted)" << std::endl;
    for (label_t i = 0; i < Q; ++i) {
        for (label_t j = 0; j < Q; ++j) std::cout << (int)(lut_mul()(i, j)) << ", ";
        std::cout << std::endl;
    }

    std::cout << "multiplicative inverse table (row and column headers "
                 "omitted)"
              << std::endl;
    for (label_t i = 0; i < Q; ++i) std::cout << (int)(lut_inv()(i)) << std::endl;

    std::cout << "multiplicative order table (row and column headers "
                 "omitted)"
              << std::endl;
    for (label_t i = 0; i < Q; ++i) std::cout << (int)(lut_mul_ord()(i)) << std::endl;

    std::cout << "element coefficients table (row and column "
                 "headers omitted)"
              << std::endl;
    for (label_t i = 0; i < Q; ++i) {
        std::cout << (int)i << ": ";
        for (uint8_t j = 0; j < m; ++j) std::cout << (int)(lut_coeff().values[i][j]) << ", ";
        std::cout << std::endl;
    }

    std::cout << "generator (with mult. order)" << std::endl;
    std::cout << get_generator() << " (" << get_generator().get_multiplicative_order() << ")" << std::endl;
}

/**
 * @brief Output stream operator for extension field elements
 * @tparam B Base field type
 * @tparam modulus Monic irreducible polynomial defining the extension
 * @param os Output stream
 * @param e Extension field element to output
 * @return Reference to the output stream
 *
 * Outputs the integer label of the extension field element.
 * The commented code shows an alternative format displaying polynomial coefficients.
 */
template <FiniteFieldType B, MOD modulus, LutMode mode>
std::ostream& operator<<(std::ostream& os, const Ext<B, modulus, mode>& e) noexcept {
    if (e.is_erased())
        os << ERASURE_MARKER;
    else
        os << (int)e.get_label();
    /*
    os << " [";
    for (size_t i = 0; i < Fq<p, modulus>::get_m(); ++i) {
        os << (int)Fq<p, modulus>::lut_coeff().values[e.get_label()][i];
    }
    os << "]";
    */
    return os;
}

/** @} */

/**
 * @class Iso
 * @brief Stack of multiple isomorphic field representations
 *
 * @tparam MAIN Primary field representation (must satisfy FiniteFieldType)
 * @tparam OTHERS Additional isomorphic field representations (all must be isomorphic to MAIN)
 *
 * The Iso class creates a unified interface for working with multiple isomorphic representations
 * of the same abstract finite field. It internally stores elements using the MAIN representation
 * while providing transparent conversion capabilities to and from all OTHERS representations.
 *
 * This is particularly useful in field towers where the same mathematical field
 * can be constructed in multiple ways (e.g., F16 as Ext<F2, irreducible1> or Ext<F4, irreducible2>).
 *
 * @section Key_Features
 * - **Transparent Conversion**: Seamless construction from any isomorphic representation
 * - **Unified Operations**: All field operations work consistently regardless of source representation
 * - **Performance**: Uses deterministic isomorphisms to avoid Frobenius factors
 * - **Type Safety**: Compile-time validation that all types are mutually isomorphic
 *
 * @section LutMode_Determination
 *
 * **The LUT mode of an Iso field is determined entirely by its MAIN field representation.**
 * All field operations (addition, multiplication, etc.) are forwarded directly to the MAIN field,
 * so the Iso inherits the MAIN field's LUT generation mode and performance characteristics.
 *
 * **Mixed LUT modes are supported**: OTHERS fields may use different LUT modes without conflict.
 * Isomorphism computations between representations occur at runtime using cached mappings,
 * regardless of individual field LUT modes.
 *
 * @code{.cpp}
 * // Mixed-mode Iso field example
 * using F16_a = Ext<Fp<2>, {1, 0, 0, 1, 1}, LutMode::CompileTime>;  // Fast startup
 * using F16_b = Ext<F4, {2, 1, 1}, LutMode::Runtime>;               // Lazy init
 * using F16 = Iso<F16_a, F16_b>;                                    // Uses CompileTime (MAIN)
 *
 * // All operations use F16_a performance characteristics
 * F16 a(100);    // Uses MAIN field's CompileTime LUTs
 * F16_b b(200);  // Source field uses Runtime LUTs
 * a = b;         // Runtime isomorphism conversion when needed
 * @endcode
 *
 * @section Usage_Example
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4_a = Ext<F2, {1, 1, 1}>;  // F4 via x² + x + 1
 * using F4_b = Ext<F2, {1, 0, 1}>;  // F4 via x² + 1
 * using F4 = Iso<F4_a, F4_b>;       // Unified F4 interface
 *
 * F4_a a(2);
 * F4_b b(3);
 * F4 result = a + b;  // Seamless conversion and operation
 * @endcode
 *
 * @see Isomorphism for the underlying conversion mechanism
 * @see FiniteFieldType for field type requirements
 */
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
class Iso : public details::Base {
    // Type safety: validate isomorphism at template instantiation
    static_assert((Isomorphic<MAIN, OTHERS> && ...), "All OTHERS must be isomorphic to MAIN");
    // Ensure we have multiple representations to unify, otherwise Iso doesn't make sense
    static_assert(sizeof...(OTHERS) > 0, "Iso requires at least two field representations");
    // Ensure all fields in the  union of MAIN and OTHERS... are pairwise distinct
    static_assert(pairwise_distinct<MAIN, OTHERS...>(), "All field representations in Iso must be pairwise distinct");
    // Comment: the last two assert that prime fields cannot occur in Isos

   public:
    using label_t = typename MAIN::label_t;

   private:
    MAIN main_;

   public:
    /**
     * @brief Default constructor - creates zero element
     *
     * Constructs the additive identity (zero) element. The zero element behaves consistently across all isomorphic
     * representations.
     */
    constexpr Iso() noexcept : main_() {}

    /**
     * @brief Integer constructor - creates element from integer label
     * @param l Integer value to convert to field element
     *
     * Constructs field element by converting integer through the MAIN representation.
     */
    constexpr Iso(int l) : main_(l) {}

    /**
     * @brief Copy constructor from MAIN representation
     * @param other Field element from primary representation to copy
     */
    constexpr Iso(const MAIN& other) noexcept : main_(other) {}

    /**
     * @brief Move constructor from MAIN representation
     * @param other Field element from primary representation to move
     */
    constexpr Iso(MAIN&& other) noexcept : main_(std::move(other)) {}

    /**
     * @brief Copy constructor from any OTHERS representation
     * @tparam OTHER One of the OTHERS... alternative representations
     * @param other Field element from alternative representation to copy
     */
    template <BelongsTo<OTHERS...> OTHER>
    constexpr Iso(const OTHER& other) noexcept : main_(MAIN(other)) {}

    /**
     * @brief Move constructor from any OTHERS representation
     * @tparam OTHER One of the OTHERS... alternative representations
     * @param other Field element from alternative representation to move
     */
    template <BelongsTo<OTHERS...> OTHER>
    constexpr Iso(OTHER&& other) noexcept : main_(MAIN(std::move(other))) {}

    /**
     * @brief Copy constructor
     * @param other Iso element to copy from
     */
    constexpr Iso(const Iso& other) noexcept = default;

    /**
     * @brief Move constructor
     * @param other Iso element to move from (will be left in valid but unspecified state)
     */
    constexpr Iso(Iso&& other) noexcept = default;

    /**
     * @brief Vector constructor for polynomial representations
     * @tparam T Component type of the vector (must be a field type)
     * @param v Vector representation to convert to field element
     *
     * Constructs field element from vector representation by delegating to
     * the MAIN field type's vector constructor.
     */
    template <FiniteFieldType T>
    Iso(const Vector<T>& v) : main_(v) {}

    /**
     * @brief Simple constructor from Extension field using Ext constructor
     * @tparam B Base field of the Extension
     * @tparam modulus Irreducible polynomial of the Extension
     * @tparam mode LUT mode of the Extension
     * @param ext Extension field element to convert from
     * @throws std::invalid_argument if conversion is not valid
     *
     * Leverages the Ext cross-field constructor to handle all conversion logic.
     *
     * @section Supported_Conversions
     * All conversion scenarios supported by the Ext constructor work automatically:
     * - **Direct**: When Ext type matches MAIN or OTHERS exactly
     * - **Upcast**: When Ext is subfield of MAIN/OTHERS (embedding, always works)
     * - **Downcast**: When MAIN/OTHERS is subfield of Ext (extraction, may throw)
     * - **Cross-field**: Via largest common subfield for cross-tower conversions
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4_a = Ext<F2, {1, 1, 1}>;         // x² + x + 1
     * using F4_b = Ext<F2, {1, 0, 1}>;         // x² + 1
     * using F16_a = Ext<F2, {1, 0, 0, 1, 1}>;  // x⁴ + x + 1
     * using F16_b = Ext<F4_a, {2, 2, 1}>;      // Tower: F2 → F4_a → F16
     * using F16 = Iso<F16_a, F16_b>;
     *
     * F4_a elem_4a(2);
     * F16 iso_from_4a(elem_4a);  // Upcast: F4_a embedded in F16_a, then to Iso
     *
     * F16_b elem_16b(5);
     * F16 iso_from_16b(elem_16b);  // Direct: F16_b matches OTHERS, isomorphism to MAIN
     * @endcode
     *
     * @note The result is always stored in the MAIN representation regardless of which conversion path is used
     * @see @ref details::largest_common_subfield_t, @ref SubfieldOf, @ref Isomorphic
     */
    template <FiniteFieldType B, MOD modulus, LutMode mode>
    Iso(const Ext<B, modulus, mode>& other);

    /**
     * @brief Cross-field constructor between Iso fields using four-branch conversion logic
     * @tparam OTHER_MAIN Main field type of the source Iso
     * @tparam OTHER_OTHERS Alternative representation types in the source Iso
     * @param other Source Iso field element to convert
     * @throws std::invalid_argument if downcast validation fails (element not in subfield)
     * @throws std::bad_alloc if memory allocation fails during conversion
     *
     * Advanced constructor enabling conversion between different Iso field representations using optimal
     * conversion paths. This constructor handles complex cross-Iso relationships and preserves field context.
     * The conversion is determined by comprehensive four-branch logic with extensive sub-case handling:
     *
     * **Branch 1 (Isomorphic)**: Direct isomorphism if MAIN ≅ OTHER_MAIN
     * **Branch 2 (Upcast)**: Multi-level embedding logic:
     *   - Input Iso ⊆ Output Iso (full Iso relationship)
     *   - Input MAIN ⊆ Output MAIN (direct MAIN relationship)
     *   - Input types ⊆ Output OTHERS (via any representation path)
     * **Branch 3 (Downcast)**: Multi-level extraction with validation:
     *   - Output Iso ⊆ Input Iso (full Iso relationship)
     *   - Output MAIN ⊆ Input MAIN (direct MAIN relationship)
     *   - Output types ⊆ Input types (via any representation path)
     * **Branch 4 (Cross-cast)**: Bridge through @ref details::largest_common_subfield_t with Iso preference mechanism
     *
     * @section Conversion_Examples
     * @code{.cpp}
     * using F16_v1 = Ext<F2, {1, 0, 0, 1, 1}>;
     * using F16_v2 = Ext<F4, {2, 2, 1}>;
     * using F256_v3 = Ext<F16_v1, {6, 13, 1}>;
     * using F16_Iso = Iso<F16_v1, F16_v2>;
     * using F256_Iso = Iso<F256_v1, F256_v2, F256_v3>;
     *
     * F16_Iso source(7);
     * F256_Iso result(source);  // Branch 2: F16_Iso ⊆ F256_Iso via F256_v3 path
     * @endcode
     *
     * @note Cross-Iso conversions use enhanced @ref SubfieldOf logic that traces all representation paths
     * @see @ref details::largest_common_subfield_t, @ref SubfieldOf, @ref Isomorphic
     */
    template <FiniteFieldType OTHER_MAIN, FiniteFieldType... OTHER_OTHERS>
    Iso(const Iso<OTHER_MAIN, OTHER_OTHERS...>& other);

    constexpr Iso operator-() const noexcept { return Iso{-main_}; }

    constexpr Iso& operator+=(const Iso& other);

    template <BelongsTo<OTHERS...> OTHER>
    constexpr Iso& operator+=(const OTHER& other);

    constexpr Iso& operator-=(const Iso& other);

    template <BelongsTo<OTHERS...> OTHER>
    constexpr Iso& operator-=(const OTHER& other);

    constexpr Iso& operator*=(const Iso& other);

    template <BelongsTo<OTHERS...> OTHER>
    constexpr Iso& operator*=(const OTHER& other);

    constexpr Iso& operator*=(int s) noexcept;

    Iso& operator/=(const Iso& other);

    template <BelongsTo<OTHERS...> OTHER>
    Iso& operator/=(const OTHER& other);

    /**
     * @brief Convert to specific isomorphic representation
     * @tparam TO Target field type (must be one of OTHERS)
     * @return Field element converted to TO representation
     *
     * Explicitly converts the internal MAIN representation to any of the
     * isomorphic OTHERS representations using the appropriate isomorphism.
     *
     * @code{.cpp}
     * Iso<F4_a, F4_b> unified(some_value);
     * F4_b specific = unified.as<F4_b>();  // Explicit conversion
     * @endcode
     */
    template <BelongsTo<OTHERS...> TO>
    constexpr TO as() const;

    /**
     * @brief Access underlying MAIN representation (const)
     * @return Const reference to internal MAIN field element
     *
     * Provides read-only access to the underlying MAIN representation
     * for inspection or performance-critical read operations.
     */
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

    /**
     * @brief Erases this element, i.e., sets it to an "outside of field" marker
     * @return Reference to this element after erasing
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     */
    constexpr Iso& erase() noexcept;

    /**
     * @brief Un-erases this element, i.e., sets it to an actual field element (the additive neutral 0)
     * @return Reference to this element after un-erasing
     */
    constexpr Iso& unerase() noexcept;

    /**
     * @brief Checks whether this element is erased
     * @return true if this element is erased, false otherwise (meaning it actually is a field element)
     */
    constexpr bool is_erased() const noexcept { return main_.is_erased(); }

    // Assignment operators
    constexpr Iso& operator=(const Iso& other);
    constexpr Iso& operator=(Iso&& other) noexcept = default;
    constexpr Iso& operator=(const MAIN& other);
    constexpr Iso& operator=(int other);

    /**
     * @brief Assignment operator for isomorphic field representations
     * @tparam OTHER One of the OTHERS template parameter types (isomorphic field representations)
     * @param other Field element from an isomorphic representation to assign
     * @return Reference to this Iso element after assignment
     *
     * Performs assignment from any of the isomorphic field representations specified in the
     * OTHERS template parameter pack. Uses the existing constructor logic with copy-and-swap
     * idiom for exception safety during the conversion process.
     *
     * @note This enables seamless assignment between different representations of the same
     *       mathematical field, such as F4_a and F4_b in Iso<F4_a, F4_b>.
     *
     * @section Usage_Example
     * @code{.cpp}
     * using F4_a = Ext<F2, {1, 1, 1}>;
     * using F4_b = Ext<F2, {1, 0, 1}>;
     * using F4 = Iso<F4_a, F4_b>;
     *
     * F4 iso_elem;
     * F4_b other_repr(3);
     * iso_elem = other_repr;  // Uses this assignment operator
     * @endcode
     */
    template <BelongsTo<OTHERS...> OTHER>
    Iso& operator=(const OTHER& other);

    /**
     * @brief Cross-field assignment from Fp type
     *
     * Performs field conversion assignment from an Fp field with the same characteristic.
     * Uses copy-and-swap idiom leveraging cross-field constructors for exception safety.
     *
     * @tparam p Prime of the source Fp field (must match characteristic of MAIN field)
     * @param other Source Fp element to assign from
     * @return Reference to this Iso element after assignment
     * @throws std::invalid_argument if conversion not possible
     */
    template <uint16_t p>
        requires(p == MAIN::get_characteristic())
    Iso& operator=(const Fp<p>& other);

    /**
     * @brief Cross-field assignment from Ext type
     *
     * Performs field conversion assignment from an Ext field with the same characteristic.
     * Uses copy-and-swap idiom leveraging enhanced cross-field constructors for exception safety.
     *
     * @tparam B Base field type of the source Ext
     * @tparam ext_modulus Modulus polynomial of the source Ext
     * @tparam mode LUT mode of the source Ext
     * @param other Source Ext element to assign from
     * @return Reference to this Iso element after assignment
     * @throws std::invalid_argument if conversion not possible
     */
    template <FiniteFieldType B, MOD ext_modulus, LutMode mode>
        requires(B::get_characteristic() == MAIN::get_characteristic())
    Iso& operator=(const Ext<B, ext_modulus, mode>& other);

    /**
     * @brief Cross-field assignment from other Iso types (cross-representation)
     *
     * Performs field conversion assignment from a different Iso field with the same characteristic.
     * Uses copy-and-swap idiom leveraging enhanced cross-field constructors for exception safety.
     *
     * @tparam OTHER_MAIN Main field type of the source Iso
     * @tparam OTHER_OTHERS Additional isomorphic field types in the source Iso group
     * @param other Source Iso element to assign from
     * @return Reference to this Iso element after assignment
     * @throws std::invalid_argument if conversion not possible
     */
    template <FiniteFieldType OTHER_MAIN, FiniteFieldType... OTHER_OTHERS>
        requires(OTHER_MAIN::get_characteristic() == MAIN::get_characteristic()) &&
                (!std::is_same_v<Iso<OTHER_MAIN, OTHER_OTHERS...>, Iso<MAIN, OTHERS...>>)
    Iso& operator=(const Iso<OTHER_MAIN, OTHER_OTHERS...>& other);

    // Equality operators
    constexpr bool operator==(const Iso& other) const noexcept { return main_ == other.main_; }
    constexpr bool operator==(const MAIN& other) const noexcept { return main_ == other; }
    constexpr bool operator!=(const Iso& other) const noexcept { return main_ != other.main_; }
    constexpr bool operator!=(const MAIN& other) const noexcept { return main_ != other; }

    // Binary arithmetic operators handled by global template functions (like Fp and Ext)

    // Stream operator - delegate to underlying value
    friend std::ostream& operator<<(std::ostream& os, const Iso& iso) { return os << iso.main_; }

    static const std::string get_info() noexcept;

    // Static methods required by FiniteFieldType concept
    static constexpr size_t get_characteristic() noexcept { return MAIN::get_characteristic(); }
    static constexpr size_t get_p() noexcept { return MAIN::get_p(); }
    static constexpr size_t get_q() noexcept { return MAIN::get_q(); }
    static constexpr size_t get_m() noexcept { return MAIN::get_m(); }
    static constexpr size_t get_size() noexcept { return MAIN::get_size(); }
    static constexpr Iso get_generator() noexcept { return Iso{MAIN::get_generator()}; }

    /**
     * @brief Check if this Iso field interface is constexpr-ready for compile-time usage
     * @return Result of MAIN::is_constexpr_ready()
     *
     * The constexpr-readiness of an Iso field is determined entirely by its MAIN field representation.
     * All field operations are forwarded to the MAIN field, so the Iso field inherits
     * the MAIN field's constexpr-readiness and interface characteristics.
     *
     * OTHERS fields may have different constexpr-readiness without conflict, as isomorphism
     * computations between different representations occur at runtime regardless
     * of individual field constexpr-readiness.
     *
     * @note This enables mixed-mode Iso fields like Iso<F16_CompileTime, F16_Runtime>
     *       where different representations can have different constexpr-readiness properties.
     *
     * @see MAIN field type for the actual constexpr-readiness determination
     * @see Isomorphism class for runtime conversion between representations
     */
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
    static constexpr label_t lut_add(label_t a, label_t b) noexcept { return lut_add()(a, b); }
    static constexpr label_t lut_mul(label_t a, label_t b) noexcept { return lut_mul()(a, b); }
    static constexpr label_t lut_neg(label_t a) noexcept { return lut_neg()(a); }
    static constexpr label_t lut_inv(label_t a) noexcept { return lut_inv()(a); }

    // Vector conversion method - handle different target types T
    template <FiniteFieldType T>
        requires((SubfieldOf<MAIN, T> || ((SubfieldOf<OTHERS, T>) || ...))) &&
                (!std::is_same_v<Iso<MAIN, OTHERS...>, T>)
    Vector<T> as_vector() const noexcept;
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
template <BelongsTo<OTHERS...> OTHER>
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(const OTHER& other) {
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
    requires(OTHER_MAIN::get_characteristic() == MAIN::get_characteristic()) &&
            (!std::is_same_v<Iso<OTHER_MAIN, OTHER_OTHERS...>, Iso<MAIN, OTHERS...>>)
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator=(const Iso<OTHER_MAIN, OTHER_OTHERS...>& other) {
    Iso temp(other);
    std::swap(*this, temp);
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <FiniteFieldType B, MOD modulus, LutMode mode>
Iso<MAIN, OTHERS...>::Iso(const Ext<B, modulus, mode>& other) {
    static_assert(MAIN::get_characteristic() == Ext<B, modulus, mode>::get_characteristic(),
                  "trying to convert between fields with different characteristic");

    using IsoType = Iso<MAIN, OTHERS...>;
    using ExtType = Ext<B, modulus, mode>;

    // Branch 1: If MAIN or any of the OTHERS is isomorphic to Ext then use the correct isomorphism
    if constexpr (Isomorphic<MAIN, ExtType>) {
        auto isomorphism = Isomorphism<ExtType, MAIN>();
        main_ = isomorphism(other);
    } else if constexpr (sizeof...(OTHERS) > 0 && (Isomorphic<OTHERS, ExtType> || ...)) {
        // Check which OTHERS type is isomorphic to ExtType
        bool conversion_done = false;
        auto try_isomorphism = [&]<typename OtherType>() {
            if constexpr (Isomorphic<OtherType, ExtType>) {
                auto isomorphism = Isomorphism<ExtType, OtherType>();
                OtherType other_repr = isomorphism(other);
                // Convert OtherType to MAIN via Iso constructor
                IsoType temp_iso(other_repr);
                main_ = temp_iso.main_;
                conversion_done = true;
            }
        };
        (try_isomorphism.template operator()<OTHERS>(), ...);

        // Branch 2 (upcast): If Ext is a subfield to MAIN or any of the OTHERS
    } else if constexpr (SubfieldOf<MAIN, ExtType>) {
        // Ext ⊆ MAIN - use cached embedding from Ext to MAIN
        auto embedding = Embedding<ExtType, MAIN>();
        main_ = embedding(other);
    } else if constexpr (sizeof...(OTHERS) > 0 && (SubfieldOf<OTHERS, ExtType> || ...)) {
        // Check which OTHERS type contains ExtType as subfield
        bool conversion_done = false;
        auto try_embedding = [&]<typename OtherType>() {
            if constexpr (SubfieldOf<OtherType, ExtType>) {
                auto embedding = Embedding<ExtType, OtherType>();
                OtherType other_repr = embedding(other);
                // Convert OtherType to MAIN via Iso constructor
                IsoType temp_iso(other_repr);
                main_ = temp_iso.main_;
                conversion_done = true;
            }
        };
        (try_embedding.template operator()<OTHERS>(), ...);

        // Branch 3 (downcast): If Iso or MAIN or any of the OTHERS is a subfield of Ext
    } else if constexpr (SubfieldOf<ExtType, IsoType>) {
        // Iso ⊆ Ext - use cached embedding to check if downcast is possible
        auto embedding = Embedding<IsoType, ExtType>();
        IsoType temp_iso = embedding.extract(other);
        main_ = temp_iso.main_;
    } else if constexpr (SubfieldOf<ExtType, MAIN>) {
        // MAIN ⊆ Ext - use cached embedding to check if downcast is possible
        auto embedding = Embedding<MAIN, ExtType>();
        main_ = embedding.extract(other);
    } else if constexpr (sizeof...(OTHERS) > 0 && (SubfieldOf<ExtType, OTHERS> || ...)) {
        // Check which OTHERS type is a superfield of ExtType
        bool conversion_done = false;
        auto try_downcast = [&]<typename OtherType>() {
            if constexpr (SubfieldOf<ExtType, OtherType>) {
                auto embedding = Embedding<OtherType, ExtType>();
                OtherType other_repr = embedding.extract(other);
                // Convert OtherType to MAIN via Iso constructor
                IsoType temp_iso(other_repr);
                main_ = temp_iso.main_;
                conversion_done = true;
            }
        };
        (try_downcast.template operator()<OTHERS>(), ...);

    } else {
        // Branch 4 (cross-cast through largest common subfield): This is the else case
        using CommonField = details::largest_common_subfield_t<IsoType, ExtType>;

        if constexpr (details::iso_info<CommonField>::is_iso) {
            // CommonField is an Iso, continue with its MAIN
            using CommonMainField = details::iso_info<CommonField>::main_type;
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
template <FiniteFieldType OTHER_MAIN, FiniteFieldType... OTHER_OTHERS>
Iso<MAIN, OTHERS...>::Iso(const Iso<OTHER_MAIN, OTHER_OTHERS...>& other) {
    static_assert(MAIN::get_characteristic() == OTHER_MAIN::get_characteristic(),
                  "trying to convert between fields with different characteristic");

    using OutputIsoType = Iso<MAIN, OTHERS...>;             // Output Iso (this)
    using InputIsoType = Iso<OTHER_MAIN, OTHER_OTHERS...>;  // Input Iso (other)

    // Branch 1: If MAIN of input Iso is isomorphic to MAIN of output Iso
    if constexpr (Isomorphic<OTHER_MAIN, MAIN>) {
        auto isomorphism = Isomorphism<OTHER_MAIN, MAIN>();
        main_ = isomorphism(other.main());

        // Branch 2 (upcast): Input types are subfields of output types
    } else if constexpr (SubfieldOf<OutputIsoType, InputIsoType>) {
        // Input Iso ⊆ Output Iso - embed via cached embedding
        auto embedding = Embedding<InputIsoType, OutputIsoType>();
        OutputIsoType temp_output = embedding(other);
        main_ = temp_output.main_;
    } else if constexpr (SubfieldOf<MAIN, OTHER_MAIN>) {
        // Input MAIN ⊆ Output MAIN - embed via cached embedding
        auto embedding = Embedding<OTHER_MAIN, MAIN>();
        main_ = embedding(other.main());
    } else if constexpr (sizeof...(OTHERS) > 0 && (SubfieldOf<OTHERS, InputIsoType> || ...)) {
        // Check if any output OTHERS contains input Iso as subfield
        bool conversion_done = false;
        auto try_upcast_iso = [&]<typename OutputOtherType>() {
            if constexpr (SubfieldOf<OutputOtherType, InputIsoType>) {
                // Input Iso ⊆ Output OTHERS - embed then convert to MAIN
                auto embedding = Embedding<InputIsoType, OutputOtherType>();
                OutputOtherType output_other = embedding(other);
                auto isomorphism = Isomorphism<OutputOtherType, MAIN>();
                main_ = isomorphism(output_other);
                conversion_done = true;
            }
        };
        (try_upcast_iso.template operator()<OTHERS>(), ...);

        if (!conversion_done && sizeof...(OTHERS) > 0 && (SubfieldOf<OTHERS, OTHER_MAIN> || ...)) {
            // Check if any output OTHERS contains input MAIN as subfield
            auto try_upcast_main = [&]<typename OutputOtherType>() {
                if constexpr (SubfieldOf<OutputOtherType, OTHER_MAIN>) {
                    // Input MAIN ⊆ Output OTHERS - embed then convert to MAIN
                    auto embedding = Embedding<OTHER_MAIN, OutputOtherType>();
                    OutputOtherType output_other = embedding(other.main());
                    auto isomorphism = Isomorphism<OutputOtherType, MAIN>();
                    main_ = isomorphism(output_other);
                    conversion_done = true;
                }
            };
            (try_upcast_main.template operator()<OTHERS>(), ...);
        }

        if (!conversion_done && sizeof...(OTHER_OTHERS) > 0 && sizeof...(OTHERS) > 0) {
            // Check if any output OTHERS contains any input OTHERS as subfield
            auto try_upcast_others = [&]<typename OutputOtherType>() {
                if (conversion_done) return;
                auto try_input_others = [&]<typename InputOtherType>() {
                    if constexpr (SubfieldOf<OutputOtherType, InputOtherType>) {
                        // Input OTHERS ⊆ Output OTHERS - embed then convert to MAIN
                        InputOtherType input_other(other);
                        auto embedding = Embedding<InputOtherType, OutputOtherType>();
                        OutputOtherType output_other = embedding(input_other);
                        auto isomorphism = Isomorphism<OutputOtherType, MAIN>();
                        main_ = isomorphism(output_other);
                        conversion_done = true;
                    }
                };
                (try_input_others.template operator()<OTHER_OTHERS>(), ...);
            };
            (try_upcast_others.template operator()<OTHERS>(), ...);
        }

        // Branch 3 (downcast): Output types are subfields of input types
    } else if constexpr (SubfieldOf<InputIsoType, OutputIsoType>) {
        // Output Iso ⊆ Input Iso - extract via cached embedding
        auto embedding = Embedding<OutputIsoType, InputIsoType>();
        OutputIsoType temp_output = embedding.extract(other);
        main_ = temp_output.main_;
    } else if constexpr (SubfieldOf<OTHER_MAIN, MAIN>) {
        // Output MAIN ⊆ Input MAIN - extract via cached embedding
        auto embedding = Embedding<MAIN, OTHER_MAIN>();
        main_ = embedding.extract(other.main());
    } else if constexpr (SubfieldOf<InputIsoType, MAIN> ||
                         (sizeof...(OTHER_OTHERS) > 0 && (SubfieldOf<OTHER_OTHERS, MAIN> || ...))) {
        // Check if output MAIN contains input types as subfield
        bool conversion_done = false;

        if constexpr (SubfieldOf<InputIsoType, MAIN>) {
            // Output MAIN ⊆ Input Iso - extract via cached embedding
            auto embedding = Embedding<MAIN, InputIsoType>();
            main_ = embedding.extract(other);
            conversion_done = true;
        }

        if (!conversion_done && sizeof...(OTHER_OTHERS) > 0) {
            // Check if output MAIN contains any input OTHERS as subfield
            auto try_downcast_others = [&]<typename InputOtherType>() {
                if constexpr (SubfieldOf<InputOtherType, MAIN>) {
                    InputOtherType input_other(other);
                    auto embedding = Embedding<MAIN, InputOtherType>();
                    main_ = embedding.extract(input_other);
                    conversion_done = true;
                }
            };
            (try_downcast_others.template operator()<OTHER_OTHERS>(), ...);
        }

    } else {
        // Branch 4 (cross-cast through largest common subfield): This is the else case
        using CommonField = details::largest_common_subfield_t<OutputIsoType, InputIsoType>;

        if constexpr (details::iso_info<CommonField>::is_iso) {
            // CommonField is an Iso, continue with its MAIN
            using CommonMainField = details::iso_info<CommonField>::main_type;

            // Find which input type has CommonField as subfield and map to it
            CommonMainField intermediate;
            bool found_input_path = false;

            if constexpr (SubfieldOf<InputIsoType, CommonMainField>) {
                // Map input Iso to CommonMainField via downcast
                auto embedding = Embedding<CommonMainField, InputIsoType>();
                intermediate = embedding.extract(other);
                found_input_path = true;
            } else if constexpr (SubfieldOf<OTHER_MAIN, CommonMainField>) {
                // Map input MAIN to CommonMainField via embedding
                auto embedding = Embedding<OTHER_MAIN, CommonMainField>();
                intermediate = embedding(other.main());
                found_input_path = true;
            } else if constexpr (sizeof...(OTHER_OTHERS) > 0) {
                // Check input OTHERS
                auto try_input_others = [&]<typename InputOtherType>() {
                    if constexpr (SubfieldOf<InputOtherType, CommonMainField>) {
                        InputOtherType input_other(other);
                        auto embedding = Embedding<InputOtherType, CommonMainField>();
                        intermediate = embedding(input_other);
                        found_input_path = true;
                    }
                };
                (try_input_others.template operator()<OTHER_OTHERS>(), ...);
            }

            // Convert CommonMainField to output MAIN
            main_ = MAIN(intermediate);  // Use Ext->Ext constructor

        } else {
            // CommonField is an Ext, continue with CommonField

            // Find which input type has CommonField as subfield and map to it
            CommonField intermediate;
            bool found_input_path = false;

            if constexpr (SubfieldOf<InputIsoType, CommonField>) {
                // Map input Iso to CommonField via downcast
                auto embedding = Embedding<CommonField, InputIsoType>();
                intermediate = embedding.extract(other);
                found_input_path = true;
            } else if constexpr (SubfieldOf<OTHER_MAIN, CommonField>) {
                // Map input MAIN to CommonField via embedding
                auto embedding = Embedding<OTHER_MAIN, CommonField>();
                intermediate = embedding(other.main());
                found_input_path = true;
            } else if constexpr (sizeof...(OTHER_OTHERS) > 0) {
                // Check input OTHERS
                auto try_input_others = [&]<typename InputOtherType>() {
                    if constexpr (SubfieldOf<InputOtherType, CommonField>) {
                        InputOtherType input_other(other);
                        auto embedding = Embedding<InputOtherType, CommonField>();
                        intermediate = embedding(input_other);
                        found_input_path = true;
                    }
                };
                (try_input_others.template operator()<OTHER_OTHERS>(), ...);
            }

            // Convert CommonField to output MAIN
            main_ = MAIN(intermediate);  // Use Ext->Ext constructor
        }
    }
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator+=(const Iso& other) {
    main_ += other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <BelongsTo<OTHERS...> OTHER>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator+=(const OTHER& other) {
    main_ += Iso(other).main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator-=(const Iso& other) {
    main_ -= other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <BelongsTo<OTHERS...> OTHER>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator-=(const OTHER& other) {
    main_ -= Iso(other).main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator*=(const Iso& other) {
    main_ *= other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <BelongsTo<OTHERS...> OTHER>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator*=(const OTHER& other) {
    main_ *= Iso(other).main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
constexpr Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator*=(int s) noexcept {
    main_ *= s;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator/=(const Iso& other) {
    main_ /= other.main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <BelongsTo<OTHERS...> OTHER>
Iso<MAIN, OTHERS...>& Iso<MAIN, OTHERS...>::operator/=(const OTHER& other) {
    main_ /= Iso(other).main_;
    return *this;
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <BelongsTo<OTHERS...> TO>
constexpr TO Iso<MAIN, OTHERS...>::as() const {
    auto phi = Isomorphism<MAIN, TO>();
    return phi(main_);
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
const std::string Iso<MAIN, OTHERS...>::get_info() noexcept {
    std::stringstream ss;
    ss << "stack of isomorphic fields, main field: ";
    ss << MAIN::get_info();
    return ss.str();
}

template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
template <FiniteFieldType T>
    requires((SubfieldOf<MAIN, T> || ((SubfieldOf<OTHERS, T>) || ...))) && (!std::is_same_v<Iso<MAIN, OTHERS...>, T>)
Vector<T> Iso<MAIN, OTHERS...>::as_vector() const noexcept {
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

namespace details {

template <FiniteFieldType T>
struct FiniteFieldHasher {
    size_t operator()(const T& e) const noexcept { return std::hash<typename T::label_t>{}(e.get_label()); }
};

}  // namespace details

}  // namespace ECC

#endif
