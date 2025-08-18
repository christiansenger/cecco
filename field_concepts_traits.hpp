/**
 * @file field_concepts_traits.hpp
 * @brief Concepts, traits, and type utilities for finite field arithmetic
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.1.0
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
 * This header provides the conceptual foundation for the finite field arithmetic library.
 * It defines concepts, type traits, and template metaprogramming utilities that enable
 * type-safe field operations and compile-time field relationship validation.
 *
 * **Core Concepts Provided:**
 * - **FiniteFieldType**: Constrains types to valid finite field implementations
 * - **SubfieldOf**: Validates mathematical subfield relationships between field types
 * - **Isomorphic**: Ensures field types have the same size and characteristic for safe conversions
 * - **SignedIntType**: Supports both standard signed integers and arbitrary precision InfInt
 * - **UnsignedIntType**: Supports both standard unsigned integers and arbitrary precision InfInt
 * - **Arithmetic types**: Complex number and rational number type validation
 *
 * **Type Traits and Utilities:**
 * - **iso_info**: Template metaprogramming utility for Iso type introspection
 * - **largest_common_subfield_t**: Automatically determines largest common subfield for cross-field operations
 *
 * @see fields.hpp for the main finite field implementation
 */

#ifndef FIELD_CONCEPTS_TRAITS_HPP
#define FIELD_CONCEPTS_TRAITS_HPP

#include <complex>
#include <concepts>
#include <string>
#include <type_traits>

#include "helpers.hpp"
// #include "InfInt.hpp" // transitive through helpers.hpp

namespace CECCO {

/**
 * @brief LUT generation mode for field operations
 */
enum class LutMode {
    CompileTime,  ///< Generate LUTs at compile-time using constexpr (default)
    RunTime       ///< Generate LUTs at runtime using lazy initialization
    // Note: Precompiled mode is handled by the USE_PRECOMPILED_LUTS macro
};

namespace details {
// Forward declaration for Base (full definition in fields.hpp)
class Base;
}  // namespace details

/**
 * @concept FieldType
 * @brief Concept for field types with complete mathematical structure
 *
 * @tparam T Type to be checked for field compliance
 *
 * This concept ensures that a type provides the complete mathematical interface
 * of a field, including all algebraic operations, constructors, and essential
 * field properties. It serves as the foundation for generic field algorithms
 * and ensures type safety for field operations.
 *
 * @section Requirements
 *
 * A FieldType must provide:
 * - **All field operations**: +, -, *, / with proper semantics
 * - **Field identities**: Additive (0) and multiplicative (1) identities
 * - **Comparison operations**: Equality testing
 * - **Element properties**: Zero testing, sign information, characteristic
 * - **Randomization**: For generating random field elements
 * - **Information**: String representation and structural information
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * template <FieldType F>
 * F quadratic_formula(const F& a, const F& b, const F& c) {
 *     F discriminant = b*b - a*c*F(4);
 *     return (-b + sqrt(discriminant)) / (a * F(2));
 * }
 * @endcode
 *
 * @note This concept is satisfied by Rationals<>, Fp&lt;&gt;, Ext<>, and Iso<>
 */
template <typename T>
concept FieldType =
    std::is_base_of_v<details::Base, T> && !std::is_same_v<details::Base, T> && requires(const T& t, T& mutable_t) {
        // Constructors (via constructibility checks)
        requires std::constructible_from<T>;       // Default constructor
        requires std::copy_constructible<T>;       // Copy constructor
        requires std::move_constructible<T>;       // Move constructor
        requires std::constructible_from<T, int>;  // From int constructor

        // Assignment operators
        { mutable_t = t } -> std::same_as<T&>;   // Field element assignment
        { mutable_t = 42 } -> std::same_as<T&>;  // Integer assignment

        // Compound assignment operators
        { mutable_t += t } -> std::same_as<T&>;
        { mutable_t -= t } -> std::same_as<T&>;
        { mutable_t *= t } -> std::same_as<T&>;
        { mutable_t /= t } -> std::same_as<T&>;

        // Comparison operators
        { t == t } -> std::same_as<bool>;

        // Unary - operator
        { -t } -> std::same_as<T>;

        // Element properties
        { t.is_zero() } -> std::same_as<bool>;
        { t.has_positive_sign() } -> std::same_as<bool>;
        { t.get_characteristic() } -> std::convertible_to<size_t>;
        { t.get_info() } -> std::convertible_to<std::string>;

        // Randomization
        requires requires { mutable_t.randomize(); };
        requires requires { mutable_t.randomize_force_change(); };
    };

/**
 * @concept FiniteFieldType
 * @brief Concept for finite field types with additional structure
 *
 * @tparam T Type to be checked for finite field compliance
 *
 * This concept refines FieldType to specifically handle finite fields, adding requirements
 * for field size and extension degree. Finite fields are characterized by having finite
 * order and non-zero characteristic.
 *
 * @section Requirements
 *
 * Beyond FieldType requirements, finite fields must provide:
 * - Prime characteristic at least two
 * - Complete field structure information (`get_size()`, `get_m()`, `get_p()`, `get_q()`)
 * - Generator element access via static `get_generator()` method
 * - Element order calculations (`get_multiplicative_order()`, `get_additive_order()`)
 *
 * @note Inter-field constructors are expected but not enforced by this concept. Such constructors enable field tower
 * operations and mathematical embeddings.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * template <FiniteFieldType F>
 * void analyze_finite_field() {
 *     std::cout << "Field size (q): " << F::get_q() << std::endl;
 *     std::cout << "Prime characteristic (p): " << F::get_p() << std::endl;
 *     std::cout << "Prime characteristic (p): " << F::get_characteristic() << std::endl;
 *     std::cout << "Extension degree (m): " << F::get_m() << std::endl;
 *     std::cout << "Generator element: " << F::get_generator() << std::endl;
 *     std::cout << "Multiplicative group order: " << F::get_size() - 1 << std::endl;
 * }
 * @endcode
 *
 * @note This concept is satisfied by Fp&lt;&gt; Ext<>, and Iso<>, but not by Rationals<>
 */

template <typename T>
concept FiniteFieldType =
    std::is_base_of_v<details::Base, T> && !std::is_same_v<details::Base, T> && requires(const T& t, T& mutable_t) {
        // Basic field requirements (from FieldType, but inlined to avoid circular dependency)
        requires std::constructible_from<T>;       // Default constructor
        requires std::copy_constructible<T>;       // Copy constructor
        requires std::move_constructible<T>;       // Move constructor
        requires std::constructible_from<T, int>;  // From int constructor

        // Assignment operators
        { mutable_t = t } -> std::same_as<T&>;   // Field element assignment
        { mutable_t = 42 } -> std::same_as<T&>;  // Integer assignment

        // Compound assignment operators
        { mutable_t += t } -> std::same_as<T&>;
        { mutable_t -= t } -> std::same_as<T&>;
        { mutable_t *= t } -> std::same_as<T&>;
        { mutable_t /= t } -> std::same_as<T&>;

        // Comparison operators
        { t == t } -> std::same_as<bool>;

        // Unary - operator
        { -t } -> std::same_as<T>;

        // Element properties
        { t.is_zero() } -> std::same_as<bool>;
        { t.has_positive_sign() } -> std::same_as<bool>;
        { t.get_characteristic() } -> std::convertible_to<size_t>;
        { t.get_info() } -> std::convertible_to<std::string>;

        // Randomization
        requires requires { mutable_t.randomize(); };
        requires requires { mutable_t.randomize_force_change(); };

        // Finite field specific requirements
        requires(T::get_characteristic() > 1);
        requires is_prime(T::get_p());

        // Static field structure information
        { t.get_size() } -> std::convertible_to<size_t>;
        { t.get_m() } -> std::convertible_to<size_t>;
        { t.get_p() } -> std::convertible_to<size_t>;
        { t.get_q() } -> std::convertible_to<size_t>;

        // Generator element
        { T::get_generator() } -> std::same_as<T>;

        // Element properties
        { t.get_multiplicative_order() } -> std::convertible_to<size_t>;
        { t.get_additive_order() } -> std::convertible_to<size_t>;
    };

/**
 * @concept SignedIntType
 * @brief Concept defining requirements for signed integer types
 *
 * @tparam T Type to be checked for signed integer compliance
 *
 * This concept constrains template parameters to types suitable for signed integer
 * arithmetic, supporting both standard signed integral types and infinite precision
 * arithmetic via InfInt.
 *
 * @section Requirements
 *
 * For a type T to satisfy SignedIntType, it must be either:
 * - A standard signed integral type (int, long, long long, etc.)
 *   - Must satisfy `std::is_integral_v<T> && std::is_signed_v<T>`
 * - The custom InfInt type for arbitrary precision arithmetic
 *   - Enables computations without overflow limitations
 *   - Required for true rational arithmetic with potentially infinite numerators/denominators
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * template <SignedIntType T>
 * class Rationals {
 *     T numerator;
 *     T denominator;
 * public:
 *     Rationals(int n = 0, int d = 1) : numerator{n}, denominator{d} {}
 * };
 *
 * // Valid instantiations:
 * Rationals<int> q1;        // Standard signed integer
 * Rationals<long long> q2;  // Larger standard type
 * Rationals<InfInt> q3;     // Infinite precision
 * @endcode
 *
 * @note This concept is primarily used by Rationals<T> to ensure the numerator
 *       and denominator types support signed arithmetic operations
 */
template <typename T>
concept SignedIntType = (std::is_integral_v<T> && std::is_signed_v<T>) || std::is_same_v<T, InfInt>;

// Forward declaration for Rationals (full definition in fields.hpp)
template <SignedIntType T>
class Rationals;

/**
 * @concept ReliablyComparableType
 * @brief Concept for types that support reliable mathematical comparison operations
 *
 * @tparam T Type to be checked for reliable comparison compliance
 *
 * This concept identifies types that can be reliably compared using standard comparison
 * operators without floating-point precision issues or other numerical instabilities.
 * It is used internally for algorithms requiring stable ordering and equality testing.
 *
 * @section Requirements
 *
 * A ReliablyComparableType must be one of:
 * - **Finite field types**: Any type satisfying FiniteFieldType (exact arithmetic)
 * - **Exact rational arithmetic**: Rationals<InfInt> with infinite precision
 * - **Signed integer types**: Any type satisfying SignedIntType for exact integer arithmetic
 *
 * @note This concept excludes floating-point types due to precision and rounding issues
 *       that can make comparison operations unreliable for mathematical algorithms.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * template <ReliablyComparableType T>
 * bool is_mathematically_equal(const T& a, const T& b) {
 *     return a == b;  // Safe for exact arithmetic types
 * }
 * @endcode
 */
template <typename T>
concept ReliablyComparableType = FiniteFieldType<T> || std::is_same_v<T, Rationals<InfInt>> || SignedIntType<T>;

/**
 * @concept ComponentType
 * @brief Concept for valid @ref Vector, @ref Polynomial, and @ref Matrix component types
 *
 * @tparam T Type to be checked for component compliance
 *
 * This concept defines the requirements for types that can serve as @ref Vector, @ref Polynomial, and @ref Matrix
 * components. It unifies field types with fundamental numeric types to provide a comprehensive type safety framework
 * for @ref Vector, @ref Polynomial, and @ref Matrix operations.
 *
 * @section Requirements
 *
 * A ComponentType must be one of:
 * - **Field types**: Any type satisfying FieldType concept
 * - **double**: IEEE 754 double-precision floating-point
 * - **std::complex<double>**: Complex numbers with double precision
 * - **Signed integer types**: Any type satisfying SignedIntType (int, long, long long, InfInt)
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * template <ComponentType T>
 * Vector<T> process_vector(const Vector<T>& v) {
 *     // Works for field types, double, std::complex<double>, and signed integers
 *     return v * T(2);
 * }
 *
 * Vector<int> int_vector(10);        // Standard integer vector
 * Vector<long long> long_vector(5);  // Long integer vector
 * Vector<InfInt> big_vector(10);     // Arbitrary-precision integer vector
 * big_vector.randomize();            // Randomizes with values in range [-100, 100]
 * @endcode
 */
template <typename T>
concept ComponentType =
    FieldType<T> || std::same_as<T, double> || std::same_as<T, std::complex<double>> || SignedIntType<T>;

/**
 * @brief Concept to check if a type belongs to a parameter pack (is same as one of the types)
 * @tparam T Type to check
 * @tparam Types Parameter pack to check against
 *
 * This concept evaluates to true if type T is identical to at least one of the types
 * in the Types parameter pack.
 *
 * @section Usage
 *
 * @code{.cpp}
 * // Instead of:
 * template <typename OTHER>
 *     requires((std::is_same_v<OTHER, OTHERS>) || ...)
 * Iso& operator=(const OTHER& other);
 *
 * // Use:
 * template <BelongsTo<OTHERS...> OTHER>
 * Iso& operator=(const OTHER& other);
 * @endcode
 *
 * @section Examples
 *
 * @code{.cpp}
 * static_assert(BelongsTo<int, float, double>);    // true - int is one of the types
 * static_assert(!BelongsTo<char, float, double>);  // false - char is not in the pack
 * static_assert(BelongsTo<F16_a, F16_a, F16_b>);   // true - F16_a matches first type
 * @endcode
 */
template <typename T, typename... Types>
concept BelongsTo = (std::is_same_v<T, Types> || ...);

/**
 * @brief Convenience macro for polynomial modulus array specification
 *
 * This macro provides a shorter syntax for specifying modulus polynomials in extension field
 * construction. It expands to std::array, allowing the use of brace-initialization syntax
 * for polynomial coefficients.
 *
 * @section Usage_Example
 * @code{.cpp}
 * using F4 = Ext<F2, {1, 1, 1}>;   // Instead of std::array<int, 3>{1, 1, 1}
 * using F16 = Ext<F4, {2, 1, 1}>;  // Shorter and more readable, don't need to use MOD
 * @endcode
 */
#define MOD std::array

// Forward declaration for Fp (full definition in fields.hpp)
template <uint16_t p>
class Fp;

// Forward declaration for Ext (full definition in fields.hpp)
template <FiniteFieldType B, MOD modulus, LutMode mode>
class Ext;

// Forward declaration for Iso (full definition in fields.hpp)
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
class Iso;

/**
 * @concept Isomorphic
 * @brief Concept for testing if two finite fields are isomorphic
 * @tparam A First field type
 * @tparam B Second field type
 *
 * This concept validates that A and B are finite fields of the same size and therefore
 * isomorphic. This enables construction of explicit field isomorphisms between different
 * representations of the same finite field (for example when a field is constructed as an extension of the same base
 * field but with different irreducible polynomials of the same degree or when it is constructed as extension of
 * different base fields, for example F64 can be constructed both from F4 and from F8)
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4_1 = Ext<F2, {1, 1, 1}>;   // 𝔽₄ ≅ 𝔽₂[x]/(x² + x + 1)
 * using F4_2 = Ext<F2, {1, 0, 1}>;   // 𝔽₄ ≅ 𝔽₂[x]/(x² + 1)
 * using F8 = Ext<F2, {1, 1, 0, 1}>;  // 𝔽₈ ≅ 𝔽₂[x]/(x³ + x + 1)
 * using F3 = Fp<3>;
 * using F64_1 = Ext<F8, {7, 1, 1}>;
 * using F64_2 = Ext<F4, {1, 2, 0, 1}>;
 *
 * static_assert(Isomorphic<F64_1, F64_2>);  // true: every field is isomorphic to itself
 * static_assert(Isomorphic<F4_1, F4_2>);    // true: both have 4 elements
 * static_assert(!Isomorphic<F4_1, F8>);     // false: 4 ≠ 8 elements
 * static_assert(!Isomorphic<F2, F3>);       // false: 2 ≠ 3 elements
 * static_assert(Isomorphic<F4_1, F4_1>);    // true: reflexive
 * static_assert(Isomorphic<F64_1, F64_2>);  // true: both have 64 elements
 * @endcode
 *
 * @note This concept only checks size equality. The actual isomorphism construction
 * is performed by the @ref CECCO::Isomorphism class, which computes explicit
 * field homomorphisms between isomorphic representations.
 *
 * @see @ref CECCO::Isomorphism for explicit isomorphism construction
 * @see @ref CECCO::FiniteFieldType for the underlying field type requirements
 */
template <typename A, typename B>
concept Isomorphic = FiniteFieldType<A> && FiniteFieldType<B> && requires { requires A::get_size() == B::get_size(); };

// Forward declaration for Iso (constraints are in the class definition)
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
class Iso;

namespace details {

/**
 * @struct degree_over_prime
 * @brief Template trait to compute total extension degree over the prime field
 * @tparam T Field type (must satisfy FiniteFieldType concept)
 *
 * This trait recursively computes the total extension degree from field T down to its underlying
 * prime field by tracing through field towers and multiplying extension degrees at each level.
 *
 * For a field tower like F₂₅₆ = Ext<F₁₆, modulus₁> where F₁₆ = Ext<F₄, modulus₂> where F₄ = Ext<F₂, modulus₃>,
 * the computation would be: degree_over_prime<F₂₅₆>::value = 2 × 2 × 4 = 16 (since F₂₅₆ = F₂^16).
 *
 * @section degree_over_prime_specializations Specializations
 * - **Prime fields Fp<p>**: Always return 1 (base case)
 * - **Extension fields Ext<B, modulus, mode>**: Return (modulus.size() - 1) × degree_over_prime<B>::value
 * - **Iso fields Iso<MAIN, OTHERS...>**: Return degree_over_prime<MAIN>::value (all components isomorphic)
 *
 * @section degree_over_prime_usage Usage Examples
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;   // Degree 2 over F2
 * using F16 = Ext<F4, {2, 1, 1}>;  // Degree 2 over F4, degree 4 over F2
 *
 * static_assert(degree_over_prime_v<F2> == 1);   // F2 has degree 1 over itself
 * static_assert(degree_over_prime_v<F4> == 2);   // F4 has degree 2 over F2
 * static_assert(degree_over_prime_v<F16> == 4);  // F16 has degree 4 over F2
 * @endcode
 */
template <FiniteFieldType T>
struct degree_over_prime;

template <FiniteFieldType T>
inline constexpr size_t degree_over_prime_v = degree_over_prime<T>::value;

// Base case: Prime fields have degree 1 over themselves
template <uint16_t p>
struct degree_over_prime<Fp<p>> {
    static constexpr size_t value = 1;
};

// Extension fields: multiply local extension degree by base field's total degree
template <FiniteFieldType B, auto modulus, LutMode mode>
struct degree_over_prime<Ext<B, modulus, mode>> {
    static constexpr size_t value = (modulus.size() - 1) * degree_over_prime<B>::value;
};

// Iso fields: use the total degree of the main field (all components are isomorphic)
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
struct degree_over_prime<Iso<MAIN, OTHERS...>> {
    static constexpr size_t value = degree_over_prime<MAIN>::value;
};

/**
 * @struct is_subfield_of
 * @brief Template trait to test recursive construction-based field relationships
 * @tparam SUPERFIELD Superfield type (must satisfy FiniteFieldType concept)
 * @tparam SUBFIELD Subfield type (must satisfy FiniteFieldType concept)
 *
 * This trait determines if SUBFIELD can be reached by recursively descending the construction
 * tower of SUPERFIELD. It implements recursive tracking down field towers through the base field
 * parameter B in Ext<B, modulus> constructions.
 *
 * The relationship is **construction-based**, not mathematical containment. For example, if F16 is
 * constructed as Ext<F2, modulus>, then is_subfield_of<F16, F2>::value is true, but is_subfield_ov<F16, F4>::value is
 * false as the field F4 simply was not constructed (even though mathematially it is "in between" F2 and F16).
 *
 * @section is_subfield_of_specializations Specializations
 * - **Prime fields**: Fp<p> contains only itself (reflexivity)
 * - **Extension fields**: Ext<B, modulus> contains B and all subfields of B (recursive descent)
 * - **Extension field reflexivity**: Ext<B, modulus> contains itself (reflexivity)
 * - **Extension-to-extension**: Cross-extension relationships via base field analysis
 * - **Iso fields**: Traces ALL paths through MAIN and OTHERS components
 * - **Iso field reflexivity**: Iso<MAIN, OTHERS...> contains itself (reflexivity)
 *
 * @section is_subfield_of_usage Usage Examples
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;
 * using F16_from_F4 = Ext<F4, {2, 1, 1}>;
 * using F16_from_F2 = Ext<F2, {1, 1, 0, 0, 1}>;
 *
 * static_assert(is_subfield_of_v<F4, F2>);            // F2 ⊆ F4 (construction-based)
 * static_assert(is_subfield_of_v<F4, F4>);            // F4 ⊆ F4 (reflexivity)
 * static_assert(is_subfield_of_v<F16_from_F4, F4>);   // F4 ⊆ F16_from_F4
 * static_assert(is_subfield_of_v<F16_from_F4, F2>);   // F2 ⊆ F16_from_F4 (through F4)
 * static_assert(!is_subfield_of_v<F16_from_F2, F4>);  // F4 ⊄ F16_from_F2 (different construction)
 * @endcode
 */
template <FiniteFieldType SUPERFIELD, FiniteFieldType SUBFIELD>
struct is_subfield_of : std::false_type {};

template <FiniteFieldType SUPER, FiniteFieldType SUB>
constexpr bool is_subfield_of_v = is_subfield_of<SUPER, SUB>::value;

// Prime field reflexivity: Fp<p> contains only itself
template <uint16_t p>
struct is_subfield_of<Fp<p>, Fp<p>> : std::true_type {};

// General extension field case: SUB ⊆ Ext<B, modulus, mode> if SUB = B or SUB ⊆ B
template <FiniteFieldType B, MOD modulus, LutMode mode, FiniteFieldType SUB>
struct is_subfield_of<Ext<B, modulus, mode>, SUB>
    : std::bool_constant<std::is_same_v<B, SUB> || is_subfield_of_v<B, SUB>> {};

// Extension field reflexivity: Ext<B, modulus, mode> contains itself
template <FiniteFieldType B, MOD modulus, LutMode mode>
struct is_subfield_of<Ext<B, modulus, mode>, Ext<B, modulus, mode>> : std::true_type {};

// Extension-to-extension: Ext<B_SUB, mod_SUB, mode_SUB> ⊆ Ext<B_SUP, mod_SUP, mode_SUP> if SUB = B_SUP or SUB ⊆ B_SUP
template <FiniteFieldType B_SUP, MOD modulus_SUP, LutMode mode_SUP, FiniteFieldType B_SUB, MOD modulus_SUB,
          LutMode mode_SUB>
struct is_subfield_of<Ext<B_SUP, modulus_SUP, mode_SUP>, Ext<B_SUB, modulus_SUB, mode_SUB>>
    : std::bool_constant<std::is_same_v<B_SUP, Ext<B_SUB, modulus_SUB, mode_SUB>> ||
                         is_subfield_of_v<B_SUP, Ext<B_SUB, modulus_SUB, mode_SUB>>> {};

// Iso components: Any component (MAIN or OTHERS) is a subfield of the Iso
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS, FiniteFieldType SUB>
    requires(std::is_same_v<SUB, MAIN> || ((std::is_same_v<SUB, OTHERS>) || ...))
struct is_subfield_of<Iso<MAIN, OTHERS...>, SUB> : std::true_type {};

// General Iso case: SUB ⊆ Iso if SUB is a subfield of any component
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS, FiniteFieldType SUB>
    requires(!std::is_same_v<SUB, MAIN> && !((std::is_same_v<SUB, OTHERS>) || ...))
struct is_subfield_of<Iso<MAIN, OTHERS...>, SUB>
    : std::bool_constant<is_subfield_of_v<MAIN, SUB> || (is_subfield_of_v<OTHERS, SUB> || ...)> {};

// Iso field reflexivity: Iso<MAIN, OTHERS...> contains itself
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
struct is_subfield_of<Iso<MAIN, OTHERS...>, Iso<MAIN, OTHERS...>> : std::true_type {};

// Cross-Iso subfield relationships: Check if any representation from SubIso is a subfield of any representation in
// SuperIso
template <FiniteFieldType SUP_MAIN, FiniteFieldType... SUP_OTHERS, FiniteFieldType SUB_MAIN,
          FiniteFieldType... SUB_OTHERS>
    requires(!std::is_same_v<Iso<SUP_MAIN, SUP_OTHERS...>, Iso<SUB_MAIN, SUB_OTHERS...>>)
struct is_subfield_of<Iso<SUP_MAIN, SUP_OTHERS...>, Iso<SUB_MAIN, SUB_OTHERS...>> {
   private:
    // Helper to check if a single sub-field is a subfield of any super-field representation
    template <FiniteFieldType SubField>
    static constexpr bool is_subfield_of_any_super_repr() {
        return is_subfield_of_v<SUP_MAIN, SubField> || (is_subfield_of_v<SUP_OTHERS, SubField> || ...);
    }

    // Check if SUB_MAIN is a subfield of any representation in SuperIso
    static constexpr bool sub_main_in_super = is_subfield_of_any_super_repr<SUB_MAIN>();

    // Check if any SUB_OTHERS is a subfield of any representation in SuperIso
    static constexpr bool any_sub_others_in_super = (is_subfield_of_any_super_repr<SUB_OTHERS>() || ...);

   public:
    static constexpr bool value = sub_main_in_super || any_sub_others_in_super;
};

}  // namespace details

/**
 * @concept SubfieldOf
 * @brief Concept for testing subfield relationships including Iso types
 * @tparam SUPERFIELD Extension field type (the larger field)
 * @tparam SUBFIELD Potential subfield type (the smaller field)
 *
 * This concept validates that SUBFIELD ⊆ SUPERFIELD using enhanced logic that handles:
 * - Traditional field tower relationships (nested construction)
 * - Iso type representations and their mathematical equivalences (traces all paths down to the prime field)
 * - Enhanced cross-Iso relationships: any representation from SubIso ⊆ any representation in SuperIso
 * - Recursive path detection through complex field hierarchies
 *
 * @warning This validates field tower relationships in the sense of construction (nested construction of extension
 * fields), not in the mathematical sense. Depending on the actual construction, the construction tower may skip one or
 * more intermediate fields from the mathematical tower.
 *
 * @note This relationship is reflexive: SubfieldOf<F, F> is always true.
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4_1 = Ext<F2, {1, 1, 1}>;     // F₄ construction 1
 * using F4_2 = Ext<F2, {1, 0, 1}>;     // F₄ construction 2 (isomorphic)
 * using F4_Iso = Iso<F4_1, F4_2>;      // Intersection field
 * using F16 = Ext<F4_Iso, {2, 1, 1}>;  // Extension using Iso as base
 *
 * // Traditional relationships
 * static_assert(SubfieldOf<F16, F16>);     // true: every field is its own subfield
 * static_assert(SubfieldOf<F16, F4_Iso>);  // true: F4_Iso ⊆ F16 (direct construction)
 * static_assert(SubfieldOf<F4_Iso, F2>);   // true: F2 ⊆ F4_Iso (via MAIN representation)
 *
 * // Enhanced Iso relationships
 * static_assert(SubfieldOf<F16, F4_1>);  // true: F4_1 ⊆ F16 (via Iso representation)
 * static_assert(SubfieldOf<F16, F4_2>);  // true: F4_2 ⊆ F16 (via Iso representation)
 * static_assert(SubfieldOf<F16, F2>);    // true: F2 ⊆ F16 (recursive path detection)
 * @endcode
 */
template <typename SUPERFIELD, typename SUBFIELD>
concept SubfieldOf = details::is_subfield_of_v<SUPERFIELD, SUBFIELD>;

/**
 * @concept ExtensionOf
 * @brief Concept for testing extension field relationships including Iso types
 * @tparam S Base field type
 * @tparam E Extension field type
 *
 * This concept validates that E is an extension of S within field towers,
 * including complex relationships via Iso types. It is simply the inverse of
 * the SubfieldOf relationship: E extends S if and only if S is a subfield of E.
 *
 * The concept handles:
 * - Traditional field tower relationships (nested construction)
 * - Iso type representations and their mathematical equivalences (search E in all paths starting from S)
 * - Multiple construction paths to the same mathematical field
 *
 * @warning This validates field tower relationships in the sense of this library (nested construction of extension
 * fields), not in the mathematical sense. Depending on the actual construction, the construction tower may skip one or
 * more intermediate fields from the mathematical tower.
 *
 * @note This relationship is reflexive: ExtensionOf<F, F> is always true.
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4_1 = Ext<F2, {1, 1, 1}>;     // F₄ construction 1
 * using F4_2 = Ext<F2, {1, 0, 1}>;     // F₄ construction 2 (isomorphic)
 * using F4_Iso = Iso<F4_1, F4_2>;      // Intersection field
 * using F16 = Ext<F4_Iso, {2, 1, 1}>;  // Extension using Iso as base
 * using F3 = Fp<3>;
 *
 * // Traditional relationships
static_assert(Isomorphic<F16, F16>);       // true: every field is its own extension
 * static_assert(ExtensionOf<F2, F4_1>);   // true: F4_1 defined based on F2
 * static_assert(ExtensionOf<F4_1, F16>);  // true: F16 defined based on F4_Iso (which contains F4_1)
 *
 * // Enhanced Iso relationships
 * static_assert(ExtensionOf<F4_1, F16>);  // true: via Iso representation
 * static_assert(ExtensionOf<F4_2, F16>);  // true: via Iso representation
 * static_assert(ExtensionOf<F2, F16>);    // true: recursive path detection
 *
 * // Negative cases
 * static_assert(!ExtensionOf<F16, F4_1>);  // false: F4_1 not defined based on F16
 * static_assert(!ExtensionOf<F3, F16>);    // false: F16 not defined based on F3
 * @endcode
 */
template <typename S, typename E>
concept ExtensionOf = SubfieldOf<E, S>;

/**
 * @concept InSameTower
 * @brief Enhanced concept for testing if two fields are in the same field tower including Iso types
 * @tparam F First field type
 * @tparam G Second field type
 *
 * This concept validates that F and G are in the same construction tower,
 * meaning one is a subfield of the other (in either direction). This is a symmetric
 * relationship useful for field conversions and compatibility checks.
 *
 * The concept now handles:
 * - Traditional field tower relationships (nested construction)
 * - Iso type representations and their mathematical equivalences
 * - Multiple construction paths connecting the same mathematical fields
 * - Complex field hierarchies with intersection fields
 *
 * @warning This validates field tower relationships in the sense of this library (nested construction of extension
 * fields), not in the mathematical sense. Depending on the actual construction, the construction tower may skip one or
 * more intermediate fields from the mathematical tower.
 *
 * @note This relationship is symmetric and reflexive: InSameTower<F, G> ↔ InSameTower<G, F> and InSameTower<F, F> is
 * always true.
 *
 * @note Isomorphic fields (like F4_1 and F4_2) are mathematically never in the same construction tower unless connected
 * via an Iso type.
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F3 = Fp<3>;
 * using F4_1 = Ext<F2, {1, 1, 1}>;     // F₄ construction 1
 * using F4_2 = Ext<F2, {1, 0, 1}>;     // F₄ construction 2 (isomorphic)
 * using F4_Iso = Iso<F4_1, F4_2>;      // Intersection field
 * using F16 = Ext<F4_Iso, {2, 1, 1}>;  // Extension using Iso as base
 * using F8 = Ext<F2, {1, 1, 0, 1}>;    // Different branch from F2
 *
 * // Traditional relationships (symmetric and reflexive)
 * static_assert(InSameTower<F2, F4_1>);    // true: F2 ⊆ F4_1
 * static_assert(InSameTower<F4_1, F2>);    // true: symmetric to above
 * static_assert(InSameTower<F4_1, F4_1>);  // true: reflexive
 * static_assert(InSameTower<F2, F16>);     // true: F2 ⊆ F16 via F4_Iso
 *
 * // Enhanced Iso relationships
 * static_assert(InSameTower<F4_1, F16>);  // true: via Iso representation
 * static_assert(InSameTower<F4_2, F16>);  // true: via Iso representation
 * static_assert(InSameTower<F16, F4_1>);  // true: symmetric to above
 *
 * // Negative cases (different towers)
 * static_assert(!InSameTower<F4_1, F4_2>);  // false: non-identical isomorphic fields
 * static_assert(!InSameTower<F8, F16>);     // false: different leaves
 * static_assert(!InSameTower<F4_1, F8>);    // false: different leaves
 * static_assert(!InSameTower<F2, F3>);      // false: different primes
 * static_assert(!InSameTower<F4_1, F3>);    // false: different characteristics
 * static_assert(!InSameTower<F16, F3>);     // false: completely separate towers
 * @endcode
 */
template <typename F, typename G>
concept InSameTower = SubfieldOf<F, G> || SubfieldOf<G, F>;

namespace details {

/**
 * @brief Compile-time type list utility for subfield calculations
 * @tparam Types Variadic list of types
 */
template <typename... Types>
struct type_list {
    static constexpr size_t size = sizeof...(Types);
};

/**
 * @brief Check if a type exists in a type_list
 * @tparam T Type to search for
 * @tparam List type_list to search in
 */
template <typename T, typename List>
struct contains;

template <typename T, typename... Types>
struct contains<T, type_list<Types...>> : std::bool_constant<(std::is_same_v<T, Types> || ...)> {};

template <typename T, typename List>
constexpr bool contains_v = contains<T, List>::value;

/**
 * @brief Union of two type_lists (removes duplicates)
 * @tparam List1 First type_list
 * @tparam List2 Second type_list
 */
template <typename List1, typename List2>
struct union_type_lists;

template <typename... Types1, typename... Types2>
struct union_type_lists<type_list<Types1...>, type_list<Types2...>> {
   private:
    template <typename T>
    using add_if_not_present = std::conditional_t<contains_v<T, type_list<Types1...>>, type_list<>, type_list<T>>;

    template <typename... Lists>
    struct concat;

    template <typename... Types>
    struct concat<type_list<Types...>> {
        using type = type_list<Types...>;
    };

    template <typename... Types1_, typename... Types2_, typename... Rest>
    struct concat<type_list<Types1_...>, type_list<Types2_...>, Rest...> {
        using type = typename concat<type_list<Types1_..., Types2_...>, Rest...>::type;
    };

   public:
    using type = typename concat<type_list<Types1...>, add_if_not_present<Types2>...>::type;
};

template <typename List1, typename List2>
using union_type_lists_t = typename union_type_lists<List1, List2>::type;

/**
 * @brief Intersection of two type_lists
 * @tparam List1 First type_list
 * @tparam List2 Second type_list
 */
template <typename List1, typename List2>
struct intersect_type_lists;

template <typename... Types1, typename... Types2>
struct intersect_type_lists<type_list<Types1...>, type_list<Types2...>> {
   private:
    template <typename T>
    using add_if_in_both = std::conditional_t<contains_v<T, type_list<Types2...>>, type_list<T>, type_list<>>;

    template <typename... Lists>
    struct concat;

    template <typename... Types>
    struct concat<type_list<Types...>> {
        using type = type_list<Types...>;
    };

    template <typename... Types1_, typename... Types2_, typename... Rest>
    struct concat<type_list<Types1_...>, type_list<Types2_...>, Rest...> {
        using type = typename concat<type_list<Types1_..., Types2_...>, Rest...>::type;
    };

   public:
    using type = typename concat<add_if_in_both<Types1>...>::type;
};

template <typename List1, typename List2>
using intersect_type_lists_t = typename intersect_type_lists<List1, List2>::type;

/**
 * @brief Find the largest field (by get_size()) in a type_list
 * @tparam List type_list of field types
 */
template <typename List>
struct largest_field_in_list;

template <typename T>
struct largest_field_in_list<type_list<T>> {
    using type = T;
};

template <typename T1, typename T2, typename... Rest>
struct largest_field_in_list<type_list<T1, T2, Rest...>> {
   private:
    using larger = std::conditional_t<(T1::get_size() >= T2::get_size()), T1, T2>;

   public:
    using type = typename largest_field_in_list<type_list<larger, Rest...>>::type;
};

template <typename List>
using largest_field_in_list_t = typename largest_field_in_list<List>::type;

/**
 * @brief Collect all subfields of a given field type at compile-time
 * @tparam F Field type (must satisfy @ref FiniteFieldType)
 *
 * This trait recursively collects all subfields of F, including F itself, into a @ref type_list.
 * The collection includes both direct subfields and all transitive subfields through the field tower.
 *
 * @section Algorithm
 *
 * - **Prime fields**: @c Fp<p> returns @c type_list<Fp<p>> (base case)
 * - **Extension fields**: @c Ext<B,mod,mode> returns @c subfields(B) ∪ {B} ∪ {Ext<B,mod,mode>}
 * - **Isomorphic fields**: @c Iso<MAIN,OTHERS...> returns union of all representation subfields
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;
 * using F16 = Ext<F4, {2, 2, 1}>;
 *
 * using subfields_F2 = collect_subfields_t<F2>;    // type_list<Fp<2>>
 * using subfields_F4 = collect_subfields_t<F4>;    // type_list<Fp<2>, Ext<F2,{1,1,1}>>
 * using subfields_F16 = collect_subfields_t<F16>;  // type_list<Fp<2>, Ext<F2,{1,1,1}>, Ext<F4,{2,2,1}>>
 * @endcode
 *
 * @note This trait is used internally by @ref largest_common_subfield_t for cross-field conversions
 * @see @ref largest_common_subfield_t
 */
template <typename F>
struct collect_subfields;

// Specialization for prime fields - base case
template <size_t p>
struct collect_subfields<Fp<p>> {
    using type = type_list<Fp<p>>;
};

// Specialization for extension fields - recursive case
template <FiniteFieldType B, MOD modulus, LutMode mode>
struct collect_subfields<Ext<B, modulus, mode>> {
   private:
    using base_subfields = typename collect_subfields<B>::type;
    using with_base = union_type_lists_t<base_subfields, type_list<B>>;

   public:
    using type = union_type_lists_t<with_base, type_list<Ext<B, modulus, mode>>>;
};

// Helper to compute union of all subfields for Iso types
template <typename... Types>
struct union_all_subfields;

template <>
struct union_all_subfields<> {
    using type = type_list<>;
};

template <typename T>
struct union_all_subfields<T> {
    using type = typename collect_subfields<T>::type;
};

template <typename T1, typename T2, typename... Rest>
struct union_all_subfields<T1, T2, Rest...> {
    using type =
        union_type_lists_t<typename collect_subfields<T1>::type, typename union_all_subfields<T2, Rest...>::type>;
};

// Specialization for isomorphic fields - union of all representations
template <typename MAIN, typename... OTHERS>
struct collect_subfields<Iso<MAIN, OTHERS...>> {
   private:
    using main_subfields = typename collect_subfields<MAIN>::type;
    using others_subfields = typename details::union_all_subfields<OTHERS...>::type;
    using all_subfields = union_type_lists_t<main_subfields, others_subfields>;

   public:
    using type = union_type_lists_t<all_subfields, type_list<Iso<MAIN, OTHERS...>>>;
};

template <typename F>
using collect_subfields_t = typename collect_subfields<F>::type;

/**
 * @brief Trait to find the largest common subfield of two fields at compile-time
 * @tparam F First field type (must satisfy @ref FiniteFieldType)
 * @tparam G Second field type (must satisfy @ref FiniteFieldType)
 *
 * This trait computes the largest field that is a subfield of both F and G, enabling
 * optimal cross-field conversions by finding the "bridge" field for conversions between
 * different field towers. Used extensively by cross-field constructors in @ref Ext and @ref Iso.
 *
 * @section Algorithm
 *
 * 1. **Trivial case**: If @c F == @c G, return @c F
 * 2. **Subfield collection**: Use @ref collect_subfields_t to gather all subfields of both fields (including Iso types)
 * 3. **Intersection**: Find common subfields using @ref intersect_type_lists_t
 * 4. **Selection**: Return the largest field (by @c get_size()) from the intersection
 * 5. **Iso preference**: If largest field is isomorphic to an input Iso parameter, return the Iso instead
 *
 * @section Performance_Characteristics
 *
 * - **Compile-time computation**: All operations performed at compile-time via template metaprogramming
 * - **Optimal selection**: Always finds the mathematically largest common subfield
 * - **Cross-tower support**: Handles conversions between different field construction paths
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F3 = Fp<3>;
 * using F9 = Ext<F3, {2, 2, 1}>;
 * using F81_a = Ext<F3, {2, 1, 0, 0, 1}>;   // Direct F3 → F81
 * using F81_b = Ext<F9, {6, 0, 1}>;         // Tower F3 → F9 → F81
 * using F27 = Ext<F3, {1, 2, 0, 1}>;        // Different tower F3 → F27
 *
 * using common_1 = largest_common_subfield_t<F81_a, F81_b>;  // F9 (optimal bridge)
 * using common_2 = largest_common_subfield_t<F81_a, F27>;    // F3 (only common field)
 * using common_3 = largest_common_subfield_t<F9, F27>;       // F3 (different towers)
 *
 * // Iso preference mechanism:
 * using F16_v1 = Ext<F2, {1, 0, 0, 1, 1}>;
 * using F16_v2 = Ext<F4, {2, 2, 1}>;
 * using F16_Iso = Iso<F16_v1, F16_v2>;
 * using F256_v3 = Ext<F16_v1, {6, 13, 1}>;
 * using common_4 = largest_common_subfield_t<F16_Iso, F256_v3>;  // F16_Iso (not F16_v1)
 *
 * // Cross-field conversion example:
 * F27 source(10);
 * F81_a target(source);  // Converts via F3 (largest common subfield)
 * @endcode
 *
 * @section Mathematical_Foundation
 *
 * For finite fields F ⊆ K and G ⊆ K in the same algebraic closure:
 * - **Existence**: The intersection F ∩ G is always a field (possibly just the prime field)
 * - **Uniqueness**: There is a unique largest common subfield
 * - **Optimality**: This trait finds the mathematically optimal conversion path
 *
 * @note Requires F and G to have the same characteristic (enforced by static_assert)
 * @see @ref collect_subfields_t
 * @see @ref intersect_type_lists_t
 * @see @ref Ext cross-field constructor
 * @see @ref Iso cross-field constructor
 */
template <typename F, typename G>
struct largest_common_subfield {
    static_assert(F::get_characteristic() == G::get_characteristic(), "Fields must have the same characteristic");

   private:
    using subfields_F = collect_subfields_t<F>;
    using subfields_G = collect_subfields_t<G>;
    using common_subfields = intersect_type_lists_t<subfields_F, subfields_G>;
    using raw_largest = largest_field_in_list_t<common_subfields>;

    // Post-processing: If the largest common subfield (an Ext) is isomorphic to one of the
    // input parameters and that parameter is an Iso, return the Iso instead of the Ext
    template <typename Largest, typename Param1, typename Param2>
    struct prefer_iso_over_isomorphic_ext {
        using type = Largest;  // Default: return the raw largest
    };

    // Specialization: If Largest is isomorphic to Param1 and Param1 is an Iso, return Param1
    template <typename Largest, typename MAIN1, typename... OTHERS1, typename Param2>
        requires Isomorphic<Largest, Iso<MAIN1, OTHERS1...>>
    struct prefer_iso_over_isomorphic_ext<Largest, Iso<MAIN1, OTHERS1...>, Param2> {
        using type = Iso<MAIN1, OTHERS1...>;
    };

    // Specialization: If Largest is isomorphic to Param2 and Param2 is an Iso, return Param2
    template <typename Largest, typename Param1, typename MAIN2, typename... OTHERS2>
        requires Isomorphic<Largest, Iso<MAIN2, OTHERS2...>>
    struct prefer_iso_over_isomorphic_ext<Largest, Param1, Iso<MAIN2, OTHERS2...>> {
        using type = Iso<MAIN2, OTHERS2...>;
    };

   public:
    using type = typename prefer_iso_over_isomorphic_ext<raw_largest, F, G>::type;
};

template <typename F, typename G>
using largest_common_subfield_t = typename largest_common_subfield<F, G>::type;

/**
 * @brief Trait to detect if a type is an Iso and extract its MAIN type
 * @tparam T Type to analyze (any field type)
 *
 * Helper trait used internally by @ref largest_common_subfield_t for Iso preference mechanism.
 * Provides compile-time detection of Iso types and access to their main representation.
 *
 * Primary template: type is not an Iso
 * - `is_iso = false`
 * - `main_type = void`
 *
 * @section Usage_Examples
 * @code{.cpp}
 * using F16_v1 = Ext<F2, {1, 0, 0, 1, 1}>;
 * using F16_v2 = Ext<F4, {2, 2, 1}>;
 * using F16_Iso = Iso<F16_v1, F16_v2>;
 *
 * static_assert(is_iso_v<F16_Iso>);       // true
 * static_assert(!is_iso_v<F16_v1>);       // false
 * using main = iso_main_type_t<F16_Iso>;  // F16_v1
 * @endcode
 *
 * @see @ref is_iso_v, @ref iso_main_type_t
 */
template <typename T>
struct iso_info {
    static constexpr bool is_iso = false;
    using main_type = void;
};

/**
 * @brief Specialization for Iso types
 *
 * Provides type information about Iso fields for template metaprogramming.
 * Extracts the MAIN type and packages OTHERS types into a tuple for iteration.
 *
 * @tparam MAIN The MAIN representation type
 * @tparam OTHERS The OTHERS representation types
 */
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
struct iso_info<Iso<MAIN, OTHERS...>> {
    static constexpr bool is_iso = true;         ///< Always true for Iso specialization
    using main_type = MAIN;                      ///< Main field representation type
    using others_tuple = std::tuple<OTHERS...>;  ///< Tuple of other field representation types
};

/**
 * @brief Helper concept to check if a type is distinct from all types in a parameter pack
 * @tparam T Type to check for distinctness
 * @tparam Types Parameter pack to check against
 *
 * This concept evaluates to true if type T is not the same as any of the types in the Types pack.
 * Used internally for validating pairwise distinctness in template parameter lists.
 */
template <typename T, typename... Types>
constexpr bool is_distinct_from_all = (!std::is_same_v<T, Types> && ...);

/**
 * @brief Constexpr function to check pairwise distinctness of all types in a parameter pack
 * @tparam Types Parameter pack of types to check
 * @return true if all types are pairwise distinct, false otherwise
 *
 * This function recursively validates that no two types in the parameter pack are identical.
 * It uses C++20 fold expressions for efficient compile-time evaluation.
 *
 * @code{.cpp}
 * static_assert(pairwise_distinct<F4, F16, F64>(), "All types must be distinct");
 * static_assert(!pairwise_distinct<F4, F4, F16>(), "Duplicate types detected");
 * @endcode
 *
 * @section Algorithm
 *
 * The algorithm works by recursively checking each type against all remaining types:
 * - For `pairwise_distinct<A, B, C>()`: checks A≠B, A≠C, then recurses on `<B, C>`
 * - For `pairwise_distinct<B, C>()`: checks B≠C
 * - Empty/single type packs return true trivially
 */
template <typename... Types>
constexpr bool pairwise_distinct() {
    if constexpr (sizeof...(Types) <= 1) {
        return true;
    } else {
        return []<typename First, typename... Rest>(std::type_identity<First>, std::type_identity<Rest>...) {
            return details::is_distinct_from_all<First, Rest...> && pairwise_distinct<Rest...>();
        }(std::type_identity<Types>{}...);
    }
}

/**
 * @brief Base class to make derived classes non-copyable
 *
 * Classes that inherit from NonCopyable cannot be copied but can still be moved.
 * This is useful for resource-managing classes or classes with unique identity.
 */
class NonCopyable {
   protected:
    constexpr NonCopyable() = default;
    ~NonCopyable() = default;

    // Delete copy operations
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;

    // Allow move operations by changing to default
    NonCopyable(NonCopyable&&) = delete;
    NonCopyable& operator=(NonCopyable&&) = delete;
};
}  // namespace details

}  // namespace CECCO

#endif