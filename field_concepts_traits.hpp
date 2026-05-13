/**
 * @file field_concepts_traits.hpp
 * @brief Concepts, traits, and type utilities for finite field arithmetic
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.1.5
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
 * Conceptual foundation for the finite field arithmetic library: C++20 concepts and traits that
 * constrain field types and let cross-field operations decide compatibility at compile time.
 *
 * Concepts visible to library users: @ref CECCO::FieldType, @ref CECCO::FiniteFieldType,
 * @ref CECCO::SubfieldOf / @ref CECCO::ExtensionOf, @ref CECCO::Isomorphic,
 * @ref CECCO::ComponentType (the umbrella for @ref CECCO::Vector / @ref CECCO::Polynomial /
 * @ref CECCO::Matrix component types), @ref CECCO::ReliablyComparableType, @ref CECCO::SignedIntType.
 *
 * Internals (in @c CECCO::details): metafunctions that compute field-tower relationships
 * (@c is_subfield_of, @c collect_subfields, @c largest_common_subfield_t) and
 * @c iso_info for Iso introspection.
 *
 * @see @ref fields.hpp for the field classes (Fp, Ext, Iso, Rationals) that satisfy these concepts
 */

#ifndef FIELD_CONCEPTS_TRAITS_HPP
#define FIELD_CONCEPTS_TRAITS_HPP

#include <complex>
#include <concepts>

#include "helpers.hpp"
/*
//transitive
#include <string>
#include <type_traits>

#include "InfInt.hpp"
*/

namespace CECCO {

/**
 * @brief LUT generation mode for field operations
 */
enum class LutMode {
    CompileTime,  ///< Generate LUTs at compile-time using constexpr (default)
    RunTime       ///< Generate LUTs at runtime using lazy initialization
};

namespace details {
// Forward declaration for Base (full definition in fields.hpp)
class Base;
}  // namespace details

/**
 * @concept FieldType
 * @brief Concept for field types: full algebraic interface
 *
 * @tparam T Candidate type
 *
 * Requires: compound assignments `+=`, `−=`, `*=`, `/=`; equality `==`; unary `−`; default,
 * copy, move, and `int` construction; assignment from `T` and from `int`; the property
 * queries `is_zero()`, `has_positive_sign()`, `get_characteristic()`, `get_info()`; and the
 * randomization methods `randomize()` / `randomize_force_change()`. The binary `+`, `−`, `*`,
 * `/` come for free from the CRTP operators in @ref fields.hpp.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * template <FieldType F>
 * F discriminant(const F& a, const F& b, const F& c) {
 *     return b * b - F(4) * a * c;
 * }
 * @endcode
 *
 * @note Satisfied by @ref CECCO::Rationals, @ref CECCO::Fp, @ref CECCO::Ext, and @ref CECCO::Iso.
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
 * @brief Refines @ref FieldType for finite fields 𝔽_{p^m}
 *
 * @tparam T Candidate type
 *
 * Adds: prime characteristic ≥ 2; field-structure queries `get_size()`, `get_m()`, `get_p()`,
 * `get_q()`; static generator access via `T::get_generator()`; element-order queries
 * `get_multiplicative_order()` and `get_additive_order()`.
 *
 * Cross-field constructors (embeddings between subfield and superfield) are expected but not
 * enforced by the concept — they live in the concrete classes.
 *
 * @note Satisfied by @ref CECCO::Fp, @ref CECCO::Ext, @ref CECCO::Iso. Not by @ref CECCO::Rationals
 * (characteristic 0).
 */

template <typename T>
concept FiniteFieldType = FieldType<T> && requires(const T& t) {
    requires(T::get_characteristic() > 1);
    requires is_prime(T::get_p());

    { t.get_size() } -> std::convertible_to<size_t>;
    { t.get_m() } -> std::convertible_to<size_t>;
    { t.get_p() } -> std::convertible_to<size_t>;
    { t.get_q() } -> std::convertible_to<size_t>;

    { T::get_generator() } -> std::same_as<T>;

    { t.get_multiplicative_order() } -> std::convertible_to<size_t>;
    { t.get_additive_order() } -> std::convertible_to<size_t>;
};

/**
 * @concept SignedIntType
 * @brief Standard signed integers or @c InfInt for arbitrary precision
 *
 * @tparam T Candidate type
 *
 * Satisfied by any type with `std::is_integral_v<T> && std::is_signed_v<T>` (i.e. `int`,
 * `long`, `long long`, …) and by @c InfInt. Used by @ref CECCO::Rationals so callers can
 * trade speed for unbounded precision in numerator and denominator.
 */
template <typename T>
concept SignedIntType = (std::is_integral_v<T> && std::is_signed_v<T>) || std::is_same_v<T, InfInt>;

// Forward declaration for Rationals (full definition in fields.hpp)
template <SignedIntType T>
class Rationals;

/**
 * @concept ReliablyComparableType
 * @brief Types whose `operator==` reflects mathematical equality
 *
 * @tparam T Candidate type
 *
 * Satisfied by @ref FiniteFieldType, @ref SignedIntType, and `Rationals<InfInt>`. Excludes
 * `double` and `std::complex<double>` because rounding makes comparison unreliable. Used by
 * algorithms that need stable equality (e.g. Hamming weight, structural matrix tests).
 */
template <typename T>
concept ReliablyComparableType = FiniteFieldType<T> || std::is_same_v<T, Rationals<InfInt>> || SignedIntType<T>;

/**
 * @concept ComponentType
 * @brief Admissible component type for @ref CECCO::Vector, @ref CECCO::Polynomial, @ref CECCO::Matrix
 *
 * @tparam T Candidate type
 *
 * Satisfied by any @ref FieldType, by `double`, by `std::complex<double>`, and by any
 * @ref SignedIntType. This is the umbrella concept used to template the linear-algebra and
 * polynomial classes.
 */
template <typename T>
concept ComponentType =
    FieldType<T> || std::same_as<T, double> || std::same_as<T, std::complex<double>> || SignedIntType<T>;

/**
 * @concept BelongsTo
 * @brief T is identical to at least one of `Types...`
 *
 * @tparam T Candidate type
 * @tparam Types Parameter pack to test against
 *
 * Used to constrain @ref CECCO::Iso operators to its `OTHERS...` representations:
 *
 * @code{.cpp}
 * template <BelongsTo<OTHERS...> OTHER>
 * Iso& operator=(const OTHER& other);
 * @endcode
 */
template <typename T, typename... Types>
concept BelongsTo = (std::is_same_v<T, Types> || ...);

/**
 * @def MOD
 * @brief Alias for `std::array`, used to spell modulus polynomials in @ref CECCO::Ext
 *
 * Lets `Ext<F2, {1, 1, 1}>` deduce the array length from the brace-initialiser instead of
 * requiring `std::array<int, 3>{…}`.
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
 * @brief A and B are finite fields of the same size (and thus isomorphic)
 *
 * @tparam A First field type
 * @tparam B Second field type
 *
 * Two finite fields of the same cardinality are isomorphic. The check is on size only — the
 * explicit isomorphism (a concrete field homomorphism) is computed by @ref CECCO::Isomorphism.
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F2   = Fp<2>;
 * using F4_a = Ext<F2, {1, 1, 1}>;     // 𝔽₄ via x² + x + 1
 * using F4_b = Ext<F2, {1, 0, 1}>;     // 𝔽₄ via x² + 1
 * using F8   = Ext<F2, {1, 1, 0, 1}>;
 * using F64_a = Ext<F8,   {7, 1, 1}>;
 * using F64_b = Ext<F4_a, {1, 2, 0, 1}>;
 *
 * static_assert(Isomorphic<F64_a, F64_b>);  // same size, different constructions
 * static_assert(Isomorphic<F4_a, F4_b>);
 * static_assert(!Isomorphic<F4_a, F8>);     // 4 ≠ 8
 * @endcode
 */
template <typename A, typename B>
concept Isomorphic = FiniteFieldType<A> && FiniteFieldType<B> && requires { requires A::get_size() == B::get_size(); };

namespace details {

/**
 * @brief Total extension degree of T over its prime field
 *
 * @tparam T Finite-field type
 *
 * Walks the construction tower and multiplies the local extension degree at each level. For
 * `F256 = Ext<F16, …>`, `F16 = Ext<F4, …>`, `F4 = Ext<F2, …>` the result is 2 · 2 · 2 = 8.
 * For an `Iso`, the value is taken from the MAIN representation (all are isomorphic).
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
 * @brief Construction-based subfield test: SUBFIELD reachable by descending SUPERFIELD's tower
 *
 * @tparam SUPERFIELD Finite-field type
 * @tparam SUBFIELD   Finite-field type
 *
 * The relationship is recorded by **construction**, not by mathematical containment. If `F16` is
 * built directly as `Ext<F2, …>`, then `is_subfield_of_v<F16, F4>` is `false` even though
 * mathematically 𝔽₄ ⊆ 𝔽₁₆ — `F4` simply does not appear in `F16`'s construction. This is the
 * primitive that backs the user-facing @ref CECCO::SubfieldOf concept.
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

// Iso super, non-Iso sub: SUB ⊆ Iso if SUB is a subfield of any component (covers reflexive
// SUB == MAIN / SUB == one of OTHERS via the per-component reflexive specializations below).
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS, FiniteFieldType SUB>
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
 * @brief SUBFIELD ⊆ SUPERFIELD as constructed in this library (Iso paths included)
 *
 * @tparam SUPERFIELD Larger field
 * @tparam SUBFIELD   Candidate smaller field
 *
 * Walks the construction tower of SUPERFIELD; for `Iso` operands, every representation is
 * inspected, so a subfield reachable through any of MAIN / OTHERS counts. Reflexive:
 * `SubfieldOf<F, F>` is always true.
 *
 * @warning This is the construction tower, not the mathematical tower. A field that exists
 * mathematically between SUBFIELD and SUPERFIELD but was never constructed (or never bridged
 * by an `Iso`) is invisible. Construct an `Iso` over equivalent representations to merge towers.
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F2     = Fp<2>;
 * using F4_a   = Ext<F2, {1, 1, 1}>;
 * using F4_b   = Ext<F2, {1, 0, 1}>;
 * using F4_Iso = Iso<F4_a, F4_b>;
 * using F16    = Ext<F4_Iso, {2, 1, 1}>;
 *
 * static_assert(SubfieldOf<F16, F4_Iso>);  // direct construction
 * static_assert(SubfieldOf<F16, F4_a>);    // via Iso: F4_a in F4_Iso ⊆ F16
 * static_assert(SubfieldOf<F16, F2>);      // recursive descent
 * @endcode
 */
template <typename SUPERFIELD, typename SUBFIELD>
concept SubfieldOf = details::is_subfield_of_v<SUPERFIELD, SUBFIELD>;

/**
 * @concept ExtensionOf
 * @brief Inverse of @ref SubfieldOf: `ExtensionOf<S, E>` iff `SubfieldOf<E, S>`
 *
 * @tparam S Smaller (base) field
 * @tparam E Candidate larger field
 *
 * Same construction-vs-mathematical caveat as @ref SubfieldOf. Reflexive.
 */
template <typename S, typename E>
concept ExtensionOf = SubfieldOf<E, S>;

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
 * @brief All subfields of F (including F itself), as a @ref type_list
 *
 * @tparam F Finite-field type
 *
 * Recurses through the construction tower: `Fp<p>` is the base case, `Ext<B, …>` extends
 * `collect_subfields_t<B> ∪ {B}` with itself, and `Iso<…>` unions the subfield lists of all
 * its representations. Used by @ref largest_common_subfield_t.
 */
template <typename F>
struct collect_subfields;

// Specialization for prime fields - base case
template <uint16_t p>
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
 * @brief Largest field that appears as a subfield of both F and G
 *
 * @tparam F Finite-field type
 * @tparam G Finite-field type (same characteristic as F; checked by `static_assert`)
 *
 * Computes the bridge field used by the cross-field constructors of @ref CECCO::Ext and
 * @ref CECCO::Iso. Procedure: intersect `collect_subfields_t<F>` with `collect_subfields_t<G>`
 * and pick the largest by `get_size()`. As a tiebreaker, if that largest field is isomorphic
 * to an `Iso` operand, the `Iso` itself is returned instead — this preserves the merged-tower
 * structure that the user explicitly built.
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * using F3   = Fp<3>;
 * using F9   = Ext<F3, {2, 2, 1}>;
 * using F81a = Ext<F3, {2, 1, 0, 0, 1}>;   // direct F3 → F81
 * using F81b = Ext<F9, {6, 0, 1}>;         // F3 → F9 → F81
 * using F27  = Ext<F3, {1, 2, 0, 1}>;
 *
 * using c1 = largest_common_subfield_t<F81a, F81b>;  // F9
 * using c2 = largest_common_subfield_t<F81a, F27>;   // F3
 * @endcode
 */
template <typename F, typename G>
struct largest_common_subfield {
    static_assert(F::get_characteristic() == G::get_characteristic(), "Fields must have the same characteristic");

   private:
    using subfields_F = collect_subfields_t<F>;
    using subfields_G = collect_subfields_t<G>;
    using common_subfields = intersect_type_lists_t<subfields_F, subfields_G>;
    using raw_largest = largest_field_in_list_t<common_subfields>;

    // Detect Iso types
    template <typename T>
    struct is_iso : std::false_type {};
    template <typename MAIN, typename... OTHERS>
    struct is_iso<Iso<MAIN, OTHERS...>> : std::true_type {};

    template <typename Largest, typename Param1, typename Param2>
    using prefer_iso =
        std::conditional_t<is_iso<Param1>::value && Isomorphic<Largest, Param1>, Param1,
                           std::conditional_t<is_iso<Param2>::value && Isomorphic<Largest, Param2>, Param2, Largest>>;

   public:
    using type = prefer_iso<raw_largest, F, G>;
};

template <typename F, typename G>
using largest_common_subfield_t = typename largest_common_subfield<F, G>::type;

/**
 * @brief Detect whether T is an `Iso` and expose its components
 *
 * @tparam T Any type
 *
 * Primary template: `is_iso = false`, `main_type = void`. The specialization for
 * `Iso<MAIN, OTHERS...>` (below) sets `is_iso = true`, `main_type = MAIN`, and packages
 * `OTHERS...` into `others_tuple`.
 */
template <typename T>
struct iso_info {
    static constexpr bool is_iso = false;
    using main_type = void;
};

/// @brief Specialization of @ref iso_info for `Iso<MAIN, OTHERS...>`
template <FiniteFieldType MAIN, FiniteFieldType... OTHERS>
struct iso_info<Iso<MAIN, OTHERS...>> {
    static constexpr bool is_iso = true;
    using main_type = MAIN;
    using others_tuple = std::tuple<OTHERS...>;
};

/// @brief True iff T is not the same as any of `Types...`
template <typename T, typename... Types>
constexpr bool is_distinct_from_all = (!std::is_same_v<T, Types> && ...);

/**
 * @brief True iff all types in the pack are pairwise distinct
 *
 * @tparam Types Parameter pack
 * @return `true` if no two types in `Types...` are identical
 *
 * Used by @ref CECCO::Iso to reject duplicate representations.
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
 * @brief Mixin that deletes the copy operations and defaults the move operations
 *
 * Inherit (protected) to mark a class movable but non-copyable.
 */
class NonCopyable {
   protected:
    constexpr NonCopyable() = default;
    ~NonCopyable() = default;

    // Delete copy operations
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;

    // Allow move operations by changing to default
    NonCopyable(NonCopyable&&) = default;
    NonCopyable& operator=(NonCopyable&&) = default;
};
}  // namespace details

}  // namespace CECCO

#endif