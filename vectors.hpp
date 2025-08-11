/**
 * @file vectors.hpp
 * @brief Vector arithmetic library
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
 * @section Description
 *
 * This header file provides a complete implementation of vector arithmetic for
 * error control coding. It supports:
 *
 * - **Generic vector operations**: Over any @ref ECC::ComponentType including finite fields,
 *   floating-point numbers, complex numbers, and signed integers
 * - **Cross-field constructors**: Safe conversions between vectors over related fields using
 *   @ref ECC::SubfieldOf, @ref ECC::ExtensionOf, and @ref ECC::largest_common_subfield_t
 * - **Matrix integration**: Bidirectional conversion Vector -> Matrix -> Vector
 * - **Performance optimizations**: High-performance O(1) caching, move semantics, and STL algorithm utilization
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
 * // Cross-field operations (field tower compatibility)
 * using F2 = Fp<2>;                        // note: used in construction of F4
 * Vector<F2> y = {1, 0, 1, 1};
 * Vector<F4> z(y);                         // Safe upcast: F₂ ⊆ F₄ (construction tower)
 * auto M = extension_vec.as_matrix<F2>();  // Convert to matrix over subfield
 * @endcode
 *
 * @section Performance_Features
 *
 * - **Lazy evaluation**: Hamming weight and burst length computed on-demand with compile-time optimized caching
 * - **Move semantics**: Optimal performance for temporary vector operations
 * - **STL integration**: Uses standard algorithms for optimal compiler optimization
 * - **Type safety**: C++20 concepts prevent invalid operations:
 *   - @ref ECC::ComponentType Ensures valid component types
 *   - @ref ECC::SubfieldOf Validates field relationship for safe conversions
 *   - @ref ECC::largest_common_subfield_t Enables generalized cross-field conversions
 *
 * @see @ref fields.hpp for fields and field arithmetic
 * @see @ref matrices.hpp for matrices and linear algebra
 * @see @ref field_concepts_traits.hpp for type constraints and field relationships (C++20 concepts)
 */

#ifndef VECTORS_HPP
#define VECTORS_HPP

// #include <algorithm> // transitive through matrices.hpp
// #include <complex> // transitive through matrices.hpp
#include <initializer_list>
// #include <iostream> // transitive through matrices.hpp
#include <numeric>
// #include <ranges> // transitive through matrices.hpp
// #include <set> // transitive through matrices.hpp
#include <unordered_set>
// #include <vector> // transitive through matrices.hpp

// #include "InfInt.hpp" // transitive through matrices.hpp
// #include "helpers.hpp" // transitive through matrices.hpp
// #include "field_concepts_traits.hpp" // transitive through matrices.hpp
#include "matrices.hpp"

namespace ECC {

template <ComponentType T>
class Vector;
template <ComponentType T>
class Polynomial;
template <ComponentType T>
class Matrix;

namespace details {
template <FiniteFieldType T>
struct FiniteFieldHasher;
}  // namespace details

template <ComponentType T>
T inner_product(const Vector<T>& lhs, const Vector<T>& rhs);
template <ComponentType T>
Vector<T> unit_vector(size_t length, size_t i);
template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& rhs) noexcept;
double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs);

/**
 * @class Vector
 * @brief Generic vector class for error control coding (ECC) and finite field applications
 *
 * @tparam T Component type satisfying @ref ECC::ComponentType concept. Supported types include:
 *   - **Finite field types**: @ref ECC::Fp, @ref ECC::Ext satisfying concept @ref ECC::FiniteFieldType
 *   - **Floating-point types**: `double` etc.
 *   - **Complex types**: `std::complex<double>` etc.
 *   - **Signed integer types**: Signed integer types including `InfInt` satisfying concept @ref ECC::SignedIntType
 *
 * This class provides comprehensive vector operations optimized for error control coding (ECC),
 * with special support for finite field arithmetic and cross-field conversions. The design
 * emphasizes performance through caching, move semantics, and STL algorithm utilization.
 *
 * @section Implementation_Notes
 *
 * - **Cross-field compatibility**: Safe conversions between related field types using concepts
 * - **ECC-specific operations**: Hamming weight, Hamming distance, burst length calculations
 * - **Performance optimization**: Lazy evaluation with compile-time optimized O(1) caching for expensive operations
 * - **Type safety**: Compile-time validation of field relationships and operations
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * // Create vectors over finite fields
 * using F4 = Ext<Fp<2>, MOD{1, 1, 1}>;
 * Vector<F4> v = {0, 1, 2, 3};
 *
 * // ECC operations
 * size_t weight = v.wH();           // Hamming weight
 * size_t burst = v.burst_length();  // Burst error length
 * // etc.
 *
 * // Cross-field operations (F₂ ⊆ F₄)
 * auto w = v.as_matrix<Fp<2>>();  // binary 4 x 2 matrix
 * // etc.
 * @endcode
 *
 * @note Vector operations require compatible dimensions. Dimension mismatches
 *          throw std::invalid_argument exceptions.
 *
 * @note Additional methods available via concept constraints:
 *       - **Finite fields only**: as_integer(), as_matrix()
 *       - **Finite fields and signed integers**: wH(), dH() (Hamming weight/distance)
 *
 * @see Concept @ref ECC::ComponentType for supported component types
 * @see Concepts @ref ECC::SubfieldOf, @ref ECC::largest_common_subfield_t for cross-field operation constraints
 * @see @ref ECC::Matrix for matrix representations and linear algebra operations
 */
template <ComponentType T>
class Vector {
    template <ComponentType U>
    friend constexpr bool operator==(const Vector<U>& lhs, const Vector<U>& rhs) noexcept
        requires ReliablyComparableType<U>;
    friend constexpr T inner_product<>(const Vector<T>& lhs, const Vector<T>& rhs);
    friend Vector unit_vector<>(size_t length, size_t i);
    friend std::ostream& operator<< <>(std::ostream& os, const Vector& rhs) noexcept;
    friend double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs);
    friend class Matrix<T>;

   public:
    // Cache configuration for this class
    enum CacheIds { Weight = 0 };

    /**
     * @brief Default constructor creating an empty vector
     *
     * Creates a vector with zero length.
     */
    constexpr Vector() noexcept : data(0) {}

    /**
     * @brief Constructs a vector of specified length with default-initialized components
     *
     * @param n Length of the vector to create
     *
     * Creates a vector with `n` components, each initialized to `T()` (zero for most types).
     * For vectors over finite fields, components are initialized to the additive identity.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector(size_t n) : data(n) {}

    /**
     * @brief Constructs a vector with all components set to a specific value
     *
     * @param n Length of the vector to create
     * @param l Value to assign to all components
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector(size_t n, const T& l);

    /**
     * @brief Constructs a vector from an initializer list
     *
     * @param l Initializer list containing the vector components
     *
     * Enables convenient vector initialization syntax:
     * @code{.cpp}
     * Vector<int> v{1, 2, 3, 4};
     * Vector<Fp<2>> w{0, 1, 1, 0};
     * @endcode
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector(const std::initializer_list<T>& l) : data(l) {}

    /**
     * @brief Copy constructor
     *
     * @param other Vector to copy from
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector(const Vector& other);

    /**
     * @brief Move constructor
     *
     * @param other Vector to move from (left in valid but unspecified state)
     */
    constexpr Vector(Vector&& other) noexcept;

    /**
     * @brief Cross-field constructor from matrix over subfield
     *
     * @tparam S Source field type that must be a subfield (from construction tower) of T
     * @param mat Matrix over subfield S to convert to vector over extension field T
     *
     * Constructs a vector by interpreting each column of the matrix as an element
     * of the extension field T. Requires that T::get_m() == mat.get_m().
     *
     * @throws std::invalid_argument if matrix dimensions are incompatible with field structure
     * @throws std::bad_alloc if memory allocation fails
     */
    template <FiniteFieldType S>
    Vector(const Matrix<S>& mat)
        requires FiniteFieldType<T> && ExtensionOf<S, T>;

    /**
     * @brief Cross-field copy constructor between fields with the same characteristic
     *
     * @tparam S Source field type that must have the same characteristic as T
     * @param other Vector over field S to convert
     *
     * Safely converts vectors between any fields with the same characteristic using
     * @ref largest_common_subfield_t as the conversion bridge. Supports conversions across
     * different field towers, not just within the same construction hierarchy.
     *
     * @throws std::invalid_argument if field components cannot be represented in target field (downcasting not
     * possible)
     * @throws std::bad_alloc if memory allocation fails
     */
    template <FieldType S>
    Vector(const Vector<S>& other)
        requires FiniteFieldType<S> && FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic());

    /**
     * @brief Constructs vector from polynomial coefficients
     *
     * @param poly Polynomial whose coefficients become vector components
     *
     * Creates a vector where componenti contains the coefficient of x^i in the polynomial.
     * The vector length equals poly.degree() + 1.
     *
     * @note If you need cross-field polynomial <-> vector conversion then first convert poly into the the other field.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector(const Polynomial<T>& poly);

    /** @name Assignment Operators
     * @{
     */

    /**
     * @brief Copy assignment operator
     *
     * @param rhs Vector to copy from
     * @return Reference to this vector after assignment
     */
    constexpr Vector& operator=(const Vector& rhs);

    /**
     * @brief Move assignment operator
     *
     * @param rhs Vector to move from (left in valid but unspecified state)
     * @return Reference to this vector after assignment
     */
    constexpr Vector& operator=(Vector&& rhs) noexcept;

    /**
     * @brief Scalar assignment operator
     *
     * @param rhs Value to assign to all components
     * @return Reference to this vector after assignment
     *
     * Sets all vector components to the specified value.
     */
    constexpr Vector& operator=(const T& rhs) noexcept;

    /**
     * @brief Cross-field assignment operator between fields with the same characteristic
     *
     * @tparam S Source field type that must have the same characteristic as T
     * @param rhs Vector over field S to convert
     * @return Reference to this vector after assignment
     *
     * Safely converts vectors between any fields with the same characteristic using
     * @ref largest_common_subfield_t as the conversion bridge. Supports conversions across
     * different field towers, not just within the same construction hierarchy.
     *
     * @throws std::invalid_argument if field components cannot be represented in target field (downcasting not
     * possible)
     * @throws std::bad_alloc if memory allocation fails
     */
    template <FieldType S>
    Vector& operator=(const Vector<S>& other)
        requires FiniteFieldType<S> && FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic());

    /** @} */

    /** @name Unary Arithmetic Operations
     * @{
     */

    /**
     * @brief Unary plus operator for lvalue references (identity)
     *
     * @return Copy of this vector (mathematical identity operation)
     */
    constexpr Vector operator+() const& noexcept { return *this; }

    /**
     * @brief Unary plus operator for rvalue references (move optimization)
     *
     * @return This vector moved (mathematical identity operation)
     */
    constexpr Vector operator+() && noexcept { return std::move(*this); }

    /**
     * @brief Unary minus operator for lvalue references
     *
     * @return New vector with all components negated
     */
    constexpr Vector operator-() const& noexcept;

    /**
     * @brief Unary minus operator for rvalue references (move optimization)
     *
     * @return This vector with all components negated in-place
     */
    constexpr Vector operator-() && noexcept;

    /** @} */

    /** @name Compound Assignment Operations
     * @{
     */

    /**
     * @brief Vector addition assignment
     *
     * @param rhs Vector to add to this vector
     * @return Reference to this vector after addition
     *
     * Performs component-wise addition: this[i] += rhs[i] for all valid indices.
     *
     * @throws std::invalid_argument if vectors have different lengths
     */
    Vector& operator+=(const Vector& rhs);

    /**
     * @brief Vector subtraction assignment
     *
     * @param rhs Vector to subtract from this vector
     * @return Reference to this vector after subtraction
     *
     * Performs component-wise subtraction: this[i] -= rhs[i] for all valid indices.
     *
     * @throws std::invalid_argument if vectors have different lengths
     */
    Vector& operator-=(const Vector& rhs);

    /**
     * @brief Scalar multiplication assignment
     *
     * @param s Scalar value to multiply with
     * @return Reference to this vector after multiplication
     *
     * Multiplies each component by the scalar: this[i] *= s for all indices.
     */
    constexpr Vector& operator*=(const T& s) noexcept;

    /**
     * @brief Scalar division assignment
     *
     * @param s Scalar value to divide by
     * @return Reference to this vector after division
     *
     * Divides each component by the scalar: this[i] /= s for all indices.
     *
     * @throws std::invalid_argument if attempting to divide by zero
     *
     * @warning Reliable results ( (v / s) *  s == v for a vector v and nonzero scalar s are only guaranteed in case T
     * fulfills concept FieldType<T>
     */
    Vector& operator/=(const T& s);

    /** @} */

    /** @name Randomization
     * @{
     */

    /**
     * @brief Randomize all vector components
     * @return Reference to this vector after randomization
     *
     * Fills the vector with random values appropriate for the component type:
     * - **Finite fields**: Using corresponding randomize member
     * - **Signed integers**: Uniform random values in range [-100, 100]
     * - **Complex numbers**: Real and imaginary part uniform random values in range [-1.0, 1.0]
     * - **Double**: Uniform random values in range [-1.0, 1.0]
     *
     * Uses the global random number generator from helpers.hpp.
     */
    Vector& randomize() noexcept;

    /** @} */

    /** @name Information and Properties
     * @{
     */

    /**
     * @brief Get the vector length
     *
     * @return Number of components in the vector
     */
    constexpr size_t get_n() const noexcept { return data.size(); }

    /**
     * @brief Check if vector is empty
     *
     * @return true if vector has zero length, false otherwise
     */
    constexpr bool is_empty() const noexcept { return data.size() == 0; }

    /**
     * @brief Check if vector is the zero vector
     *
     * @return true if all components equal T(0), false otherwise
     *
     * @note Being the zero vector implies being non-empty.
     *
     * @note Only for types fulfilling ECC::ReliablyComparableType.
     */
    constexpr bool is_zero() const noexcept
        requires ReliablyComparableType<T>;

    /**
     * @brief Check if all components are pairwise distinct
     *
     * @return true if no two components have the same value, false otherwise
     *
     * @note Only for types fulfilling ECC::ReliablyComparableType.
     */
    constexpr bool is_pairwisedistinct() const
        requires ReliablyComparableType<T>;

    /**
     * @brief Compute Hamming weight (number of non-zero components) for discrete types
     *
     * @return Number of non-zero components
     *
     * Uses lazy evaluation with O(1) compile-time optimized caching for optimal performance on repeated calls.
     *
     * @note Only for types fulfilling ECC::ReliablyComparableType.
     */
    size_t wH() const noexcept
        requires ReliablyComparableType<T>
    {
        return cache.template get_or_compute<Weight>([this] { return calculate_weight(); });
    }

    /**
     * @brief Compute burst length (minimum interval containing all non-zero components)
     *
     * @return Length of burst
     *
     * For a vector with first non-zero componentat position L and last at position R,
     * returns R - L + 1. Returns 0 for the empty vector and the zero vector.
     */
    constexpr size_t burst_length() const noexcept;

    /**
     * @brief Compute cyclic burst length (burst length with wraparound)
     *
     * @return Length of cyclic burst
     *
     * Considers the vector as circular, finding the shortest arc that contains all
     * non-zero components. Returns 0 for the empty vector and the zero vector.
     */
    constexpr size_t cyclic_burst_length() const noexcept;

    /** @} */

    /** @name Component Access and Manipulation
     * @{
     */

    /**
     * @brief Set component value by copy
     *
     * @param i Index of component to modify (0-based)
     * @param c New value to assign
     * @return Reference to this vector after setting component
     *
     * Updates the componentat position i.
     *
     * @throws std::invalid_argument if index is out of bounds
     */
    /**
     * @brief Set component value using perfect forwarding
     *
     * @tparam U Type that can be converted to T
     * @param i Index of component to set (0-based)
     * @param c Value to forward into the component
     * @return Reference to this vector after modification
     *
     * Efficiently forwards the value into the specified component position.
     * Handles both lvalue and rvalue references optimally.
     *
     * @throws std::invalid_argument if index i is out of bounds
     */
    template <typename U>
    Vector& set_component(size_t i, U&& c)
        requires std::convertible_to<std::decay_t<U>, T>;

    /**
     * @brief Access component by index (const version)
     *
     * @param i Index of component to access
     * @return Const reference to the componentat position i
     *
     * Provides safe, bounds-checked (read-opnly) access to vector components.
     *
     * @throws std::invalid_argument if index is out of bounds
     */
    const T& operator[](size_t i) const;

    /**
     * @brief Extract subvector by copy (lvalue version)
     *
     * @param i Starting index for extraction
     * @param w Width (number of components) to extract
     * @return New vector containing components [i, i+w)
     *
     * Creates a new vector from a contiguous subsequence of this vector.
     *
     * @throws std::invalid_argument if [i, i+w) extends beyond vector bounds
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector get_subvector(size_t i, size_t w) const&;

    /**
     * @brief Extract subvector in-place (rvalue version)
     *
     * @param i Starting index for extraction
     * @param w Width (number of components) to keep
     * @return Reference to this vector, modified to contain only components [i, i+w)
     *
     * Efficient version that modifies this vector when called on temporaries.
     *
     * @throws std::invalid_argument if [i, i+w) extends beyond vector bounds
     */
    Vector& get_subvector(size_t i, size_t w) &&;

    /**
     * @brief Replace subvector with another vector (copy version)
     *
     * @param v Vector to copy into this vector
     * @param i Starting position for replacement
     * @return Reference to this vector after modification
     *
     * Replaces components [i, i+v.get_n()) with components from vector v.
     *
     * @throws std::invalid_argument if replacement would extend beyond vector bounds
     */
    Vector& set_subvector(const Vector& v, size_t i);

    /**
     * @brief Replace subvector with another vector (move version)
     *
     * @param v Vector to move into this vector
     * @param i Starting position for replacement
     * @return Reference to this vector after modification
     *
     * More efficient version for temporary vectors.
     *
     * @throws std::invalid_argument if replacement would extend beyond vector bounds
     */
    Vector& set_subvector(Vector&& v, size_t i);

    /**
     * @brief Append another vector to the end
     *
     * @param rhs Vector to append (on the right)
     * @return Reference to this vector after appending
     *
     * Extends this vector by concatenating rhs to the end.
     * The result has length get_n() + rhs.get_n().
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    /**
     * @brief Append another vector to the end using perfect forwarding
     *
     * @tparam U Type that can be converted to Vector
     * @param rhs Vector to append (on the right)
     * @return Reference to this vector after appending
     *
     * Extends this vector by concatenating rhs to the end.
     * Handles both lvalue and rvalue references optimally.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    template <typename U>
    Vector& append(U&& rhs)
        requires std::convertible_to<std::decay_t<U>, Vector>;

    /**
     * @brief Prepend another vector to the beginning using perfect forwarding
     *
     * @tparam U Type that can be converted to Vector
     * @param lhs Vector to prepend (on the left)
     * @return Reference to this vector after prepending
     *
     * Extends this vector by concatenating lhs to the beginning.
     * Handles both lvalue and rvalue references optimally.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    template <typename U>
    Vector& prepend(U&& lhs)
        requires std::convertible_to<std::decay_t<U>, Vector>;

    /**
     * @brief Delete specified components from the vector
     *
     * @param v Vector of indices to delete (automatically deduplicated)
     * @return Reference to this vector after deletion
     *
     * Removes components at the specified indices, compacting the remaining components.
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     * @throws std::bad_alloc if memory allocation fails during compaction
     */
    Vector& delete_components(const std::vector<size_t>& v);

    /**
     * @brief Delete specified component from the vector
     *
     * @param i Index of component to delete
     * @return Reference to this vector after deletion
     *
     * Removes component at the specified index, compacting the remaining components.
     *
     * @throws std::invalid_argument if any index index iis out of bounds
     * @throws std::bad_alloc if memory allocation fails during compaction
     */
    Vector& delete_component(size_t i) { return delete_components({i}); }

    /**
     * @brief Erases specified components from the vector (flags them as erasures)
     *
     * @param v Vector of indices to erase (automatically deduplicated)
     * @return Reference to this vector after erasing
     *
     * Erases components at the specified indices.
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     *
     * @note Only available for field types (since erasure flag/erase() is required)
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Vector& erase_components(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Erases specified component from the vector (flags it as erasure)
     *
     * @param i Index of component to erase
     * @return Reference to this vector after erasing
     *
     * Erases component at the specified index.
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     *
     * @note Only available for field types (since erasure flag/erase() is required)
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Vector& erase_component(size_t i)
        requires FieldType<T>
    {
        return erase_components({i});
    }

    /**
     * @brief Un-erases specified components from the vector (removes the erasure flag))
     *
     * @param v Vector of indices to un-erase (automatically deduplicated)
     * @return Reference to this vector after un-erasing
     *
     * Un-erases components at the specified indices and sets them to 0.
     *
     * @note Only available for field types (since erasure flag/unerase() is required)
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Vector& unerase_components(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Un-erases specified component from the vector (flags it as erasure)
     *
     * @param i Index of component to un-erase
     * @return Reference to this vector after un-erasing
     *
     * Erases component at the specified index and sets it to 0.
     *
     * @note Only available for field types (since erasure flag/unerase() is required)
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Vector& unerase_component(size_t i)
        requires FieldType<T>
    {
        return unerase_components({i});
    }

    /**
     * @brief Pad vector to specified length with zeros at front
     *
     * @param n Target length for the vector
     * @return Reference to this vector after padding
     *
     * If current length >= n, vector remains unchanged.
     * If current length < n, zeros are prepended to reach target length.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector& pad_front(size_t n);

    /**
     * @brief Pad vector to specified length with zeros at back
     *
     * @param n Target length for the vector
     * @return Reference to this vector after padding
     *
     * If current length >= n, vector remains unchanged.
     * If current length < n, zeros are appended to reach target length.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector& pad_back(size_t n);

    /** @} */

    /** @name Transformations
     * @{
     */

    /**
     * @brief Rotate vector components to the left
     *
     * @param i Number of positions to rotate left
     * @return Reference to this vector after rotation
     *
     * Performs circular left shift: componentat position j moves to position (j-i) mod n.
     */
    constexpr Vector& rotate_left(size_t i) noexcept;

    /**
     * @brief Rotate vector components to the right
     *
     * @param i Number of positions to rotate right
     * @return Reference to this vector after rotation
     *
     * Performs circular right shift: componentat position j moves to position (j+i) mod n.
     * Equivalent to rotate_left(get_n() - i).
     */
    constexpr Vector& rotate_right(size_t i) noexcept;

    /**
     * @brief Reverse the order of all components
     *
     * @return Reference to this vector after reversal
     *
     * Component at position i moves to position (n-1-i).
     */
    constexpr Vector& reverse() noexcept;

    /**
     * @brief Fill all vector components with specified value
     *
     * @param value Value to assign to all components
     * @return Reference to this vector after filling
     *
     * Sets every component to the specified value..
     */
    constexpr Vector& fill(const T& value) noexcept;

    /** @} */

    /** @name Finite Field Specific Operations
     * @{
     */

    /**
     * @brief Interpret vector as a single integer (finite fields only)
     *
     * @return Integer representation treating vector components as base-q digits
     *
     * Converts the vector to an integer using positional notation with base q = T::get_size().
     * Component i contributes data[n-1-i] * q^i to the result.
     *
     * @note Result may overflow for large vectors; use with caution
     * @note Available only for finite field types.
     */
    size_t as_integer() const noexcept
        requires FiniteFieldType<T>;

    /**
     * @brief Convert to matrix over subfield (finite fields only)
     *
     * @tparam S Subfield type satisfying SubfieldOf<T, S>
     * @return Matrix where each column represents one vector component over the subfield
     *
     * Converts each extension field componentto its vector representation over the subfield S,
     * arranging these as columns of a matrix.
     *
     * @throws std::bad_alloc if memory allocation fails
     *
     * @note Matrix dimensions are [Ext : Fp<p>] * get_n(), where [Ext : Fp<p>] = extension degree of Ext over Fp<p>.
     * @note If at the j-th component of this has the erasure flag, then the whole j-th column of the returned matrix
     * has the erasure flag.
     */
    template <FiniteFieldType S>
    Matrix<S> as_matrix() const noexcept
        requires FiniteFieldType<T> && SubfieldOf<T, S> && (!std::is_same_v<T, S>);

    /** @} */

   private:
    std::vector<T> data;

    /// High-performance O(1) cache for expensive operations (Hamming weight, etc.) - uses compile-time optimized array
    /// storage
    mutable details::Cache<details::CacheEntry<Weight, size_t>> cache;

    constexpr size_t calculate_weight() const noexcept
        requires ReliablyComparableType<T>;
};

template <ComponentType T>
Vector<T>::Vector(size_t n, const T& l) : data(n) {
    std::fill(data.begin(), data.end(), l);
}

template <ComponentType T>
Vector<T>::Vector(const Vector<T>& other) : data(other.data), cache(other.cache) {}

template <ComponentType T>
constexpr Vector<T>::Vector(Vector<T>&& other) noexcept : data(std::move(other.data)), cache(std::move(other.cache)) {}

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
template <FieldType S>
Vector<T>::Vector(const Vector<S>& other)
    requires FiniteFieldType<S> && FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
{
    data.resize(other.get_n());
    for (size_t i = 0; i < other.get_n(); ++i) {
        data[i] = T(other[i]);  // Uses enhanced cross-field constructors
    }
}

template <ComponentType T>
Vector<T>::Vector(const Polynomial<T>& poly) {
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
    // std::cout << "vector copy assignment" << std::endl;
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::operator=(Vector<T>&& rhs) noexcept {
    if (this == &rhs) return *this;
    data = std::move(rhs.data);
    cache = std::move(rhs.cache);
    // std::cout << "vector move assignment" << std::endl;
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::operator=(const T& rhs) noexcept {
    std::fill(data.begin(), data.end(), rhs);
    cache.invalidate();
    return *this;
}

template <ComponentType T>
template <FieldType S>
Vector<T>& Vector<T>::operator=(const Vector<S>& other)
    requires FiniteFieldType<S> && FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
{
    data.resize(other.get_n());
    for (size_t i = 0; i < other.get_n(); ++i) {
        data[i] = T(other[i]);  // Uses enhanced cross-field constructors
    }
    return *this;
}

template <ComponentType T>
constexpr Vector<T> Vector<T>::operator-() const& noexcept {
    Vector res(*this);
    std::for_each(res.data.begin(), res.data.end(), [](T& v) { v = -v; });
    return res;  // move elision
}

template <ComponentType T>
constexpr Vector<T> Vector<T>::operator-() && noexcept {
    std::for_each(data.begin(), data.end(), [](T& v) { v = -v; });
    cache.invalidate();
    return std::move(*this);
}

template <ComponentType T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& rhs) {
    if (data.size() != rhs.data.size()) throw std::invalid_argument("trying to add vectors of different lengths");
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<T>{});
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& rhs) {
    if (data.size() != rhs.data.size())
        throw std::invalid_argument(
            "trying to subtract vectors of different "
            "lengths");
    std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::minus<T>{});
    cache.invalidate();
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::operator*=(const T& s) noexcept {
    if (s == T(0))
        fill(T(0));
    else
        std::for_each(data.begin(), data.end(), [&s](T& v) { v *= s; });

    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::operator/=(const T& s) {
    if (s == T(0)) throw std::invalid_argument("trying to divide components of vector by zero");
    std::for_each(data.begin(), data.end(), [&s](T& v) { v /= s; });
    return *this;
}

template <ComponentType T>
template <typename U>
Vector<T>& Vector<T>::append(U&& rhs)
    requires std::convertible_to<std::decay_t<U>, Vector>
{
    const Vector<T>& rhs_ref = std::forward<U>(rhs);
    data.reserve(data.size() + rhs_ref.data.size());
    data.insert(data.end(), rhs_ref.data.begin(), rhs_ref.data.end());
    cache.invalidate();
    return *this;
}

template <ComponentType T>
template <typename U>
Vector<T>& Vector<T>::prepend(U&& lhs)
    requires std::convertible_to<std::decay_t<U>, Vector>
{
    const Vector<T>& lhs_ref = std::forward<U>(lhs);
    data.reserve(data.size() + lhs_ref.data.size());
    data.insert(data.begin(), lhs_ref.data.begin(), lhs_ref.data.end());
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::delete_components(const std::vector<size_t>& v) {
    if (v.empty()) return *this;

    // Validate and create sorted set of unique indices (deduplicate)
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= data.size()) throw std::invalid_argument("trying to delete non-existent component");
    }

    // Single-pass filtering: copy components that should be kept
    std::vector<T> new_data;
    new_data.reserve(data.size() - indices.size());

    for (size_t i = 0; i < data.size(); ++i) {
        if (indices.find(i) == indices.end()) {
            new_data.push_back(std::move(data[i]));
        }
    }

    data = std::move(new_data);
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::erase_components(const std::vector<size_t>& v)
    requires FieldType<T>
{
    if (v.empty()) return *this;

    // Validate and create sorted set of unique indices (deduplicate)
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= data.size()) throw std::invalid_argument("trying to erase non-existent component");
    }

    for_each(indices.crbegin(), indices.crend(), [&](auto i) { data[i].erase(); });

    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::unerase_components(const std::vector<size_t>& v)
    requires FieldType<T>
{
    if (v.empty()) return *this;

    // Validate and create sorted set of unique indices (deduplicate)
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= data.size()) throw std::invalid_argument("trying to un-erase non-existent component");
    }

    for_each(indices.crbegin(), indices.crend(), [&](auto i) { data[i].unerase(); });
    ;

    cache.invalidate();
    return *this;
}

/**
 * @brief Delete single component from vector
 * @tparam T Vector component type
 * @param v Source vector
 * @param i Index of component to delete
 * @return New vector with component at index i removed
 */
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

    std::vector<T> new_data(n);
    std::copy(data.begin(), data.end(), new_data.begin() + (n - data.size()));
    data = std::move(new_data);
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::pad_back(size_t n) {
    if (n <= data.size()) return *this;

    data.resize(n);  // Automatically fills with T() (zeros)
    cache.invalidate();
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::fill(const T& value) noexcept {
    std::fill(data.begin(), data.end(), value);
    if (value == T(0))
        cache.template set<Weight>(0);
    else
        cache.template set<Weight>(data.size());

    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::rotate_left(size_t i) noexcept {
    std::rotate(data.begin(), data.begin() + i, data.end());
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::rotate_right(size_t i) noexcept {
    std::rotate(data.rbegin(), data.rbegin() + i, data.rend());
    return *this;
}

template <ComponentType T>
constexpr Vector<T>& Vector<T>::reverse() noexcept {
    std::reverse(data.begin(), data.end());
    return *this;
}

template <ComponentType T>
template <typename U>
Vector<T>& Vector<T>::set_component(size_t i, U&& c)
    requires std::convertible_to<std::decay_t<U>, T>
{
    if (i >= data.size()) throw std::invalid_argument("trying to access non-existent element");
    data[i] = std::forward<U>(c);
    if (data[i] != T(0)) cache.invalidate();
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
    for (size_t j = 0; j < v.get_n(); ++j) {
        data[i + j] = v.data[j];
    }
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Vector<T>& Vector<T>::set_subvector(Vector&& v, size_t i) {
    if (i + v.get_n() > data.size())
        throw std::invalid_argument(
            "trying to replace subvector with "
            "vector of incompatible length");
    for (size_t j = 0; j < v.get_n(); ++j) {
        data[i + j] = std::move(v.data[j]);
    }
    cache.invalidate();
    return *this;
}

template <ComponentType T>
constexpr bool Vector<T>::is_zero() const noexcept
    requires ReliablyComparableType<T>
{
    return std::all_of(data.cbegin(), data.cend(), [](const T& x) { return x == T(0); });
}

template <ComponentType T>
constexpr bool Vector<T>::is_pairwisedistinct() const
    requires ReliablyComparableType<T>
{
    if constexpr (FiniteFieldType<T>) {
        std::unordered_set<T, details::FiniteFieldHasher<T>> unique_elems;
        for (const auto& elem : data)
            if (!unique_elems.insert(elem).second) return false;
    } else if constexpr (ReliablyComparableType<T> ||
                         std::is_same_v<T, InfInt>) {  // everything InfInt needs special treatment
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = i + 1; j < data.size(); ++j) {
                if (data[i] == data[j]) return false;
            }
        }
    } else {  // must be signed integer
        std::unordered_set<T> unique_elems;
        for (const auto& elem : data)
            if (!unique_elems.insert(elem).second) return false;
    }

    return true;
}

template <ComponentType T>
constexpr size_t Vector<T>::calculate_weight() const noexcept
    requires ReliablyComparableType<T>
{
    return data.size() - std::count(data.cbegin(), data.cend(), T(0));
}

template <ComponentType T>
constexpr size_t Vector<T>::burst_length() const noexcept {
    // Find first non-zero element
    auto first_nonzero = std::find_if(data.begin(), data.end(), [](const T& x) { return x != T(0); });

    if (first_nonzero == data.end()) return 0;  // All zeros

    // Find last non-zero component(search from end)
    auto last_nonzero = std::find_if(data.rbegin(), data.rend(), [](const T& x) { return x != T(0); });

    size_t L = std::distance(data.begin(), first_nonzero);
    size_t R = data.size() - 1 - std::distance(data.rbegin(), last_nonzero);

    return R - L + 1;
}

template <ComponentType T>
constexpr size_t Vector<T>::cyclic_burst_length() const noexcept {
    if (data.empty()) return 0;

    // Handle all-zero vector
    if (burst_length() == 0) return 0;

    size_t n = data.size();
    size_t max_zero_run = 0;
    size_t current_zero_run = 0;

    // First pass: find longest zero run within the vector
    for (size_t i = 0; i < n; ++i) {
        if (data[i] == T(0)) {
            current_zero_run++;
            max_zero_run = std::max(max_zero_run, current_zero_run);
        } else {
            current_zero_run = 0;
        }
    }

    // Second pass: check for wraparound zeros only if needed
    if (data[0] == T(0) && data[n - 1] == T(0)) {
        // Count zeros from start
        size_t zeros_from_start = 0;
        for (size_t i = 0; i < n && data[i] == T(0); ++i) {
            zeros_from_start++;
        }

        // Count zeros from end
        size_t zeros_from_end = 0;
        for (size_t i = n; i > 0 && data[i - 1] == T(0); --i) {
            zeros_from_end++;
        }

        // Update max if wraparound creates longer run
        if (zeros_from_start + zeros_from_end < n) {  // Avoid double counting all-zero case
            max_zero_run = std::max(max_zero_run, zeros_from_start + zeros_from_end);
        }
    }

    return n - max_zero_run;
}

template <ComponentType T>
Vector<T>& Vector<T>::randomize() noexcept {
    if constexpr (FieldType<T>) {
        std::for_each(data.begin(), data.end(), std::mem_fn(&T::randomize));
    } else if constexpr (std::same_as<T, double>) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::for_each(data.begin(), data.end(), [&](double& val) { val = dist(gen()); });
    } else if constexpr (std::same_as<T, std::complex<double>>) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::for_each(data.begin(), data.end(),
                      [&](std::complex<double>& val) { val = std::complex<double>(dist(gen()), dist(gen())); });
    } else if constexpr (SignedIntType<T>) {
        std::uniform_int_distribution<long long> dist(-100, 100);
        std::for_each(data.begin(), data.end(), [&](T& val) { val = T(dist(gen())); });
    }
    cache.invalidate();
    return *this;
}

template <ComponentType T>
size_t Vector<T>::as_integer() const noexcept
    requires FiniteFieldType<T>
{
    // Create index sequence for parallel processing
    auto indices = std::views::iota(size_t{0}, data.size());

    return std::transform_reduce(indices.begin(), indices.end(), size_t{0}, std::plus<size_t>{}, [this](size_t i) {
        return data[data.size() - i - 1].get_label() * sqm<size_t>(T::get_size(), i);
    });
}

template <ComponentType T>
template <FiniteFieldType S>
Matrix<S> Vector<T>::as_matrix() const noexcept
    requires FiniteFieldType<T> && SubfieldOf<T, S> && (!std::is_same_v<T, S>)
{
    const auto v = data[0].template as_vector<S>();
    const auto m = v.get_n();
    Matrix<S> res(m, data.size());

    Matrix<S> temp(v);
    temp.transpose();
    res.set_submatrix(0, 0, temp);

    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].is_erased()) {
            for (size_t j = 0; j < m; ++j) {
                res(j, i).erase();
            }
        } else {
            Matrix<S> temp(data[i].template as_vector<S>());
            temp.transpose();
            res.set_submatrix(0, i, temp);
        }
    }

    return res;
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
constexpr Vector<T> operator*(const Vector<T>& lhs, const T& rhs) noexcept {
    Vector res(lhs);
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator*(Vector<T>&& lhs, const T& rhs) noexcept {
    Vector res(std::move(lhs));
    res *= rhs;
    return res;
}

/*
 * T * vector
 */

template <ComponentType T>
constexpr Vector<T> operator*(const T& lhs, const Vector<T>& rhs) noexcept {
    Vector res(rhs);
    res *= lhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator*(const T& lhs, Vector<T>&& rhs) noexcept {
    Vector res(std::move(rhs));
    res *= lhs;
    return res;
}

/*
 * vector / T
 */

template <ComponentType T>
constexpr Vector<T> operator/(const Vector<T>& lhs, const T& rhs) {
    Vector res(lhs);
    res /= rhs;
    return res;
}

template <ComponentType T>
constexpr Vector<T> operator/(Vector<T>&& lhs, const T& rhs)
    requires FieldType<T>
{
    Vector res(std::move(lhs));
    res /= rhs;
    return res;
}

template <ComponentType T>
Vector<T> convolve(const Vector<T>& v, const Vector<T>& w) {
    return Vector(Polynomial(v) * Polynomial(w));
}

template <ComponentType T>
Vector<T> randomize(const Vector<T>& v) {
    Vector<T> result = v;
    result.randomize();
    return result;
}

template <ComponentType T>
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
Vector<T> set_component(Vector<T>&& v, size_t i, size_t j, const T& c) {
    Matrix<T> res(std::move(v));
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
constexpr Vector<T> set_subvector(const Vector<T>& v, size_t start, size_t end, const Vector<T>& w) {
    return v.set_subvector(start, end, w);
}

template <ComponentType T>
constexpr Vector<T> set_subvector(Vector<T>&& v, size_t start, size_t end, const Vector<T>& w) {
    return std::move(v).set_subvector(start, end, w);
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

template <FiniteFieldType T>
constexpr auto as_integer(const Vector<T>& v) {
    return v.as_integer();
}

template <FiniteFieldType T, FiniteFieldType S>
constexpr auto as_matrix(const Vector<T>& v) {
    return v.template as_matrix<S>();
}

/**
 * @brief Compute inner product (dot product) of two vectors
 *
 * @tparam T Vector component type
 * @param lhs First vector
 * @param rhs Second vector
 * @return Inner product ⟨lhs, rhs⟩ = Σᵢ lhs[i] * rhs[i]
 *
 * Computes the standard inner product. For finite fields, multiplication follows field arithmetic rules.
 * For complex numbers, uses standard complex multiplication (not conjugate).
 *
 * @throws std::invalid_argument if vectors have different lengths
 */
template <ComponentType T>
T inner_product(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate inner product of "
            "vectors of different lengths");
    return std::inner_product(lhs.data.cbegin(), lhs.data.cend(), rhs.data.begin(), T(0));
}

template <ComponentType T>
constexpr bool operator==(const Vector<T>& lhs, const Vector<T>& rhs) noexcept
    requires ReliablyComparableType<T>
{
    if (lhs.data.size() != rhs.data.size()) return false;
    return lhs.data == rhs.data;
}

template <ComponentType T>
constexpr bool operator!=(const Vector<T>& lhs, const Vector<T>& rhs) noexcept
    requires ReliablyComparableType<T>
{
    return !(lhs == rhs);
}

template <ComponentType T>
Vector<T> unit_vector(size_t length, size_t i) {
    if (i >= length) throw std::invalid_argument("trying to create invalid unit vector");
    Vector<T> res(length);
    res.set_component(i, T(1));
    res.cache.template set<Vector<T>::Weight>(1);
    return res;
}

template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& rhs) noexcept {
    os << "( ";
    for (auto it = rhs.data.cbegin(); it != rhs.data.cend(); ++it) {
        std::cout << *it;
        if (it != rhs.data.cend() - 1) {
            os << ", ";
        }
    }
    os << " )";
    return os;
}

/** @name Error control coding-related functions
 * @{
 */

/**
 * @brief Compute Hamming weight of a vector
 *
 * @tparam T Vector component type (must satisfy @ref ECC::FiniteFieldType or @ref ECC::SignedIntType)
 * @param v Vector to analyze
 * @return Number of non-zero components in the vector
 *
 * @note Not available for types T, where precise comparison for zero is not possible (all floating point types,
 * Rationals<T> with T != InfInt.
 */
template <ComponentType T>
constexpr size_t wH(const Vector<T>& v) noexcept
    requires ReliablyComparableType<T>
{
    return v.wH();
}

/**
 * @brief Compute Hamming distance between two vectors of discrete types
 *
 * @tparam T Vector component type (must satisfy @ref ECC::FiniteFieldType or @ref ECC::SignedIntType)
 * @param lhs First vector
 * @param rhs Second vector
 * @return Hamming distance dₕ(lhs, rhs) = wₕ(lhs - rhs)
 *
 * The Hamming distance is the number of positions where two vectors differ.
 *
 * @note Only for types fulfilling ECC::ReliablyComparableType.
 *
 * @throws std::invalid_argument if vectors have different lengths
 */
template <ComponentType T>
    requires ReliablyComparableType<T>
size_t dH(const Vector<T>& lhs, const Vector<T>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between vectors of different "
            "lengths");
    return (lhs - rhs).wH();
}

template <ComponentType T>
    requires ReliablyComparableType<T>
size_t dH(Vector<T>&& lhs, const Vector<T>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between vectors of different "
            "lengths");
    return (std::move(lhs) - rhs).wH();
}

template <ComponentType T>
    requires ReliablyComparableType<T>
size_t dH(const Vector<T>& lhs, Vector<T>&& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between vectors of different "
            "lengths");
    return (lhs - std::move(rhs)).wH();
}
template <ComponentType T>
    requires ReliablyComparableType<T>
size_t dH(Vector<T>&& lhs, Vector<T>&& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate Hamming distance between vectors of different "
            "lengths");
    return (std::move(lhs) - std::move(rhs)).wH();
}

/**
 * @brief Compute burst length of a vector
 *
 * @tparam T Vector component type (must satisfy @ref ECC::FiniteFieldType or @ref ECC::SignedIntType)
 * @param v Vector to analyze
 * @return Length of burst
 *
 * @note Not available for types T, where precise comparison for zero is not possible (all floating point types,
 * Rationals<T> with T != InfInt.
 *
 * @see Vector::burst_length() for detailed explanation
 */
template <ComponentType T>
constexpr size_t burst_length(const Vector<T>& v) noexcept
    requires ReliablyComparableType<T>
{
    return v.burst_length();
}

/**
 * @brief Compute cyclic burst length of a vector
 *
 * @tparam T Vector component type (must satisfy @ref ECC::FiniteFieldType or @ref ECC::SignedIntType)
 * @param v Vector to analyze
 * @return Length of cyclic burst
 *
 * @note Not available for types T, where precise comparison for zero is not possible (all floating point types,
 * Rationals<T> with T != InfInt.
 *
 * @see Vector::cyclic_burst_length() for detailed explanation
 */
template <ComponentType T>
constexpr size_t cyclic_burst_length(const Vector<T>& v) noexcept
    requires FiniteFieldType<T> || std::is_same_v<T, Rationals<InfInt>> || SignedIntType<T>
{
    return v.cyclic_burst_length();
}

/**
 * @brief Compute Euclidean distance between two complex vectors
 *
 * @param lhs First complex vector
 * @param rhs Second complex vector
 * @return Euclidean distance ||lhs - rhs||₂ = √(Σᵢ |lhs[i] - rhs[i]|²)
 *
 * Computes the standard L2 norm distance in the complex plane.
 *
 * @throws std::invalid_argument if vectors have different lengths
 */
inline double dE(const Vector<std::complex<double>>& lhs, const Vector<std::complex<double>>& rhs) {
    if (lhs.get_n() != rhs.get_n())
        throw std::invalid_argument(
            "trying to calculate euclidean distance between vectors of different "
            "lengths");

    double sum_of_squares = std::transform_reduce(
        lhs.data.begin(), lhs.data.end(), rhs.data.begin(), 0.0, std::plus<double>{},
        [](const std::complex<double>& l, const std::complex<double>& r) { return powl(abs(l - r), 2); });

    return sqrt(sum_of_squares);
}

/** @} */

}  // namespace ECC

#endif
