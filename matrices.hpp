/**
 * @file matrices.hpp
 * @brief Matrix arithmetic library
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
 * Dense and structured matrices over any @ref CECCO::ComponentType (finite fields, floating-point,
 * complex, signed integers). Provides REF/RREF (with binary-field fast paths), cached rank,
 * determinant, nullspace, characteristic polynomial, and matrix inversion. Cross-field
 * operations bridge through @ref CECCO::largest_common_subfield_t.
 *
 * The class template `Matrix<T>` automatically tracks structural type
 * (@ref CECCO::details::matrix_type_t — Zero, Identity, Diagonal, Vandermonde, Toeplitz)
 * to enable specialized fast paths. The type tag is transparent to callers; factories
 * (`IdentityMatrix`, `DiagonalMatrix`, `VandermondeMatrix`, …) all return `Matrix<T>`.
 *
 * @code{.cpp}
 * // Factory + arithmetic
 * auto I = IdentityMatrix<double>(3);
 * auto D = DiagonalMatrix(Vector<double>{1, 2, 3});
 * auto M = I + D;                              // 3×3, type tag stays Diagonal
 *
 * // Finite-field linear algebra
 * Matrix<Fp<7>> P = {{1, 2}, {3, 4}};
 * size_t r = P.rank();                         // Cached
 * auto null_basis = P.basis_of_nullspace();
 * @endcode
 *
 * @see @ref fields.hpp, @ref vectors.hpp, @ref field_concepts_traits.hpp
 */

#ifndef MATRICES_HPP
#define MATRICES_HPP

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <set>
#include <sstream>
#include <unordered_map>

#include "field_concepts_traits.hpp"

/*
// transitive
#include <algorithm>
#include <complex>
#include <concepts>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "InfInt.hpp"
#include "helpers.hpp"
*/

namespace CECCO {

namespace details {

/**
 * @brief Matrix structural type tag for optimization (internal)
 *
 * Maintained automatically by `Matrix<T>` operations; mutations that break a structure
 * demote the tag to `Generic`. Specialized algorithms dispatch on this tag to avoid
 * full O(mn) traversals when possible.
 */
enum matrix_type_t : uint8_t {
    /**
     * @brief Generic  matrix with arbitrary elements
     */
    Generic,

    /**
     * @brief Zero matrix with all elements equal to zero
     */
    Zero,

    /**
     * @brief Diagonal matrix (square) with non-zero elements only on the main diagonal
     */
    Diagonal,

    /**
     * @brief Identity matrix with ones on diagonal and zeros elsewhere
     */
    Identity,

    /**
     * @brief Vandermonde matrix with arithmetic progressions (of pairwise distinct elements) in columns
     */
    Vandermonde,

    /**
     * @brief Toeplitz matrix T_{i,j} = t_{i-j} with constant diagonals
     */
    Toeplitz
};

}  // namespace details

template <ComponentType T>
class Vector;
template <ComponentType T>
class Polynomial;
template <ComponentType T>
class Matrix;

template <ComponentType T>
constexpr Matrix<T> ZeroMatrix(size_t m, size_t n);
template <ComponentType T>
constexpr Matrix<T> IdentityMatrix(size_t m);
template <ComponentType T>
constexpr Matrix<T> DiagonalMatrix(const Vector<T>& v);
template <ComponentType T>
constexpr Matrix<T> ToeplitzMatrix(const Vector<T>& v, size_t m, size_t n);
template <ComponentType T>
constexpr Matrix<T> VandermondeMatrix(const Vector<T>& v, size_t m);
template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& rhs);

/**
 * @brief Dense m × n matrix over a @ref CECCO::ComponentType
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType (finite field, `double`,
 *           `std::complex<double>`, or signed integer including `InfInt`)
 *
 * Components are stored row-major in a contiguous buffer. A structural tag
 * (@ref details::matrix_type_t) tracks `Zero` / `Identity` / `Diagonal` / `Vandermonde` /
 * `Toeplitz` structures and dispatches to specialised fast paths; mutating operations that
 * break a tracked structure demote the tag to `Generic` and structure is not re-detected.
 * Dimension mismatches in arithmetic throw `std::invalid_argument`.
 *
 * Methods that need division (REF/RREF, inversion, nullspace, determinant, characteristic
 * polynomial, …) are gated by `requires FieldType<T>`; eigenvalue computation by
 * `requires FiniteFieldType<T>`. Methods that compare against zero (Hamming weight, rank,
 * structural tests, …) are gated by `requires ReliablyComparableType<T>`.
 *
 * Cross-field constructors and assignment operators between two finite fields of the same
 * characteristic route through @ref CECCO::details::largest_common_subfield_t, so matrices
 * over fields from disjoint construction towers can interoperate.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using F4 = Ext<Fp<2>, MOD{1, 1, 1}>;
 * Matrix<F4> M = {{1, 0}, {1, 1}};
 * size_t r = M.rank();                          // cached
 * auto null_basis = M.basis_of_nullspace();
 * auto chi = M.characteristic_polynomial();
 * @endcode
 */
template <ComponentType T>
class Matrix {
    friend constexpr Matrix<T> IdentityMatrix<>(size_t m);
    friend constexpr Matrix<T> DiagonalMatrix<>(const Vector<T>& v);
    friend constexpr Matrix<T> ToeplitzMatrix<>(const Vector<T>& v, size_t m, size_t n);
    friend constexpr Matrix<T> VandermondeMatrix<>(const Vector<T>& v, size_t m);
    template <ReliablyComparableType U>
    friend constexpr bool operator==(const Matrix<U>& lhs, const Matrix<U>& rhs);
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix& rhs);
    template <ComponentType>
    friend class Matrix;

   public:
    /** @name Constructors
     * @{
     */

    /// @brief Default constructor: empty matrix
    constexpr Matrix() noexcept = default;

    /// @brief m × n zero matrix (tag @ref details::Zero)
    constexpr Matrix(size_t m, size_t n) : data(m * n), m(m), n(n), type(details::Zero) {}

    /**
     * @brief m × n matrix with every component equal to `l`
     *
     * Tag is @ref details::Zero if `l == T(0)`, otherwise @ref details::Generic.
     */
    Matrix(size_t m, size_t n, const T& l);

    /**
     * @brief m × n matrix from a flat initializer list (row-major)
     *
     * @throws std::invalid_argument if `l.size() != m * n`
     */
    constexpr Matrix(size_t m, size_t n, std::initializer_list<T> l);

    /**
     * @brief From nested initializer lists, e.g. `{{1, 2, 3}, {4, 5, 6}}`
     *
     * Rows of unequal length are zero-padded to the longest row.
     */
    Matrix(std::initializer_list<std::initializer_list<T>> l);

    /// @brief 1 × n row matrix from a `Vector<T>`
    Matrix(const Vector<T>& v);

    constexpr Matrix(const Matrix& other)
        : data(other.data),
          m(other.m),
          n(other.n),
          transposed(other.transposed),
          type(other.type),
          cache(other.cache) {}

    constexpr Matrix(Matrix&& other) noexcept
        : data(std::move(other.data)),
          m(other.m),
          n(other.n),
          transposed(other.transposed),
          type(other.type),
          cache(std::move(other.cache)) {}

    /**
     * @brief Cross-field conversion between two finite fields of the same characteristic
     *
     * @tparam S Source field type (`Matrix<S>`); must share characteristic with T
     *
     * Converts component by component via T's cross-field constructor, which routes through
     * @ref CECCO::details::largest_common_subfield_t and so handles disjoint construction towers.
     * Propagates `std::invalid_argument` if any component is not representable in T.
     */
    template <FiniteFieldType S>
    constexpr Matrix(const Matrix<S>& other)
        requires FiniteFieldType<T> && (T::get_characteristic() == S::get_characteristic());

    /**
     * @brief Read from a PPM (P3) image file using the 64-entry colormap of @ref export_as_ppm
     *
     * Pixels whose RGB does not match any colormap entry are erased (under
     * @ref CECCO_ERASURE_SUPPORT) or replaced by a random field element.
     *
     * @note Convert other formats with ImageMagick:
     * @code{.sh}
     * magick input.png -alpha remove -background black +dither -remap palette.ppm -compress none output.ppm
     * @endcode
     */
    Matrix(const std::string& filename)
        requires FiniteFieldType<T> && (T::get_size() <= 64);

    /** @} */

    /** @name Assignment Operators
     * @{
     */

    constexpr Matrix& operator=(const Matrix& rhs);
    constexpr Matrix& operator=(Matrix&& rhs) noexcept;

    /// @brief Cross-field assignment (same semantics as the cross-field constructor)
    template <FiniteFieldType S>
        requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
    constexpr Matrix& operator=(const Matrix<S>& other);

    /** @} */

    /** @name Unary Arithmetic Operations
     * @{
     */

    /// @brief Unary `+` (lvalue): returns a copy
    constexpr Matrix operator+() const& { return *this; }
    /// @brief Unary `+` (rvalue): returns the rvalue itself
    constexpr Matrix operator+() && noexcept { return std::move(*this); }

    /// @brief Unary `−` (lvalue): returns a new matrix with each component negated
    constexpr Matrix operator-() const&;
    /// @brief Unary `−` (rvalue): negates components in place
    constexpr Matrix operator-() &&;

    /** @} */

    /** @name Compound Assignment Operations
     * @{
     */

    /**
     * @brief Component-wise addition `this(i, j) += rhs(i, j)`
     *
     * @throws std::invalid_argument if dimensions differ
     */
    Matrix& operator+=(const Matrix& rhs);

    /**
     * @brief Component-wise subtraction `this(i, j) -= rhs(i, j)`
     *
     * @throws std::invalid_argument if dimensions differ
     */
    Matrix& operator-=(const Matrix& rhs);

    /**
     * @brief Matrix multiplication `*this = *this · rhs`
     *
     * @throws std::invalid_argument if `this->get_n() != rhs.get_m()`
     */
    Matrix& operator*=(const Matrix& rhs);

    /// @brief Multiply every component by the scalar `s`
    constexpr Matrix& operator*=(const T& s);

    /**
     * @brief Divide every component by the scalar `s`
     *
     * @throws std::invalid_argument if `s == T(0)`
     * @note Round-trip `(M / s) * s == M` is only guaranteed when T satisfies @ref CECCO::FieldType
     * (otherwise integer rounding may corrupt components).
     */
    Matrix& operator/=(const T& s);

    /** @} */

    /** @name Randomization
     * @{
     */

    /**
     * @brief Fill matrix with random values
     *
     * @return Reference to this matrix after randomization
     *
     * Distribution depends on the component type: finite-field types draw uniformly from the
     * field; signed integers from [−100, 100]; `double` and the real/imaginary parts of
     * `std::complex<double>` from [−1.0, 1.0].
     *
     * @note Tag becomes @ref details::Generic — incidental structure (e.g. an accidental zero
     * matrix) is not re-detected.
     */
    Matrix& randomize();

    /** @} */

    /** @name Information and Properties
     * @{
     */

    /**
     * @brief Get the number of rows
     *
     * @return Number of rows in the matrix
     */
    constexpr size_t get_m() const noexcept { return m; }

    /**
     * @brief Get the number of columns
     *
     * @return Number of columns in the matrix
     */
    constexpr size_t get_n() const noexcept { return n; }

    /**
     * @brief Check if matrix is empty
     *
     * @return true if matrix has zero rows or columns, false otherwise
     */
    constexpr bool is_empty() const noexcept { return m == 0 || n == 0; }

    /**
     * @brief Check if the matrix is zero, caching the result via the type tag
     *
     * @return true iff every component equals T(0)
     *
     * On a positive result the tag is updated to @ref details::Zero, so subsequent calls and
     * other tag-aware fast paths short-circuit.
     */
    constexpr bool is_zero() {
        if (type == details::Zero) return true;
        const bool b = std::ranges::all_of(data, [](const T& v) { return v == T(0); });
        if (b) type = details::Zero;
        return b;
    }

    /**
     * @brief Check if the matrix is zero (no caching)
     *
     * @return true iff every component equals T(0)
     */
    constexpr bool is_zero() const {
        if (type == details::Zero) return true;
        return std::ranges::all_of(data, [](const T& v) { return v == T(0); });
    }

    /// @brief Hamming weight: number of non-zero, non-erased components; cached on first call
    size_t wH() const
        requires ReliablyComparableType<T>
    {
        return cache.template get_or_compute<Weight>([this] { return calculate_weight(); });
    }

    /**
     * @brief Matrix rank, computed once and cached
     *
     * @return Dimension of the row space (equivalently, of the column space)
     *
     * Uses row reduction to echelon form. Subsequent calls return the cached value until a
     * mutating operation invalidates it.
     */
    size_t rank() const
        requires FieldType<T>;

    /**
     * @brief Check if matrix is invertible
     *
     * @return true iff the matrix is square and has full rank
     */
    bool is_invertible() const
        requires FieldType<T>
    {
        return m == n && rank() == m;
    }

    /**
     * @brief Main diagonal as a vector
     *
     * @return Vector containing entries (i, i) for i = 0, …, m−1
     *
     * @throws std::invalid_argument if the matrix is not square
     */
    Vector<T> diagonal() const;

    /**
     * @brief Characteristic polynomial det(λI − A)
     *
     * @return Polynomial of degree m
     *
     * Computed via the Samuelson–Berkowitz algorithm in the general case; the
     * @ref details::Diagonal and @ref details::Vandermonde tags trigger closed-form shortcuts.
     *
     * @throws std::invalid_argument if the matrix is not square or is empty
     */
    Polynomial<T> characteristic_polynomial() const
        requires FieldType<T>;

    /**
     * @brief Basis of the nullspace (right kernel)
     *
     * @return Matrix whose rows span { x : A xᵀ = 0 }; empty matrix if the nullspace is trivial
     *
     * Computed by row reduction.
     */
    Matrix<T> basis_of_nullspace() const
        requires FieldType<T>;

    /**
     * @brief Alias for @ref basis_of_nullspace
     *
     * @return Matrix whose rows form a basis of the kernel
     */
    Matrix<T> basis_of_kernel() const
        requires FieldType<T>
    {
        return basis_of_nullspace();
    }

    /**
     * @brief Matrix determinant
     *
     * @return det(A); T(0) for singular matrices
     *
     * Algorithm depends on the type tag: closed-form for @ref details::Identity,
     * @ref details::Zero, and @ref details::Diagonal; Samuelson–Berkowitz otherwise.
     *
     * @throws std::invalid_argument if the matrix is not square or is empty
     */
    T determinant() const
        requires FieldType<T>;

    /**
     * @brief Eigenvalues lying in the underlying finite field
     *
     * @return Roots of the characteristic polynomial that lie in T
     *
     * Eigenvalues that exist only in an extension of T are omitted.
     *
     * @throws std::invalid_argument if the matrix is not square
     */
    std::vector<T> eigenvalues() const
        requires FiniteFieldType<T>;

    /**
     * @brief Enumerate every vector in the row space
     *
     * @return All q^rank vectors in span(rows), where q = |T|
     *
     * @warning Size grows as q^rank — only practical for small fields and small rank.
     */
    std::vector<Vector<T>> rowspace() const
        requires FieldType<T>;

    /**
     * @brief Alias for @ref rowspace
     *
     * @return Every vector in the span of the rows
     */
    std::vector<Vector<T>> span() const
        requires FieldType<T>
    {
        return rowspace();
    }

    /** @} */

    /** @name Component Access and Manipulation
     * @{
     */

    /**
     * @brief Set component (i, j) by perfect forwarding
     *
     * @param i Row index (0-based)
     * @param j Column index (0-based)
     * @param c Value to assign; bound by lvalue or rvalue reference
     * @return Reference to this matrix after modification
     *
     * @throws std::invalid_argument if either index is out of bounds
     */
    template <typename U>
    Matrix& set_component(size_t i, size_t j, U&& c)
        requires std::convertible_to<std::decay_t<U>, T>;

    /**
     * @brief Access component (i, j) (read-only)
     *
     * @param i Row index (0-based)
     * @param j Column index (0-based)
     * @return Const reference to the component at (i, j)
     *
     * @throws std::invalid_argument if either index is out of bounds
     */
    const T& operator()(size_t i, size_t j) const;

    /**
     * @brief Extract row i as a vector
     *
     * @param i Row index
     * @return Vector containing the components of row i
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Vector<T> get_row(size_t i) const;

    /**
     * @brief Extract column j as a (row) vector
     *
     * @param j Column index
     * @return Vector containing the components of column j; transposed, since @ref Vector models row vectors
     *
     * @throws std::invalid_argument if j is out of bounds
     */
    Vector<T> get_col(size_t j) const;

    /**
     * @brief Extract submatrix from region [i, i+h) × [j, j+w)
     *
     * @param i Starting row index
     * @param j Starting column index
     * @param h Height (number of rows)
     * @param w Width (number of columns)
     * @return Submatrix of shape h × w
     *
     * @throws std::invalid_argument if the region extends beyond the matrix
     */
    Matrix<T> get_submatrix(size_t i, size_t j, size_t h, size_t w) const;

    /**
     * @brief Overwrite the region starting at (i, j) with N
     *
     * @param i Starting row index
     * @param j Starting column index
     * @param N Source matrix; its shape must fit within this matrix from (i, j)
     * @return Reference to this matrix after the assignment
     *
     * @throws std::invalid_argument if the region would extend beyond bounds
     */
    Matrix<T>& set_submatrix(size_t i, size_t j, const Matrix& N);

    /**
     * @brief Concatenate other to the right (column-wise)
     *
     * @param other Matrix to append on the right; must have the same number of rows as this
     * @return Reference to this matrix after the join
     *
     * @throws std::invalid_argument if row counts differ
     */
    Matrix<T>& horizontal_join(const Matrix& other);

    /**
     * @brief Concatenate other below (row-wise)
     *
     * @param other Matrix to append below; must have the same number of columns as this
     * @return Reference to this matrix after the join
     *
     * @throws std::invalid_argument if column counts differ
     */
    Matrix<T>& vertical_join(const Matrix& other);

    /**
     * @brief Block-diagonal join: this in the upper-left, other in the lower-right
     *
     * @param other Matrix placed in the lower-right block
     * @return Reference to this matrix after the join
     *
     * Off-diagonal blocks are filled with zeros.
     */
    Matrix<T>& diagonal_join(const Matrix& other);

    /**
     * @brief Kronecker (tensor) product with other
     *
     * @param other Right operand
     * @return Reference to this matrix after the product
     *
     * If this is m × n and other is p × q, the result is mp × nq.
     */
    Matrix<T>& Kronecker_product(const Matrix& other);

    /**
     * @brief Swap rows i and j
     *
     * @param i First row index
     * @param j Second row index
     * @return Reference to this matrix after the swap
     *
     * @throws std::invalid_argument if either index is out of bounds
     */
    Matrix<T>& swap_rows(size_t i, size_t j);

    /**
     * @brief Swap columns i and j
     *
     * @param i First column index
     * @param j Second column index
     * @return Reference to this matrix after the swap
     *
     * @throws std::invalid_argument if either index is out of bounds
     */
    Matrix<T>& swap_columns(size_t i, size_t j);

    /**
     * @brief row[i] ← s · row[i]
     *
     * @param s Scalar multiplier
     * @param i Row index
     * @return Reference to this matrix after the scaling
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Matrix<T>& scale_row(const T& s, size_t i);

    /**
     * @brief col[i] ← s · col[i]
     *
     * @param s Scalar multiplier
     * @param i Column index
     * @return Reference to this matrix after the scaling
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Matrix<T>& scale_column(const T& s, size_t i);

    /**
     * @brief row[j] ← row[j] + s · row[i]
     *
     * @param s Scalar multiplier on the source row
     * @param i Source row index
     * @param j Destination row index
     * @return Reference to this matrix after the update
     *
     * @throws std::invalid_argument if either index is out of bounds
     */
    Matrix<T>& add_scaled_row(const T& s, size_t i, size_t j);

    /**
     * @brief col[j] ← col[j] + s · col[i]
     *
     * @param s Scalar multiplier on the source column
     * @param i Source column index
     * @param j Destination column index
     * @return Reference to this matrix after the update
     *
     * @throws std::invalid_argument if either index is out of bounds
     */
    Matrix<T>& add_scaled_column(const T& s, size_t i, size_t j);

    /**
     * @brief row[j] ← row[j] + row[i]
     *
     * @param i Source row index
     * @param j Destination row index
     * @return Reference to this matrix after the update
     *
     * @throws std::invalid_argument if either index is out of bounds
     */
    Matrix<T>& add_row(size_t i, size_t j) {
        if (i >= m || j >= m)
            throw std::invalid_argument("trying to add row to other row, at least one of them is non-existent");
        return add_scaled_row(T(1), i, j);
    }

    /**
     * @brief col[j] ← col[j] + col[i]
     *
     * @param i Source column index
     * @param j Destination column index
     * @return Reference to this matrix after the update
     *
     * @throws std::invalid_argument if either index is out of bounds
     */
    Matrix<T>& add_column(size_t i, size_t j) {
        if (i >= n || j >= n)
            throw std::invalid_argument("trying to add column to other column, at least one of them is non-existent");
        transpose();
        add_scaled_row(T(1), i, j);
        transpose();
        return *this;
    }

    /**
     * @brief Delete the columns whose indices appear in v
     *
     * @param v Column indices (deduplicated internally)
     * @return Reference to this matrix after deletion
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix<T>& delete_columns(const std::vector<size_t>& v);

    /**
     * @brief Delete column i (single-index convenience for @ref delete_columns)
     *
     * @param i Column index
     * @return Reference to this matrix after deletion
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Matrix<T>& delete_column(size_t i) { return delete_columns({i}); }

    /**
     * @brief Delete the rows whose indices appear in v
     *
     * @param v Row indices (deduplicated internally)
     * @return Reference to this matrix after deletion
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix<T>& delete_rows(const std::vector<size_t>& v);

    /**
     * @brief Delete row i (single-index convenience for @ref delete_rows)
     *
     * @param i Row index
     * @return Reference to this matrix after deletion
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Matrix<T>& delete_row(size_t i) { return delete_rows({i}); }

#ifdef CECCO_ERASURE_SUPPORT

    /**
     * @brief Flag component (i, j) as erased
     *
     * @param i Row index
     * @param j Column index
     * @return Reference to this matrix after erasing
     *
     * Marks the component as erased; field arithmetic on erased elements is undefined — see
     * @ref CECCO_ERASURE_SUPPORT and the erase()/unerase() interface in fields.hpp.
     *
     * @warning An erased element can no longer participate in field operations or property
     * queries. Correct handling of erased elements is the caller's responsibility.
     *
     * @throws std::invalid_argument if (i, j) is out of bounds
     */
    Matrix<T>& erase_component(size_t i, size_t j)
        requires FieldType<T>;

    /**
     * @brief Clear the erasure flag on component (i, j)
     *
     * @param i Row index
     * @param j Column index
     * @return Reference to this matrix after un-erasing
     *
     * @throws std::invalid_argument if (i, j) is out of bounds
     */
    Matrix<T>& unerase_component(size_t i, size_t j)
        requires FieldType<T>;

    /**
     * @brief Flag every component of the specified columns as erased
     *
     * @param v Column indices (deduplicated internally)
     * @return Reference to this matrix after erasing
     *
     * @warning See @ref erase_component for the semantics and caller obligations.
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix<T>& erase_columns(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Erase column i (single-index convenience for @ref erase_columns)
     *
     * @param i Column index
     * @return Reference to this matrix after erasing
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Matrix<T>& erase_column(size_t i)
        requires FieldType<T>
    {
        return erase_columns({i});
    }

    /**
     * @brief Clear erasure flags on every component of the specified columns
     *
     * @param v Column indices (deduplicated internally)
     * @return Reference to this matrix after un-erasing
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix<T>& unerase_columns(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Un-erase column i (single-index convenience for @ref unerase_columns)
     *
     * @param i Column index
     * @return Reference to this matrix after un-erasing
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Matrix<T>& unerase_column(size_t i)
        requires FieldType<T>
    {
        return unerase_columns({i});
    }

    /**
     * @brief Flag every component of the specified rows as erased
     *
     * @param v Row indices (deduplicated internally)
     * @return Reference to this matrix after erasing
     *
     * @warning See @ref erase_component for the semantics and caller obligations.
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix& erase_rows(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Erase row i (single-index convenience for @ref erase_rows)
     *
     * @param i Row index
     * @return Reference to this matrix after erasing
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Matrix& erase_row(size_t i)
        requires FieldType<T>
    {
        return erase_rows({i});
    }

    /**
     * @brief Clear erasure flags on every component of the specified rows
     *
     * @param v Row indices (deduplicated internally)
     * @return Reference to this matrix after un-erasing
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix& unerase_rows(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Un-erase row i (single-index convenience for @ref unerase_rows)
     *
     * @param i Row index
     * @return Reference to this matrix after un-erasing
     *
     * @throws std::invalid_argument if i is out of bounds
     */
    Matrix& unerase_row(size_t i)
        requires FieldType<T>
    {
        return unerase_rows({i});
    }

#endif

    /** @} */

    /** @name Transformations
     * @{
     */

    /**
     * @brief Reverse the row order
     *
     * @return Reference to this matrix after the reversal
     */
    Matrix<T>& reverse_rows();

    /**
     * @brief Reverse the column order
     *
     * @return Reference to this matrix after the reversal
     */
    Matrix<T>& reverse_columns();

    /**
     * @brief Set every component to s
     *
     * @param s Value assigned to every component
     * @return Reference to this matrix after filling
     *
     * @note Tag becomes @ref details::Zero if s == T(0), otherwise @ref details::Generic.
     */
    constexpr Matrix<T>& fill(const T& s);

    /**
     * @brief Transpose in place
     *
     * @return Reference to this matrix after transposition
     *
     * For an m × n matrix the result is n × m, with component (i, j) moved to (j, i).
     */
    constexpr Matrix<T>& transpose();

    /**
     * @brief Row echelon form (REF), with a binary-field fast path
     *
     * @param rank Optional out-parameter; if non-null, receives the rank
     * @return Reference to this matrix after the reduction
     *
     * Forward Gaussian elimination only — cheaper than RREF when only the rank or a
     * triangularised form is needed. For Fp<2>, pivot scaling is skipped at compile time. Rank
     * is cached when @p rank is non-null.
     */
    Matrix<T>& ref(size_t* rank = nullptr)
        requires FieldType<T>;

    /**
     * @brief Reduced row echelon form (RREF)
     *
     * @param rank Optional out-parameter; if non-null, receives the rank
     * @return Reference to this matrix after the reduction
     *
     * Two phases: forward elimination (REF) followed by backward elimination. Rank is always
     * cached.
     */
    Matrix<T>& rref(size_t* rank = nullptr)
        requires FieldType<T>;

    /**
     * @brief Invert in place
     *
     * @return Reference to this matrix after inversion
     *
     * Gaussian elimination with partial pivoting. Requires a square non-singular matrix.
     *
     * @throws std::invalid_argument if the matrix is not square or is singular
     */
    Matrix<T>& invert()
        requires FieldType<T>;

    /** @} */

    /** @name Finite Field Specific Operations
     * @{
     */

    /**
     * @brief Reinterpret the columns as elements of a superfield
     *
     * @tparam S Superfield of T (same construction tower)
     * @return Vector over S, one component per column
     *
     * Each column is read as the coordinate representation of an element of S over T.
     */
    template <FiniteFieldType S>
    constexpr Vector<S> as_vector() const
        requires FiniteFieldType<T> && ExtensionOf<T, S> && (!std::is_same_v<T, S>);

    /**
     * @brief Flatten in row-major order
     *
     * @return Vector of length m·n; component (i, j) maps to index i·n + j
     */
    Vector<T> to_vector() const;

    /**
     * @brief Export the matrix as a PPM (P3) image
     *
     * @param filename Output path
     *
     * Each component is mapped through a built-in 64-entry colormap (black → blue → green →
     * yellow → white). The image is n wide and m tall.
     *
     * @note With @ref CECCO_ERASURE_SUPPORT, erased components render in red.
     */
    void export_as_ppm(const std::string& filename) const
        requires FiniteFieldType<T> && (T::get_size() <= 64);

    /** @} */

   private:
    /// @brief Component storage, row-major
    std::vector<T> data;

    /// @brief Number of rows
    size_t m = 0;
    /// @brief Number of columns
    size_t n = 0;

    /// @brief When true, accesses interpret storage as transposed (no data movement)
    bool transposed = false;

    /// @brief Structural tag — see @ref details::matrix_type_t
    details::matrix_type_t type = details::Zero;

    /// @brief Cache for matrix rank and Hamming weight (invalidated by mutating operations)
    enum CacheIds { Rank = 0, Weight = 1 };
    mutable details::Cache<details::CacheEntry<Rank, size_t>, details::CacheEntry<Weight, size_t>> cache;

    size_t calculate_weight() const
        requires ReliablyComparableType<T>;

    /**
     * @brief Matrix multiplication kernel with compile-time transpose dispatch
     * @tparam this_transposed Compile-time flag: true if left matrix (*this) is transposed
     * @tparam rhs_transposed Compile-time flag: true if right matrix (rhs) is transposed
     * @param this_data Raw pointer to left matrix data
     * @param rhs_data Raw pointer to right matrix data
     * @param res_data Raw pointer to result matrix data
     * @param M Number of rows in result
     * @param K Inner dimension (cols of this, rows of rhs)
     * @param N Number of columns in result
     * @param BS Block size for cache tiling
     */
    template <bool this_transposed, bool rhs_transposed>
    static void multiply_kernel(const T* this_data, const T* rhs_data, T* res_data, size_t M, size_t K, size_t N,
                                size_t BS) {
        for (size_t ii = 0; ii < M; ii += BS) {
            const size_t imax = std::min(ii + BS, M);
            for (size_t kk = 0; kk < K; kk += BS) {
                const size_t kmax = std::min(kk + BS, K);
                for (size_t jj = 0; jj < N; jj += BS) {
                    const size_t jmax = std::min(jj + BS, N);
                    for (size_t i = ii; i < imax; ++i) {
                        for (size_t k = kk; k < kmax; ++k) {
                            // Compile-time dispatch for optimal addressing
                            const size_t this_idx = this_transposed ? (i + k * M) : (i * K + k);
                            const T aik = this_data[this_idx];
                            const size_t res_row_offset = i * N;
                            for (size_t j = jj; j < jmax; ++j) {
                                // Compile-time dispatch for optimal addressing
                                const size_t rhs_idx = rhs_transposed ? (k + j * K) : (k * N + j);
                                res_data[res_row_offset + j] += aik * rhs_data[rhs_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Elimination helper for a single row operation
     * @tparam transposed True if matrix is transposed, false otherwise
     * @param data Raw matrix data
     * @param m Number of rows
     * @param n Number of columns
     * @param target_row Row to be eliminated
     * @param pivot_row Row containing the pivot
     * @param start_col Starting column for elimination
     * @param f Scaling factor
     */
    template <bool transposed>
    static void eliminate_row_kernel(T* data, size_t m, size_t n, size_t target_row, size_t pivot_row,
                                     size_t start_col, const T& f) {
        if constexpr (!transposed) {
            // Row-major access
            T* target_data = data + target_row * n;
            const T* pivot_data = data + pivot_row * n;
            for (size_t j = start_col; j < n; ++j) {
                target_data[j] -= f * pivot_data[j];
            }
        } else {
            // Column-major access
            for (size_t j = start_col; j < n; ++j) {
                data[target_row + j * m] -= f * data[pivot_row + j * m];
            }
        }
    }

    /**
     * @brief REF elimination kernel with compile-time transpose dispatch
     * @tparam transposed True if matrix is transposed, false otherwise
     * @param data Raw matrix data
     * @param m Number of rows
     * @param n Number of columns
     * @param h Current pivot row
     * @param k Current pivot column
     * @return Number of rows processed (new h value)
     */
    template <bool transposed>
    static size_t ref_elimination_kernel(T* data, size_t m, size_t n, size_t h, size_t k) {
        while (h < m && k < n) {
            // find pivot
            size_t p = m;  // Default: no pivot found

            for (size_t row = h; row < m; ++row) {
                const size_t idx = transposed ? (row + k * m) : (row * n + k);
                if (data[idx] != T(0)) {
                    p = row;
                    break;
                }
            }

            if (p == m) {  // no pivot in column -> proceed to next column
                ++k;
            } else {
                // this->swap_rows(h, p);
                if constexpr (!transposed)
                    std::swap_ranges(data + h * n, data + (h + 1) * n, data + p * n);
                else
                    for (size_t j = 0; j < n; ++j) std::swap(data[h + j * m], data[p + j * m]);

                const size_t pivot_idx = transposed ? (h + k * m) : (h * n + k);
                const T pivot = data[pivot_idx];

                // this->scale_row(T(1) / pivot, h);
                const T inv = T(1) / pivot;
                if constexpr (!transposed) {
                    for (size_t j = k; j < n; ++j) data[h * n + j] *= inv;
                } else {
                    for (size_t j = k; j < n; ++j) data[h + j * m] *= inv;
                }

                // Forward elimination only - eliminate entries BELOW pivot
                for (size_t i = h + 1; i < m; ++i) {
                    const size_t f_idx = transposed ? (i + k * m) : (i * n + k);
                    const T f = data[f_idx];

                    if (f != T(0)) eliminate_row_kernel<transposed>(data, m, n, i, h, k, f);
                }

                ++h;
                ++k;
            }
        }
        return h;
    }

    /**
     * @brief RREF backward elimination kernel with compile-time transpose dispatch
     * @tparam transposed True if matrix is transposed, false otherwise
     * @param data Raw matrix data
     * @param m Number of rows
     * @param n Number of columns
     * @param r Matrix rank (number of pivots to process)
     */
    template <bool transposed>
    void rref_backward_elimination_kernel(T* data, size_t m, size_t n, size_t r) {
        for (size_t i = 0; i < r; ++i) {
            const size_t pivot_row = r - 1 - i;  // Process from last to first

            // find pivot
            size_t pivot_col = 0;
            while (pivot_col < n) {
                const size_t pivot_idx = transposed ? (pivot_row + pivot_col * m) : (pivot_row * n + pivot_col);
                if (data[pivot_idx] != T(0)) {
                    break;
                }
                ++pivot_col;
            }

            if (pivot_col < n) {
                // Backward elimination only - eliminate entries ABOVE pivot
                for (size_t row = 0; row < pivot_row; ++row) {
                    const size_t f_idx = transposed ? (row + pivot_col * m) : (row * n + pivot_col);
                    const T f = data[f_idx];

                    if (f != T(0)) eliminate_row_kernel<transposed>(data, m, n, row, pivot_row, pivot_col, f);
                }
            }
        }
    }

    /**
     * @brief Calculate matrix rank (private implementation for caching)
     */
    size_t calculate_rank() const
        requires FieldType<T>;
};

/* Matrix member function implementations */

template <ComponentType T>
Matrix<T>::Matrix(size_t m, size_t n, const T& l)
    : data(m * n), m(m), n(n), type(l == T(0) ? details::Zero : details::Generic) {
    std::fill(data.begin(), data.end(), l);
    if (l != T(0))
        cache.template set<Rank>(1);
    else
        cache.template set<Rank>(0);
}

template <ComponentType T>
constexpr Matrix<T>::Matrix(size_t m, size_t n, std::initializer_list<T> l)
    : data(l), m(m), n(n), type(details::Generic) {
    if (l.size() != m * n) {
        throw std::invalid_argument(
            "number of elements in initializer list does not correspond to number of rows and columns specified");
    }
}

template <ComponentType T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> l) : m(l.size()), n(0), type(details::Generic) {
    if (m == 0) return;
    for (auto it = l.begin(); it != l.end(); ++it) {
        if (it->size() > n) n = it->size();
    }
    if (n == 0) return;
    data.resize(m * n);

    size_t i = 0;
    for (const auto& row : l) {
        size_t j = 0;
        for (const auto& val : row) set_component(i, j++, val);
        ++i;
    }
}

template <ComponentType T>
Matrix<T>::Matrix(const Vector<T>& v) : data(v.get_n()), m(1), n(v.get_n()), type(details::Toeplitz) {
    // Direct copy from vector data to matrix data for single row
    std::copy(v.data.begin(), v.data.end(), data.begin());
    cache.template set<Rank>(v.is_zero() ? 0 : 1);
}

template <ComponentType T>
template <FiniteFieldType S>
constexpr Matrix<T>::Matrix(const Matrix<S>& other)
    requires FiniteFieldType<T> && (T::get_characteristic() == S::get_characteristic())
    : data(other.get_m() * other.get_n()),
      m(other.get_m()),
      n(other.get_n()),
      type((other.type == details::Zero || other.type == details::Identity || other.type == details::Toeplitz)
               ? other.type
               : details::Generic) {
    if (!other.transposed) {
        std::transform(other.data.begin(), other.data.end(), data.begin(),
                       [](const S& elem) { return T(elem); });  // Uses enhanced cross-field constructors
    } else {
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) data[i * n + j] = T(other(i, j));  // Uses enhanced cross-field constructors
    }
    cache.invalidate();
}

template <ComponentType T>
Matrix<T>::Matrix(const std::string& filename)
    requires FiniteFieldType<T> && (T::get_size() <= 64)
{
    std::ifstream in(filename);

    auto next_token = [&](std::string& tok) -> void {
        for (;;) {
            in >> tok;
            if (tok.empty()) continue;
            if (tok[0] == '#') {
                in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                continue;
            }
            return;
        }
    };

    std::string tok;
    next_token(tok);  // "P3"
    next_token(tok);
    size_t w = std::stoul(tok);
    next_token(tok);
    size_t h = std::stoul(tok);
    next_token(tok); /* maxval */  // assumed 255

    *this = Matrix(h, w);

    // "RGB to a" map
    static const auto rgb_to_a = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        m.reserve(64);
        for (uint8_t a = 0; a < 64; ++a) {
            const uint32_t key = (uint32_t(details::colormap[a][0]) << 16) | (uint32_t(details::colormap[a][1]) << 8) |
                                 uint32_t(details::colormap[a][2]);
            m.emplace(key, a);
        }
        return m;
    }();

    auto label_from_a = [&](uint8_t a) -> size_t { return ((T::get_size() - 1) * (63 - static_cast<size_t>(a))) / 63; };

    for (size_t i = 0; i < h; ++i) {
        for (size_t j = 0; j < w; ++j) {
            next_token(tok);
            uint8_t r = static_cast<uint8_t>(std::stoi(tok));
            next_token(tok);
            uint8_t g = static_cast<uint8_t>(std::stoi(tok));
            next_token(tok);
            uint8_t b = static_cast<uint8_t>(std::stoi(tok));

            const uint32_t key = (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);

            const auto it = rgb_to_a.find(key);
            if (it != rgb_to_a.end()) {
                set_component(i, j, T(label_from_a(it->second)));
            } else {
#ifdef CECCO_ERASURE_SUPPORT
                erase_component(i, j);
#else
                set_component(i, j, T(0).randomize());
#endif
            }
        }
    }
}

template <ComponentType T>
constexpr Matrix<T>& Matrix<T>::operator=(const Matrix& rhs) {
    if (this == &rhs) return *this;
    data = rhs.data;
    m = rhs.m;
    n = rhs.n;
    transposed = rhs.transposed;
    type = rhs.type;
    cache = rhs.cache;
    return *this;
}

template <ComponentType T>
constexpr Matrix<T>& Matrix<T>::operator=(Matrix&& rhs) noexcept {
    data = std::move(rhs.data);
    m = rhs.m;
    n = rhs.n;
    transposed = rhs.transposed;
    type = rhs.type;
    cache = std::move(rhs.cache);
    return *this;
}

template <ComponentType T>
template <FiniteFieldType S>
    requires FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
constexpr Matrix<T>& Matrix<T>::operator=(const Matrix<S>& other) {
    data.resize(other.get_m() * other.get_n());
    std::transform(other.data.cbegin(), other.data.cend(), data.begin(),
                   [&](const S& e) { return T(e); });  // Uses enhanced cross-field constructors
    m = other.get_m();
    n = other.get_n();
    transposed = other.transposed;
    type = other.type;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
constexpr Matrix<T> Matrix<T>::operator-() const& {
    auto res = *this;
    auto rank_backup = res.cache.template get<Rank>();
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        std::ranges::for_each(res.data, [](T& x) { x = -x; });
        if (type == details::Vandermonde) res.type = details::Generic;

    } else if (type == details::Zero) {
        /* no-op */
    } else if (type == details::Diagonal || type == details::Identity) {
        for (size_t mu = 0; mu < m; ++mu) res.data[mu * n + mu] = -res.data[mu * n + mu];
        if constexpr (FiniteFieldType<T>) {
            if (T::get_characteristic() != 2 && type == details::Identity) res.type = details::Diagonal;
        } else {
            if (type == details::Identity) res.type = details::Diagonal;
        }
    }
    if (rank_backup) res.cache.template set<Rank>(*rank_backup);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> Matrix<T>::operator-() && {
    auto rank_backup = cache.template get<Rank>();
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        std::ranges::for_each(data, [](T& x) { x = -x; });
        if (type == details::Vandermonde) {
            type = details::Generic;
        }
    } else if (type == details::Zero) {
        /* no-op */
    } else if (type == details::Diagonal || type == details::Identity) {
        for (size_t mu = 0; mu < m; ++mu) data[mu * n + mu] = -data[mu * n + mu];
        if constexpr (FiniteFieldType<T>) {
            if (T::get_characteristic() != 2 && type == details::Identity) type = details::Diagonal;
        } else {
            if (type == details::Identity) type = details::Diagonal;
        }
    }
    if (rank_backup) cache.template set<Rank>(*rank_backup);
    return std::move(*this);
}

template <ComponentType T>
Matrix<T>& Matrix<T>::operator+=(const Matrix& rhs) {
    if (m != rhs.m || n != rhs.n)
        throw std::invalid_argument(
            "trying to add two matrices of different "
            "dimensions");
    if (type == details::Zero) {
        *this = rhs;
    } else if (rhs.type == details::Zero) {
        /* no-op */
    } else if ((type == details::Diagonal && rhs.type == details::Diagonal) ||
               (type == details::Identity && rhs.type == details::Identity) ||
               (type == details::Diagonal && rhs.type == details::Identity) ||
               (type == details::Identity && rhs.type == details::Diagonal)) {
        for (size_t mu = 0; mu < m; ++mu) data[mu * n + mu] += rhs.data[mu * n + mu];
        if (type == details::Identity) type = details::Diagonal;
    } else {
        if (!transposed && !rhs.transposed) {
            std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<T>{});
            if (type != details::Generic && !(type == details::Toeplitz && rhs.type == details::Toeplitz))
                type = details::Generic;
        } else {
            for (size_t mu = 0; mu < m; ++mu)
                for (size_t nu = 0; nu < n; ++nu) set_component(mu, nu, (*this)(mu, nu) + rhs(mu, nu));
        }
    }
    if (std::all_of(data.cbegin(), data.cend(), [](const T& x) { return x == T(0); })) {
        this->type = details::Zero;
    }
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::operator-=(const Matrix& rhs) {
    if (m != rhs.m || n != rhs.n)
        throw std::invalid_argument(
            "trying to subtract two matrices of different "
            "dimensions");
    operator+=(-rhs);
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::operator*=(const Matrix& rhs) {
    if (n != rhs.m)
        throw std::invalid_argument(
            "trying to multiply two matrices "
            "with incompatible dimensions");
    if (type == details::Zero || rhs.type == details::Zero) {
        *this = Matrix(m, rhs.n);
    } else if (type == details::Identity) {
        *this = rhs;
    } else if (rhs.type == details::Identity) {
        /* no-op */
    } else if (type == details::Diagonal && rhs.type == details::Diagonal) {
        for (size_t mu = 0; mu < m; ++mu) data[mu * n + mu] *= rhs.data[mu * n + mu];
    } else if (type == details::Diagonal) {
        auto res = rhs;
        for (size_t mu = 0; mu < m; ++mu) {
            auto s = (*this)(mu, mu);
            for (size_t nu = 0; nu < rhs.n; ++nu) res.set_component(mu, nu, res(mu, nu) * s);
        }
        *this = std::move(res);
    } else if (rhs.type == details::Diagonal) {
        for (size_t nu = 0; nu < n; ++nu) {
            const auto& s = rhs(nu, nu);
            for (size_t mu = 0; mu < m; ++mu) set_component(mu, nu, (*this)(mu, nu) * s);
        }
    } else {
        const size_t M = get_m();
        const size_t K = get_n();
        const size_t N = rhs.get_n();
        Matrix<T> res(M, N, T(0));

        const T* this_data = this->data.data();
        const T* rhs_data = rhs.data.data();
        T* res_data = res.data.data();

        // Tune this block/tile size for architecture. 48, 64, 96, 128 are reasonable.
        constexpr size_t BS = 64;

        // Branch ONCE on transpose flags - dispatch to optimized kernels with zero duplication
        if (!this->transposed && !rhs.transposed) {
            multiply_kernel<false, false>(this_data, rhs_data, res_data, M, K, N, BS);
        } else if (this->transposed && !rhs.transposed) {
            multiply_kernel<true, false>(this_data, rhs_data, res_data, M, K, N, BS);
        } else if (!this->transposed && rhs.transposed) {
            multiply_kernel<false, true>(this_data, rhs_data, res_data, M, K, N, BS);
        } else {
            multiply_kernel<true, true>(this_data, rhs_data, res_data, M, K, N, BS);
        }
        res.type = details::Generic;
        *this = std::move(res);
    }
    if (std::all_of(this->data.cbegin(), this->data.cend(), [](const T& elem) { return elem == T(0); })) {
        this->type = details::Zero;
    }
    cache.invalidate();
    return *this;
}

template <ComponentType T>
constexpr Matrix<T>& Matrix<T>::operator*=(const T& s) {
    if (s == T(0)) {
        *this = Matrix(m, n);
    } else if (s == T(1) || type == details::Zero) {
        /* no-op */
    } else if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        std::ranges::for_each(data, [&s](T& x) { x *= s; });
        if (type == details::Vandermonde) type = details::Generic;
    } else if (type == details::Diagonal || type == details::Identity) {
        for (size_t mu = 0; mu < m; ++mu) data[mu * n + mu] *= s;
        if (type == details::Identity) type = details::Diagonal;
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::operator/=(const T& s) {
    if (s == T(0)) throw std::invalid_argument("trying to divide components of matrix by zero");
    operator*=(T(1) / s);
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::randomize() {
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
    type = details::Generic;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
size_t Matrix<T>::calculate_weight() const
    requires ReliablyComparableType<T>
{
    if (m == 0 || n == 0) return 0;

    if (type == details::Zero) {
        return 0;
    } else {
        size_t res = data.size() - std::count(data.cbegin(), data.cend(), T(0));
#ifdef CECCO_ERASURE_SUPPORT
        if constexpr (FieldType<T>) res -= std::count_if(data.cbegin(), data.cend(), [](T x) { return x.is_erased(); });
#endif
        return res;
    }
    throw std::logic_error("calculate_weight(): unhandled matrix type");
}

template <ComponentType T>
size_t Matrix<T>::rank() const
    requires FieldType<T>
{
    return cache.template get_or_compute<Rank>([this] { return calculate_rank(); });
}

template <ComponentType T>
Vector<T> Matrix<T>::diagonal() const {
    if (m != n) throw std::invalid_argument("trying to extract diagonal of non-square matrix");
    Vector<T> res(n);
    for (size_t i = 0; i < n; ++i) res.data[i] = (*this)(i, i);
    return res;
}

template <ComponentType T>
Polynomial<T> Matrix<T>::characteristic_polynomial() const
    requires FieldType<T>
{
    if (m != n) throw std::invalid_argument("trying to calculate characteristic polynomial of non-square matrix");
    if (m == 0) throw std::invalid_argument("trying to calculate characteristic polynomial of empty matrix");
    if (m == 1) return Polynomial<T>({-(*this)(0, 0), 1});
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        // Samuelson–Berkowitz algorithm

        // calculate details::Toeplitz matrices
        std::vector<Matrix<T>> TM;  // details::Toeplitz matrices
        TM.reserve(m);
        for (size_t i = 0; i < m; ++i) {
            // partition matrix
            T a = (*this)(i, i);
            if (m - i == 1) {
                Vector<T> v(2);
                v.set_component(0, -a);
                v.set_component(1, T(1));
                TM.push_back(ToeplitzMatrix(v, 2, 1));
            } else if (m - i == 2) {
                const Matrix C = get_submatrix(i + 1, i, m - i - 1, 1);
                const Matrix R = get_submatrix(i, i + 1, 1, m - i - 1);
                Vector<T> v(4);
                v.set_component(0, -(R * C)(0, 0));
                v.set_component(1, -a);
                v.set_component(2, T(1));
                TM.push_back(ToeplitzMatrix(v, 3, 2));
            } else {
                const Matrix C = get_submatrix(i + 1, i, m - i - 1, 1);
                const Matrix R = get_submatrix(i, i + 1, 1, m - i - 1);
                Matrix A = get_submatrix(i + 1, i + 1, m - i - 1, m - i - 1);
                const Matrix X = A;
                Vector<T> v(2 * (m - i));
                for (size_t j = 0; j < m - i - 2; ++j) {
                    v.set_component(m - i - 3 - j, -((R * A * C)(0, 0)));
                    A *= X;
                }
                v.set_component(m - i - 2, -(R * C)(0, 0));
                v.set_component(m - i - 1, -a);
                v.set_component(m - i, T(1));
                TM.push_back(ToeplitzMatrix(v, m - i + 1, m - i));
            }
        }

        // multiply details::Toeplitz matrices together
        Matrix P = IdentityMatrix<T>(m + 1);
        for (auto it = TM.cbegin(); it != TM.cend(); ++it) {
            P *= *it;
        }

        // extract polynomial from solution/column vector
        Polynomial<T> res;
        for (size_t i = 0; i <= m; ++i) res.set_coefficient(m - i, P(i, 0));

        // Samuelson-Berkowitz raw produces det(A - λI) = (-1)^m · det(λI - A);
        // negate for odd m to obtain the standard monic det(λI - A).
        if (m % 2) res *= T(-1);

        return res;
    } else if (type == details::Zero) {
        Polynomial<T> res({0, 1});
        return res ^ m;
    } else if (type == details::Diagonal) {
        Polynomial<T> res({1});
        for (size_t mu = 0; mu < m; ++mu) res *= Polynomial<T>({-(*this)(mu, mu), T(1)});
        return res;
    } else if (type == details::Identity) {
        Polynomial<T> res({-1, 1});
        return res ^ m;
    }
    throw std::logic_error("characteristic_polynomial(): unhandled matrix type");
}

template <ComponentType T>
Matrix<T> Matrix<T>::basis_of_nullspace() const
    requires FieldType<T>
{
    if (type == details::Zero) return IdentityMatrix<T>(n);
    if (type == details::Identity) return Matrix<T>(1, n, T(0));
    if (type == details::Diagonal) {
        std::vector<size_t> zero_positions;
        for (size_t i = 0; i < m; ++i)
            if ((*this)(i, i) == T(0)) zero_positions.push_back(i);
        if (zero_positions.empty()) return Matrix<T>(1, n, T(0));
        Matrix<T> B(zero_positions.size(), n);
        for (size_t k = 0; k < zero_positions.size(); ++k) B.set_component(k, zero_positions[k], T(1));
        return B;
    }

    Matrix<T> temp(*this);
    size_t r = 0;
    temp.rref(&r);

    if (n - r == 0) return Matrix<T>(1, n, T(0));

    Matrix B(n - r, n);

    std::vector<size_t> mocols;  // "minus one columns"
    mocols.reserve(n - r);
    size_t i = 0;
    for (size_t j = 0; j < n; ++j) {
        if (i < m) {
            if (temp(i, j) == T(1))
                ++i;
            else
                mocols.push_back(j);
        } else {
            mocols.push_back(j);
        }
    }

    size_t offset = 0;
    for (size_t i = 0; i < m + offset + 1; ++i) {
        for (size_t k = offset; k < n - r; ++k) {
            if (i + offset == mocols[k]) {
                B.set_component(k, i + offset, -T(1));
                ++offset;
            } else {
                B.set_component(k, i + offset, temp(i, mocols[k]));
            }
        }
    }

    return B;
}

template <ComponentType T>
T Matrix<T>::determinant() const
    requires FieldType<T>
{
    if (m != n) throw std::invalid_argument("trying to calculate determinant of non-square matrix");
    if (m == 0) throw std::invalid_argument("trying to calculate determinant of empty matrix");
    if (m == 1) return ((*this))(0, 0);
    if (type == details::Generic || type == details::Toeplitz) {
        // char_poly is monic det(λI - A), so [0] = (-1)^m · det(A).
        const auto c = characteristic_polynomial()[0];
        return (m % 2 == 0) ? c : -c;
    } else if (type == details::Vandermonde) {
        T s(1);
        for (size_t mu = 1; mu < m; ++mu)
            for (size_t i = 0; i < mu; ++i) s *= (*this)(1, mu) - (*this)(1, i);
        return s;
    } else if (type == details::Zero) {
        return T(0);
    } else if (type == details::Diagonal) {
        T s(1);
        for (size_t i = 0; i < m; ++i) {
            if ((*this)(i, i) == T(0)) return T(0);
            s *= (*this)(i, i);
        }
        return s;
    } else if (type == details::Identity) {
        return T(1);
    }
    throw std::logic_error("determinant(): unhandled matrix type");
}

template <ComponentType T>
std::vector<T> Matrix<T>::eigenvalues() const
    requires FiniteFieldType<T>
{
    const auto p = characteristic_polynomial();
    std::vector<T> res;
    for (size_t j = 0; j < T::get_size(); ++j) {
        T element = T(j);
        if (p(element) == T(0)) res.push_back(element);
    }
    return res;
}

template <ComponentType T>
std::vector<Vector<T>> Matrix<T>::rowspace() const
    requires FieldType<T>
{
    const size_t r = rank();
    if (sqm<InfInt>(T::get_size(), r) > sqm<InfInt>(10, 10)) {
        throw std::out_of_range("row space too big (more than 10^10 elements) to compute all elements");
    }
    const auto size = sqm<size_t>(sqm<size_t>(T::get_p(), T::get_m()), r);
    std::vector<Vector<T>> res;
    res.reserve(size);
    for (size_t counter = 0; counter < size; ++counter) {
        Vector<T> temp(n);
        for (size_t i = 0; i < r; ++i) {
            T scalar(counter / sqm<size_t>(sqm<size_t>(T::get_p(), T::get_m()), i) %
                     sqm<size_t>(T::get_p(), T::get_m()));
            if (scalar != T(0)) {
                temp += scalar * get_row(r - i - 1);
            }
        }
        res.push_back(std::move(temp));
    }
    // sort vectors so that two spans consisting of the same vectors compare (==) to equal
    std::sort(res.begin(), res.end(), [](Vector<T>& a, Vector<T>& b) { return a.as_integer() > b.as_integer(); });
    return res;
}

template <ComponentType T>
template <typename U>
Matrix<T>& Matrix<T>::set_component(size_t i, size_t j, U&& c)
    requires std::convertible_to<std::decay_t<U>, T>
{
    if (i >= m || j >= n) throw std::invalid_argument("trying to access non-existent component of matrix");

    T& old_value = (!transposed) ? data[i * n + j] : data[i + j * m];

    T new_value(std::forward<U>(c));
    if (old_value == new_value) return *this;

    switch (type) {
        case details::Generic:
            break;

        case details::Diagonal:
            if (i != j) type = details::Generic;
            break;

        case details::Zero:
            if (i == j && m == n)
                type = details::Diagonal;
            else
                type = details::Generic;
            break;

        case details::Identity:
            if (i == j)
                type = details::Diagonal;  // we already know it is not one
            else
                type = details::Generic;  // we already know it is not zero
            break;

        case details::Vandermonde:
        case details::Toeplitz:
            type = details::Generic;
            break;

        default:
            type = details::Generic;
            break;
    }

    if (!transposed)
        data[i * n + j] = new_value;
    else
        data[i + j * m] = new_value;

    cache.invalidate();

    return *this;
}

template <ComponentType T>
const T& Matrix<T>::operator()(size_t i, size_t j) const {
    if (i >= m || j >= n) throw std::invalid_argument("trying to access non-existent component of matrix");
    if (!transposed) return data[i * n + j];
    return data[i + j * m];
}

template <ComponentType T>
Vector<T> Matrix<T>::get_row(size_t i) const {
    if (i >= m) throw std::invalid_argument("trying to access non-existent row");
    Vector<T> res(n);
    if (!transposed) {
        std::copy(data.begin() + i * n, data.begin() + (i + 1) * n, res.data.begin());
    } else {
        for (size_t j = 0; j < n; ++j) res.data[j] = (*this)(i, j);
    }
    return res;
}

template <ComponentType T>
Vector<T> Matrix<T>::get_col(size_t j) const {
    if (j >= n) throw std::invalid_argument("trying to access non-existent column");
    Vector<T> res(m);
    if (transposed) {
        std::copy(data.begin() + j * m, data.begin() + (j + 1) * m, res.data.begin());
    } else {
        for (size_t i = 0; i < m; ++i) res.data[i] = (*this)(i, j);
    }
    return res;
}

template <ComponentType T>
Matrix<T> Matrix<T>::get_submatrix(size_t i, size_t j, size_t h, size_t w) const {
    if (i + h > m || j + w > n)
        throw std::invalid_argument(
            "trying to extract a submatrix with incompatible "
            "dimensions");
    Matrix res(h, w);
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        if (!transposed && !res.transposed) {
            for (size_t mu = 0; mu < h; ++mu)
                std::copy(data.begin() + (i + mu) * n + j, data.begin() + (i + mu) * n + j + w,
                          res.data.begin() + mu * w);
            res.type = details::Generic;
        } else {
            for (size_t mu = 0; mu < h; ++mu)
                for (size_t nu = 0; nu < w; ++nu) res.set_component(mu, nu, (*this)(i + mu, j + nu));
        }
        if (type == details::Vandermonde && i == 0) {
            res.type = details::Vandermonde;
        } else if (type == details::Toeplitz) {
            res.type = details::Toeplitz;
        }
    } else if (type == details::Zero) {
        /* no-op */
    } else if (type == details::Diagonal || type == details::Identity) {
        for (size_t mu = 0; mu < h; ++mu)
            for (size_t nu = 0; nu < w; ++nu)
                if (i + mu == j + nu) res.set_component(mu, nu, (*this)(i + mu, j + nu));
        if (i == j) {
            if (type == details::Diagonal && h == w) {
                res.type = details::Diagonal;
            } else if (type == details::Identity) {
                res.type = details::Identity;
            }
        }
    }
    return res;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::set_submatrix(size_t i, size_t j, const Matrix& N) {
    if (m < i + N.m || n < j + N.n)
        throw std::invalid_argument(
            "trying to replace submatrix with "
            "matrix of incompatible dimensions");

    if (!transposed && !N.transposed) {
        for (size_t mu = 0; mu < N.m; ++mu)
            std::copy(N.data.begin() + mu * N.n, N.data.begin() + (mu + 1) * N.n, data.begin() + (i + mu) * n + j);
    } else {
        for (size_t mu = 0; mu < N.m; ++mu)
            for (size_t nu = 0; nu < N.n; ++nu) set_component(i + mu, j + nu, N(mu, nu));
    }
    type = details::Generic;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::horizontal_join(const Matrix& other) {
    if (m != other.m)
        throw std::invalid_argument(
            "trying to horizontally join two "
            "matrices of incompatible dimensions");
    if (type == details::Zero && other.type == details::Zero) {
        *this = ZeroMatrix<T>(m, n + other.n);
    } else {
        Matrix temp(m, n + other.n);
        temp.set_submatrix(0, 0, *this);
        temp.set_submatrix(0, n, other);
        *this = std::move(temp);
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::vertical_join(const Matrix& other) {
    if (n != other.n)
        throw std::invalid_argument(
            "trying to vertically join two "
            "matrices of incompatible dimensions");
    if (type == details::Zero && other.type == details::Zero) {
        *this = ZeroMatrix<T>(m + other.m, n);
    } else {
        Matrix temp(m + other.m, n);
        temp.set_submatrix(0, 0, *this);
        temp.set_submatrix(m, 0, other);
        *this = std::move(temp);
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::diagonal_join(const Matrix& other) {
    if (type == details::Zero && other.type == details::Zero) {
        *this = ZeroMatrix<T>(m + other.m, n + other.n);
    } else if (type == details::Identity && other.type == details::Identity) {
        *this = IdentityMatrix<T>(m + other.m);
    } else if ((type == details::Diagonal && other.type == details::Identity) ||
               (type == details::Identity && other.type == details::Diagonal) ||
               (type == details::Diagonal && other.type == details::Diagonal)) {
        *this = DiagonalMatrix<T>(concatenate(diagonal(), other.diagonal()));
    } else {
        Matrix temp(m + other.m, n + other.n);
        temp.set_submatrix(0, 0, *this);
        temp.set_submatrix(m, n, other);
        *this = std::move(temp);
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::Kronecker_product(const Matrix& other) {
    if (type == details::Zero || other.type == details::Zero) {
        *this = ZeroMatrix<T>(m * other.m, n * other.n);
    } else if (type == details::Identity && other.type == details::Identity) {
        *this = IdentityMatrix<T>(m * other.m);
    } else if (type == details::Diagonal || type == details::Identity) {
        if (other.type == details::Diagonal || other.type == details::Identity) {
            auto d1 = diagonal();
            auto d2 = other.diagonal();
            Vector<T> d(m * other.m);
            for (size_t idx = 0; idx < m * other.m; ++idx) d.set_component(idx, d1[idx / other.m] * d2[idx % other.m]);
            *this = DiagonalMatrix<T>(std::move(d));
        } else {
            Matrix temp(m * other.m, n * other.n);
            for (size_t mu = 0; mu < m; ++mu) {
                if (type == details::Identity)
                    temp.set_submatrix(mu * other.m, mu * other.n, other);
                else
                    temp.set_submatrix(mu * other.m, mu * other.n, (*this)(mu, mu) * other);
            }
            *this = std::move(temp);
        }
    } else {
        Matrix temp(m * other.m, n * other.n);
        for (size_t mu = 0; mu < m; ++mu)
            for (size_t nu = 0; nu < n; ++nu) temp.set_submatrix(mu * other.m, nu * other.n, (*this)(mu, nu) * other);
        *this = std::move(temp);
    }

    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::swap_rows(size_t i, size_t j) {
    if (i >= m || j >= m) throw std::invalid_argument("trying to swap non-existent row(s)");
    if (i == j) return *this;
    if (type != details::Zero) {
        if (!transposed) {
            std::swap_ranges(data.begin() + i * n, data.begin() + (i + 1) * n, data.begin() + j * n);
        } else {
            auto rank_backup = cache.template get<Rank>();
            for (size_t nu = 0; nu < n; ++nu) {
                const auto temp = (*this)(i, nu);
                set_component(i, nu, (*this)(j, nu));
                set_component(j, nu, temp);
            }
            if (rank_backup) cache.template set<Rank>(*rank_backup);
        }
        type = details::Generic;
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::swap_columns(size_t i, size_t j) {
    transpose();
    swap_rows(i, j);
    transpose();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::scale_row(const T& s, size_t i) {
    if (i >= m) throw std::invalid_argument("trying to scale non-existent row");
    if (s == T(1)) return *this;
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        if (!transposed) {
            std::for_each(data.begin() + i * n, data.begin() + (i + 1) * n, [&s](T& x) { x *= s; });
            if (type == details::Vandermonde || type == details::Toeplitz) type = details::Generic;

        } else {
            for (size_t nu = 0; nu < n; ++nu) data[i + nu * m] *= s;
            if (type == details::Vandermonde || type == details::Toeplitz) type = details::Generic;
        }
    } else if (type == details::Zero) {
        /* no-op */
    } else if (type == details::Diagonal || type == details::Identity) {
        data[i * n + i] *= s;
        if (type == details::Identity) type = details::Diagonal;
    }
    if (s == T(0)) cache.template invalidate<Rank>();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::scale_column(const T& s, size_t i) {
    transpose();
    scale_row(s, i);
    transpose();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::add_scaled_row(const T& s, size_t i, size_t j) {
    if (i >= m || j >= m)
        throw std::invalid_argument("trying to add scaled row to other row, at least one of them is non-existent");
    if (s == T(0)) return *this;
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        if (!transposed) {
            std::transform(data.begin() + j * n, data.begin() + (j + 1) * n, data.begin() + i * n, data.begin() + j * n,
                           [&s](const T& target, const T& source) { return target + s * source; });
            if (type == details::Vandermonde || type == details::Toeplitz) type = details::Generic;
        } else {
            for (size_t nu = 0; nu < n; ++nu) data[j + nu * m] += s * data[i + nu * m];
            if (type == details::Vandermonde || type == details::Toeplitz) type = details::Generic;
        }
    } else if (type == details::Zero) {
        /* no-op */
    } else if (type == details::Diagonal || type == details::Identity) {
        data[j * n + i] += data[i * n + i] * s;
        type = details::Generic;
    }
    if (i == j) cache.template invalidate<Rank>();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::add_scaled_column(const T& s, size_t i, size_t j) {
    transpose();
    add_scaled_row(s, i, j);
    transpose();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::delete_columns(const std::vector<size_t>& v) {
    if (v.empty()) return *this;

    // Validate and create sorted set of unique indices (deduplicate)
    // Careful: implicit sorting is ascending, so need to use reverse iterators in next loop!
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= n) throw std::invalid_argument("trying to delete non-existent column");
    }

    for (auto it = indices.crbegin(); it != indices.crend(); ++it) {
        Matrix left = get_submatrix(0, 0, m, *it);
        const Matrix right = get_submatrix(0, *it + 1, m, n - (*it + 1));
        *this = left.horizontal_join(std::move(right));
    }

    type = details::Generic;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::delete_rows(const std::vector<size_t>& v) {
    if (v.empty()) return *this;

    // Validate and create sorted set of unique indices (deduplicate)
    // Careful: implicit sorting is ascending, so need to use reverse iterators in next loop!
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= m) throw std::invalid_argument("trying to delete non-existent row");
    }

    for (auto it = indices.crbegin(); it != indices.crend(); ++it) {
        Matrix top = get_submatrix(0, 0, *it, n);
        Matrix bottom = get_submatrix(*it + 1, 0, m - (*it + 1), n);
        *this = top.vertical_join(bottom);
    }

    type = details::Generic;
    cache.invalidate();
    return *this;
}

#ifdef CECCO_ERASURE_SUPPORT

template <ComponentType T>
Matrix<T>& Matrix<T>::erase_component(size_t i, size_t j)
    requires FieldType<T>
{
    if (i >= m || j >= n) throw std::invalid_argument("trying to erase component at invalid index");

    if (!transposed)
        data[i * n + j].erase();
    else
        data[i + j * m].erase();

    type = details::Generic;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::unerase_component(size_t i, size_t j)
    requires FieldType<T>
{
    if (i >= m || j >= n) throw std::invalid_argument("trying to un-erase component at invalid index");

    if (!transposed)
        data[i * n + j].unerase();
    else
        data[i + j * m].unerase();

    type = details::Generic;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::erase_columns(const std::vector<size_t>& v)
    requires FieldType<T>
{
    if (v.empty()) return *this;

    // Validate and create sorted set of unique indices (deduplicate)
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= n) throw std::invalid_argument("trying to erase non-existent column");
    }

    // Apply erase using std::for_each
    std::for_each(indices.crbegin(), indices.crend(), [&](auto col) {
        for (size_t row = 0; row < m; ++row) erase_component(row, col);
    });

    type = details::Generic;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::erase_rows(const std::vector<size_t>& v)
    requires FieldType<T>
{
    if (v.empty()) return *this;

    // Validate and create sorted set of unique indices (deduplicate)
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= m) throw std::invalid_argument("trying to erase non-existent row");
    }

    // Apply erase using std::for_each
    std::for_each(indices.crbegin(), indices.crend(), [&](auto row) {
        for (size_t col = 0; col < n; ++col) erase_component(row, col);
    });

    type = details::Generic;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::unerase_columns(const std::vector<size_t>& v)
    requires FieldType<T>
{
    // Validate and create sorted set of unique indices (deduplicate)
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= n) throw std::invalid_argument("trying to un-erase non-existent column");
    }

    // Apply erase using std::for_each
    std::for_each(indices.crbegin(), indices.crend(), [&](auto col) {
        for (size_t row = 0; row < m; ++row) unerase_component(row, col);
    });

    type = details::Generic;
    cache.invalidate();
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::unerase_rows(const std::vector<size_t>& v)
    requires FieldType<T>
{
    // Validate and create sorted set of unique indices (deduplicate)
    std::set<size_t> indices(v.begin(), v.end());
    for (size_t idx : indices) {
        if (idx >= m) throw std::invalid_argument("trying to un-erase non-existent row");
    }

    // Apply erase using std::for_each
    std::for_each(indices.crbegin(), indices.crend(), [&](auto row) {
        for (size_t col = 0; col < n; ++col) unerase_component(row, col);
    });

    type = details::Generic;
    cache.invalidate();
    return *this;
}

#endif

template <ComponentType T>
Matrix<T>& Matrix<T>::reverse_rows() {
    if (type != details::Zero) {
        if (!transposed) {
            // For non-transposed matrices, reverse row-wise using STL
            for (size_t mu = 0; mu < m / 2; ++mu)
                std::swap_ranges(data.begin() + mu * n, data.begin() + (mu + 1) * n, data.begin() + (m - 1 - mu) * n);
        } else {
            auto rank_backup = cache.template get<Rank>();
            for (size_t mu = 0; mu < m / 2; ++mu)
                for (size_t nu = 0; nu < n; ++nu) {
                    const auto temp = (*this)(mu, nu);
                    set_component(mu, nu, (*this)(m - 1 - mu, nu));
                    set_component(m - 1 - mu, nu, temp);
                }
            if (rank_backup) cache.template set<Rank>(*rank_backup);
        }
        type = details::Generic;
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::reverse_columns() {
    if (type != details::Zero) {
        if (!transposed) {
            // For non-transposed matrices, reverse elements within each row
            for (size_t mu = 0; mu < m; ++mu) std::reverse(data.begin() + mu * n, data.begin() + (mu + 1) * n);
        } else {
            auto rank_backup = cache.template get<Rank>();
            for (size_t mu = 0; mu < m; ++mu)
                for (size_t nu = 0; nu < n / 2; ++nu) {
                    const auto temp = (*this)(mu, nu);
                    set_component(mu, nu, (*this)(mu, n - 1 - nu));
                    set_component(mu, n - 1 - nu, temp);
                }
            if (rank_backup) cache.template set<Rank>(*rank_backup);
        }
        if (type != details::Vandermonde) {
            type = details::Generic;
        }
    }
    return *this;
}

template <ComponentType T>
constexpr Matrix<T>& Matrix<T>::fill(const T& s) {
    std::fill(data.begin(), data.end(), s);
    if (s == T(0))
        type = details::Zero;
    else
        type = details::Generic;

    if (s != T(0))
        cache.template set<Rank>(1);
    else
        cache.template set<Rank>(0);
    return *this;
}

template <ComponentType T>
constexpr Matrix<T>& Matrix<T>::transpose() {
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        transposed = !transposed;
        std::swap(m, n);
        if (type == details::Vandermonde) type = details::Generic;
    } else if (type == details::Zero) {
        std::swap(m, n);
    } else if (type == details::Diagonal || type == details::Identity) {
        /* no-op */
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::ref(size_t* rank)
    requires FieldType<T>
{
    if (type == details::Generic || type == details::Toeplitz) {
        size_t h = 0;
        size_t k = 0;

        // Branch ONCE on transpose flag - dispatch to optimized elimination kernels
        if (!this->transposed)
            h = ref_elimination_kernel<false>(this->data.data(), m, n, h, k);
        else
            h = ref_elimination_kernel<true>(this->data.data(), m, n, h, k);

        cache.template set<Rank>(h);
        if (rank != nullptr) *rank = h;
        if (type == details::Toeplitz) type = details::Generic;

    } else if (type == details::Vandermonde) {
        // For Vandermonde matrices, calculate RREF (as special case of REF)
        return rref(rank);
    } else if (type == details::Diagonal) {
        std::vector<size_t> zero_rows;
        size_t r = 0;
        for (size_t i = 0; i < std::min(m, n); ++i) {
            if ((*this)(i, i) == T(0))
                zero_rows.push_back(i);
            else
                ++r;
        }

        if (!zero_rows.empty()) {
            this->delete_rows(zero_rows);
            this->vertical_join(ZeroMatrix<T>(zero_rows.size(), n));
            type = details::Generic;  // No longer purely diagonal
        }

        cache.template set<Rank>(r);
        if (rank != nullptr) *rank = r;
    } else if (type == details::Identity) {
        cache.template set<Rank>(m);
        if (rank != nullptr) *rank = m;
    } else if (type == details::Zero) {
        cache.template set<Rank>(0);
        if (rank != nullptr) *rank = 0;
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::rref(size_t* rank)
    requires FieldType<T>
{
    if (type == details::Generic || type == details::Toeplitz) {
        size_t r = 0;
        this->ref(&r);

        // Branch ONCE on transpose flag - dispatch to optimized backward elimination kernels
        if (!this->transposed)
            rref_backward_elimination_kernel<false>(this->data.data(), m, n, r);
        else
            rref_backward_elimination_kernel<true>(this->data.data(), m, n, r);

        cache.template set<Rank>(r);
        if (rank != nullptr) *rank = r;
        if (r == m && m == n)
            type = details::Identity;
        else if (type == details::Toeplitz)
            type = details::Generic;
    } else if (type == details::Vandermonde) {
        if (m == n) {
            // Case 1: Square details::Vandermonde -> I
            *this = IdentityMatrix<T>(m);
            if (rank != nullptr) *rank = m;
            type = details::Identity;

        } else if (m < n) {
            // Case 2: Wide details::Vandermonde [W | A] with W, A details::Vandermonde -> [I | W^-1 A]
            auto W = this->get_submatrix(0, 0, m, m);
            auto A = this->get_submatrix(0, m, m, n - m);

            W.invert();  // Use known details::Vandermonde inverse of W
            auto M = W * A;

            *this = IdentityMatrix<T>(m);
            this->horizontal_join(std::move(M));

            cache.template set<Rank>(m);
            if (rank != nullptr) *rank = m;
            type = details::Generic;

        } else {
            // Case 3: Tall details::Vandermonde (more rows than columns) -> I above of Z
            *this = IdentityMatrix<T>(n);
            this->vertical_join(ZeroMatrix<T>(m - n, n));

            cache.template set<Rank>(n);
            if (rank != nullptr) *rank = n;
            type = details::Toeplitz;  // very special case of details::Toeplitz...
        }
    } else if (type == details::Diagonal) {
        std::vector<size_t> zero_rows;
        size_t r = 0;
        for (size_t i = 0; i < std::min(m, n); ++i) {
            if ((*this)(i, i) == T(0)) {
                zero_rows.push_back(i);
            } else {
                set_component(i, i, T(1));  // Normalize to 1 for RREF
                ++r;
            }
        }

        if (!zero_rows.empty()) {
            this->delete_rows(zero_rows);
            this->vertical_join(ZeroMatrix<T>(zero_rows.size(), n));
            type = details::Generic;
        } else {
            type = details::Identity;
        }

        cache.template set<Rank>(r);
        if (rank != nullptr) *rank = r;
    } else if (type == details::Zero) {
        cache.template set<Rank>(0);
        if (rank != nullptr) *rank = 0;
    } else if (type == details::Identity) {
        cache.template set<Rank>(m);
        if (rank != nullptr) *rank = m;
    }
    return *this;
}

template <ComponentType T>
Matrix<T>& Matrix<T>::invert()
    requires FieldType<T>
{
    if (type == details::Generic || type == details::Toeplitz) {
        if (m != n) throw std::invalid_argument("trying to invert a non-square matrix");
        const auto I = IdentityMatrix<T>(m);
        Matrix temp(m, 2 * m);
        temp.set_submatrix(0, 0, *this);
        temp.set_submatrix(0, m, I);
        temp.rref();
        if (temp.get_submatrix(0, 0, m, m) != I)
            throw std::invalid_argument("trying to invert a non-invertible matrix");
        *this = temp.get_submatrix(0, m, m, m);
        if (type == details::Toeplitz) type = details::Generic;

    } else if (type == details::Vandermonde) {
        if (m != n) throw std::invalid_argument("trying to invert a non-square details::Vandermonde matrix");
        std::vector<Polynomial<T>> Lagrange_polynomials(m);
        std::fill(Lagrange_polynomials.begin(), Lagrange_polynomials.end(), Polynomial<T>(1));
        for (size_t i = 0; i < m; ++i)
            for (size_t k = 0; k < m; ++k) {
                if (k == i) continue;
                Lagrange_polynomials[i] *= Polynomial<T>({-(*this)(1, k), 1}) / ((*this)(1, i) - (*this)(1, k));
            }
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < m; ++j) set_component(i, j, Lagrange_polynomials[i][j]);
    } else if (type == details::Zero) {
        throw std::invalid_argument("trying to invert a non-invertible matrix/a zero matrix");
    } else if (type == details::Diagonal) {
        // Check for zero diagonal elements and invert
        for (size_t mu = 0; mu < m; ++mu) {
            if ((*this)(mu, mu) == T(0))
                throw std::invalid_argument(
                    "trying to invert a non-invertible matrix/a diagonal matrix with at least one zero on the "
                    "diagonal");
            data[mu * n + mu] = T(1) / data[mu * n + mu];
        }
    } else if (type == details::Identity) {
        /* no-op */
    }
    return *this;
}

template <ComponentType T>
template <FiniteFieldType S>
constexpr Vector<S> Matrix<T>::as_vector() const
    requires FiniteFieldType<T> && ExtensionOf<T, S> && (!std::is_same_v<T, S>)
{
    const auto m = S().template as_vector<T>().get_n();
    if (m != get_m())
        throw std::invalid_argument("trying to create superfield vector from subfield matrix, wrong number of rows");

    Vector<S> res(get_n());
    Matrix<T> Tp(*this);
    Tp.transpose();
    for (size_t i = 0; i < get_n(); ++i) {
        const auto temp = Tp.get_row(i);
        res.set_component(i, S(temp));
    }
    return res;
}

template <ComponentType T>
Vector<T> Matrix<T>::to_vector() const {
    const size_t m = get_m();
    const size_t n = get_n();

    Vector<T> v(m * n);

    size_t k = 0;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            v.set_component(k, (*this)(i, j));
            ++k;
        }
    }
    return v;
}

template <ComponentType T>
void Matrix<T>::export_as_ppm(const std::string& filename) const
    requires FiniteFieldType<T> && (T::get_size() <= 64)
{
    std::ofstream file;
    file.open(filename);

    file << "P3" << std::endl;  // PPM with P3 format
    file << get_n() << " " << get_m() << std::endl;
    file << 255 << std::endl;

    for (size_t i = 0; i < get_m(); ++i) {
        for (size_t j = 0; j < get_n(); ++j) {
#ifdef CECCO_ERASURE_SUPPORT
            if ((*this)(i, j).is_erased()) {
                file << std::setw(3) << 255 << " " << std::setw(3) << 0 << " " << std::setw(3) << 0 << "  ";
                if (j == get_n() - 1) file << std::endl;
                continue;
            }
#endif

            const auto label = (*this)(i, j).get_label();

            const size_t a = 63 - 63 * static_cast<double>(label) / (T::get_size() - 1);

            const uint8_t r = details::colormap[a][0];
            const uint8_t g = details::colormap[a][1];
            const uint8_t b = details::colormap[a][2];

            file << std::setw(3) << static_cast<int>(r) << " " << std::setw(3) << static_cast<int>(g) << " "
                 << std::setw(3) << static_cast<int>(b) << "  ";
        }
        file << std::endl;
    }

    file.close();
}

template <ComponentType T>
size_t Matrix<T>::calculate_rank() const
    requires FieldType<T>
{
    Matrix<T> temp(*this);
    size_t r = 0;
    temp.ref(&r);  // Use REF instead of RREF for better performance
    return r;
}

/* free functions wrt. Matrix */

/*
 * matrix + matrix
 */

template <ComponentType T>
constexpr Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator+(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res += rhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator+(const Matrix<T>& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(std::move(rhs));
    res += lhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator+(Matrix<T>&& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(std::move(lhs));
    res += rhs;
    return res;
}

/*
 * matrix - matrix
 */

template <ComponentType T>
constexpr Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator-(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator-(const Matrix<T>& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(-std::move(rhs));
    res += lhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator-(Matrix<T>&& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

/*
 * matrix * matrix
 */

template <ComponentType T>
constexpr Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator*(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

/*
 * vector * matrix
 */

template <ComponentType T>
constexpr Vector<T> operator*(const Vector<T>& lhs, const Matrix<T>& rhs) {
    return (Matrix<T>(lhs) * rhs).get_row(0);
}

template <ComponentType T>
constexpr Vector<T> operator*(Vector<T>&& lhs, const Matrix<T>& rhs) {
    return (Matrix<T>(std::move(lhs)) * rhs).get_row(0);
}

/*
 * matrix * T
 */

template <ComponentType T>
constexpr Matrix<T> operator*(const Matrix<T>& lhs, const T& rhs) {
    Matrix<T> res(lhs);
    res *= rhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator*(Matrix<T>&& lhs, const T& rhs) {
    Matrix<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

/*
 * T * matrix
 */

template <ComponentType T>
constexpr Matrix<T> operator*(const T& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(rhs);
    res *= lhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator*(const T& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(std::move(rhs));
    res *= lhs;
    return res;
}

/*
 * matrix / T
 */

template <ComponentType T>
constexpr Matrix<T> operator/(const Matrix<T>& lhs, const T& rhs) {
    Matrix<T> res(lhs);
    res /= rhs;
    return res;
}

template <ComponentType T>
constexpr Matrix<T> operator/(Matrix<T>&& lhs, const T& rhs) {
    Matrix<T> res(std::move(lhs));
    res /= rhs;
    return res;
}

template <ComponentType T>
Matrix<T> randomize(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.randomize();
    return res;
}

template <ComponentType T>
Matrix<T> randomize(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.randomize();
    return res;
}

template <ReliablyComparableType T>
constexpr size_t wH(const Matrix<T>& M) {
    return M.wH();
}

template <ComponentType T>
Matrix<T> set_component(auto&& M, size_t i, size_t j, const T& c)
    requires std::convertible_to<std::decay_t<decltype(M)>, Matrix<T>>
{
    Matrix<T> res(std::forward<decltype(M)>(M));
    res.set_component(i, j, c);
    return res;
}

template <ComponentType T>
Matrix<T> get_submatrix(const Matrix<T>& M, size_t i, size_t j, size_t h, size_t w) {
    Matrix<T> res(M);
    return res.get_submatrix(i, j, h, w);
}

template <ComponentType T>
Matrix<T> get_submatrix(Matrix<T>&& M, size_t i, size_t j, size_t h, size_t w) {
    Matrix<T> res(std::move(M));
    return res.get_submatrix(i, j, h, w);
}

template <ComponentType T>
Matrix<T> set_submatrix(const Matrix<T>& M, size_t i, size_t j, const Matrix<T>& N) {
    Matrix<T> res(M);
    res.set_submatrix(i, j, N);
    return res;
}

template <ComponentType T>
Matrix<T> set_submatrix(Matrix<T>&& M, size_t i, size_t j, const Matrix<T>& N) {
    Matrix<T> res(std::move(M));
    res.set_submatrix(i, j, N);
    return res;
}

template <ComponentType T>
Matrix<T> horizontal_join(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res.horizontal_join(rhs);
    return res;
}

template <ComponentType T>
Matrix<T> horizontal_join(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res.horizontal_join(rhs);
    return res;
}

template <ComponentType T>
Matrix<T> horizontal_join(Matrix<T>&& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(std::move(lhs));
    res.horizontal_join(std::move(rhs));
    return res;
}

template <ComponentType T>
Matrix<T> vertical_join(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res.vertical_join(rhs);
    return res;
}

template <ComponentType T>
Matrix<T> vertical_join(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res.vertical_join(rhs);
    return res;
}

template <ComponentType T>
Matrix<T> vertical_join(Matrix<T>&& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(std::move(lhs));
    res.vertical_join(std::move(rhs));
    return res;
}

template <ComponentType T>
Matrix<T> diagonal_join(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res.diagonal_join(rhs);
    return res;
}

template <ComponentType T>
Matrix<T> diagonal_join(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res.diagonal_join(rhs);
    return res;
}

template <ComponentType T>
Matrix<T> diagonal_join(Matrix<T>&& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(std::move(lhs));
    res.diagonal_join(std::move(rhs));
    return res;
}

template <ComponentType T>
Matrix<T> Kronecker_product(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res.Kronecker_product(rhs);
    return res;
}

template <ComponentType T>
Matrix<T> Kronecker_product(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res.Kronecker_product(rhs);
    return res;
}

template <ComponentType T>
Matrix<T> Kronecker_product(Matrix<T>&& lhs, Matrix<T>&& rhs) {
    Matrix<T> res(std::move(lhs));
    res.Kronecker_product(std::move(rhs));
    return res;
}

template <ComponentType T>
Matrix<T> swap_rows(const Matrix<T>& M, size_t i, size_t j) {
    Matrix<T> res(M);
    res.swap_rows(i, j);
    return res;
}

template <ComponentType T>
Matrix<T> swap_columns(const Matrix<T>& M, size_t i, size_t j) {
    Matrix<T> res(M);
    res.swap_columns(i, j);
    return res;
}

template <ComponentType T>
Matrix<T> swap_rows(Matrix<T>&& M, size_t i, size_t j) {
    Matrix<T> res(std::move(M));
    res.swap_rows(i, j);
    return res;
}

template <ComponentType T>
Matrix<T> swap_columns(Matrix<T>&& M, size_t i, size_t j) {
    Matrix<T> res(std::move(M));
    res.swap_columns(i, j);
    return res;
}

template <ComponentType T>
Matrix<T> scale_row(const Matrix<T>& M, const T& s, size_t i) {
    Matrix<T> res(M);
    res.scale_row(s, i);
    return res;
}

template <ComponentType T>
Matrix<T> scale_column(const Matrix<T>& M, const T& s, size_t i) {
    Matrix<T> res(M);
    res.scale_column(s, i);
    return res;
}

template <ComponentType T>
Matrix<T> scale_row(Matrix<T>&& M, const T& s, size_t i) {
    Matrix<T> res(std::move(M));
    res.scale_row(s, i);
    return res;
}

template <ComponentType T>
Matrix<T> scale_column(Matrix<T>&& M, const T& s, size_t i) {
    Matrix<T> res(std::move(M));
    res.scale_column(s, i);
    return res;
}

template <ComponentType T>
Matrix<T> add_scaled_row(const Matrix<T>& M, const T& s, size_t i, size_t j) {
    Matrix<T> res(M);
    res.add_scaled_row(s, i, j);
    return res;
}

template <ComponentType T>
Matrix<T> add_scaled_column(const Matrix<T>& M, const T& s, size_t i, size_t j) {
    Matrix<T> res(M);
    res.add_scaled_column(s, i, j);
    return res;
}

template <ComponentType T>
Matrix<T> add_scaled_row(Matrix<T>&& M, const T& s, size_t i, size_t j) {
    Matrix<T> res(std::move(M));
    res.add_scaled_row(s, i, j);
    return res;
}

template <ComponentType T>
Matrix<T> add_scaled_column(Matrix<T>&& M, const T& s, size_t i, size_t j) {
    Matrix<T> res(std::move(M));
    res.add_scaled_column(s, i, j);
    return res;
}

template <ComponentType T>
Matrix<T> add_row(const Matrix<T>& M, size_t i, size_t j) {
    Matrix<T> res(M);
    res.add_row(i, j);
    return res;
}

template <ComponentType T>
Matrix<T> add_column(const Matrix<T>& M, size_t i, size_t j) {
    Matrix<T> res(M);
    res.add_column(i, j);
    return res;
}

template <ComponentType T>
Matrix<T> add_row(Matrix<T>&& M, size_t i, size_t j) {
    Matrix<T> res(std::move(M));
    res.add_row(i, j);
    return res;
}

template <ComponentType T>
Matrix<T> add_column(Matrix<T>&& M, size_t i, size_t j) {
    Matrix<T> res(std::move(M));
    res.add_column(i, j);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> delete_columns(const Matrix<T>& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(lhs);
    res.delete_columns(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> delete_columns(Matrix<T>&& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(std::move(lhs));
    res.delete_columns(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> delete_column(const Matrix<T>& lhs, size_t i) {
    Matrix<T> res(lhs);
    res.delete_column(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> delete_column(Matrix<T>&& lhs, size_t i) {
    Matrix<T> res(std::move(lhs));
    res.delete_column(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> delete_rows(const Matrix<T>& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(lhs);
    res.delete_rows(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> delete_rows(Matrix<T>&& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(std::move(lhs));
    res.delete_rows(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> delete_row(const Matrix<T>& lhs, size_t i) {
    Matrix<T> res(lhs);
    res.delete_row(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> delete_row(Matrix<T>&& lhs, size_t i) {
    Matrix<T> res(std::move(lhs));
    res.delete_row(i);
    return res;
}

#ifdef CECCO_ERASURE_SUPPORT

template <ComponentType T>
constexpr Matrix<T> erase_component(const Matrix<T>& lhs, size_t i, size_t j) {
    Matrix<T> res(lhs);
    res.erase_component(i, j);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_component(Matrix<T>&& lhs, size_t i, size_t j) {
    Matrix<T> res(std::move(lhs));
    res.erase_component(i, j);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_component(const Matrix<T>& lhs, size_t i, size_t j) {
    Matrix<T> res(lhs);
    res.unerase_component(i, j);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_component(Matrix<T>&& lhs, size_t i, size_t j) {
    Matrix<T> res(std::move(lhs));
    res.unerase_component(i, j);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_columns(const Matrix<T>& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(lhs);
    res.erase_columns(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_columns(Matrix<T>&& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(std::move(lhs));
    res.erase_columns(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_column(const Matrix<T>& lhs, size_t i) {
    Matrix<T> res(lhs);
    res.erase_column(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_column(Matrix<T>&& lhs, size_t i) {
    Matrix<T> res(std::move(lhs));
    res.erase_column(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_columns(const Matrix<T>& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(lhs);
    res.unerase_columns(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_columns(Matrix<T>&& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(std::move(lhs));
    res.unerase_columns(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_column(const Matrix<T>& lhs, size_t i) {
    Matrix<T> res(lhs);
    res.unerase_column(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_column(Matrix<T>&& lhs, size_t i) {
    Matrix<T> res(std::move(lhs));
    res.unerase_column(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_rows(const Matrix<T>& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(lhs);
    res.erase_rows(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_rows(Matrix<T>&& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(std::move(lhs));
    res.erase_rows(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_row(const Matrix<T>& lhs, size_t i) {
    Matrix<T> res(lhs);
    res.erase_row(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> erase_row(Matrix<T>&& lhs, size_t i) {
    Matrix<T> res(std::move(lhs));
    res.erase_row(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_rows(const Matrix<T>& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(lhs);
    res.unerase_rows(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_rows(Matrix<T>&& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(std::move(lhs));
    res.unerase_rows(v);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_row(const Matrix<T>& lhs, size_t i) {
    Matrix<T> res(lhs);
    res.unerase_row(i);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> unerase_row(Matrix<T>&& lhs, size_t i) {
    Matrix<T> res(std::move(lhs));
    res.unerase_row(i);
    return res;
}

#endif

template <ComponentType T>
Matrix<T> reverse_rows(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.reverse_rows();
    return res;
}

template <ComponentType T>
Matrix<T> reverse_rows(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.reverse_rows();
    return res;
}

template <ComponentType T>
Matrix<T> reverse_columns(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.reverse_columns();
    return res;
}

template <ComponentType T>
Matrix<T> reverse_columns(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.reverse_columns();
    return res;
}

template <ComponentType T>
constexpr Matrix<T> fill(const Matrix<T>& m, const T& value) {
    Matrix<T> res(m);
    res.fill(value);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> fill(Matrix<T>&& m, const T& value) {
    Matrix<T> res(std::move(m));
    res.fill(value);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> transpose(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.transpose();
    return res;
}

template <ComponentType T>
constexpr Matrix<T> transpose(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.transpose();
    return res;
}

template <FieldType T>
Matrix<T> rref(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.rref();
    return res;
}

template <FieldType T>
Matrix<T> rref(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.rref();
    return res;
}

template <ComponentType T>
Matrix<T> inverse(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.invert();
    return res;
}

template <ComponentType T>
Matrix<T> inverse(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.invert();
    return res;
}

/**
 * @brief Equality of two matrices
 *
 * @return true iff dimensions match and all components are equal
 *
 * Tag-aware fast paths handle structured operands (e.g. two Toeplitz matrices compare only
 * their first row and column).
 */
template <ReliablyComparableType T>
constexpr bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    if (lhs.m != rhs.m || lhs.n != rhs.n) return false;

    if (lhs.m == 0) {
        return true;
    } else if (lhs.m == 1) {
        return lhs.get_row(0) == rhs.get_row(0);
    } else if (lhs.n == 1) {
        return lhs.get_col(0) == rhs.get_col(0);
    } else if (lhs.type == details::Toeplitz && rhs.type == details::Toeplitz) {
        // Compare left column (in reverse order for details::Toeplitz structure)
        for (size_t mu = 0; mu < lhs.m; ++mu)
            if (lhs(lhs.m - 1 - mu, 0) != rhs(lhs.m - 1 - mu, 0)) return false;
        // Compare top row (excluding first element)
        for (size_t nu = 1; nu < lhs.n; ++nu)
            if (lhs(0, nu) != rhs(0, nu)) return false;
    } else if (lhs.type == details::Vandermonde && rhs.type == details::Vandermonde) {
        return lhs.get_row(1) == rhs.get_row(1);
    } else if ((lhs.type == details::Diagonal && rhs.type == details::Diagonal) ||
               (lhs.type == details::Diagonal && rhs.type == details::Identity) ||
               (lhs.type == details::Identity && rhs.type == details::Diagonal)) {
        return lhs.diagonal() == rhs.diagonal();
    } else if (lhs.type == details::Identity && rhs.type == details::Identity) {
        /* no-op */
    } else if ((lhs.type == details::Zero && rhs.type != details::Zero) ||
               (lhs.type != details::Zero && rhs.type == details::Zero)) {
        return false;
    } else {
        if (!lhs.transposed && !rhs.transposed) {
            return std::equal(lhs.data.begin(), lhs.data.end(), rhs.data.begin());
        } else {
            for (size_t mu = 0; mu < lhs.m; ++mu)
                for (size_t nu = 0; nu < lhs.n; ++nu)
                    if (lhs(mu, nu) != rhs(mu, nu)) return false;
            return true;
        }
    }
    return true;
}

/**
 * @brief Negation of @ref operator==
 *
 * @return true iff dimensions or any component differ
 */
template <ReliablyComparableType T>
constexpr bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    return !(lhs == rhs);
}

/**
 * @brief Pretty-print the matrix with column alignment and bracket borders
 *
 * @return Reference to @p os for chaining
 *
 * An empty matrix is printed as "(empty matrix)".
 */
template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& rhs) {
    if (rhs.m == 0 || rhs.n == 0) {
        os << "  (empty matrix)";
        return os;
    }
    size_t max = 0;
    std::stringstream ss;
    for (size_t i = 0; i < rhs.m; ++i)
        for (size_t j = 0; j < rhs.n; ++j) {
            ss << rhs(i, j);
            max = std::max(ss.str().length(), max);
            ss.str(std::string());  // clear stringstream
        }
    os << "  " << (rhs.m == 1 ? "(" : "⌈");
    for (size_t j = 0; j + 1 < rhs.n; ++j) {
        os << std::setw(max) << rhs(0, j);
        os << " ";  // must be in extra line due to set::setw()
    }
    os << std::setw(max) << rhs(0, rhs.n - 1);
    os << (rhs.m == 1 ? ")" : "⌉");
    if (rhs.m > 1) os << std::endl;
    if (rhs.m > 2) {
        for (size_t i = 1; i + 1 < rhs.m; ++i) {
            os << "  |";
            for (size_t j = 0; j + 1 < rhs.n; ++j) {
                os << std::setw(max) << rhs(i, j);
                os << " ";  // must be in extra line due to set::setw()
            }
            os << std::setw(max) << rhs(i, rhs.n - 1);
            os << "|" << std::endl;
        }
    }
    if (rhs.m > 1) {
        os << "  ⌊";
        for (size_t j = 0; j + 1 < rhs.n; ++j) {
            os << std::setw(max) << rhs(rhs.m - 1, j);
            os << " ";  // must be in extra line due to set::setw()
        }
        os << std::setw(max) << rhs(rhs.m - 1, rhs.n - 1);
        os << "⌋";
    }
    return os;
}

/*
 * factories
 */

/**
 * @brief m × n matrix of zeros (tag @ref details::Zero)
 *
 * @param m Number of rows
 * @param n Number of columns
 * @return Newly constructed zero matrix
 */
template <ComponentType T>
constexpr Matrix<T> ZeroMatrix(size_t m, size_t n) {
    return Matrix<T>(m, n);
}

/**
 * @brief m × m identity matrix I_m (tag @ref details::Identity)
 *
 * @param m Side length
 * @return Newly constructed identity matrix
 */
template <ComponentType T>
constexpr Matrix<T> IdentityMatrix(size_t m) {
    auto res = Matrix<T>(m, m);
    for (size_t i = 0; i < m; ++i) res.set_component(i, i, T(1));
    res.type = details::Identity;
    return res;
}

/**
 * @brief Permutation matrix P with P_{i, perm[i]} = 1
 *
 * @param perm Permutation of {0, …, m-1}; m is inferred from `perm.size()`
 * @return m × m permutation matrix; the identity permutation returns @ref IdentityMatrix
 *
 * @throws std::invalid_argument if perm contains out-of-range or duplicate indices
 */
template <ComponentType T>
Matrix<T> PermutationMatrix(const std::vector<size_t>& perm) {
    const size_t m = perm.size();

    std::vector<bool> seen(m, false);
    bool is_identity_perm = true;
    for (size_t i = 0; i < m; ++i) {
        if (perm[i] >= m || seen[perm[i]])
            throw std::invalid_argument("Trying to construct permutation matrix from a list that is not a permutation");
        seen[perm[i]] = true;
        if (perm[i] != i) is_identity_perm = false;
    }

    if (is_identity_perm) return IdentityMatrix<T>(m);

    auto res = Matrix<T>(m, m);
    for (size_t i = 0; i < m; ++i) res.set_component(i, perm[i], T(1));

    return res;
}

/**
 * @brief m × m exchange matrix (ones on the anti-diagonal)
 *
 * @param m Side length
 * @return Matrix with E_{i, j} = 1 iff i + j = m − 1
 */
template <ComponentType T>
constexpr Matrix<T> ExchangeMatrix(size_t m) {
    auto res = IdentityMatrix<T>(m);
    res.reverse_columns();
    return res;
}

/**
 * @brief Diagonal matrix with diagonal v (tag @ref details::Diagonal)
 *
 * @param v Vector of diagonal entries; the result is `v.length()` × `v.length()`
 * @return Diagonal matrix with v[i] on position (i, i), zeros elsewhere
 */
template <ComponentType T>
constexpr Matrix<T> DiagonalMatrix(const Vector<T>& v) {
    const size_t m = v.get_n();
    Matrix<T> res(m, m);
    for (size_t i = 0; i < m; ++i) res.set_component(i, i, v[i]);
    res.type = details::Diagonal;
    return res;
}

/**
 * @brief m × n Toeplitz matrix from its diagonal entries (tag @ref details::Toeplitz)
 *
 * @param v Diagonal values, ordered so that v[i − j + n − 1] sits on entry (i, j); length m + n − 1
 * @param m Number of rows
 * @param n Number of columns
 * @return Toeplitz matrix; each descending diagonal is constant
 *
 * @throws std::invalid_argument if `v.length() != m + n - 1`
 */
template <ComponentType T>
constexpr Matrix<T> ToeplitzMatrix(const Vector<T>& v, size_t m, size_t n) {
    if (v.get_n() != m + n - 1)
        throw std::invalid_argument(
            "vector for constructing m x n details::Toeplitz matrix must have "
            "length m+n-1");
    Matrix<T> res(m, n);

    // Fill first column: v[0] to v[m-1] in reverse order
    for (size_t i = 0; i < m; ++i) res.set_component(m - 1 - i, 0, v[i]);

    // Fill first row: v[m-1] to v[m+n-2]
    for (size_t j = 1; j < n; ++j) res.set_component(0, j, v[m - 1 + j]);

    // Fill remaining elements using diagonal copy pattern
    for (size_t i = 1; i < m; ++i)
        for (size_t j = 1; j < n; ++j) res.set_component(i, j, res(i - 1, j - 1));
    res.type = details::Toeplitz;
    return res;
}

/**
 * @brief m × n Hankel matrix from its anti-diagonal entries
 *
 * @param v Anti-diagonal values; length m + n − 1
 * @param m Number of rows
 * @param n Number of columns
 * @return Hankel matrix; each ascending anti-diagonal is constant
 *
 * Constructed as `ToeplitzMatrix(reverse(v), m, n) * ExchangeMatrix(n)`.
 *
 * @throws std::invalid_argument if `v.length() != m + n - 1`
 */
template <ComponentType T>
constexpr Matrix<T> HankelMatrix(const Vector<T>& v, size_t m, size_t n) {
    return ToeplitzMatrix<T>(reverse(v), m, n) * ExchangeMatrix<T>(n);
}

/**
 * @brief Vandermonde matrix V_{i, j} = v[j]^i (tag @ref details::Vandermonde)
 *
 * @param v Evaluation points; must be pairwise distinct
 * @param m Number of rows (i.e. the highest power is m − 1)
 * @return m × `v.length()` Vandermonde matrix
 *
 * @throws std::invalid_argument if v is empty, has duplicates, or m is zero
 */
template <ComponentType T>
constexpr Matrix<T> VandermondeMatrix(const Vector<T>& v, size_t m) {
    const size_t n = v.get_n();
    if (n == 0)
        throw std::invalid_argument(
            "vector for constructing details::Vandermonde matrix must have "
            "at least one element");
    if (m == 0) throw std::invalid_argument("trying to construct details::Vandermonde matrix with zero rows");
    if (!v.is_pairwise_distinct())
        throw std::invalid_argument(
            "vector for constructing details::Vandermonde matrix must have pairwise distinct elements");

    Matrix<T> res(m, n);

    // First row: all ones

    for (size_t i = 0; i < n; ++i) res.set_component(0, i, T(1));

    if (m > 1) {
        // Second row: copy from vector v
        for (size_t i = 0; i < n; ++i) res.set_component(1, i, v[i]);

        // Remaining rows: compute powers
        for (size_t j = 2; j < m; ++j)
            for (size_t i = 0; i < n; ++i) res.set_component(j, i, res(j - 1, i) * v[i]);
    }
    res.type = details::Vandermonde;
    return res;
}

/**
 * @brief m × m upper shift matrix (ones on the superdiagonal)
 *
 * @param m Side length
 * @return Matrix with U_{i, i+1} = 1, all other entries 0
 */
template <ComponentType T>
constexpr Matrix<T> UpperShiftMatrix(size_t m) {
    Matrix<T> res(m, m);
    for (size_t i = 0; i + 1 < m; ++i) res.set_component(i, i + 1, T(1));
    return res;
}

/**
 * @brief m × m lower shift matrix (ones on the subdiagonal); transpose of @ref UpperShiftMatrix
 *
 * @param m Side length
 * @return Matrix with L_{i+1, i} = 1, all other entries 0
 */
template <ComponentType T>
constexpr Matrix<T> LowerShiftMatrix(size_t m) {
    return transpose(UpperShiftMatrix<T>(m));
}

/**
 * @brief Companion matrix of a monic polynomial p(x) = x^n + a_{n−1} x^{n−1} + … + a_0
 *
 * @param poly Monic polynomial of degree n
 * @return n × n matrix whose characteristic polynomial is `poly`
 *
 * Built from a lower shift matrix; the last column holds the negated coefficients of `poly`.
 *
 * @throws std::invalid_argument if `poly` is not monic
 */
template <ComponentType T>
constexpr Matrix<T> CompanionMatrix(const Polynomial<T>& poly) {
    if (!poly.is_monic()) throw std::invalid_argument("companion matrices only defined for monic polynomials");
    Matrix<T> res(transpose(UpperShiftMatrix<T>(poly.degree())));

    // Fill last column with negated polynomial coefficients
    for (size_t i = 0; i < poly.degree(); ++i) res.set_component(i, poly.degree() - 1, -poly[i]);

    return res;
}

}  // namespace CECCO

#endif
