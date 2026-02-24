/**
 * @file matrices.hpp
 * @brief Matrix arithmetic library
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
 * This header file provides a complete implementation of matrix arithmetic and many linear algebra
 * operations. It supports:
 *
 * - **details::Generic matrix operations**: Over any @ref CECCO::ComponentType including finite fields,
 *   floating-point numbers, complex numbers, and signed integers
 * - **Specialized matrix types**: details::Zero, details::Identity, details::Diagonal, details::Vandermonde, and
 * details::Toeplitz matrices with optimized operations
 * - **Optimized linear algebra operations**: REF/RREF with binary field optimizations, cached rank computation,
 *   determinant, nullspace, characteristic polynomial, eigenvalue computation, and matrix inversion
 * - **Cross-field operations**: Safe conversions between matrices over related fields using
 *   @ref CECCO::SubfieldOf, @ref CECCO::ExtensionOf, and @ref CECCO::largest_common_subfield_t
 * - **Vector integration**: Bidirectional conversion Matrix -> Vector -> Matrix
 * - **Performance optimizations**: STL algorithms, move semantics, and type-specific optimizations
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * Matrix<int> U = {{1, 2, 3}, {4, 5, 6}};  // 2x3 matrix
 * Matrix<int> V(2, 3, 7);                  // 2x3 matrix filled with 7s
 * auto W = U + V;                          // Element-wise addition
 * auto X = U * V.transpose();              // Matrix multiplication
 *
 * // Special matrices (factories, results are type Matrix)
 * auto I = IdentityMatrix<double>(3);  // 3x3 identity matrix
 * auto Z = ZeroMatrix<double>(2, 4);   // 2x4 zero matrix
 * Vector<double> v = {1, 2, 3};
 * auto D = DiagonalMatrix(v);          // 3x3 diagonal matrix
 *
 * // Finite field matrices
 * Matrix<Fp<7>> P = {{1, 2}, {3, 4}};
 * auto det = P.determinant();  // Determinant
 * size_t rank = P.rank();      // Cached rank computation
 * P.rref();                    // Bring into RREF
 *
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;
 * Matrix<F4> Q = {{0, 1}, {2, 3}};
 * auto nullspace = Q.basis_of_nullspace();         // Nullspace basis
 * auto char_poly = Q.characteristic_polynomial();  // Characteristic polynomial
 *
 * // Cross-field operations (field tower compatibility)
 * Vector<F4> r(10);
 * r.randomize();
 * auto R = r.as_matrix<F2>();  // Convert vector over superfield to matrix over subfield
 * Matrix<F4> S(R);             // Safe upcast: F₂ ⊆ F₄
 * @endcode
 *
 * @section Matrix_Types
 *
 * The library supports several types of matrices:
 * - **Generic**: General dense matrices with arbitrary elements
 * - **Zero**: Matrices with all zero elements
 * - **Identity**: Identity matrices
 * - **Diagonal**: Diagonal matrices
 * - **Vandermonde**: Vandermonde matrices
 * - **Toeplitz**: Toeplitz matrices
 *
 * @note There is only one class template Matrix<T>! The type of matrix is only for internal use and transparent to the
 * user of Matrix<T>. Example: Both IdentityMatrix<double>(3) and ZeroMatrix<double>(2, 4) return an instance
 * of class (template) Matrix<T>!
 *
 * @section Performance_Features
 *
 * - **Optimized algorithms**: REF (Row Echelon Form) for efficient rank computation, binary field optimizations using
 * constexpr if
 * - **High-performance caching**: Rank computation uses caching
 * - **Move semantics**: Optimal performance for temporary matrix operations
 * - **STL integration**: Uses standard algorithms for optimal compiler optimization
 * - **Type safety**: C++20 concepts prevent invalid operations:
 *   - @ref CECCO::ComponentType Ensures valid component types
 *   - @ref CECCO::largest_common_subfield_t Enables generalized cross-field conversions
 *
 * @see @ref fields.hpp for fields and field arithmetic
 * @see @ref vectors.hpp for vectors and associated operations
 * @see @ref field_concepts_traits.hpp for type constraints and field relationships (C++20 concepts)
 */

#ifndef MATRICES_HPP
#define MATRICES_HPP

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <set>
#include <sstream>
#include <unordered_map>
// #include <string> // transitive through field_concepts_traits.hpp
// #include <vector> // transitive through helpers.hpp

#include "field_concepts_traits.hpp"
// #include "helpers.hpp" // transitive through field_concepts_traits.hpp
// #include "InfInt.hpp" // transitive through field_concepts_traits.hpp

namespace CECCO {

namespace details {

/**
 * @brief Enumeration of matrix types for operations (internal use only)
 *
 * This enum enables type-specific optimizations by tracking the structural properties
 * of matrices. Different matrix types allow for specialized algorithms that can
 * dramatically improve performance.
 *
 * @note Matrix type information is automatically maintained during operations.
 *       Operations that break the structure will demote to details::Generic type.
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
std::ostream& operator<<(std::ostream& os, const Matrix<T>& rhs) noexcept;

/**
 * @class Matrix
 * @brief Generic matrix class for error control coding (CECCO) and finite field applications
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType concept. Supported types include:
 *   - **Finite field types**: @ref CECCO::Fp, @ref CECCO::Ext satisfying @ref CECCO::FiniteFieldType
 *   - **Floating-point type**: `double`
 *   - **Complex type**: `std::complex<double>`
 *   - **Signed integer types**: Signed integer types including `InfInt` satisfying @ref CECCO::SignedIntType
 *
 * The Matrix class provides a linear algebra framework for error control
 * coding applications. It supports both dense and structured matrix types with automatic
 * optimization based on matrix structure.
 *
 * @section Implementation_Notes
 *
 * - **Cross-field compatibility**: Safe conversions between related field types using concepts
 * - **CECCO-specific operations**: Hamming weight, Hamming distance, burst length calculations
 * - **Type safety**: Compile-time validation of field relationships and operations
 *
 * - **Automatic type optimization**: Recognizes and optimizes for special matrix structures
 *   (Zero, Identity, Diagonal, Vandermonde, Toeplitz) with significant
 * performance gains
 * - **Cross-field compatibility**: Safe conversions between matrices over related fields
 *   using C++20 concepts for field tower relationships
 * - **Optimized linear algebra**: REF/RREF with binary field optimizations, cached rank computation,
 *   determinant, nullspace, eigenvalues, matrix inversion, characteristic polynomial computation
 *
 * @section Matrix_Types
 *
 * The library supports several types of matrices:
 * - **Generic**: General dense matrices with arbitrary elements
 * - **Zero**: Matrices with all zero elements
 * - **Identity**: Identity matrices
 * - **Diagonal**: Diagonal matrices
 * - **Vandermonde**: Vandermonde matrices
 * - **Toeplitz**: Toeplitz matrices
 *
 * @section Usage_Examples
 *
 * @code{.cpp}
 * Matrix<int> U = {{1, 2, 3}, {4, 5, 6}};  // 2x3 matrix
 * Matrix<int> V(2, 3, 7);                  // 2x3 matrix filled with 7s
 * auto W = U + V;                          // Element-wise addition
 * auto X = U * V.transpose();              // Matrix multiplication
 *
 * // Special matrices (factories, results are type Matrix)
 * auto I = IdentityMatrix<double>(3);  // 3x3 identity matrix
 * auto Z = ZeroMatrix<double>(2, 4);   // 2x4 zero matrix
 * Vector<double> v = {1, 2, 3};
 * auto D = DiagonalMatrix(v);          // 3x3 diagonal matrix
 *
 * // Finite field matrices
 * Matrix<Fp<7>> P = {{1, 2}, {3, 4}};
 * auto det = P.determinant();  // Determinant
 * size_t rank = P.rank();      // Cached rank computation
 * P.rref();                    // Bring into RREF
 *
 * using F2 = Fp<2>;
 * using F4 = Ext<F2, {1, 1, 1}>;
 * Matrix<F4> Q = {{0, 1}, {2, 3}};
 * auto nullspace = Q.basis_of_nullspace();         // Nullspace basis
 * auto char_poly = Q.characteristic_polynomial();  // Characteristic polynomial
 *
 * // Cross-field operations (field tower compatibility)
 * Vector<F4> r(10);
 * r.randomize();
 * auto R = r.as_matrix<F2>();  // Convert vector over superfield to matrix over subfield
 * Matrix<F4> S(R);             // Safe upcast: F₂ ⊆ F₄
 * @endcode
 *
 * @section Template Constraints
 *
 * - **Basic operations**: Available for all @ref CECCO::ComponentType
 * - **Field operations**: RREF, inversion, nullspace require @ref CECCO::FieldType
 * - **Cross-field operations**: Require same characteristic using @ref CECCO::largest_common_subfield_t
 * - **Finite field specific**: Some operations require @ref CECCO::FiniteFieldType
 *
 * @warning Matrix operations assume compatible dimensions. Operations on incompatible
 *          matrices throw `std::invalid_argument` exceptions.
 *
 * @note The class maintains strong exception safety guarantees. Failed operations
 *       leave the matrix in its original state.
 *
 * @see @ref CECCO::Vector for vector operations and matrix-vector conversions
 * @see @ref CECCO::details::matrix_type_t for matrix type optimizations
 * @see @ref CECCO::ComponentType for supported component types
 * @see @ref CECCO::FieldType, @ref CECCO::FiniteFieldType for field operation constraints
 * @see @ref CECCO::largest_common_subfield_t for cross-field operation requirements
 */
template <ComponentType T>
class Matrix {
    friend constexpr Matrix<T> IdentityMatrix<>(size_t m);
    friend constexpr Matrix<T> DiagonalMatrix<>(const Vector<T>& v);
    friend constexpr Matrix<T> ToeplitzMatrix<>(const Vector<T>& v, size_t m, size_t n);
    friend constexpr Matrix<T> VandermondeMatrix<>(const Vector<T>& v, size_t m);
    template <ReliablyComparableType U>
    friend constexpr bool operator==(const Matrix<U>& lhs, const Matrix<U>& rhs) noexcept;
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix& rhs) noexcept;
    template <ComponentType>
    friend class Matrix;

   public:
    /**
     * @brief Default constructor creating an empty matrix
     *
     * Creates a matrix with zero dimensions (0 × 0). The matrix is considered empty
     * and has @ref details::Zero type by default.
     */
    constexpr Matrix() noexcept : data(0) {}

    /**
     * @brief Constructs a zero matrix of specified dimensions
     *
     * @param m Number of rows
     * @param n Number of columns
     *
     * Creates an m × n zero matrix with all elements initialized to T(0).
     * Automatically sets matrix type to @ref details::Zero for optimization.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    constexpr Matrix(size_t m, size_t n) : data(m * n), m(m), n(n), type(details::Zero) {}

    /**
     * @brief Constructs a matrix filled with a specific value from T
     *
     * @param m Number of rows
     * @param n Number of columns
     * @param l Value to assign to all elements
     *
     * Creates an m × n matrix with all elements set to the specified value.
     * Matrix type is automatically set to @ref details::Zero if value is T(0), otherwise @ref details::Generic.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Matrix(size_t m, size_t n, const T& l);

    /**
     * @brief Constructs a matrix from a flat initializer list
     *
     * @param m Number of rows
     * @param n Number of columns
     * @param l Initializer list containing m*n elements from T in row-major order
     *
     * Creates an m × n matrix from elements provided in row-major order.
     *
     * @throws std::invalid_argument if initializer list size doesn't match m*n
     * @throws std::bad_alloc if memory allocation fails
     *
     * @code{.cpp}
     * Matrix<int> M(2, 3, {1, 2, 3, 4, 5, 6});  // 2 × 3 matrix
     * @endcode
     */
    constexpr Matrix(size_t m, size_t n, std::initializer_list<T> l);

    /**
     * @brief Constructs a matrix from nested initializer lists
     *
     * @param l Nested initializer list where each inner list represents a row
     *
     * Creates a matrix from a 2D initializer list structure. Dimensions are
     * automatically determined from the initializer list structure.
     *
     * @throws std::bad_alloc if memory allocation fails
     *
     * @code{.cpp}
     * Matrix<int> M = {{1, 2, 3}, {4, 5, 6}};  // 2 × 3 matrix
     * @endcode
     *
     * @note Rows with different lengths are zero-padded to match the longest row
     */
    Matrix(std::initializer_list<std::initializer_list<T>> l);

    /**
     * @brief Constructs a single-row matrix from a vector
     *
     * @param v Vector to convert into a 1 × n matrix
     *
     * Creates a 1 × n matrix (row vector) from the provided vector.
     * Matrix type is set to @ref details::Toeplitz for optimization.
     *
     * @throws std::bad_alloc if memory allocation fails
     *
     * @code{.cpp}
     * Vector<int> v = {1, 2, 3, 4};
     * Matrix<int> M(v);  // 1 × 4 matrix
     * @endcode
     */
    Matrix(const Vector<T>& v);

    /**
     * @brief Copy constructor
     *
     * @param other Matrix to copy from
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    constexpr Matrix(const Matrix& other)
        : data(other.data),
          m(other.m),
          n(other.n),
          transposed(other.transposed),
          type(other.type),
          cache(other.cache) {}

    /**
     * @brief Move constructor
     *
     * @param other Matrix to move from (left in valid but unspecified state)
     */
    constexpr Matrix(Matrix&& other) noexcept
        : data(std::move(other.data)),
          m(other.m),
          n(other.n),
          transposed(other.transposed),
          type(other.type),
          cache(std::move(other.cache)) {}

    /**
     * @brief Cross-field copy constructor for finite fields with the same characteristic
     *
     * @tparam S Source finite field type that must have the same characteristic as T
     * @param other Matrix over finite field S to copy from
     *
     * Safely converts matrices between any finite fields with the same characteristic using
     * @ref largest_common_subfield_t as the conversion bridge. Supports conversions across
     * different field towers, not just within the same construction hierarchy.
     *
     * @throws std::invalid_argument if field components cannot be represented in target field (downcasting not
     * possible)
     * @throws std::bad_alloc if memory allocation fails
     *
     * @note Available for any finite field types with matching characteristics
     */
    template <FiniteFieldType S>
    constexpr Matrix(const Matrix<S>& other)
        requires FiniteFieldType<T> && (T::get_characteristic() == S::get_characteristic());

    /**
     * @brief Construct matrix from a PPM (P3) image file
     *
     * @param filename Path to the input PPM file (ASCII P3 format)
     *
     * Reads a PPM image and constructs an m × n matrix (m = height, n = width)
     * by mapping each pixel's RGB triplet back to a field element label using the
     * same 64-entry colormap as @ref export_as_ppm. Pixels whose RGB values do not
     * match any colormap entry are treated as erased (when @c CECCO_ERASURE_SUPPORT
     * is defined) or set to a random field element.
     *
     * @note Only available for finite field types with field size at most 64
     * @note The input file must use the colormap palette. To convert any image
     *       (PNG, JPG, etc.) into the appropriate PPM format, use ImageMagick:
     * @code{.sh}
     * magick input.png -alpha remove -background black +dither -remap palette.ppm -compress none output.ppm
     * @endcode
     *
     * @code{.cpp}
     * using F4 = Ext<Fp<2>, MOD{1, 1, 1}>;
     * Matrix<F4> M("matrix.ppm");
     * @endcode
     */
    Matrix(const std::string& filename)
        requires FiniteFieldType<T> && (T::get_size() <= 64);

    /** @name Assignment Operators
     * @{
     */

    /**
     * @brief Copy assignment operator
     *
     * @param rhs Matrix to copy from
     * @return Reference to this matrix after assignment
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    constexpr Matrix& operator=(const Matrix& rhs);

    /**
     * @brief Move assignment operator
     *
     * @param rhs Matrix to move from (left in valid but unspecified state)
     * @return Reference to this matrix after assignment
     */
    constexpr Matrix& operator=(Matrix&& rhs) noexcept;

    /**
     * @brief Cross-field assignment operator between fields with the same characteristic
     *
     * @tparam S Source field type that must have the same characteristic as T
     * @param rhs Matrix over field S to convert
     * @return Reference to this matrix after assignment
     *
     * Safely converts matrices between any fields with the same characteristic using
     * @ref largest_common_subfield_t as the conversion bridge. Supports conversions across
     * different field towers, not just within the same construction hierarchy.
     *
     * @throws std::invalid_argument if field components cannot be represented in target field (downcasting not
     * possible)
     * @throws std::bad_alloc if memory allocation fails
     *
     * @note Available for any field types with matching characteristics
     */
    template <FieldType S>
    constexpr Matrix& operator=(const Matrix<S>& other)
        requires FiniteFieldType<S> && FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic());

    /** @} */

    /** @name Unary Arithmetic Operations
     * @{
     */

    /**
     * @brief Unary plus operator for lvalue references (identity)
     *
     * @return Copy of this matrix (mathematical identity operation)
     */
    constexpr Matrix operator+() const& noexcept { return *this; }

    /**
     * @brief Unary plus operator for rvalue references (move optimization)
     *
     * @return This matrix moved (mathematical identity operation)
     */
    constexpr Matrix operator+() && noexcept { return std::move(*this); }

    /**
     * @brief Unary minus operator for lvalue references
     *
     * @return New matrix with all elements negated
     */
    constexpr Matrix operator-() const& noexcept;

    /**
     * @brief Unary minus operator for rvalue references (move optimization)
     *
     * @return This matrix with all elements negated in-place
     *
     * @note This modifies the matrix in-place (move operation)
     */
    constexpr Matrix operator-() && noexcept;

    /** @} */

    /** @name Compound Assignment Operations
     * @{
     */

    /**
     * @brief Matrix addition assignment
     *
     * @param rhs Matrix to add to this matrix
     * @return Reference to this matrix after addition
     *
     * Performs element-wise addition: this[i,j] += rhs[i,j] for all valid indices.
     * Matrices must have identical dimensions.
     *
     * @throws std::invalid_argument if matrices have different dimensions
     */
    Matrix& operator+=(const Matrix& rhs);

    /**
     * @brief Matrix subtraction assignment
     *
     * @param rhs Matrix to subtract from this matrix
     * @return Reference to this matrix after subtraction
     *
     * Performs element-wise subtraction: this[i,j] -= rhs[i,j] for all valid indices.
     * Matrices must have identical dimensions.
     *
     * @throws std::invalid_argument if matrices have different dimensions
     */
    Matrix& operator-=(const Matrix& rhs);

    /**
     * @brief Matrix multiplication assignment
     *
     * @param rhs Matrix to multiply with this matrix
     * @return Reference to this matrix after multiplication
     *
     * Performs matrix multiplication: this = this * rhs.
     * Number of columns in this matrix must equal number of rows in rhs.
     *
     * @throws std::invalid_argument if matrix dimensions are incompatible
     */
    Matrix& operator*=(const Matrix& rhs);

    /**
     * @brief Scalar multiplication assignment
     *
     * @param s Scalar value to multiply with
     * @return Reference to this matrix after multiplication
     *
     * Multiplies each matrix element by the scalar: this[i,j] *= s for all elements.
     */
    constexpr Matrix& operator*=(const T& s);

    /**
     * @brief Scalar division assignment
     *
     * @param s Nonzero scalar value to divide by
     * @return Reference to this matrix after division
     *
     * Divides each matrix element by the scalar: this[i,j] /= s for all elements.
     *
     * @throws std::invalid_argument if attempting to divide by zero
     *
     * @warning Reliable results ( (M / s) *  s == M for a matrix M and nonzero scalar s are only guaranteed in case T
     * fulfills concept FieldType<T>
     */
    Matrix& operator/=(const T& s);

    /** @} */

    /** @name Randomization
     * @{
     */

    /**
     * @brief Fill matrix with random values
     * @return Reference to this matrix after randomization
     *
     * Fills the matrix with random values appropriate for the component type:
     * - **Finite fields**: Using field-specific randomization
     * - **Signed integers**: Uniform random values in range [-100, 100]
     * - **Complex numbers**: Real and imaginary parts uniform random in [-1.0, 1.0]
     * - **Double**: Uniform random values in range [-1.0, 1.0]
     *
     * @note Matrix type becomes @ref details::Generic after randomization, actual structure is not checked
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
     * @brief Check if matrix is zero
     *
     * Sets type to details::Zero in case the matrix is zero.
     *
     * @return true if matrix has only zero components, false otherwise
     *
     * @see @ref CECCO::details::matrix_type_t for matrix type optimizations
     */
    constexpr bool is_zero() noexcept {
        if (type == details::Zero) return true;
        const bool b = std::ranges::all_of(data, [](const T& v) { return v == T(0); });
        if (b) type = details::Zero;
        return b;
    }

    /**
     * @brief Check if matrix is zero
     *
     * @return true if matrix has only zero components, false otherwise
     */
    constexpr bool is_zero() const noexcept {
        if (type == details::Zero) return true;
        return std::ranges::all_of(data, [](const T& v) { return v == T(0); });
    }

    /**
     * @brief Compute Hamming weight (number of non-zero elements)
     *
     * @return Number of non-zero elements in the matrix
     *
     * Counts the number of elements that are not equal to T(0).
     *
     * @note Only for types fulfilling CECCO::ReliablyComparableType.
     */
    constexpr size_t wH() const noexcept
        requires ReliablyComparableType<T>;

    /**
     * @brief Compute matrix rank with caching
     *
     * @return Rank of the matrix (dimension of row/column space)
     *
     * Computes the rank using Gaussian elimination (REF algorithm).
     * Uses caching for repeated calls.
     *
     * @note Only available for field types (requires division)
     */
    size_t rank() const
        requires FieldType<T>;

    /**
     * @brief Check if matrix is invertible
     *
     * @return true if matrix is square and has full rank, false otherwise
     *
     * @note Only available for field types
     */
    bool is_invertible() const
        requires FieldType<T>
    {
        return m == n && rank() == m;
    }

    /**
     * @brief Extract the main diagonal as a vector
     *
     * @return Vector containing the diagonal elements
     *
     * Returns a vector containing the elements on the main diagonal. For square matrices only.
     *
     * @throws std::invalid_argument if matrix is not square
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector<T> diagonal() const;

    /**
     * @brief Compute characteristic polynomial
     *
     * @return Characteristic polynomial det(λI - A)
     *
     * Computes the characteristic polynomial using the Samuelson-Berkowitz algorithm.
     * For square matrices only. The result is a polynomial of degree m.
     *
     * @throws std::invalid_argument if matrix is not square or empty
     *
     * @note Specialized algorithms for structured matrices (@ref details:Diagonal, @ref details::Vandermonde)
     * @note Only available for field types
     */
    Polynomial<T> characteristic_polynomial() const
        requires FieldType<T>;

    /**
     * @brief Compute basis for the nullspace (kernel)
     *
     * @return Matrix whose rows form a basis for the nullspace
     *
     * Computes a basis for the nullspace using Gaussian elimination.
     * The nullspace consists of all row vectors x such that Ax^T = 0.
     *
     * @note Only available for field types
     */
    Matrix<T> basis_of_nullspace() const
        requires FieldType<T>;

    /**
     * @brief Compute basis for the kernel (alias for nullspace)
     *
     * @return Matrix whose rows form a basis for the kernel
     *
     * Equivalent to basis_of_nullspace(). The kernel and nullspace are
     * the same mathematical concept.
     *
     * @note Only available for field types
     */
    Matrix<T> basis_of_kernel() const
        requires FieldType<T>
    {
        return basis_of_nullspace();
    }

    /**
     * @brief Compute matrix determinant
     *
     * @return Determinant of the matrix
     *
     * Computes the determinant using algorithms based on matrix type.
     * For square matrices only.
     *
     * @throws std::invalid_argument if matrix is not square or empty
     *
     * @note Returns T(0) for singular matrices
     * @note Only available for field types that support division operations
     */
    T determinant() const
        requires FieldType<T>;

    /**
     * @brief Compute eigenvalues of the matrix
     *
     * @return Vector of eigenvalues
     *
     * Computes eigenvalues by finding roots of the characteristic polynomial.
     * For square matrices only.
     *
     * @throws std::invalid_argument if matrix is not square
     *
     * @note Only available for finite field types
     */
    std::vector<T> eigenvalues() const
        requires FiniteFieldType<T>;

    /**
     * @brief Computes all vectors of the row space
     *
     * @return Container of all vectors of the row space
     *
     * @warning The row space can be exceedingly large!!
     *
     * @note Only available for field types
     */
    std::vector<Vector<T>> rowspace() const
        requires FieldType<T>;

    /**
     * @brief Compute span of matrix rows (alias for row space)
     *
     * @return Container of all vectors of the row space
     *
     * Equivalent to rowspace(). Computes the span of the matrix rows.
     *
     * @note Only available for field types
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
     * @brief Set component value using perfect forwarding
     *
     * @tparam U Type that can be converted to T
     * @param i Row index (0-based)
     * @param j Column index (0-based)
     * @param c Value to forward into the component
     * @return Reference to this matrix after modification
     *
     * Efficiently forwards the value into the specified component position.
     * Handles both lvalue and rvalue references optimally.
     *
     * @throws std::invalid_argument if either index is out of bounds
     *
     * @note Matrix type may change decay after this operation
     */
    template <typename U>
    Matrix& set_component(size_t i, size_t j, U&& c)
        requires std::convertible_to<std::decay_t<U>, T>;

    /**
     * @brief Access matrix element (const)
     *
     * @param i Row index (0-based)
     * @param j Column index (0-based)
     * @return Const reference to element at position (i ,j)
     *
     * Provides read-only access to matrix elements with bounds checking.
     *
     * @throws std::invalid_argument if indices are out of bounds
     */
    const T& operator()(size_t i, size_t j) const;

    /**
     * @brief Extract a row as a vector
     *
     * @param i Row index to extract
     * @return Vector containing the elements of row i
     *
     * Creates a new vector containing all elements from the specified row.
     *
     * @throws std::invalid_argument if row index is out of bounds
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector<T> get_row(size_t i) const;

    /**
     * @brief Extract a column as a vector
     *
     * @param j Column index to extract
     * @return Vector containing the elements of column j
     *
     * Creates a new vector containing all elements from the specified column. This implies that the column is
     * transposed (since @ref Vector realizes only row vectors).
     *
     * @throws std::invalid_argument if column index is out of bounds
     * @throws std::bad_alloc if memory allocation fails
     */
    Vector<T> get_col(size_t j) const;

    /**
     * @brief Extract a submatrix
     *
     * @param i Starting row index
     * @param j Starting column index
     * @param h Height (number of rows) of submatrix
     * @param w Width (number of columns) of submatrix
     * @return Submatrix containing elements from the specified region
     *
     * Extracts a submatrix from the region [i:i+h, j:j+w).
     *
     * @throws std::invalid_argument if submatrix extends beyond matrix bounds
     * @throws std::bad_alloc if memory allocation fails
     */
    Matrix<T> get_submatrix(size_t i, size_t j, size_t h, size_t w) const;

    /**
     * @brief Set a submatrix region
     *
     * @param i Starting row index for placement
     * @param j Starting column index for placement
     * @param N Matrix to copy into this matrix
     * @return Reference to this matrix after submatrix assignment
     *
     * Copies the contents of matrix N into this matrix starting at position (i, j).
     * The target region must fit within this matrix's bounds.
     *
     * @throws std::invalid_argument if submatrix would extend beyond bounds
     *
     * @note Matrix type may change to details::Generic after this operation
     */
    Matrix<T>& set_submatrix(size_t i, size_t j, const Matrix& N);

    /**
     * @brief Join another matrix horizontally (concatenate columns)
     *
     * @param other Matrix to join to the right
     * @return Reference to this matrix after horizontal join
     *
     * Concatenates the columns of another matrix to the right of this matrix.
     * The matrices must have the same number of rows.
     *
     * @throws std::invalid_argument if matrices have different numbers of rows
     * @throws std::bad_alloc if memory allocation fails
     *
     * @code{.cpp}
     * Matrix<int> A = {{1, 2}, {3, 4}};  // 2 × 2 matrix
     * Matrix<int> B = {{5, 6}, {7, 8}};  // 2 × 2 matrix
     * A.horizontal_join(B);              // A becomes 2 × 4 matrix [[1,2,5,6], [3,4,7,8]]
     * @endcode
     */
    Matrix<T>& horizontal_join(const Matrix& other);

    /**
     * @brief Join another matrix vertically (concatenate rows)
     *
     * @param other Matrix to join below
     * @return Reference to this matrix after vertical join
     *
     * Concatenates the rows of another matrix below this matrix.
     * The matrices must have the same number of columns.
     *
     * @throws std::invalid_argument if matrices have different numbers of columns
     * @throws std::bad_alloc if memory allocation fails
     *
     * @code{.cpp}
     * Matrix<int> A = {{1, 2}, {3, 4}};  // 2 × 2 matrix
     * Matrix<int> B = {{5, 6}, {7, 8}};  // 2 × 2 matrix
     * A.vertical_join(B);                // A becomes 4 × 2 matrix [[1,2], [3,4], [5,6], [7,8]]
     * @endcode
     */
    Matrix<T>& vertical_join(const Matrix& other);

    /**
     * @brief Join another matrix diagonally (block diagonal)
     *
     * @param other Matrix to join diagonally
     * @return Reference to this matrix after diagonal join
     *
     * Creates a block diagonal matrix with this matrix in the upper-left
     * and the other matrix in the lower-right. Off-diagonal blocks are zero.
     *
     * @throws std::bad_alloc if memory allocation fails
     *
     * @code{.cpp}
     * Matrix<int> A = {{1, 2}, {3, 4}};  // 2 × 2 matrix
     * Matrix<int> B = {{5, 6}, {7, 8}};  // 2 × 2 matrix
     * A.diagonal_join(B);                // A becomes 4 × 4 block diagonal matrix
     * @endcode
     */
    Matrix<T>& diagonal_join(const Matrix& other) noexcept;

    /**
     * @brief Compute Kronecker product with another matrix
     *
     * @param other Matrix to compute Kronecker product with
     * @return Reference to this matrix after Kronecker product
     *
     * Computes the Kronecker product (tensor product) of this matrix with another.
     * If this matrix is m × n and other is p × q, the result is mp × nq.
     *
     * @throws std::bad_alloc if memory allocation fails
     */
    Matrix<T>& Kronecker_product(const Matrix& other);

    /**
     * @brief Swap two rows of the matrix
     *
     * @param i Index of first row to swap
     * @param j Index of second row to swap
     * @return Reference to this matrix after row swap
     *
     * Exchanges the contents of rows i and j.
     *
     * @throws std::invalid_argument if row indices are out of bounds
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& swap_rows(size_t i, size_t j);

    /**
     * @brief Swap two columns of the matrix
     *
     * @param i Index of first column to swap
     * @param j Index of second column to swap
     * @return Reference to this matrix after column swap
     *
     * Exchanges the contents of columns i and j.
     *
     * @throws std::invalid_argument if column indices are out of bounds
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& swap_columns(size_t i, size_t j);

    /**
     * @brief Scale a row by a scalar value
     *
     * @param s Scalar value to multiply the row by
     * @param i Index of row to scale
     * @return Reference to this matrix after row scaling
     *
     * Multiplies all elements in row i by the scalar s. This is a fundamental
     * row operation used in Gaussian elimination.
     *
     * @throws std::invalid_argument if row index is out of bounds
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& scale_row(const T& s, size_t i);

    /**
     * @brief Scale a column by a scalar value
     *
     * @param s Scalar value to multiply the column by
     * @param i Index of column to scale
     * @return Reference to this matrix after column scaling
     *
     * Multiplies all elements in column i by the scalar s. This is a fundamental
     * column operation.
     *
     * @throws std::invalid_argument if column index is out of bounds
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& scale_column(const T& s, size_t i);

    /**
     * @brief Add a scaled row to another row
     *
     * @param s Scalar value to multiply row i by before adding
     * @param i Index of source row to scale and add
     * @param j Index of destination row to add to
     * @return Reference to this matrix after row operation
     *
     * Performs the operation: row[j] += s * row[i].
     *
     * @throws std::invalid_argument if row indices are out of bounds
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& add_scaled_row(const T& s, size_t i, size_t j);

    /**
     * @brief Add a scaled column to another column
     *
     * @param s Scalar value to multiply column i by before adding
     * @param i Index of source column to scale and add
     * @param j Index of destination column to add to
     * @return Reference to this matrix after column operation
     *
     * Performs the operation: column[j] += s * column[i].
     *
     * @throws std::invalid_argument if column indices are out of bounds
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& add_scaled_column(const T& s, size_t i, size_t j);

    /**
     * @brief Add one row to another row
     *
     * @param i Index of source row to add
     * @param j Index of destination row to add to
     * @return Reference to this matrix after row operation
     *
     * Performs the operation: row[j] += row[i].
     *
     * @throws std::invalid_argument if row indices are out of bounds
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& add_row(size_t i, size_t j) {
        if (i >= m || j >= m)
            throw std::invalid_argument("trying to add row to other row, at least one of them is non-existent");
        return add_scaled_row(T(1), i, j);
    }

    /**
     * @brief Add one column to another column
     *
     * @param i Index of source column to add
     * @param j Index of destination column to add to
     * @return Reference to this matrix after column operation
     *
     * Performs the operation: column[j] += column[i].
     *
     * @throws std::invalid_argument if column indices are out of bounds
     *
     * @note Matrix type may change to @ref details::Generic after this operation
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
     * @brief Delete specified columns from the matrix
     *
     * @param v Vector of column indices to delete (automatically deduplicated)
     * @return Reference to this matrix after column deletion
     *
     * Removes the specified columns from the matrix.
     *
     * @throws std::invalid_argument if any column index is out of bounds
     * @throws std::bad_alloc if memory allocation fails
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& delete_columns(const std::vector<size_t>& v);

    /**
     * @brief Delete a single column from the matrix
     *
     * @param i Index of column to delete
     * @return Reference to this matrix after column deletion
     *
     * Removes the specified column from the matrix.
     *
     * @throws std::invalid_argument if column index is out of bounds
     * @throws std::bad_alloc if memory allocation fails
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& delete_column(size_t i) { return delete_columns({i}); }

    /**
     * @brief Delete specified rows from the matrix
     *
     * @param v Vector of row indices to delete (automatically deduplicated)
     * @return Reference to this matrix after row deletion
     *
     * Removes the specified rows from the matrix.
     *
     * @throws std::invalid_argument if any row index is out of bounds
     * @throws std::bad_alloc if memory allocation fails
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& delete_rows(const std::vector<size_t>& v);

    /**
     * @brief Delete a single row from the matrix
     *
     * @param i Index of row to delete
     * @return Reference to this matrix after row deletion
     *
     * Removes the specified row from the matrix.
     *
     * @throws std::invalid_argument if row index is out of bounds
     * @throws std::bad_alloc if memory allocation fails
     *
     * @note Matrix type may change to @ref details::Generic after this operation
     */
    Matrix<T>& delete_row(size_t i) { return delete_rows({i}); }

#ifdef CECCO_ERASURE_SUPPORT

    /**
     * @brief Erases specified component from the matrix (flags it as erasure)
     *
     * @param i row i
     * @param j column j
     * @return Reference to this matrix after erasing
     *
     * Erases the component at position (i, j), cf. @ref Field::erase
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     *
     * @note Only available for field types (since erasure flag/erase() is required)
     *
     * @throws std::invalid_argument if (i, j) is out of bounds
     */
    Matrix<T>& erase_component(size_t i, size_t j)
        requires FieldType<T>;

    /**
     * @brief Un-erases specified component from the matrix (removes the erasure flag from it)
     *
     * @param i row i
     * @param j column j
     * @return Reference to this matrix after un-erasing
     *
     * Un-erases the component at position (i, j), cf. @ref Field::erase
     *
     * @note Only available for field types (since erasure flag/unerase() is required)
     *
     * @throws std::invalid_argument if (i, j) is out of bounds
     */
    Matrix<T>& unerase_component(size_t i, size_t j)
        requires FieldType<T>;

    /**
     * @brief Erases specified columns from the matrix (flags their components as erasures)
     *
     * @param v Vector of column indices to erase (automatically deduplicated)
     * @return Reference to this matrix after erasing
     *
     * Erases the components of all specified columns, cf. @ref Field::erase
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     *
     * @note Only available for field types (since erasure flag/erase() is required)
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix<T>& erase_columns(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Erases specified column from the matrix (flags its components as erasures)
     *
     * @param i Index of column to delete
     * @return Reference to this matrix after erasing
     *
     * Erases all components of the specified column.
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     *
     * @note Only available for field types (since erasure flag/erase() is required)
     *
     * @throws std::invalid_argument if index i is out of bounds
     */
    Matrix<T>& erase_column(size_t i)
        requires FieldType<T>
    {
        return erase_columns({i});
    }

    /**
     * @brief Un-erases specified columns from the matrix (removes the erasure flag from their components)
     *
     * @param v Vector of column indices to un-erase (automatically deduplicated)
     * @return Reference to this matrix after un-erasing
     *
     * Un-erases the components of all specified column.
     *
     * @note Only available for field types (since erasure flag/unerase() is required)
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix<T>& unerase_columns(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Un-erases specified column from the matrix (removes the erasure flag from its components)
     *
     * @param i Index of column to delete
     * @return Reference to this matrix after un-erasing
     *
     * Un-Erases all components of the specified column.
     *
     * @note Only available for field types (since erasure flag/unerase() is required)
     *
     * @throws std::invalid_argument if index i is out of bounds
     */
    Matrix<T>& unerase_column(size_t i)
        requires FieldType<T>
    {
        return unerase_columns({i});
    }

    /**
     * @brief Erases specified rows from the matrix (flags their components as erasures)
     *
     * @param v Vector of row indices to erase (automatically deduplicated)
     * @return Reference to this matrix after erasing
     *
     * Erases the components of all specified rows.
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     *
     * @note Only available for field types (since erasure flag/erase() is required)
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix& erase_rows(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Erases specified row from the matrix (flags its components as erasures)
     *
     * @param i Index of row to delete
     * @return Reference to this matrix after erasing
     *
     * Erases all components of the specified row.
     *
     * @warning Once a field element has been erased, it can no longer be used as a normal field element, i.e. field
     * operations, property queries, etc. will return incorrect results or throw errors. The correct use of erased field
     * elements is the responsibility of the user!
     *
     * @note Only available for field types (since erasure flag/erase() is required)
     *
     * @throws std::invalid_argument if index i is out of bounds
     */
    Matrix& erase_row(size_t i)
        requires FieldType<T>
    {
        return erase_rows({i});
    }

    /**
     * @brief Un-erases specified rows from the matrix (removes the erasure flag from their components)
     *
     * @param v Vector of row indices to un-erase (automatically deduplicated)
     * @return Reference to this matrix after un-erasing
     *
     * Un-erases the components of all specified row.
     *
     * @note Only available for field types (since erasure flag/unerase() is required)
     *
     * @throws std::invalid_argument if any index in v is out of bounds
     */
    Matrix& unerase_rows(const std::vector<size_t>& v)
        requires FieldType<T>;

    /**
     * @brief Un-erases specified row from the matrix (removes the erasure flag from its components)
     *
     * @param i Index of row to delete
     * @return Reference to this matrix after un-erasing
     *
     * Un-Erases all components of the specified row.
     *
     * @note Only available for field types (since erasure flag/unerase() is required)
     *
     * @throws std::invalid_argument if index i is out of bounds
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
     * @brief Reverse the order of matrix rows
     *
     * @return Reference to this matrix after row reversal
     *
     * Reverses the order of rows: first row becomes last, second becomes
     * second-to-last, etc.
     */
    Matrix<T>& reverse_rows();

    /**
     * @brief Reverse the order of matrix columns
     *
     * @return Reference to this matrix after column reversal
     *
     * Reverses the order of columns: first column becomes last, second becomes
     * second-to-last, etc.
     */
    Matrix<T>& reverse_columns();

    /**
     * @brief Fill all matrix elements with specified value
     *
     * @param s Value to assign to all elements
     * @return Reference to this matrix after filling
     *
     * Sets every element to the specified value.
     *
     * @note Matrix type is updated to @ref details::Zero if value is T(0), otherwise becomes @ref details::Generic.
     */
    constexpr Matrix<T>& fill(const T& s) noexcept;

    /**
     * @brief Transpose the matrix in-place
     *
     * @return Reference to this matrix after transposition
     *
     * Transposes the matrix by swapping rows and columns. For an m × n matrix,
     * the result is an n × m matrix where element (i, j) becomes element (j, i).
     *
     * @code{.cpp}
     * Matrix<int> A = {{1, 2, 3}, {4, 5, 6}};  // 2 × 3 matrix
     * A.transpose();                           // Now 3 × 2 matrix
     * @endcode
     */
    constexpr Matrix<T>& transpose();

    /**
     * @brief Row echelon form (REF) computation with binary field optimization
     *
     * @param rank Optional pointer to store the computed rank
     * @return Reference to this matrix after REF computation
     *
     * Converts matrix to row echelon form using forward Gaussian elimination only.
     * More efficient than RREF when only the rank is needed. Uses constexpr if to optimize
     * binary field operations (Fp<2>) by eliminating unnecessary pivot scaling.
     * If rank pointer is provided, stores the matrix rank and caches it for future use.
     *
     * @note Only available for field types
     *
     * @code{.cpp}
     * Matrix<double> A = {{2, 1, 3}, {1, 0, 1}, {1, 1, 1}};
     * size_t rank;
     * A.ref(&rank);  // A is now in REF, rank contains the rank (and the rank is cached)
     * @endcode
     */
    Matrix<T>& ref(size_t* rank = nullptr)
        requires FieldType<T>;

    /**
     * @brief Reduced row echelon form (RREF) computation
     *
     * @param rank Optional pointer to store the computed rank
     * @return Reference to this matrix after RREF computation
     *
     * Converts matrix to reduced row echelon form using two-phase algorithm:
     * forward elimination (REF) followed by backward elimination.
     * If rank pointer is provided, stores the matrix rank. Always caches it for future use.
     *
     * @note Only available for field types
     *
     * @code{.cpp}
     * Matrix<double> A = {{2, 1, 3}, {1, 0, 1}, {1, 1, 1}};
     * size_t rank;
     * A.rref(&rank);  // A is now in RREF, rank contains the rank, rank is cached
     * @endcode
     */
    Matrix<T>& rref(size_t* rank = nullptr)
        requires FieldType<T>;

    /**
     * @brief Invert the matrix in-place
     *
     * @return Reference to this matrix after inversion
     *
     * Computes the matrix inverse using Gaussian elimination with partial pivoting.
     * The matrix must be square and non-singular (non-zero determinant).
     *
     * @note Only available for field types (requires division)
     * @note Matrix type becomes details::Generic after inversion (except for @ref details::Identity)
     *
     * @throws std::invalid_argument if matrix is not square or singular
     *
     * @code{.cpp}
     * Matrix<double> A = {{1, 2}, {3, 4}};
     * A.invert();  // A is now its own inverse
     * @endcode
     */
    Matrix<T>& invert()
        requires FieldType<T>;

    /** @} */

    /** @name Finite Field Specific Operations
     * @{
     */

    /**
     * @brief Convert matrix to vector over superfield
     *
     * @tparam S Superfield of T (in same construction tower)
     * @return Vector representation of the rows over the superfield
     *
     * Converts the matrix to a vector over a superfield by interpreting each
     * matrix column as an element of the superfield.
     *
     * @throws std::bad_alloc if memory allocation fails
     *
     * @note Only available for finite field types in the same construction tower
     *
     * @code{.cpp}
     * using F2 = Fp<2>;
     * using F4 = Ext<F2, MOD{1, 1, 1}>;
     * Matrix<F2> M = {{1, 0}, {1, 1}};
     * auto vec = M.as_vector<F4>();  // Convert to vector over F4
     * @endcode
     */
    template <FiniteFieldType S>
    constexpr Vector<S> as_vector() const
        requires FiniteFieldType<T> && ExtensionOf<T, S> && (!std::is_same_v<T, S>);

    /**
     * @brief Flatten matrix into a vector by row-major concatenation
     *
     * @return Vector of length m × n containing all matrix elements
     *
     * Concatenates all rows of the matrix into a single vector. For an m × n matrix,
     * element (i, j) maps to vector index i · n + j.
     *
     * @code{.cpp}
     * Matrix<double> M = {{1, 2, 3}, {4, 5, 6}};
     * auto v = M.to_vector();  // v = (1, 2, 3, 4, 5, 6)
     * @endcode
     */
    Vector<T> to_vector() const;

    /**
     * @brief Export matrix as a PPM image file
     *
     * @param filename Path to the output PPM file
     *
     * Writes the matrix to a PPM (P3) image file where each field element is mapped
     * to an RGB color using a built-in 64-entry colormap (black → blue → green → yellow → white).
     * The image has width n and height m.
     *
     * @note Only available for finite field types with field size at most 64
     * @note When @c CECCO_ERASURE_SUPPORT is defined, erased elements are rendered in red
     *
     * @code{.cpp}
     * using F4 = Ext<Fp<2>, MOD{1, 1, 1}>;
     * Matrix<F4> M(8, 8);
     * M.export_as_ppm("matrix.ppm");
     * @endcode
     */
    void export_as_ppm(const std::string& filename) const
        requires FiniteFieldType<T> && (T::get_size() <= 64);

    /** @} */

   private:
    /**
     * @brief Matrix data storage
     *
     * Stores matrix elements in row-major order as a contiguous vector.
     */
    std::vector<T> data;

    /**
     * @brief Number of rows
     */
    size_t m = 0;

    /**
     * @brief Number of columns
     */
    size_t n = 0;

    /**
     * @brief Transpose state flag
     *
     * When true, matrix operations interpret the storage as transposed
     * for performance optimization without data movement.
     */
    bool transposed = false;

    /**
     * @brief Matrix type for optimization
     *
     * Tracks the structural type of the matrix to enable type-specific
     * optimizations and algorithms.
     */
    details::matrix_type_t type = details::Zero;

    static inline const uint8_t colormap[64][3] = {
        {0, 0, 0},       {0, 0, 24},      {0, 0, 40},      {0, 0, 56},      {0, 0, 72},      {0, 0, 88},
        {0, 0, 104},     {0, 0, 120},     {0, 0, 136},     {0, 0, 152},     {0, 0, 167},     {0, 0, 183},
        {0, 0, 199},     {0, 0, 215},     {0, 0, 231},     {0, 0, 252},     {0, 6, 253},     {0, 24, 232},
        {0, 40, 216},    {0, 56, 200},    {0, 72, 184},    {0, 88, 168},    {0, 104, 152},   {0, 120, 136},
        {0, 136, 120},   {0, 152, 104},   {0, 167, 88},    {0, 183, 72},    {0, 199, 56},    {0, 215, 41},
        {0, 231, 25},    {0, 249, 6},     {6, 255, 0},     {24, 255, 0},    {40, 255, 0},    {56, 255, 0},
        {72, 255, 0},    {88, 255, 0},    {104, 255, 0},   {120, 255, 0},   {136, 255, 0},   {152, 255, 0},
        {167, 255, 0},   {183, 255, 0},   {199, 255, 0},   {215, 255, 0},   {231, 255, 0},   {249, 255, 0},
        {255, 255, 6},   {255, 255, 24},  {255, 255, 40},  {255, 255, 56},  {255, 255, 72},  {255, 255, 88},
        {255, 255, 104}, {255, 255, 120}, {255, 255, 136}, {255, 255, 152}, {255, 255, 167}, {255, 255, 183},
        {255, 255, 199}, {255, 255, 215}, {255, 255, 231}, {255, 255, 255}};

    /**
     * @brief Cache system for expensive computations
     */
    enum CacheIds { Rank = 0 };
    mutable details::Cache<details::CacheEntry<Rank, size_t>> cache;

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
    static void multiply_kernel(const T* __restrict__ this_data, const T* __restrict__ rhs_data,
                                T* __restrict__ res_data, size_t M, size_t K, size_t N, size_t BS) {
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
    static void eliminate_row_kernel(T* __restrict__ data, size_t m, size_t n, size_t target_row, size_t pivot_row,
                                     size_t start_col, const T& f) {
        if constexpr (!transposed) {
            // Row-major access
            T* __restrict__ target_data = data + target_row * n;
            const T* __restrict__ pivot_data = data + pivot_row * n;
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
     * @brief High-performance REF elimination kernel with compile-time transpose dispatch
     * @tparam transposed True if matrix is transposed, false otherwise
     * @param data Raw matrix data
     * @param m Number of rows
     * @param n Number of columns
     * @param h Current pivot row
     * @param k Current pivot column
     * @return Number of rows processed (new h value)
     */
    template <bool transposed>
    static size_t ref_elimination_kernel(T* __restrict__ data, size_t m, size_t n, size_t h, size_t k) {
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
     * @brief High-performance RREF backward elimination kernel with compile-time transpose dispatch
     * @tparam transposed True if matrix is transposed, false otherwise
     * @param data Raw matrix data
     * @param m Number of rows
     * @param n Number of columns
     * @param r Matrix rank (number of pivots to process)
     */
    template <bool transposed>
    void rref_backward_elimination_kernel(T* __restrict__ data, size_t m, size_t n, size_t r) {
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
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> l)
    : m(l.size()), n(0), type(details::Generic) {
    if (m == 0) return;
    for (auto it = l.begin(); it != l.end(); ++it) {
        if (it->size() > n) n = it->size();
    }
    if (n == 0) return;
    data.resize(m * n);

    // Use enumerate-style iteration with STL algorithms
    auto row_indices = std::views::iota(size_t{0}, m);
    auto row_it = l.begin();
    std::ranges::for_each(row_indices, [&](size_t i) {
        auto col_indices = std::views::iota(size_t{0}, row_it->size());
        auto col_it = row_it->begin();
        std::ranges::for_each(col_indices, [&](size_t j) {
            set_component(i, j, *col_it);
            ++col_it;
        });
        ++row_it;
    });
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
        const auto indices = std::views::iota(size_t{0}, m * n);
        std::ranges::transform(indices, data.begin(), [&](size_t idx) {
            size_t i = idx / n;
            size_t j = idx % n;
            return T(other(i, j));  // Uses enhanced cross-field constructors
        });
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
            const uint32_t key =
                (uint32_t(colormap[a][0]) << 16) | (uint32_t(colormap[a][1]) << 8) | uint32_t(colormap[a][2]);
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
template <FieldType S>
constexpr Matrix<T>& Matrix<T>::operator=(const Matrix<S>& other)
    requires FiniteFieldType<S> && FiniteFieldType<T> && (S::get_characteristic() == T::get_characteristic())
{
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
constexpr Matrix<T> Matrix<T>::operator-() const& noexcept {
    auto res = *this;
    auto rank_backup = res.cache.template get<Rank>();
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        std::ranges::for_each(res.data, [](T& x) { x = -x; });
        if (type == details::Vandermonde) res.type = details::Generic;

    } else if (type == details::Zero) {
        // continue;
    } else if (type == details::Diagonal || type == details::Identity) {
        const auto indices = std::views::iota(size_t{0}, m);
        std::ranges::for_each(indices, [this, &res](size_t mu) { res.data[mu * n + mu] = -res.data[mu * n + mu]; });
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
constexpr Matrix<T> Matrix<T>::operator-() && noexcept {
    auto rank_backup = cache.template get<Rank>();
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        std::ranges::for_each(data, [](T& x) { x = -x; });
        if (type == details::Vandermonde) {
            type = details::Generic;
        }
    } else if (type == details::Zero) {
        // continue;
    } else if (type == details::Diagonal || type == details::Identity) {
        const auto indices = std::views::iota(size_t{0}, m);
        std::ranges::for_each(indices, [this](size_t mu) { data[mu * n + mu] = -data[mu * n + mu]; });
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
        // continue;
    } else if ((type == details::Diagonal && rhs.type == details::Diagonal) ||
               (type == details::Identity && rhs.type == details::Identity) ||
               (type == details::Diagonal && rhs.type == details::Identity) ||
               (type == details::Identity && rhs.type == details::Diagonal)) {
        const auto indices = std::views::iota(size_t{0}, m);
        std::ranges::for_each(indices, [this, &rhs](size_t mu) { data[mu * n + mu] += rhs.data[mu * n + mu]; });
        if (type == details::Identity) type = details::Diagonal;
    } else {
        if (!transposed && !rhs.transposed) {
            // Optimized element-wise addition for non-transposed matrices
            std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<T>{});
            if (type != details::Generic && !(type == details::Toeplitz && rhs.type == details::Toeplitz))
                type = details::Generic;
        } else {
            const auto indices = std::views::iota(size_t{0}, m * n);
            std::ranges::for_each(indices, [this, &rhs](size_t idx) {
                size_t mu = idx / n;
                size_t nu = idx % n;
                set_component(mu, nu, (*this)(mu, nu) + rhs(mu, nu));
            });
        }
    }
    // Check if result is zero matrix using STL algorithm
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
        // continue;
    } else if (type == details::Diagonal && rhs.type == details::Diagonal) {
        const auto indices = std::views::iota(size_t{0}, m);
        std::ranges::for_each(indices, [this, &rhs](size_t mu) { data[mu * n + mu] *= rhs.data[mu * n + mu]; });
    } else if (type == details::Diagonal) {
        auto res = rhs;
        const auto indices = std::views::iota(size_t{0}, m);
        std::ranges::for_each(indices, [this, &res, &rhs](size_t mu) {
            auto s = (*this)(mu, mu);
            auto nu_indices = std::views::iota(size_t{0}, rhs.n);
            std::ranges::for_each(nu_indices, [&](size_t nu) { res.set_component(mu, nu, res(mu, nu) * s); });
        });
        *this = std::move(res);
    } else if (rhs.type == details::Diagonal) {
        const auto indices = std::views::iota(size_t{0}, n);
        std::ranges::for_each(indices, [this, &rhs](size_t nu) {
            const auto& s = rhs(nu, nu);
            auto mu_indices = std::views::iota(size_t{0}, m);
            std::ranges::for_each(mu_indices, [&](size_t mu) { set_component(mu, nu, (*this)(mu, nu) * s); });
        });
    } else {
        const size_t M = get_m();
        const size_t K = get_n();
        const size_t N = rhs.get_n();
        Matrix<T> res(M, N, T(0));

        // Get raw data pointers for direct access (bypass operator())
        const T* __restrict__ this_data = this->data.data();
        const T* __restrict__ rhs_data = rhs.data.data();
        T* __restrict__ res_data = res.data.data();

        // Tune this block/tile size for archicture. 48, 64, 96, 128 are ressonable.
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
        // continue;
    } else if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        std::ranges::for_each(data, [&s](T& x) { x *= s; });
        if (type == details::Vandermonde) type = details::Generic;
    } else if (type == details::Diagonal || type == details::Identity) {
        const auto indices = std::views::iota(size_t{0}, m);
        std::ranges::for_each(indices, [this, &s](size_t mu) { data[mu * n + mu] *= s; });
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
constexpr size_t Matrix<T>::wH() const noexcept
    requires ReliablyComparableType<T>
{
    if (type == details::Generic || type == details::Vandermonde || type == details::Toeplitz) {
        return data.size() - std::count(data.cbegin(), data.cend(), T(0));
    } else if (type == details::Zero) {
        return 0;
    } else if (type == details::Diagonal) {
        auto diagonal_indices = std::views::iota(size_t{0}, m);
        return std::count_if(diagonal_indices.begin(), diagonal_indices.end(),
                             [this](size_t i) { return (*this)(i, i) != T(0); });
    } else if (type == details::Identity) {
        return m;
    }
    assert(false && "wH(): should never be here");
    return 0;  // dummy
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
    const auto indices = std::views::iota(size_t{0}, n);
    std::transform(indices.begin(), indices.end(), res.data.begin(), [this](size_t i) { return (*this)(i, i); });
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
                auto j_indices = std::views::iota(size_t{0}, m - i - 2);
                std::ranges::for_each(j_indices, [&](size_t j) {
                    v.set_component(m - i - 3 - j, -((R * A * C)(0, 0)));
                    A *= X;
                });
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
        auto i_indices = std::views::iota(size_t{0}, m + 1);
        std::ranges::for_each(i_indices,
                              [&](size_t i) { res.set_coefficient(m - i, const_cast<const Matrix<T>&>(P)(i, 0)); });

        // for odd n: negate characteristic polynomial *ToDo: verify*
        if (m % 2) res *= T(-1);

        return res;
    } else if (type == details::Zero) {
        Polynomial<T> res({0, 1});
        return res ^ m;
    } else if (type == details::Diagonal) {
        Polynomial<T> res({1});
        auto mu_indices = std::views::iota(size_t{0}, m);
        std::ranges::for_each(mu_indices, [&](size_t mu) { res *= Polynomial<T>({(*this)(mu, mu), -1}); });
        return res;
    } else if (type == details::Identity) {
        Polynomial<T> res({-1, 1});
        return res ^ m;
    }
    assert(false && "characteristic_polynomial(): should never be here");
    return Polynomial<T>();  // dummy
}

template <ComponentType T>
Matrix<T> Matrix<T>::basis_of_nullspace() const
    requires FieldType<T>
{
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
        return characteristic_polynomial()[0];
    } else if (type == details::Vandermonde) {
        const auto indices = std::views::iota(size_t{0}, m);
        return std::accumulate(indices.begin(), indices.end(), T(1), [this](T acc, size_t mu) {
            if (mu == 0) return acc;
            auto inner_indices = std::views::iota(size_t{0}, mu);
            T row_product =
                std::accumulate(inner_indices.begin(), inner_indices.end(), T(1),
                                [this, mu](T prod, size_t i) { return prod * ((*this)(1, mu) - (*this)(1, i)); });
            return acc * row_product;
        });
    } else if (type == details::Zero) {
        return T(0);
    } else if (type == details::Diagonal) {
        auto diagonal_indices = std::views::iota(size_t{0}, m);

        // Check for any zero on diagonal first
        if (std::any_of(diagonal_indices.begin(), diagonal_indices.end(),
                        [this](size_t i) { return (*this)(i, i) == T(0); })) {
            return T(0);
        }

        // Compute product of diagonal elements
        return std::accumulate(diagonal_indices.begin(), diagonal_indices.end(), T(1),
                               [this](T acc, size_t i) { return acc * (*this)(i, i); });
    } else if (type == details::Identity) {
        return T(1);
    }
    assert(false && "determinant(): should never be here");
    return T(0);  // dummy
}

template <ComponentType T>
std::vector<T> Matrix<T>::eigenvalues() const
    requires FiniteFieldType<T>
{
    const auto p = characteristic_polynomial();
    std::vector<T> res;
    const auto indices = std::views::iota(size_t{0}, T::get_size());

    // Collect all j where p(T(j)) == 0
    std::ranges::for_each(indices, [&p, &res](size_t j) {
        T element = T(j);
        if (p(element) == T(0)) {
            res.push_back(element);
        }
    });
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
            if (i != j)
                type = details::Generic;
            else
                type = details::Diagonal;
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
        const auto indices = std::views::iota(size_t{0}, n);
        std::transform(indices.begin(), indices.end(), res.data.begin(), [this, i](size_t j) { return (*this)(i, j); });
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
        const auto indices = std::views::iota(size_t{0}, m);
        std::transform(indices.begin(), indices.end(), res.data.begin(), [this, j](size_t i) { return (*this)(i, j); });
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
            // Optimized row-wise copying for non-transposed matrices
            auto mu_indices = std::views::iota(size_t{0}, h);
            std::ranges::for_each(mu_indices, [&](size_t mu) {
                std::copy(data.begin() + (i + mu) * n + j, data.begin() + (i + mu) * n + j + w,
                          res.data.begin() + mu * w);
            });
            res.type = details::Generic;
        } else {
            // Fall back to element-wise access for transposed matrices
            const auto indices = std::views::iota(size_t{0}, h * w);
            std::ranges::for_each(indices, [&](size_t idx) {
                size_t mu = idx / w;
                size_t nu = idx % w;
                res.set_component(mu, nu, (*this)(i + mu, j + nu));
            });
        }
        if (type == details::Vandermonde && i == 0) {
            res.type = details::Vandermonde;
        } else if (type == details::Toeplitz) {
            res.type = details::Toeplitz;
        }
    } else if (type == details::Zero) {
        // continue;
    } else if (type == details::Diagonal || type == details::Identity) {
        const auto indices = std::views::iota(size_t{0}, h * w);
        std::ranges::for_each(indices, [&](size_t idx) {
            size_t mu = idx / w;
            size_t nu = idx % w;
            if (i + mu == j + nu) {
                res.set_component(mu, nu, (*this)(i + mu, j + nu));
            }
        });
        if (i == j) {
            if (type == details::Diagonal) {
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

    // For non-transposed matrices, use row-wise copying when possible
    if (!transposed && !N.transposed) {
        auto mu_indices = std::views::iota(size_t{0}, N.m);
        std::ranges::for_each(mu_indices, [&](size_t mu) {
            std::copy(N.data.begin() + mu * N.n, N.data.begin() + (mu + 1) * N.n, data.begin() + (i + mu) * n + j);
        });
    } else {
        // Fall back to element-wise access for transposed matrices
        const auto indices = std::views::iota(size_t{0}, N.m * N.n);
        std::ranges::for_each(indices, [&](size_t idx) {
            size_t mu = idx / N.n;
            size_t nu = idx % N.n;
            set_component(i + mu, j + nu, N(mu, nu));
        });
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
Matrix<T>& Matrix<T>::diagonal_join(const Matrix& other) noexcept {
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
            const auto indices = std::views::iota(size_t{0}, m * other.m);
            std::ranges::for_each(indices,
                                  [&](size_t idx) { d.set_component(idx, d1[idx / other.m] * d2[idx % other.m]); });
            *this = DiagonalMatrix<T>(std::move(d));
        } else {
            Matrix temp(m * other.m, n * other.n);
            const auto indices = std::views::iota(size_t{0}, m);
            std::ranges::for_each(indices, [this, &temp, &other](size_t mu) {
                if (type == details::Identity)
                    temp.set_submatrix(mu * other.m, mu * other.n, other);
                else
                    temp.set_submatrix(mu * other.m, mu * other.n, (*this)(mu, mu) * other);
            });
            *this = std::move(temp);
        }
    } else {
        Matrix temp(m * other.m, n * other.n);
        const auto indices = std::views::iota(size_t{0}, m * n);
        std::ranges::for_each(indices, [&](size_t idx) {
            size_t mu = idx / n;
            size_t nu = idx % n;
            temp.set_submatrix(mu * other.m, nu * other.n, (*this)(mu, nu) * other);
        });
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
            const auto indices = std::views::iota(size_t{0}, n);
            std::ranges::for_each(indices, [this, i, j](size_t nu) {
                const auto temp = (*this)(i, nu);
                set_component(i, nu, (*this)(j, nu));
                set_component(j, nu, temp);
            });
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
            const auto indices = std::views::iota(size_t{0}, n);
            std::ranges::for_each(indices, [this, i, &s](size_t nu) { data[i + nu * m] *= s; });
            if (type == details::Vandermonde || type == details::Toeplitz) type = details::Generic;
        }
    } else if (type == details::Zero) {
        // continue;
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
            const auto indices = std::views::iota(size_t{0}, n);
            std::ranges::for_each(indices, [this, i, j, &s](size_t nu) { data[j + nu * m] += s * data[i + nu * m]; });
            if (type == details::Vandermonde || type == details::Toeplitz) type = details::Generic;
        }
    } else if (type == details::Zero) {
        // continue;
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
            const auto indices = std::views::iota(size_t{0}, m / 2);
            std::ranges::for_each(indices, [this](size_t mu) {
                std::swap_ranges(data.begin() + mu * n, data.begin() + (mu + 1) * n, data.begin() + (m - 1 - mu) * n);
            });
        } else {
            auto rank_backup = cache.template get<Rank>();
            const auto indices = std::views::iota(size_t{0}, m / 2 * n);
            std::ranges::for_each(indices, [this](size_t idx) {
                size_t mu = idx / n;
                size_t nu = idx % n;
                const auto temp = (*this)(mu, nu);
                set_component(mu, nu, (*this)(m - 1 - mu, nu));
                set_component(m - 1 - mu, nu, temp);
            });
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
            const auto indices = std::views::iota(size_t{0}, m);
            std::ranges::for_each(
                indices, [this](size_t mu) { std::reverse(data.begin() + mu * n, data.begin() + (mu + 1) * n); });
        } else {
            auto rank_backup = cache.template get<Rank>();
            const auto indices = std::views::iota(size_t{0}, m * (n / 2));
            std::ranges::for_each(indices, [this](size_t idx) {
                size_t mu = idx / (n / 2);
                size_t nu = idx % (n / 2);
                const auto temp = (*this)(mu, nu);
                set_component(mu, nu, (*this)(mu, n - 1 - nu));
                set_component(mu, n - 1 - nu, temp);
            });
            if (rank_backup) cache.template set<Rank>(*rank_backup);
        }
        if (type != details::Vandermonde) {
            type = details::Generic;
        }
    }
    return *this;
}

template <ComponentType T>
constexpr Matrix<T>& Matrix<T>::fill(const T& s) noexcept {
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
        // continue;
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
        auto i_indices = std::views::iota(size_t{0}, m);
        std::ranges::for_each(i_indices, [this, &Lagrange_polynomials](size_t i) {
            auto k_indices = std::views::iota(size_t{0}, m);
            std::ranges::for_each(k_indices, [this, &Lagrange_polynomials, i](size_t k) {
                if (k == i) return;
                Lagrange_polynomials[i] *= Polynomial<T>({-(*this)(1, k), 1}) / ((*this)(1, i) - (*this)(1, k));
            });
        });
        const auto indices = std::views::iota(size_t{0}, m * m);
        std::ranges::for_each(indices, [this, &Lagrange_polynomials](size_t idx) {
            size_t i = idx / m;
            size_t j = idx % m;
            set_component(i, j, Lagrange_polynomials[i][j]);
        });
    } else if (type == details::Zero) {
        throw std::invalid_argument("trying to invert a non-invertible matrix/a zero matrix");
    } else if (type == details::Diagonal) {
        // Check for zero diagonal elements first
        auto diag_indices = std::views::iota(size_t{0}, m);
        auto zero_diag = std::ranges::find_if(diag_indices, [this](size_t mu) { return (*this)(mu, mu) == T(0); });
        if (zero_diag != diag_indices.end()) {
            throw std::invalid_argument(
                "trying to invert a non-invertible matrix/a diagonal matrix with at least one zero on the "
                "diagonal");
        }

        // Invert all diagonal elements
        std::ranges::for_each(diag_indices, [this](size_t mu) { data[mu * n + mu] = T(1) / data[mu * n + mu]; });
    } else if (type == details::Identity) {
        // continue;
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
    auto i_indices = std::views::iota(size_t{0}, get_n());
    std::ranges::for_each(i_indices, [&](size_t i) {
        const auto temp = Tp.get_row(i);
        res.set_component(i, S(temp));
    });
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

            const uint8_t r = colormap[a][0];
            const uint8_t g = colormap[a][1];
            const uint8_t b = colormap[a][2];

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
Matrix<T> randomize(Matrix<T>&& M) noexcept {
    Matrix<T> res(std::move(M));
    res.randomize();
    return res;
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
Matrix<T> reverse_rows(Matrix<T>&& M) noexcept {
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
Matrix<T> reverse_columns(Matrix<T>&& M) noexcept {
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
constexpr Matrix<T> fill(Matrix<T>&& m, const T& value) noexcept {
    Matrix<T> res(std::move(m));
    res.fill(value);
    return res;
}

template <ComponentType T>
constexpr Matrix<T> transpose(const Matrix<T>& M) noexcept {
    Matrix<T> res(M);
    res.transpose();
    return res;
}

template <ComponentType T>
constexpr Matrix<T> transpose(Matrix<T>&& M) noexcept {
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
 * @brief Matrix equality comparison operator
 * @ingroup matrix_arithmetic
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param lhs Left-hand side matrix
 * @param rhs Right-hand side matrix
 * @return true if matrices are equal, false otherwise
 *
 * Two matrices are equal if they have the same dimensions and all corresponding
 * elements are equal. Uses optimized comparison algorithms for structured matrices.
 */
template <ReliablyComparableType T>
constexpr bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) noexcept {
    if (lhs.m != rhs.m || lhs.n != rhs.n) return false;

    if (lhs.m == 0) {
        return true;
    } else if (lhs.m == 1) {
        return lhs.get_row(0) == rhs.get_row(0);
    } else if (lhs.n == 1) {
        return lhs.get_col(0) == rhs.get_col(0);
    } else if (lhs.type == details::Toeplitz && rhs.type == details::Toeplitz) {
        // Compare left column (in reverse order for details::Toeplitz structure)
        const auto indices = std::views::iota(size_t{0}, lhs.m);
        if (!std::ranges::equal(indices, indices,
                                [&](size_t mu, size_t) { return lhs(lhs.m - 1 - mu, 0) == rhs(lhs.m - 1 - mu, 0); })) {
            return false;
        }
        // Compare top row (excluding first element)
        auto col_indices = std::views::iota(size_t{1}, lhs.n);
        if (!std::ranges::equal(col_indices, col_indices,
                                [&](size_t nu, size_t) { return lhs(0, nu) == rhs(0, nu); })) {
            return false;
        }
    } else if (lhs.type == details::Vandermonde && rhs.type == details::Vandermonde) {
        return lhs.get_row(1) == rhs.get_row(1);
    } else if ((lhs.type == details::Diagonal && rhs.type == details::Diagonal) ||
               (lhs.type == details::Diagonal && rhs.type == details::Identity) ||
               (lhs.type == details::Identity && rhs.type == details::Diagonal)) {
        return lhs.diagonal() == rhs.diagonal();
    } else if (lhs.type == details::Identity && rhs.type == details::Identity) {
        // continue;
    } else if ((lhs.type == details::Zero && rhs.type != details::Zero) ||
               (lhs.type != details::Zero && rhs.type == details::Zero)) {
        return false;
    } else {
        if (!lhs.transposed && !rhs.transposed) {
            // Optimized comparison for non-transposed matrices
            return std::equal(lhs.data.begin(), lhs.data.end(), rhs.data.begin());
        } else {
            const auto indices = std::views::iota(size_t{0}, lhs.m * lhs.n);
            return std::ranges::all_of(indices, [&](size_t idx) {
                size_t mu = idx / lhs.n;
                size_t nu = idx % lhs.n;
                return lhs(mu, nu) == rhs(mu, nu);
            });
        }
    }
    return true;
}

/**
 * @brief Matrix inequality comparison operator
 * @ingroup matrix_arithmetic
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param lhs Left-hand side matrix
 * @param rhs Right-hand side matrix
 * @return true if matrices are not equal, false otherwise
 *
 * Equivalent to !(lhs == rhs). Two matrices are not equal if they have different
 * dimensions or any corresponding elements differ.
 */
template <ReliablyComparableType T>
constexpr bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs) noexcept {
    return !(lhs == rhs);
}

/**
 * @brief Matrix output stream operator
 * @ingroup matrix_utilities
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param os Output stream to write to
 * @param rhs Matrix to output
 * @return Reference to the output stream
 *
 * Outputs the matrix in a formatted mathematical notation with proper brackets
 * and alignment. Empty matrices are displayed as "(empty matrix)".
 *
 * @note This operation is noexcept
 */
template <ComponentType T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& rhs) noexcept {
    if (rhs.m == 0 || rhs.n == 0) {
        os << "(empty matrix)";
        return os;
    }
    size_t max = 0;
    std::stringstream ss;
    const auto indices = std::views::iota(size_t{0}, rhs.m * rhs.n);
    std::ranges::for_each(indices, [&rhs, &ss, &max](size_t idx) {
        size_t i = idx / rhs.n;
        size_t j = idx % rhs.n;
        ss << rhs(i, j);
        max = std::max(ss.str().length(), max);
        ss.str(std::string());  // clear stringstream
    });
    os << (rhs.m == 1 ? "(" : "⌈");
    if (rhs.n > 1) {
        const auto indices = std::views::iota(size_t{0}, rhs.n - 1);
        std::ranges::for_each(indices, [&rhs, &os, max](size_t j) {
            os << std::setw(max) << rhs(0, j);
            os << " ";  // must be in extra line due to set::setw()
        });
    }
    os << std::setw(max) << rhs(0, rhs.n - 1);
    os << (rhs.m == 1 ? ")" : "⌉");
    if (rhs.m > 1) os << std::endl;
    if (rhs.m > 2) {
        auto i_indices = std::views::iota(size_t{1}, rhs.m - 1);
        std::ranges::for_each(i_indices, [&](size_t i) {
            os << "|";
            auto j_indices = std::views::iota(size_t{0}, rhs.n - 1);
            std::ranges::for_each(j_indices, [&](size_t j) {
                os << std::setw(max) << rhs(i, j);
                os << " ";  // must be in extra line due to set::setw()
            });
            os << std::setw(max) << rhs(i, rhs.n - 1);
            os << "|" << std::endl;
        });
    }
    if (rhs.m > 1) {
        os << "⌊";
        auto j_indices = std::views::iota(size_t{0}, rhs.n - 1);
        std::ranges::for_each(j_indices, [&](size_t j) {
            os << std::setw(max) << rhs(rhs.m - 1, j);
            os << " ";  // must be in extra line due to set::setw()
        });
        os << std::setw(max) << rhs(rhs.m - 1, rhs.n - 1);
        os << "⌋";
    }
    return os;
}

/*
 * factories
 */

/**
 * @brief Create a zero matrix of specified dimensions
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param m Number of rows
 * @param n Number of columns
 * @return m×n matrix with all elements equal to T(0)
 *
 * Creates a zero matrix with all elements set to the additive identity.
 *
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * auto Z = ZeroMatrix<double>(3, 4);  // 3×4 zero matrix
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> ZeroMatrix(size_t m, size_t n) {
    return Matrix<T>(m, n);
}

/**
 * @brief Create an identity matrix of specified size
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param m Dimension of the square identity matrix
 * @return m×m identity matrix with ones on diagonal and zeros elsewhere
 *
 * Creates an identity matrix I_m where element (i,j) equals 1 if i=j,
 * and 0 otherwise.
 *
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * auto I = IdentityMatrix<double>(4);  // 4×4 identity matrix
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> IdentityMatrix(size_t m) {
    auto res = Matrix<T>(m, m);
    const auto indices = std::views::iota(size_t{0}, m);
    std::ranges::for_each(indices, [&res](size_t i) { res.set_component(i, i, T(1)); });
    res.type = details::Identity;
    return res;
}

/**
 * @brief Create an exchange matrix (anti-diagonal identity matrix)
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param m Dimension of the square exchange matrix
 * @return m×m exchange matrix with ones on anti-diagonal and zeros elsewhere
 *
 * Creates an exchange matrix (also known as permutation matrix) with ones on the
 * anti-diagonal and zeros elsewhere. Element (i,j) equals 1 if i+j=m-1, and 0 otherwise.
 *
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * auto E = ExchangeMatrix<int>(3);  // [[0,0,1], [0,1,0], [1,0,0]]
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> ExchangeMatrix(size_t m) {
    auto res = IdentityMatrix<T>(m);
    res.reverse_columns();
    return res;
}

/**
 * @brief Create a diagonal matrix from a vector
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param v Vector containing the diagonal elements
 * @return n×n diagonal matrix with v[i] on position (i,i)
 *
 * Creates a diagonal matrix where element (i,j) equals v[i] if i=j,
 * and 0 otherwise.
 *
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * Vector<double> diag = {1, 2, 3};
 * auto D = DiagonalMatrix(diag);  // [[1,0,0], [0,2,0], [0,0,3]]
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> DiagonalMatrix(const Vector<T>& v) {
    const size_t m = v.get_n();
    Matrix<T> res(m, m);
    const auto indices = std::views::iota(size_t{0}, m);
    std::ranges::for_each(indices, [&res, &v](size_t i) { res.set_component(i, i, v[i]); });
    res.type = details::Diagonal;
    return res;
}

/**
 * @brief Create a Toeplitz matrix from a vector
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param v Vector containing the values for diagonals (length must be m+n-1)
 * @param m Number of rows
 * @param n Number of columns
 * @return m×n Toeplitz matrix where element (i,j) = v[i-j+n-1]
 *
 * Creates a Toeplitz matrix where each descending diagonal contains identical elements.
 * The vector v must have length m+n-1, providing values for all diagonals.
 *
 * @throws std::invalid_argument if v.length() != m+n-1
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * Vector<int> diags = {1, 2, 3, 4, 5};  // For 3×3 matrix: 3+3-1=5
 * auto T = ToeplitzMatrix(diags, 3, 3); // details::Toeplitz with constant diagonals
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> ToeplitzMatrix(const Vector<T>& v, size_t m, size_t n) {
    if (v.get_n() != m + n - 1)
        throw std::invalid_argument(
            "vector for constructing m x n details::Toeplitz matrix must have "
            "length m+n-1");
    Matrix<T> res(m, n);

    // Fill first column: v[0] to v[m-1] in reverse order
    auto row_indices = std::views::iota(size_t{0}, m);
    std::ranges::for_each(row_indices, [&](size_t i) { res.set_component(m - 1 - i, 0, v[i]); });

    // Fill first row: v[m-1] to v[m+n-2]
    if (n > 1) {
        auto col_indices = std::views::iota(size_t{1}, n);
        std::ranges::for_each(col_indices, [&](size_t j) { res.set_component(0, j, v[m - 1 + j]); });
    }

    // Fill remaining elements using diagonal copy pattern
    auto i_indices = std::views::iota(size_t{1}, m);
    std::ranges::for_each(i_indices, [&](size_t i) {
        if (n > 1) {
            auto col_indices = std::views::iota(size_t{1}, n);
            std::ranges::for_each(col_indices, [&](size_t j) { res.set_component(i, j, res(i - 1, j - 1)); });
        }
    });
    res.type = details::Toeplitz;
    return res;
}

/**
 * @brief Create a Hankel matrix from a vector
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param v Vector containing the anti-diagonal values (length must be m+n-1)
 * @param m Number of rows
 * @param n Number of columns
 * @return m×n Hankel matrix where element (i,j) depends on i+j
 *
 * Creates a Hankel matrix where each ascending anti-diagonal contains identical elements.
 * A Hankel matrix is related to a details::Toeplitz matrix through matrix multiplication.
 *
 * @throws std::invalid_argument if v.length() != m+n-1
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * Vector<int> antidiags = {1, 2, 3, 4, 5};  // For 3×3 matrix
 * auto H = HankelMatrix(antidiags, 3, 3); // Hankel matrix
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> HankelMatrix(const Vector<T>& v, size_t m, size_t n) {
    return ToeplitzMatrix<T>(reverse(v), m, n) * ExchangeMatrix<T>(n);
}

/**
 * @brief Create a Vandermonde matrix from evaluation points
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param v Vector of evaluation points (must have pairwise distinct elements)
 * @param m Number of rows (degree of polynomials)
 * @return m×n Vandermonde matrix where element (i,j) = v[j]^i
 *
 * Creates a Vandermonde matrix used in polynomial interpolation and evaluation.
 * Element (i,j) contains the j-th evaluation point raised to the i-th power.
 *
 * @throws std::invalid_argument if v is empty or has duplicate elements
 * @throws std::invalid_argument if m is zero
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * Vector<double> points = {1, 2, 3, 4};  // Evaluation points
 * auto V = VandermondeMatrix(points, 3); // [[1,1,1,1], [1,2,3,4], [1,4,9,16]]
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> VandermondeMatrix(const Vector<T>& v, size_t m) {
    const size_t n = v.get_n();
    if (n == 0)
        throw std::invalid_argument(
            "vector for constructing details::Vandermonde matrix must have "
            "at least one element");
    if (m == 0) throw std::invalid_argument("trying to construct details::Vandermonde matrix with zero rows");
    if (!v.is_pairwisedistinct())
        throw std::invalid_argument(
            "vector for constructing details::Vandermonde matrix must have pairwise distinct elements");

    Matrix<T> res(m, n);

    // First row: all ones using std::fill
    if (!res.transposed) {
        std::fill(res.data.begin(), res.data.begin() + n, T(1));
    } else {
        const auto indices = std::views::iota(size_t{0}, n);
        std::ranges::for_each(indices, [&](size_t i) { res.set_component(0, i, T(1)); });
    }

    if (m > 1) {
        // Second row: copy from vector v
        if (!res.transposed) {
            std::copy(v.data.begin(), v.data.end(), res.data.begin() + n);
        } else {
            const auto indices = std::views::iota(size_t{0}, n);
            std::ranges::for_each(indices, [&](size_t i) { res.set_component(1, i, v[i]); });
        }

        // Remaining rows: compute powers using std::ranges::for_each
        auto j_indices = std::views::iota(size_t{2}, m);
        std::ranges::for_each(j_indices, [&](size_t j) {
            const auto indices = std::views::iota(size_t{0}, n);
            std::ranges::for_each(indices, [&](size_t i) { res.set_component(j, i, res(j - 1, i) * v[i]); });
        });
    }
    res.type = details::Vandermonde;
    return res;
}

/**
 * @brief Create an upper shift matrix (superdiagonal matrix)
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param m Dimension of the square shift matrix
 * @return m×m upper shift matrix with ones on the superdiagonal
 *
 * Creates a matrix with ones on the superdiagonal (elements (i,i+1) = 1) and zeros elsewhere.
 * Used in companion matrices and linear system representations.
 *
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * auto U = UpperShiftMatrix<int>(4);  // [[0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]]
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> UpperShiftMatrix(size_t m) {
    Matrix<T> res(m, m);
    if (m > 1) {
        const auto indices = std::views::iota(size_t{0}, m - 1);
        std::ranges::for_each(indices, [&](size_t i) { res.set_component(i, i + 1, T(1)); });
    }
    return res;
}

/**
 * @brief Create a lower shift matrix (subdiagonal matrix)
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param m Dimension of the square shift matrix
 * @return m×m lower shift matrix with ones on the subdiagonal
 *
 * Creates a matrix with ones on the subdiagonal (elements (i+1,i) = 1) and zeros elsewhere.
 * Equivalent to the transpose of the upper shift matrix.
 *
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * auto L = LowerShiftMatrix<int>(4);  // [[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0]]
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> LowerShiftMatrix(size_t m) {
    return transpose(UpperShiftMatrix<T>(m));
}

/**
 * @brief Create a companion matrix for a monic polynomial
 * @ingroup matrix_factories
 *
 * @tparam T Component type satisfying @ref CECCO::ComponentType
 * @param poly Monic polynomial p(x) = x^n + a_{n-1}x^{n-1} + ... + a_1x + a_0
 * @return n×n companion matrix with characteristic polynomial p(x)
 *
 * Creates a companion matrix whose characteristic polynomial equals the input polynomial.
 * The matrix has the form of a lower shift matrix with the last column containing
 * the negated polynomial coefficients.
 *
 * @throws std::invalid_argument if polynomial is not monic
 * @throws std::bad_alloc if memory allocation fails
 *
 * @code{.cpp}
 * Polynomial<double> p = {1, 2, 3, 1};  // 1 + 2x + 3x² + x³ (monic)
 * auto C = CompanionMatrix(p);  // 3×3 companion matrix
 * @endcode
 */
template <ComponentType T>
constexpr Matrix<T> CompanionMatrix(const Polynomial<T>& poly) {
    if (!poly.is_monic()) throw std::invalid_argument("companion matrices only defined for monic polynomials");
    Matrix<T> res(transpose(UpperShiftMatrix<T>(poly.get_degree())));

    // Fill last column with negated polynomial coefficients
    const auto indices = std::views::iota(size_t{0}, poly.get_degree());
    std::ranges::for_each(indices, [&](size_t i) { res.set_component(i, poly.get_degree() - 1, -poly[i]); });

    return res;
}

}  // namespace CECCO

#endif
