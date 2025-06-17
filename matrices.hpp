/*
   Copyright 2025 Christian Senger <senger@inue.uni-stuttgart.de>

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   v1.0
*/

/* ToDo:
    - Toeplitz: rref, determinant, etc. based on Bareiss algorithm? Levinson algorithm?
*/

#ifndef MATRICES_HPP
#define MATRICES_HPP

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "InfInt.hpp"
#include "polynomials.hpp"
#include "vectors.hpp"

namespace ECC {

enum matrix_type_t : uint8_t { Generic, Zero, Diagonal, Identity, Vandermonde, Toeplitz };

template <class T>
class Matrix;

template <class T>
Matrix<T> ZeroMatrix(size_t m, size_t n);
template <class T>
Matrix<T> IdentityMatrix(size_t m);
template <class T>
Matrix<T> DiagonalMatrix(const Vector<T>& v);
template <class T>
Matrix<T> ToeplitzMatrix(const Vector<T>& v, size_t m, size_t n);
template <class T>
Matrix<T> VandermondeMatrix(const Vector<T>& v, size_t m);
template <class T>
bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) noexcept;
template <class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& rhs) noexcept;

template <class T>
class Matrix {
    friend Matrix<T> IdentityMatrix<>(size_t m);
    friend Matrix<T> DiagonalMatrix<>(const Vector<T>& v);
    friend Matrix<T> ToeplitzMatrix<>(const Vector<T>& v, size_t m, size_t n);
    friend Matrix<T> VandermondeMatrix<>(const Vector<T>& v, size_t m);
    friend bool operator== <>(const Matrix& lhs, const Matrix& rhs) noexcept;
    friend std::ostream& operator<< <>(std::ostream& os, const Matrix& rhs) noexcept;

   public:
    /* constructors */
    Matrix() : data(0), m(0), n(0), transposed(false), type(Zero) {}
    Matrix(size_t m, size_t n) : data(m * n), m(m), n(n), transposed(false), type(Zero) {}
    Matrix(size_t m, size_t n, const T& l) : data(m * n), m(m), n(n), transposed(false), type(l == 0 ? Zero : Generic) {
        std::fill(data.begin(), data.end(), l);
    }
    Matrix(size_t m, size_t n, std::initializer_list<T> l) : data(l), m(m), n(n), transposed(false), type(Generic) {
        if (l.size() != m * n) {
            throw std::invalid_argument("number of elements in initializer list does not correspond to number of rows and columns specified");
        }
    }
    Matrix(std::initializer_list<std::initializer_list<T>> l) : m(l.size()), n(0), transposed(false), type(Generic) {
        if (m == 0) return;
        for (auto it = l.begin(); it != l.end(); ++it) {
            if (it->size() > n) n = it->size();
        }
        if (n == 0) return;
        data.resize(m * n);
        size_t i = 0;
        size_t j = 0;
        for (auto it1 = l.begin(); it1 != l.end(); ++it1) {
            for (auto it2 = it1->begin(); it2 != it1->end(); ++it2) {
                (*this)(i, j) = *it2;
                ++j;
            }
            ++i;
            j = 0;
        }
    }
    Matrix(const Vector<T>& v) : data(v.get_n()), m(1), n(v.get_n()), transposed(false), type(Toeplitz) {
        for (size_t i = 0; i < n; ++i) {
            (*this)(0, i) = v[i];
        }
    }
    Matrix(const Matrix& other)
        : data(other.data), m(other.m), n(other.n), transposed(other.transposed), type(other.type) {}
    Matrix(Matrix&& other) noexcept
        : data(std::move(other.data)), m(other.m), n(other.n), transposed(other.transposed), type(other.type) {}

    template <class S>
    Matrix(Matrix&& other) noexcept
        requires(std::is_base_of_v<T, S>)
    {
        data.resize(other.get_m() * other.get_n());
        for (size_t i = 0; i < other.get_m(); ++i) {
            for (size_t j = 0; j < other.get_n(); ++j) {
                auto temp = other(i, j).template as_vector<T>();
                (*this)(i, j) = temp[0];
                for (size_t l = 1; l < S::get_m(); ++l) {
                    if (temp[l] != T(0)) {
                        std::cout << "Warning: data loss while converting superfield matrix to (prime) subfield matrix!"
                                  << std::endl;
                        break;
                    }
                }
            }
        }
    }

    template <class S>
    Matrix(const Matrix<S>& other) noexcept
        requires(std::is_base_of_v<S, T>)
    {
        data.resize(other.get_m() * other.get_n());
        m = other.get_m();
        n = other.get_n();
        transposed = false;
        type = Generic;
        for (size_t i = 0; i < other.get_m(); ++i) {
            for (size_t j = 0; j < other.get_n(); ++j) {
                (*this)(i, j) = T(other(i, j));
            }
        }
    }

    /* assignment operators */
    Matrix& operator=(const Matrix& rhs) {
        if (*this == rhs) return *this;
        data = rhs.data;
        m = rhs.m;
        n = rhs.n;
        transposed = rhs.transposed;
        type = rhs.type;
        return *this;
    }

    Matrix& operator=(Matrix&& rhs) noexcept {
        data = std::move(rhs.data);
        m = rhs.m;
        n = rhs.n;
        transposed = rhs.transposed;
        type = rhs.type;
        return *this;
    }

    /* non-modifying operations */
    Matrix operator+() const { return *this; }

    Matrix operator-() const {
        auto res = *this;
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            for (size_t mu = 0; mu < m; ++mu) {
                for (size_t nu = 0; nu < n; ++nu) {
                    res(mu, nu) *= -T(1);
                }
            }
            if (type == Vandermonde) {
                res.type = Generic;
            }
        } else if (type == Zero) {
            // continue;
        } else if (type == Diagonal || type == Identity) {
            for (size_t mu = 0; mu < m; ++mu) {
                res(mu, mu) *= -T(1);
            }
            if (type == Identity) {
                res.type = Diagonal;
            }
        }
        return res;
    }

    /* modifying operations */
    Matrix<T>& transpose() {
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            transposed = !transposed;
            std::swap(m, n);
            if (type == Vandermonde || type == Toeplitz) {
                type = Generic;
            }
        } else if (type == Zero) {
            std::swap(m, n);
        } else if (type == Diagonal || type == Identity) {
            // continue;
        }
        return *this;
    }

    Matrix<T>& rref(size_t* rank = nullptr) {
        if (type == Generic || type == Vandermonde || type == Toeplitz || type == Diagonal) {
            size_t h = 0;
            size_t k = 0;

            auto* cthis = const_cast<const Matrix<T>*>(this);

            while (h < m && k < n) {
                // find pivot (some nonzero element in column k)
                size_t p = 0;
                for (p = h; p < m; ++p) {
                    if ((*cthis)(p, k) != T(0)) break;
                }

                if (p == m) {  // no pivot in column -> proceed to next column
                    ++k;
                } else {
                    this->swap_rows(h, p);

                    // scale pivot row so that pivot becomes 1
                    const T pivot = (*cthis)(h, k);
                    this->scale_row(T(1) / pivot, h);

                    for (size_t i = 0; i < m; ++i) {
                        if (i == h)
                            continue;  // nothing to do for pivot
                                       // row

                        // scaling factor for row i
                        const T f = (*cthis)(i, k);

                        // update all components
                        for (size_t j = k; j < n; ++j) {
                            (*this)(i, j) -= f * (*cthis)(h, j);
                        }
                    }

                    ++h;
                    ++k;
                }
            }
            if (rank != nullptr) *rank = h;
            if (type == Vandermonde || type == Toeplitz) {
                type = Generic;
            }
        } else if (type == Zero) {
            if (rank != nullptr) *rank = 0;
        } else if (type == Identity) {
            if (rank != nullptr) *rank = m;
        }
        return *this;
    }

    Matrix<T>& invert() {
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            if (m != n) throw std::invalid_argument("trying to invert a non-square matrix");
            const auto I = IdentityMatrix<T>(m);
            Matrix temp(m, 2 * m);
            temp.set_submatrix(*this, 0, 0);
            temp.set_submatrix(I, 0, m);
            temp.rref();
            if (temp.get_submatrix(0, 0, m, m) != I)
                throw std::invalid_argument("trying to invert a non-invertible matrix");
            *this = temp.get_submatrix(0, m, m, m);
            if (type == Vandermonde || type == Toeplitz) {
                type = Generic;
            }
        } else if (type == Zero) {
            throw std::invalid_argument("trying to invert a non-invertible matrix/a zero matrix");
        } else if (type == Diagonal) {
            for (size_t mu = 0; mu < m; ++mu) {
                if ((*this)(mu, mu) == T(0)) {
                    throw std::invalid_argument(
                        "trying to invert a non-invertible matrix/a diagonal matrix with at least one zero on the "
                        "diagonal");
                } else {
                    (*this)(mu, mu) = T(1) / (*this)(mu, mu);
                }
            }
        } else if (type == Identity) {
            // continue;
        }
        return *this;
    }

    Matrix<T>& swap_rows(size_t i, size_t j) {
        if (i >= m || j >= m) throw std::invalid_argument("trying to swap non-existent row(s)");
        if (i == j) return *this;
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            for (size_t nu = 0; nu < n; ++nu) {
                std::swap((*this)(i, nu), (*this)(j, nu));
            }
            if (type == Vandermonde || type == Toeplitz) {
                type = Generic;
            }
        } else if (type == Zero) {
            // continue;
        } else if (type == Diagonal || type == Identity) {
            std::swap((*this)(i, i), (*this)(j, j));
            type = Generic;
        }
        return *this;
    }

    Matrix<T>& scale_row(const T& s, size_t i) {
        if (i >= m) throw std::invalid_argument("trying to scale non-existent row");
        if (s == T(1)) return *this;
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            for (size_t nu = 0; nu < n; ++nu) {
                (*this)(i, nu) *= s;
            }
            if (type == Vandermonde || type == Toeplitz) {
                type = Generic;
            }
        } else if (type == Zero) {
            // continue;
        } else if (type == Diagonal || type == Identity) {
            (*this)(i, i) *= s;
            if (type == Identity) {
                type = Diagonal;
            }
        }
        return *this;
    }

    Matrix<T>& add_scaled_row(const T& s, size_t i, size_t j) {
        if (i >= m || j >= m)
            throw std::invalid_argument("trying to add scaled row to other row, at least one of them is non-existent");
        if (i == j || s == T(0)) return *this;
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            for (size_t nu = 0; nu < n; ++nu) {
                (*this)(j, nu) += s * (*this)(i, nu);
            }
            if (type == Vandermonde || type == Toeplitz) {
                type = Generic;
            }
        } else if (type == Zero) {
            // continue;
        } else if (type == Diagonal || type == Identity) {
            (*this)(j, i) += s * (*this)(i, i);
            type = Generic;
        }
        return *this;
    }

    Matrix<T>& add_row(size_t i, size_t j) {
        if (i >= m || j >= m)
            throw std::invalid_argument("trying to add row to other row, at least one of them is non-existent");
        return add_scaled_row(T(1), i, j);
    }

    Matrix<T>& horizontal_join(const Matrix& other) {
        if (m != other.m)
            throw std::invalid_argument(
                "trying to horizontally join two "
                "matrices of incompatible dimensions");
        Matrix temp(m, n + other.n);
        temp.set_submatrix(*this, 0, 0);
        temp.set_submatrix(other, 0, n);
        *this = std::move(temp);
        return *this;
    }

    Matrix<T>& vertical_join(const Matrix& other) {
        if (n != other.n)
            throw std::invalid_argument(
                "trying to vertically join two "
                "matrices of incompatible dimensions");
        Matrix temp(m + other.m, n);
        temp.set_submatrix(*this, 0, 0);
        temp.set_submatrix(other, m, 0);
        *this = std::move(temp);
        return *this;
    }

    Matrix<T>& diagonal_join(const Matrix& other) noexcept {
        Matrix temp(m + other.m, n + other.n);
        temp.set_submatrix(*this, 0, 0);
        temp.set_submatrix(other, m, n);
        *this = std::move(temp);
        return *this;
    }

    Matrix<T>& Kronecker_product(const Matrix& other) {
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            Matrix temp(m * other.m, n * other.n);
            for (size_t mu = 0; mu < m; ++mu) {
                for (size_t nu = 0; nu < n; ++nu) {
                    temp.set_submatrix((*this)(mu, nu) * other, mu * other.m, nu * other.n);
                }
            }
            *this = std::move(temp);
        } else if (type == Zero) {
            *this = Matrix(m * other.m, n * other.n);
        } else if (type == Diagonal || type == Identity) {
            Matrix temp(m * other.m, n * other.n);
            for (size_t mu = 0; mu < m; ++mu) {
                temp.set_submatrix((*this)(mu, mu) * other, mu * other.m, mu * other.n);
            }
            *this = std::move(temp);
        }
        return *this;
    }

    Matrix<T>& delete_columns(std::vector<size_t> v) {
        std::sort(v.begin(), v.end(), std::greater<>());
        for (auto it = v.cbegin(); it != v.cend(); ++it) {
            if (*it >= n) throw std::invalid_argument("trying to delete non-existent column");
            Matrix left = get_submatrix(0, 0, m, *it);
            const Matrix right = get_submatrix(0, *it + 1, m, n - (*it + 1));
            *this = left.horizontal_join(right);
        }
        return *this;
    }

    Matrix<T>& delete_rows(std::vector<size_t> v) {
        std::sort(v.begin(), v.end(), std::greater<>());
        for (auto it = v.cbegin(); it != v.cend(); ++it) {
            if (*it >= m) throw std::invalid_argument("trying to delete non-existent row");
            Matrix left = get_submatrix(0, 0, *it, n);
            const Matrix right = get_submatrix(*it + 1, 0, m - (*it + 1), n);
            *this = left.vertical_join(right);
        }
        return *this;
    }

    Matrix<T>& reverse_rows() {
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            for (size_t mu = 0; mu < m / 2; ++mu) {
                for (size_t nu = 0; nu < n; ++nu) {
                    std::swap((*this)(mu, nu), (*this)(m - 1 - mu, nu));
                }
            }
            if (type == Vandermonde || type == Toeplitz) {
                type = Generic;
            }
        } else if (type == Zero) {
            // continue;
        } else if (type == Diagonal || type == Identity) {
            for (size_t mu = 0; mu < m / 2; ++mu) {
                std::swap((*this)(mu, mu), (*this)(m - 1 - mu, mu));
            }
            type = Generic;
        }
        return *this;
    }

    Matrix<T>& reverse_columns() {
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            for (size_t mu = 0; mu < m; ++mu) {
                for (size_t nu = 0; nu < n / 2; ++nu) {
                    std::swap((*this)(mu, nu), (*this)(mu, n - 1 - nu));
                }
            }
            if (type == Toeplitz) {
                type = Generic;
            }
        } else if (type == Zero) {
            // continue;
        } else if (type == Diagonal || type == Identity) {
            for (size_t nu = 0; nu < n / 2; ++nu) {
                std::swap((*this)(nu, nu), (*this)(nu, n - 1 - nu));
            }
            type = Generic;
        }
        return *this;
    }

    /* getters */
    size_t get_m() const { return m; }

    size_t get_n() const { return n; }

    bool is_empty() const { return m == 0 || n == 0; }

    size_t wH() const {
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            return data.size() - std::count(data.cbegin(), data.cend(), T(0));
        } else if (type == Zero) {
            return 0;
        } else if (type == Diagonal) {
            size_t res = 0;
            for (size_t mu = 0; mu < m; ++mu) {
                if ((*this)(mu, mu) != T(0)) {
                    ++res;
                }
            }
            return res;
        } else if (type == Identity) {
            return m;
        }
        assert("wH(): should never be here");
        return 0;  // dummy
    }

    size_t rank() const {
        Matrix<T> temp(*this);
        size_t r = 0;
        temp.rref(&r);
        return r;
    }

    bool is_invertible() const { return m == n && rank() == m; }

    Vector<T> diagonal() const {
        if (m != n) throw std::invalid_argument("trying to extract diagonal of non-square matrix");
        Vector<T> res(n);
        for (size_t i = 0; i < n; ++i) {
            res.set_component(i, (*this)(i, i));
        }
        return res;
    }

    Polynomial<T> characteristic_polynomial() const {
        if (m != n) throw std::invalid_argument("trying to calculate characteristic polynomial of non-square matrix");
        if (m == 0) throw std::invalid_argument("trying to calculate characteristic polynomial of empty matrix");
        if (m == 1) return Polynomial<T>({-(*this)(0, 0), 1});
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            // Samuelson–Berkowitz algorithm

            // calculate Toeplitz matrices
            std::vector<Matrix<T>> TM;  // Toeplitz matrices
            TM.reserve(m);
            for (size_t i = 0; i < m; ++i) {
                // partion matrix
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

            // multiply Toeplitz matrices together
            Matrix P = IdentityMatrix<T>(m + 1);
            for (auto it = TM.cbegin(); it != TM.cend(); ++it) {
                P *= *it;
            }

            // extract polynomial from solution/column vector
            Polynomial<T> res;
            for (size_t i = 0; i < m + 1; ++i) {
                res.set_coeff(m - i, const_cast<const Matrix<T>&>(P)(i, 0));
            }

            // for odd n: negate characteristic polynomial *ToDo: verify*
            if (m % 2) res *= T(-1);

            return res;
        } else if (type == Zero) {
            return Polynomial<T>({0});
        } else if (type == Diagonal) {
            Polynomial<T> res({1});
            for (size_t mu = 0; mu < m; ++mu) {
                res *= Polynomial<T>({(*this)(mu, mu), -1});
            }
            return res;
        } else if (type == Identity) {
            Polynomial<T> res({0, 1});
            return res ^ m;
        }
        assert("characteristic_polynomial(): should never be here");
        return Polynomial<T>();  // dummy
    }

    Matrix<T> basis_of_nullspace() const {
        Matrix<T> temp(*this);
        size_t r = 0;
        temp.rref(&r);
        Matrix B(n - r, n);

        std::vector<size_t> mocols;
        size_t i = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i < m) {
                if (temp(i, j) == T(1)) {
                    ++i;
                } else {
                    mocols.push_back(j);
                }
            } else {
                mocols.push_back(j);
            }
        }

        size_t offset = 0;
        for (size_t i = 0; i < m + offset + 1; ++i) {
            for (size_t k = offset; k < n - r; ++k) {
                if (i + offset == mocols[k]) {
                    B(k, i + offset) = -T(1);
                    ++offset;
                } else {
                    B(k, i + offset) = temp(i, mocols[k]);
                }
            }
        }
        return B;
    }

    Matrix<T> basis_of_kernel() const { return basis_of_nullspace(); }

    T determinant() const {
        if (m != n) throw std::invalid_argument("trying to calculate determinant of non-square matrix");
        if (m == 0) throw std::invalid_argument("trying to calculate determinant of empty matrix");
        if (m == 1) return ((*this))(0, 0);
        if (type == Generic || type == Toeplitz) {
            return characteristic_polynomial()[0];
        } else if (type == Vandermonde) {
            T res = T(1);
            for (size_t mu = 0; mu < m; ++mu) {
                for (size_t i = 0; i < mu; ++i) {
                    res *= (*this)(1, mu) - (*this)(1, i);
                }
            }
            return res;
        } else if (type == Zero) {
            return T(0);
        } else if (type == Diagonal) {
            T res = T(1);
            for (size_t mu = 0; mu < m; ++mu) {
                const T temp = (*this)(mu, mu);
                if (temp == T(0)) {
                    return T(0);
                } else {
                    res *= temp;
                }
            }
            return res;
        } else if (type == Identity) {
            return T(1);
        }
        assert("determinant(): should never be here");
        return T(0);  // dummy
    }

    template <bool b = std::is_base_of_v<ECC::Base, T>>
    std::vector<T> eigenvalues() const
        requires(b)
    {
        const auto p = characteristic_polynomial();
        std::vector<T> res;
        for (size_t j = 0; j < T::get_size(); ++j) {
            if (p(T(j)) == T(0)) {
                res.push_back(T(j));
            }
        }
        return res;
    }

    template <bool b = std::is_base_of_v<ECC::Base, T>>
    std::vector<Vector<T>> rowspace() const
        requires(b)
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
        return res;
    }

    template <bool b = std::is_base_of_v<ECC::Base, T>>
    std::vector<Vector<T>> get_span() const
        requires(b)
    {
        return rowspace();
    }

    /* operational assignments */
    Matrix& operator+=(const Matrix& rhs) {
        if (m != rhs.m || n != rhs.n)
            throw std::invalid_argument(
                "trying to add two matrices of different "
                "dimensions");
        if (type == Zero) {
            *this = rhs;
        } else if (rhs.type == Zero) {
            // continue;
        } else if ((type == Diagonal && rhs.type == Diagonal) || (type == Identity && rhs.type == Identity) ||
                   (type == Diagonal && rhs.type == Identity) || (type == Identity && rhs.type == Diagonal)) {
            for (size_t mu = 0; mu < m; ++mu) {
                (*this)(mu, mu) += rhs(mu, mu);
            }
        } else {
            for (size_t mu = 0; mu < m; ++mu) {
                for (size_t nu = 0; nu < n; ++nu) {
                    (*this)(mu, nu) += rhs(mu, nu);
                }
            }
        }
        bool zeroflag = true;
        for (auto it = this->data.cbegin(); it != this->data.cend(); ++it) {
            if (*it != T(0)) {
                zeroflag = false;
                break;
            }
        }
        if (zeroflag) this->type = Zero;
        return *this;
    }

    Matrix& operator-=(const Matrix& rhs) {
        if (m != rhs.m || n != rhs.n)
            throw std::invalid_argument(
                "trying to subtract two matrices of different "
                "dimensions");
        operator+=(-rhs);
        return *this;
    }

    Matrix& operator*=(const Matrix& rhs) {
        if (n != rhs.m)
            throw std::invalid_argument(
                "trying to multiply two matrices "
                "with incompatible dimensions");
        if (type == Zero || rhs.type == Zero) {
            *this = Matrix(m, rhs.n);
        } else if (type == Identity) {
            *this = rhs;
        } else if (rhs.type == Identity) {
            // continue;
        } else if (type == Diagonal && rhs.type == Diagonal) {
            for (size_t mu = 0; mu < m; ++mu) {
                (*this)(mu, mu) *= rhs(mu, mu);
            }
        } else if (type == Diagonal) {
            auto res = rhs;
            for (size_t mu = 0; mu < m; ++mu) {
                auto s = (*this)(mu, mu);
                for (size_t nu = 0; nu < rhs.n; ++nu) {
                    res(mu, nu) *= s;
                }
            }
            *this = std::move(res);
        } else if (rhs.type == Diagonal) {
            for (size_t nu = 0; nu < n; ++nu) {
                const auto& s = rhs(nu, nu);
                for (size_t mu = 0; mu < m; ++mu) {
                    (*this)(mu, nu) *= s;
                }
            }
        } else {
            Matrix<T> res(m, rhs.n);
            for (size_t mu = 0; mu < m; ++mu) {
                for (size_t nu = 0; nu < n; ++nu) {
                    if (const_cast<const Matrix&>(*this)(mu, nu) == T(0)) continue;  // skip a few OPs if possible
                    for (size_t i = 0; i < res.n; ++i) {
                        const T prod = (const_cast<const Matrix&>(*this))(mu, nu) * rhs(nu, i);
                        if (prod == T(0)) continue;
                        res(mu, i) += prod;
                    }
                }
            }
            *this = std::move(res);
        }
        bool zeroflag = true;
        for (auto it = this->data.cbegin(); it != this->data.cend(); ++it) {
            if (*it != T(0)) {
                zeroflag = false;
                break;
            }
        }
        if (zeroflag) this->type = Zero;
        return *this;
    }

    Matrix& operator*=(const T& s) {
        if (s == T(0)) {
            *this = Matrix(m, n);
        } else if (s == T(1) || type == Zero) {
            // continue;
        } else if (type == Generic || type == Vandermonde || type == Toeplitz) {
            for (size_t mu = 0; mu < m; ++mu) {
                for (size_t nu = 0; nu < n; ++nu) {
                    (*this)(mu, nu) *= s;
                }
            }
        } else if (type == Diagonal || type == Identity) {
            for (size_t mu = 0; mu < m; ++mu) {
                (*this)(mu, mu) *= s;
            }
        }
        return *this;
    }

    Matrix& operator/=(const T& s) {
        if (s == T(0)) throw std::invalid_argument("trying to divide components of matrix by zero");
        operator*=(T(1) / s);
        return *this;
    }

    /* component access */
    T& operator()(size_t i, size_t j) {
        if (i >= m || j >= n) throw std::invalid_argument("trying to access non-existent element of matrix");
        if (type == Vandermonde || type == Toeplitz || type == Zero || type == Identity ||
            (type == Diagonal && i != j)) {
            type = Generic;
        }
        if (!transposed) return data[i * n + j];
        return data[i + j * m];
    }

    const T& operator()(size_t i, size_t j) const {
        if (i >= m || j >= n) throw std::invalid_argument("trying to access non-existent element of matrix");
        if (!transposed) return data[i * n + j];
        return data[i + j * m];
    }

    Vector<T> get_row(size_t i) const {
        if (i >= m) throw std::invalid_argument("trying to access non-existent row");
        Vector<T> res(n);
        for (size_t j = 0; j < n; ++j) {
            res.set_component(j, (*this)(i, j));
        }
        return res;
    }

    Vector<T> get_col(size_t j) const {
        if (j >= n) throw std::invalid_argument("trying to access non-existent column");
        Vector<T> res(m);
        for (size_t i = 0; i < m; ++i) {
            res.set_component(i, (*this)(i, j));
        }
        return res;
    }

    Matrix<T> get_submatrix(size_t i, size_t j, size_t h, size_t w) const {
        if (i + h > m || j + w > n)
            throw std::invalid_argument(
                "trying to extract a submatrix with incompatible "
                "dimensions");
        Matrix res(h, w);
        if (type == Generic || type == Vandermonde || type == Toeplitz) {
            for (size_t mu = 0; mu < h; ++mu) {
                for (size_t nu = 0; nu < w; ++nu) {
                    res(mu, nu) = (*this)(i + mu, j + nu);
                }
            }
            if (type == Vandermonde && i == 0) {
                res.type = Vandermonde;
            } else if (type == Toeplitz) {
                res.type = Toeplitz;
            }
        } else if (type == Zero) {
            // continue;
        } else if (type == Diagonal || type == Identity) {
            for (size_t mu = 0; mu < h; ++mu) {
                for (size_t nu = 0; nu < w; ++nu) {
                    if (i + mu == j + nu) {
                        res(mu, nu) = (*this)(i + mu, j + nu);
                    }
                }
            }
            if (i == j) {
                if (type == Diagonal) {
                    res.type = Diagonal;
                } else if (type == Identity) {
                    res.type = Identity;
                }
            }
        }
        return res;
    }

    Matrix<T>& set_submatrix(const Matrix& M, size_t i, size_t j) {
        if (m < i + M.m || n < j + M.n)
            throw std::invalid_argument(
                "trying to replace submatrix with "
                "matrix of incompatible dimensions");
        for (size_t mu = 0; mu < M.m; ++mu) {
            for (size_t nu = 0; nu < M.n; ++nu) {
                (*this)(i + mu, j + nu) = M(mu, nu);
            }
        }
        type = Generic;
        return *this;
    }

    /* randomization */
    template <bool b = std::is_base_of_v<ECC::Base, T>>
    void randomize()
        requires(b)
    {
        std::for_each(data.begin(), data.end(), std::mem_fn(&T::randomize));
        type = Generic;
    }

    /* vector as matrix over subfield */
    // template <class S, class U = T, typename std::enable_if<!std::is_arithmetic<U>::value>::type>
    template <class S, bool b = is_finite_field<T>()>
    Vector<S> as_vector() const noexcept
        requires(b)
    {
        Vector<S> res(get_n());
        Matrix<T> Tp(*this);
        Tp.transpose();
        for (size_t i = 0; i < get_n(); ++i) {
            auto temp = Tp.get_row(i);
            res.set_component(i, S(temp));
        }
        return res;
    }

   private:
    std::vector<T> data;
    size_t m;
    size_t n;
    bool transposed;
    matrix_type_t type;
};

/* free functions wrt. Matrix */

template <class T>
size_t wH(const Matrix<T>& M) noexcept {
    return M.wH();
}

template <class T>
Matrix<T> transpose(const Matrix<T>& M) noexcept {
    Matrix<T> res(M);
    res.transpose();
    return res;
}

template <class T>
Matrix<T> transpose(Matrix<T>&& M) noexcept {
    Matrix<T> res(std::move(M));
    res.transpose();
    return res;
}

template <class T>
Matrix<T> rref(const Matrix<T>& M) noexcept {
    Matrix<T> res(M);
    res.rref();
    return res;
}

template <class T>
Matrix<T> rref(Matrix<T>&& M) noexcept {
    Matrix<T> res(std::move(M));
    res.rref();
    return res;
}

template <class T>
Matrix<T> inverse(const Matrix<T>& M) noexcept {
    Matrix<T> res(M);
    res.invert();
    return res;
}

template <class T>
Matrix<T> inverse(Matrix<T>&& M) noexcept {
    Matrix<T> res(std::move(M));
    res.invert();
    return res;
}

template <class T>
T determinant(const Matrix<T>& M) noexcept {
    return M.determinant();
}

template <class T>
Matrix<T> horizontal_join(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res.horizontal_join(rhs);
    return res;
}

template <class T>
Matrix<T> horizontal_join(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res.horizontal_join(rhs);
    return res;
}

template <class T>
Matrix<T> vertical_join(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res.vertical_join(rhs);
    return res;
}

template <class T>
Matrix<T> diagonal_join(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res.diagonal_join(rhs);
    return res;
}

template <class T>
Matrix<T> diagonal_join(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res.diagonal_join(rhs);
    return res;
}

template <class T>
Matrix<T> vertical_join(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res.vertical_join(rhs);
    return res;
}

template <class T>
Matrix<T> Kronecker_product(const Matrix<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(lhs);
    res.Kronecker_product(rhs);
    return res;
}

template <class T>
Matrix<T> Kronecker_product(Matrix<T>&& lhs, const Matrix<T>& rhs) {
    Matrix<T> res(std::move(lhs));
    res.Kronecker_product(rhs);
    return res;
}

template <class T>
Matrix<T> delete_columns(const Matrix<T>& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(lhs);
    res.delete_columns(v);
    return res;
}

template <class T>
Matrix<T> delete_columns(Matrix<T>&& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(std::move(lhs));
    res.delete_columns(v);
    return res;
}

template <class T>
Matrix<T> delete_rows(const Matrix<T>& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(lhs);
    res.delete_rows(v);
    return res;
}

template <class T>
Matrix<T> delete_rows(Matrix<T>&& lhs, const std::vector<size_t>& v) {
    Matrix<T> res(std::move(lhs));
    res.delete_rows(v);
    return res;
}

template <class T>
Matrix<T> reverse_columns(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.reverse_columns();
    return res;
}

template <class T>
Matrix<T> reverse_columns(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.reverse_columns();
    return res;
}

template <class T>
Matrix<T> reverse_rows(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.reverse_rows();
    return res;
}

template <class T>
Matrix<T> reverse_rows(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.reverse_rows();
    return res;
}

/*
 * matrix + matrix
 */

template <class T>
Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) noexcept {
    Matrix<T> res(lhs);
    res += rhs;
    return res;
}

template <class T>
Matrix<T> operator+(const Matrix<T>& lhs, Matrix<T>&& rhs) noexcept {
    Matrix<T> res(std::move(rhs));
    res += lhs;
    return res;
}

template <class T>
Matrix<T> operator+(Matrix<T>&& lhs, const Matrix<T>& rhs) noexcept {
    Matrix<T> res(std::move(lhs));
    res += rhs;
    return res;
}

/*
 * matrix - matrix
 */

template <class T>
Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) noexcept {
    Matrix<T> res(lhs);
    res -= rhs;
    return res;
}

template <class T>
Matrix<T> operator-(const Matrix<T>& lhs, Matrix<T>&& rhs) noexcept {
    Matrix<T> res(std::move(rhs));
    res -= lhs;
    return -res;
}

template <class T>
Matrix<T> operator-(Matrix<T>&& lhs, const Matrix<T>& rhs) noexcept {
    Matrix<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

template <class T>
Matrix<T> operator-(Matrix<T>&& lhs, Matrix<T>&& rhs) noexcept {
    Matrix<T> res(std::move(lhs));
    res -= rhs;
    return res;
}

/*
 * matrix * matrix
 */

template <class T>
Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) noexcept {
    Matrix<T> res(lhs);
    res *= rhs;
    return res;
}

template <class T>
Matrix<T> operator*(Matrix<T>&& lhs, const Matrix<T>& rhs) noexcept {
    Matrix<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

/*
 * vector * matrix
 */

template <class T>
Vector<T> operator*(const Vector<T>& lhs, const Matrix<T>& rhs) {
    Matrix<T> temp(lhs);
    return (temp * rhs).get_row(0);
}

/*
 * matrix * T
 */

template <class T>
Matrix<T> operator*(const Matrix<T>& lhs, const T& rhs) noexcept {
    Matrix<T> res(lhs);
    res *= rhs;
    return res;
}

template <class T>
Matrix<T> operator*(Matrix<T>&& lhs, const T& rhs) noexcept {
    Matrix<T> res(std::move(lhs));
    res *= rhs;
    return res;
}

/*
 * T * matrix
 */

template <class T>
Matrix<T> operator*(const T& lhs, const Matrix<T>& rhs) noexcept {
    Matrix<T> res(rhs);
    res *= lhs;
    return res;
}

template <class T>
Matrix<T> operator*(const T& lhs, Matrix<T>&& rhs) noexcept {
    Matrix<T> res(std::move(rhs));
    res *= lhs;
    return res;
}

/*
 * matrix / T
 */

template <class T>
Matrix<T> operator/(const Matrix<T>& lhs, const T& rhs) noexcept {
    Matrix<T> res(lhs);
    res /= rhs;
    return res;
}

template <class T>
Matrix<T> operator/(Matrix<T>&& lhs, const T& rhs) noexcept {
    Matrix<T> res(std::move(lhs));
    res /= rhs;
    return res;
}

/*
 * comparison
 */

template <class T>
bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs) noexcept {
    if (lhs.m != rhs.m || lhs.n != rhs.n) return false;
    if (lhs.type == Toeplitz && rhs.type == Toeplitz) {
        for (size_t mu = 0; mu < lhs.m; ++mu) {
            if (lhs(lhs.m - 1 - mu, 0) != rhs(lhs.m - 1 - mu, 0)) return false;
        }
        for (size_t nu = 1; nu < lhs.n; ++nu) {
            if (lhs(0, nu) != rhs(0, nu)) return false;
        }
    } else if ((lhs.type == Diagonal && rhs.type == Diagonal) || (lhs.type == Diagonal && rhs.type == Identity) ||
               (lhs.type == Identity && rhs.type == Diagonal)) {
        for (size_t mu = 0; mu < lhs.m; ++mu) {
            if (lhs(mu, mu) != rhs(mu, mu)) return false;
        }
    } else if (lhs.type == Identity && rhs.type == Identity) {
        // continue;
    } else if ((lhs.type == Zero && rhs.type != Zero) || (lhs.type != Zero && rhs.type == Zero)) {
        return false;
    } else {
        for (size_t mu = 0; mu < lhs.m; ++mu) {
            for (size_t nu = 0; nu < lhs.n; ++nu) {
                if (lhs(mu, nu) != rhs(mu, nu)) return false;
            }
        }
    }
    return true;
}

template <class T>
bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs) noexcept {
    return !(lhs == rhs);
}

/*
 * randomization
 */

template <class T>
Matrix<T> randomize(const Matrix<T>& M) {
    Matrix<T> res(M);
    res.randomize();
    return res;
}

template <class T>
Matrix<T> randomize(Matrix<T>&& M) {
    Matrix<T> res(std::move(M));
    res.randomize();
    return res;
}

/*
 * output
 */

template <class T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& rhs) noexcept {
    if (rhs.m == 0 || rhs.n == 0) {
        os << "(empty matrix)";
        return os;
    }
    size_t max = 0;
    std::stringstream ss;
    for (size_t i = 0; i < rhs.m; ++i) {
        for (size_t j = 0; j < rhs.n; ++j) {
            ss << rhs(i, j);
            max = std::max(ss.str().length(), max);
            ss.str(std::string());  // clear stringstream
        }
    }
    os << (rhs.m == 1 ? "(" : "⌈");
    for (size_t j = 0; j < rhs.n - 1; ++j) {
        os << std::setw(max) << rhs(0, j);
        os << " ";  // must be in extra line due to set::setw()
    }
    os << std::setw(max) << rhs(0, rhs.n - 1);
    os << (rhs.m == 1 ? ")" : "⌉");
    if (rhs.m > 1) os << std::endl;
    for (size_t i = 1; i < rhs.m - 1; ++i) {
        os << "|";
        for (size_t j = 0; j < rhs.n - 1; ++j) {
            os << std::setw(max) << rhs(i, j);
            os << " ";  // must be in extra line due to set::setw()
        }
        os << std::setw(max) << rhs(i, rhs.n - 1);
        os << "|" << std::endl;
    }
    if (rhs.m > 1) {
        os << "⌊";
        for (size_t j = 0; j < rhs.n - 1; ++j) {
            os << std::setw(max) << rhs(rhs.m - 1, j);
            os << " ";  // must be in extra line due to set::setw()
        }
        os << std::setw(max) << rhs(rhs.m - 1, rhs.n - 1);
        os << "⌋";
    }
    return os;
}

template <class T>
Matrix<T> ZeroMatrix(size_t m, size_t n) {
    return Matrix<T>(m, n);
}

template <class T>
Matrix<T> IdentityMatrix(size_t m) {
    auto res = Matrix<T>(m, m);
    for (size_t i = 0; i < m; ++i) {
        res(i, i) = T(1);
    }
    res.type = Identity;
    return res;
}

template <class T>
Matrix<T> ExchangeMatrix(size_t m) {
    auto res = IdentityMatrix<T>(m);
    res.reverse_columns();
    return res;
}

template <class T>
Matrix<T> DiagonalMatrix(const Vector<T>& v) {
    const size_t m = v.get_n();
    Matrix<T> res(m, m);
    for (size_t i = 0; i < m; ++i) {
        res(i, i) = v[i];
    }
    res.type = Diagonal;
    return res;
}

template <class T>
Matrix<T> ToeplitzMatrix(const Vector<T>& v, size_t m, size_t n) {
    if (v.get_n() != m + n - 1)
        throw std::invalid_argument(
            "vector for constructing m x n Toeplitz matrix must have "
            "length m+n-1");
    Matrix<T> res(m, n);
    for (size_t i = 0; i < m; ++i) {
        res(m - 1 - i, 0) = v[i];
    }
    for (size_t j = 1; j < n; ++j) {
        res(0, j) = v[m - 1 + j];
    }
    for (size_t i = 1; i < m; ++i) {
        for (size_t j = 1; j < n; ++j) {
            res(i, j) = const_cast<const Matrix<T>&>(res)(i - 1, j - 1);
        }
    }
    res.type = Toeplitz;
    return res;
}

template <class T>
Matrix<T> HankelMatrix(const Vector<T>& v, size_t m, size_t n) {
    return ToeplitzMatrix<T>(reverse(v), m, n) * ExchangeMatrix<T>(n);
}

template <class T>
Matrix<T> VandermondeMatrix(const Vector<T>& v, size_t m) {
    const size_t n = v.get_n();
    if (n == 0)
        throw std::invalid_argument(
            "vector for constructing Vandermonde matrix must have "
            "at least one element");
    if (m == 0) throw std::invalid_argument("trying to construct Vandermonde matrix with zero rows");
    std::vector<T> temp;
    for (size_t i = 0; i < n; ++i) {
        if (temp.end() == std::find(temp.begin(), temp.end(), v[i])) temp.push_back(v[i]);
    }
    if (temp.size() < n)
        throw std::invalid_argument("vector for constructing Vandermonde matrix must have pairwise distinct elements");

    Matrix<T> res(m, n);
    for (size_t i = 0; i < n; ++i) {
        res(0, i) = T(1);
    }
    if (m > 1) {
        for (size_t i = 0; i < n; ++i) {
            res(1, i) = v[i];
            for (size_t j = 2; j < m; ++j) {
                res(j, i) = res(j - 1, i) * v[i];
            }
        }
    }
    res.type = Vandermonde;
    return res;
}

template <class T>
Matrix<T> UpperShiftMatrix(size_t m) {
    Matrix<T> res(m, m);
    for (size_t i = 0; i < m - 1; ++i) {
        res(i, i + 1) = T(1);
    }
    return res;
}

template <class T>
Matrix<T> LowerShiftMatrix(size_t m) {
    return transpose(UpperShiftMatrix<T>(m));
}

template <class T>
Matrix<T> CompanionMatrix(const Polynomial<T>& poly) {
    if (!poly.is_monic()) throw std::invalid_argument("companion matrices only defined for monic polynomials");
    Matrix<T> res(transpose(UpperShiftMatrix<T>(poly.get_degree())));
    for (size_t i = 0; i < poly.get_degree(); ++i) {
        res(i, poly.get_degree() - 1) = -poly[i];
    }
    return res;
}

}  // namespace ECC

#endif
