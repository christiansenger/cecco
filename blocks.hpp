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

#ifndef BLOCKS_HPP
#define BLOCKS_HPP

#include <complex>
#include <random>
#include <type_traits>

#include "fields.hpp"
#include "matrices.hpp"
#include "vectors.hpp"

namespace ECC {

template <class IN, class OUT = IN>
class Block {
   public:
    virtual OUT operator()(const IN& in) = 0;
    // virtual OUT operator()(IN&& in)=0;
};

template <class IN, class OUT = IN>
OUT operator>>(const IN& in, Block<IN, OUT>& block) {
    // std::cout << "non-destructive" << std::endl;
    return block(in);
}

template <class IN, class OUT = IN>
Vector<OUT> operator>>(const Vector<IN>& in, Block<IN, OUT>& block) {
    // std::cout << "non-destructive" << std::endl;
    Vector<OUT> res(in.get_n());
    for (size_t i = 0; i < in.get_n(); ++i) {
        res.set_component(i, block(in[i]));
    }
    return res;
}

template <class IN, class OUT = IN>
Matrix<OUT> operator>>(const Matrix<IN>& in, Block<IN, OUT>& block) {
    // std::cout << "non-destructive" << std::endl;
    Matrix<OUT> res(in.get_m(), in.get_n());
    for (size_t i = 0; i < in.get_m(); ++i) {
        for (size_t j = 0; j < in.get_n(); ++j) {
            res(i, j) = block(in(i, j));
        }
    }
    return res;
}

template <class T>
T operator>>(T&& in, Block<T, T>& block) {
    // std::cout << "destructive" << std::endl;
    in = block(in);
    return std::forward<T>(in);
}

template <class T>
Vector<T> operator>>(Vector<T>&& in, Block<T, T>& block) {
    // std::cout << "destructive" << std::endl;
    for (size_t i = 0; i < in.get_n(); ++i) {
        in.set_component(i, block(in[i]));
    }
    return std::move(in);
}

template <class T>
Matrix<T> operator>>(Matrix<T>&& in, Block<T, T>& block) {
    // std::cout << "destructive" << std::endl;
    for (size_t i = 0; i < in.get_m(); ++i) {
        for (size_t j = 0; j < in.get_n(); ++j) {
            in(i, j) = block(in(i, j));
        }
    }
    return std::move(in);
}

template <class T>
class DMC : public Block<T> {
   public:
    DMC(double pe) : dist(pe) {
        if (pe != 0.0 && pe < 0.000000001) throw std::out_of_range("pe too small");
        failures_before_hit = dist(gen);
    }

    T operator()(const T& in) noexcept {
        if (dist.p() == 0.0) return in;
        T res(in);
        if (trials == failures_before_hit) {
            res.randomize_force_change();
            trials = 0;
            failures_before_hit = dist(gen);
        } else {
            ++trials;
        }
        return res;
    }

   private:
    std::geometric_distribution<unsigned int> dist;
    unsigned int trials{0};
    unsigned int failures_before_hit;
};

using BSC = DMC<Fp<2>>;

class Mapper {
   public:
    virtual double getEb() const = 0;
};

class NRZEncoder : public Block<Fp<2>, std::complex<double>>, public Mapper {
   public:
    NRZEncoder(double a, double b) : a(a), b(b) {}

    double getEb() const noexcept override { return pow(a, 2.0) + pow(b, 2.0) / 4.0; }  // assuming symbol duration is Delta=1

    double get_a() const { return a; }
    double get_b() const { return b; }

    std::complex<double> operator()(const Fp<2>& in) noexcept override {
        if (in == Fp<2>(0)) {
            return {a - b / 2.0, 0};
        } else {
            return {a + b / 2.0, 0};
        }
    }

    Vector<std::complex<double>> operator()(const Vector<Fp<2>>& in) noexcept {
        Vector<std::complex<double>> res(in.get_n());
        for (size_t i = 0; i < in.get_n(); ++i) {
            res.set_component(i, operator()(in[i]));
        }
        return res;
    }

   private:
    const double a{};
    const double b{};
};

class BPSKEncoder : public NRZEncoder {
   public:
    BPSKEncoder() : NRZEncoder(0.0, 2.0) {}
};

class AWGN : public Block<std::complex<double>> {
   public:
    /*
     * AWGN: sigma=sqrt(No/2)
     * EbNo=10^(EbNodB/10)
     * No=Eb/10^(EbNodB/10)
     */
    AWGN(const Mapper& mapper, double EbNodB)
        : dist(0, sqrt(0.5 * mapper.getEb() / pow(10, EbNodB / 10))),
          g(&gen),
          pe(0.5 * erfc(sqrt(pow(10, EbNodB / 10)))) {}

    double get_variance() const noexcept { return pow(dist.stddev(), 2); }

    double get_standard_deviation() const noexcept { return dist.stddev(); }

    double get_pe() const noexcept { return pe; }

    void set_generator(std::mt19937& gen) { g = &gen; }

    std::complex<double> operator()(const std::complex<double>& in) noexcept override {
        std::complex<double> res(in.real() + dist(*g), in.imag() + dist(*g));
        return res;
    }

   private:
    std::normal_distribution<double> dist;
    std::mt19937* g;
    const double pe{};
};

class NRZDecoder : public Block<std::complex<double>, Fp<2>> {
   public:
    NRZDecoder(const NRZEncoder& nrz) : a(nrz.get_a()) {}

    Fp<2> operator()(const std::complex<double>& in) noexcept override {
        if (in.real() >= a) {
            return {1};
        } else {
            return {0};
        }
    }

    Vector<Fp<2>> operator()(const Vector<std::complex<double>>& in) noexcept {
        Vector<Fp<2>> res(in.get_n());
        for (size_t i = 0; i < in.get_n(); ++i) {
            res.set_component(i, operator()(in[i]));
        }
        return res;
    }

   private:
    const double a{};
};

class BPSKDecoder : public NRZDecoder {
   public:
    BPSKDecoder() : NRZDecoder(BPSKEncoder()) {}
};

class LLRCalculator : public Block<std::complex<double>, double> {
   public:
    LLRCalculator(const NRZEncoder& nrz, const AWGN& transmission)
        : a(nrz.get_a()), b(nrz.get_b()), sigmasq(pow(transmission.get_standard_deviation(), 2.0)) {}

    double operator()(const std::complex<double>& in) noexcept override { return b * (a - in.real()) / sigmasq; }

    Vector<double> operator()(const Vector<std::complex<double>>& in) noexcept {
        Vector<double> res(in.get_n());
        for (size_t i = 0; i < in.get_n(); ++i) {
            res.set_component(i, operator()(in[i]));
        }
        return res;
    }

   private:
    const double a{};
    const double b{};
    const double sigmasq{};
};

}  // namespace ECC

#endif