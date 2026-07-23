/**
 * @file helpers.hpp
 * @brief Utility functions and mathematical helpers
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.2.2
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
 * This header collects small utilities used by the algebraic types: random number generation,
 * integer arithmetic, square-and-multiply exponentiation, double-and-add multiplication,
 * caching, maxima, constexpr floor, and divisibility tests.
 */

#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "InfInt.hpp"

namespace CECCO {

/**
 * @brief Thread-local random number generator with shared seeding policy
 *
 * Provides one `std::mt19937` engine per thread. `seed()` selects deterministic seeding;
 * `use_hardware_seed()` returns to seeding from `std::random_device`. A seed change is observed
 * on the next call to @ref get in each thread.
 *
 * @note Use the static interface; the class has no object state.
 */
class RNG {
   public:
    /**
     * @brief Get thread-local random number generator
     * @return Reference to thread-local std::mt19937 generator
     *
     * Initializes the engine on first access. If the global seed generation changed since
     * the previous access in this thread, reseeds before returning.
     */
    static std::mt19937& get() {
        static thread_local std::mt19937 generator{get_initial_seed()};
        if (should_reseed()) generator.seed(get_current_seed());
        return generator;
    }

    /**
     * @brief Set deterministic seed for all threads
     * @param seed Base seed value (combined with a per-thread registration index)
     *
     * Selects deterministic seeding. Each thread derives its seed from @p seed and an index
     * assigned on the thread's first use of the generator; the first thread (index 0) uses
     * @p seed unchanged. Single-threaded runs are therefore fully reproducible. In
     * multithreaded runs the set of streams is fixed, but their assignment to threads follows
     * first-use order, which scheduling may permute across runs.
     */
    static void seed(uint32_t seed) {
        use_deterministic_seed.store(true);
        deterministic_seed_value.store(seed);
        reseed_generation.fetch_add(1);  // Signal all threads to reseed
    }

    /**
     * @brief Enable hardware-based random seeding
     *
     * Selects seeding from `std::random_device`.
     */
    static void use_hardware_seed() {
        use_deterministic_seed.store(false);
        reseed_generation.fetch_add(1);  // Signal all threads to reseed
    }

   private:
    static inline std::atomic<bool> use_deterministic_seed{false};
    static inline std::atomic<uint32_t> deterministic_seed_value{0};
    static inline std::atomic<uint32_t> reseed_generation{0};

    static uint32_t get_initial_seed() {
        if (use_deterministic_seed.load()) {
            // Weyl increment (2^32 / golden ratio) decorrelates the per-thread seeds
            return deterministic_seed_value.load() + 0x9e3779b9u * thread_index();
        } else {
            static thread_local std::random_device rd;
            return rd();
        }
    }

    static uint32_t thread_index() {
        static std::atomic<uint32_t> count{0};
        static thread_local const uint32_t index = count.fetch_add(1);
        return index;
    }

    static uint32_t get_current_seed() { return get_initial_seed(); }

    static bool should_reseed() {
        static thread_local uint32_t last_seen_generation = 0;
        uint32_t current_generation = reseed_generation.load();
        if (current_generation != last_seen_generation) {
            last_seen_generation = current_generation;
            return true;
        }
        return false;
    }
};

/**
 * @brief Current thread's random number generator
 * @return Reference to thread-local random number generator
 *
 * Equivalent to @ref RNG::get.
 */
inline std::mt19937& gen() { return RNG::get(); }

/**
 * @brief Indices of all maximum elements
 * @tparam T Element type (must support operator< for comparison)
 * @param v Input vector
 * @return Indices i with `v[i] == max(v)`; empty if @p v is empty
 *
 * Uses two passes over @p v.
 */
template <class T>
std::vector<size_t> find_maxima(const std::vector<T>& v) {
    std::vector<size_t> indices;
    if (v.empty()) return indices;
    const auto max_value = *std::max_element(v.begin(), v.end());
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == max_value) indices.push_back(i);
    }
    return indices;
}

/**
 * @brief Primality test by trial division
 * @tparam T Unsigned integer type
 * @param a Number to test for primality
 * @return true if a is prime, false otherwise
 *
 * Tests odd divisors up to √a. Returns false for a ≤ 1 and even a > 2.
 */
template <class T>
constexpr bool is_prime(T a) noexcept {
    if (a == 2) return true;
    if (a <= 1 || !(a & 1)) return false;
    // find "smaller half" of factorization (if factor > sqrt(a) there must be a factor < sqrt(a))
    for (T b = 3; b * b <= a; b += 2)
        if ((a % b) == 0) return false;
    return true;
}

/**
 * @brief Greatest common divisor and optional Bézout coefficients
 * @tparam T Signed integral type
 * @param a First integer
 * @param b Second integer
 * @param s Pointer to store Bézout coefficient for a (optional)
 * @param t Pointer to store Bézout coefficient for b (optional)
 * @return Greatest common divisor of a and b
 *
 * If @p s and @p t are non-null, stores coefficients satisfying `a*s + b*t = gcd(a,b)`.
 */
template <class T>
constexpr T GCD(T a, T b, T* s = nullptr, T* t = nullptr) {
    static_assert((std::is_integral_v<T> && std::is_signed_v<T>) || std::is_same_v<T, InfInt>,
                  "GCD requires signed integral type or InfInt");
    if (s != nullptr && t != nullptr) {  // extended EA
        *s = T(1);
        *t = T(0);
        T u = T(0);
        T v = T(1);
        while (b != T(0)) {
            const T q = a / b;
            T b1 = std::move(b);
            b = a - q * b1;
            a = std::move(b1);
            T u1 = std::move(u);
            u = *s - q * u1;
            *s = std::move(u1);
            T v1 = std::move(v);
            v = *t - q * v1;
            *t = std::move(v1);
        }
    } else {  // "normal" EA
        while (b != T(0)) {
            const T q = a / b;
            T b1 = std::move(b);
            b = a - q * b1;
            a = std::move(b1);
        }
    }
    return a;
}

/**
 * @brief Multiplicative inverse modulo a prime
 * @tparam p Prime modulus (must be prime for correct results)
 * @tparam T Signed integer type
 * @param a Element to invert
 * @return Modular inverse a^(-1) mod p
 *
 * Uses the extended Euclidean algorithm.
 */
template <uint16_t p, class T>
constexpr T modinv(T a) {
    static_assert(is_prime(p), "p is not a prime");
    T s, t;
    GCD<T>(std::move(a), T(p), &s, &t);  // don't actually need the gcd
    T result = s % T(p);
    return result < 0 ? result + T(p) : result;
}

/**
 * @brief Factorial a!
 * @tparam T Integer type
 * @param a Non-negative integer
 * @return `a! = a * (a-1) * ... * 2 * 1`
 */
template <class T>
T fac(T a) {
    T res = 1;
    while (a > 1) {
        res *= a;
        --a;
    }
    return res;
}

/**
 * @brief Binomial coefficient C(n,k)
 * @tparam T Integer type
 * @param n Total number of items
 * @param k Number of items to choose
 * @return `C(n,k) = n! / (k! * (n-k)!)`
 *
 * Uses the multiplicative formula and the symmetry `C(n,k) = C(n,n-k)`.
 */
template <class T>
T bin(const T& n, T k) {
    if (k > n) return 0;
    if (k == 0 || n == k) return 1;
    if (k > n - k) k = n - k;  // symmetry
    T res = 1;
    for (T i = 1; i <= k; ++i) res = res * (n - k + i) / i;
    return res;
}

/**
 * @brief Binomial coefficient specialization for @ref InfInt
 * @param n Total number of items
 * @param k Number of items to choose
 * @return `C(n,k)` as an @ref InfInt
 *
 * Builds numerator and denominator separately, then performs one division.
 */
template <>
inline InfInt bin(const InfInt& n, InfInt k) {
    if (k > n) return 0;
    if (k == 0 || n == k) return 1;
    if (n == 0) return 0;
    if (k > n - k) k = n - k;  // symmetry
    InfInt numerator = 1;
    InfInt denominator = 1;
    for (InfInt i = 1; i <= k; ++i) {
        numerator *= n + 1 - i;
        denominator *= i;
    }
    return numerator / denominator;
}

/**
 * @brief Base-2 logarithm of a positive @ref InfInt of arbitrary magnitude
 * @param v Positive integer
 * @return `log2(v)` as `long double`
 * @throws std::invalid_argument if v <= 0
 *
 * Evaluates the leading decimal digits and shifts by the remaining length, avoiding any
 * conversion of @p v to a builtin integer type (which would overflow beyond 2^64).
 */
inline long double log2(const InfInt& v) {
    if (v <= 0) throw std::invalid_argument("log2 requires a positive argument!");
    const std::string s = v.to_string();
    const size_t lead = std::min<size_t>(s.size(), 18);
    return std::log2(std::stold(s.substr(0, lead))) + static_cast<long double>(s.size() - lead) * std::log2(10.0L);
}

/**
 * @brief Conversion of an @ref InfInt of arbitrary magnitude to `long double`
 * @param v Integer
 * @return Nearest representable value (about 18 significant decimal digits are preserved)
 */
inline long double to_long_double(const InfInt& v) {
    const std::string s = v.to_string();
    const size_t sign = (s[0] == '-') ? 1 : 0;
    const size_t digits = s.size() - sign;
    const size_t lead = std::min<size_t>(digits, 18);
    const long double head = std::stold(s.substr(0, sign + lead));
    return head * std::pow(10.0L, static_cast<long double>(digits - lead));
}

/**
 * @brief Sierpinski triangle (Pascal's triangle modulo p)
 *
 * Provides binomial coefficients `C(n,k) mod p` without overflow or multi-precision arithmetic.
 */
class SierpinskiTriangle {
   public:
    /**
     * @brief Precompute `C(n,k) mod p` for all n <= n_max and k <= k_max
     * @throws std::invalid_argument if p is zero
     */
    SierpinskiTriangle(size_t n_max, size_t k_max, size_t p)
        : n_max(n_max), k_max(k_max), table((n_max + 1) * (k_max + 1)) {
        if (p == 0) throw std::invalid_argument("SierpinskiTriangle modulus must be positive!");
        for (size_t n = 0; n <= n_max; ++n) {
            table[n * (k_max + 1)] = 1 % p;
            for (size_t k = 1; k <= std::min(n, k_max); ++k)
                table[n * (k_max + 1) + k] =
                    (table[(n - 1) * (k_max + 1) + k - 1] + table[(n - 1) * (k_max + 1) + k]) % p;
        }
    }

    /// @brief `C(n,k) mod p`, zero whenever k > n (even beyond the precomputed range)
    /// @throws std::invalid_argument if k <= n and n > n_max or k > k_max
    size_t operator()(size_t n, size_t k) const {
        if (k > n) return 0;
        if (n > n_max || k > k_max)
            throw std::invalid_argument("SierpinskiTriangle argument(s) beyond precomputed range!");
        return table[n * (k_max + 1) + k];
    }

   private:
    size_t n_max;
    size_t k_max;
    std::vector<size_t> table;
};

/**
 * @brief Exponentiation by square-and-multiply
 * @tparam T Type supporting multiplication and, for negative exponents, division
 * @param b Base value
 * @param e Exponent
 * @return `b^e`
 *
 * For negative exponents, computes `(1/b)^|e|`.
 *
 * @note Returns `T(1)` for e = 0.
 * @throws std::overflow_error if T is an integral type and the result would overflow it
 * @throws std::invalid_argument for negative exponents when T is integral or has no division
 */
template <class T>
constexpr T sqm(T b, int e) {
    static_assert(std::is_integral_v<decltype(e)>, "exponent must be integral type");
    if (e == 0) return T(1);
    if (e < 0) {
        if constexpr (std::is_integral_v<T> || std::is_same_v<T, InfInt> || !requires(T x) { T(1) / x; }) {
            throw std::invalid_argument("sqm: negative exponent requires an invertible base type!");
        } else {
            b = T(1) / b;
            if (e == std::numeric_limits<int>::min())
                throw std::invalid_argument(
                    "Exponent e too large!");  // INT_MIN might be INT_MAX+1, potential problem in next line
            e = -e;
        }
    }
    // square and multiply
    T temp(1);
    unsigned int exp = static_cast<unsigned int>(e);
    if constexpr (std::is_integral_v<T>) {
        const auto mul = [](T x, T y) -> T {
            if constexpr (std::is_unsigned_v<T>) {
                if (x != T(0) && y > std::numeric_limits<T>::max() / x)
                    throw std::overflow_error("sqm: integer overflow");
            } else {
                constexpr T max = std::numeric_limits<T>::max();
                constexpr T min = std::numeric_limits<T>::min();
                if (x != T(0) && y != T(0)) {
                    if (x > T(0) ? (y > T(0) ? x > max / y : y < min / x) : (y > T(0) ? x < min / y : y < max / x))
                        throw std::overflow_error("sqm: integer overflow");
                }
            }
            return x * y;
        };
        while (exp > 0) {
            if (exp & 1) temp = mul(temp, b);
            exp >>= 1;
            if (exp > 0) b = mul(b, b);
        }
    } else {
        while (exp > 0) {
            if (exp & 1) temp *= b;
            exp >>= 1;
            if (exp > 0) b *= b;
        }
    }
    return temp;
}

/**
 * @brief Scalar multiplication by double-and-add
 * @tparam T Type supporting addition and unary minus
 * @param b Multiplicand
 * @param m Integer multiplier
 * @return `b * m`
 *
 * For negative multipliers, computes `(-b) * |m|`.
 *
 * @note Returns `T(0)` for m = 0.
 */
template <class T>
constexpr T daa(T b, int m) {
    static_assert(std::is_integral_v<decltype(m)>, "multiplicand must be integral type");
    if (m == 0) return T(0);
    if (m < 0) {
        b = -b;
        if (m == std::numeric_limits<int>::min())
            throw std::invalid_argument(
                "Multiplier m too large!");  // INT_MIN might be INT_MAX+1, potential problem in next line
        m = -m;
    }
    // double and add
    T temp(0);
    unsigned int um = static_cast<unsigned int>(m);
    while (um > 0) {
        if (um & 1) temp += b;
        um >>= 1;
        if (um > 0) b += b;
    }
    return temp;
}

namespace details {

/**
 * @brief Reservoir-sampling acceptance test for randomized tie-breaking
 * @param count Number of candidates tied for best so far, including the current one (≥ 1)
 * @return true with probability 1/count
 *
 * Reservoir sampling with a reservoir of size one, Algorithm R (Wikipedia, "Reservoir
 * sampling"): for the i-th stream item draw `randomInteger(1, i)` and keep it iff the draw is 1,
 * i.e. with probability 1/i. Passing the running tie count (incremented for the current
 * candidate) thus leaves every tied candidate equally likely to be the one finally kept. Used by
 * the Viterbi, BCJR, and standard-array tie-breaks.
 *
 * @code{.cpp}
 * if (metric < best) { best = metric; choice = cand; ties = 1; }
 * else if (metric == best && details::reservoir_accept(++ties)) choice = cand;
 * @endcode
 */
inline bool reservoir_accept(size_t count) { return std::uniform_int_distribution<size_t>(1, count)(gen()) == 1; }

/**
 * @brief Constexpr floor function
 * @param x Floating-point value
 * @return Floor of x (largest integer ≤ x)
 *
 * Alternative to `std::floor` for constant evaluation.
 *
 * @note Returns `double`, matching `std::floor`.
 */
constexpr double floor_constexpr(double x) {
    long int i = static_cast<long int>(x);
    return (x < 0 && x != i) ? i - 1 : i;
}

/**
 * @brief Cache entry specification
 * @tparam ID Entry identifier
 * @tparam T Stored value type
 *
 * Associates an entry ID with its value type for @ref Cache.
 *
 * @note IDs must be unique within one @ref Cache instance.
 */
template <auto ID, typename T>
struct CacheEntry {
    static constexpr auto id = ID;
    using type = T;
};

/**
 * @brief Heterogeneous cache indexed by entry ID
 *
 * @tparam ENTRIES Pack of @ref CacheEntry types
 *
 * Each ID gets its own `std::optional<T>` slot in a `std::tuple`, so different IDs may share
 * the same value type without ambiguity. Lookup is by compile-time ID; an unknown ID is a
 * compile-time error.
 *
 * @warning Not thread-safe for concurrent writes. Use external synchronisation if multiple
 * threads may call `set()`, `invalidate()`, or `operator()` simultaneously.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * using Entry1 = CacheEntry<0, std::vector<int>>;
 * using Entry2 = CacheEntry<1, double>;
 * using Entry3 = CacheEntry<5, std::string>;  // IDs need not be consecutive
 * Cache<Entry1, Entry2, Entry3> cache;
 *
 * cache.set<0>(std::vector<int>{1, 2, 3});
 * auto& vec = cache.get_or_compute<0>([] { return std::vector<int>{4, 5, 6}; });
 * if (cache.is_set<0>()) cache.invalidate<0>();
 * @endcode
 */
template <typename... ENTRIES>
class Cache {
   private:
    // ID -> position in the ENTRIES pack. Two constrained partial specialisations select
    // lazily, so the recursion only instantiates the not-yet-matched tail.
    template <auto ID, size_t I, typename...>
    struct index_finder;

    template <auto ID, size_t I, typename First, typename... Rest>
        requires(First::id == ID)
    struct index_finder<ID, I, First, Rest...> {
        static constexpr size_t value = I;
    };

    template <auto ID, size_t I, typename First, typename... Rest>
        requires(First::id != ID)
    struct index_finder<ID, I, First, Rest...> {
        static constexpr size_t value = index_finder<ID, I + 1, Rest...>::value;
    };

    template <auto ID>
    static constexpr size_t index_for = index_finder<ID, 0, ENTRIES...>::value;

    template <auto ID>
    using type_for = typename std::tuple_element_t<index_for<ID>, std::tuple<ENTRIES...>>::type;

    // One slot per entry, indexed by position. Independent slots avoid the duplicate-types
    // ambiguity that a `std::variant<monostate, T1, T2, …>` would have when two entries
    // happen to share the same value type.
    mutable std::tuple<std::optional<typename ENTRIES::type>...> slots{};

   public:
    Cache() = default;

    template <auto ID>
    bool is_set() const noexcept {
        return std::get<index_for<ID>>(slots).has_value();
    }

    template <auto ID, typename TYPE>
    void set(TYPE&& value) const {
        std::get<index_for<ID>>(slots) = static_cast<type_for<ID>>(std::forward<TYPE>(value));
    }

    template <auto ID>
    bool invalidate() const noexcept {
        auto& slot = std::get<index_for<ID>>(slots);
        const bool was_set = slot.has_value();
        slot.reset();
        return was_set;
    }

    bool invalidate() const noexcept {
        return std::apply(
            [](auto&... s) {
                const bool any = (s.has_value() || ...);
                (s.reset(), ...);
                return any;
            },
            slots);
    }

    template <auto ID>
    const type_for<ID>& operator()(auto&& calculate_func) const {
        auto& slot = std::get<index_for<ID>>(slots);
        if (!slot.has_value()) slot = calculate_func();
        return *slot;
    }

    template <auto ID>
    std::optional<type_for<ID>> get() const {
        return std::get<index_for<ID>>(slots);
    }

    template <auto ID>
    const type_for<ID>& get_or_compute(auto&& calculate_func) const {
        return operator()<ID>(std::forward<decltype(calculate_func)>(calculate_func));
    }
};

/**
 * @brief Thread-safe single-value cache
 * @tparam T Value type stored in the cache
 *
 * Stores one optional value together with a `std::once_flag`. Use `call_once()` to guard
 * lazy initialization when multiple threads may read the same object. Copying or moving a
 * cache copies or moves the stored value, if any, and creates a fresh once flag.
 *
 * @note Classes containing `OnceCache` members need no hand-written copy or move operations:
 * compiler-generated memberwise operations use the cache's copy and move operations, transferring
 * the cached value while giving each destination cache its own fresh `std::once_flag`.
 *
 * @warning Concurrent reads through `call_once()` are thread-safe. Assignment, `emplace()`, and
 * manual value changes are not synchronization points and must not race with other operations.
 *
 * @section Usage_Example
 *
 * @code{.cpp}
 * mutable OnceCache<size_t> weight;
 *
 * weight.call_once([this] {
 *     if (weight.has_value()) return;
 *     weight.emplace(calculate_weight());
 * });
 *
 * return weight.value();
 * @endcode
 */
template <class T>
class OnceCache {
   public:
    OnceCache() = default;

    OnceCache(const OnceCache& other) {
        if (other.has_value()) data.emplace(other.value());
    }

    OnceCache(OnceCache&& other) {
        if (other.has_value()) data.emplace(std::move(other.value()));
    }

    OnceCache& operator=(const OnceCache& other) {
        if (this != &other) {
            reset();
            if (other.has_value()) data.emplace(other.value());
        }
        return *this;
    }

    OnceCache& operator=(OnceCache&& other) {
        if (this != &other) {
            reset();
            if (other.has_value()) data.emplace(std::move(other.value()));
        }
        return *this;
    }

    OnceCache& operator=(const T& value) {
        data = value;
        return *this;
    }

    OnceCache& operator=(T&& value) {
        data = std::move(value);
        return *this;
    }

    template <class F>
    void call_once(F&& f) const {
        std::call_once(flag, std::forward<F>(f));
    }

    template <class... Args>
    T& emplace(Args&&... args) const {
        return data.emplace(std::forward<Args>(args)...);
    }

    bool has_value() const noexcept { return data.has_value(); }

    explicit operator bool() const noexcept { return has_value(); }

    T& value() { return data.value(); }

    const T& value() const { return data.value(); }

    T& operator*() { return *data; }

    const T& operator*() const { return *data; }

    T* operator->() { return &*data; }

    const T* operator->() const { return &*data; }

    void reset() const {
        data.reset();
        flag.~once_flag();
        new (&flag) std::once_flag();
    }

   private:
    mutable std::optional<T> data;
    mutable std::once_flag flag;
};

inline std::string basename(const char* path) {
    std::string s(path);

    const auto pos = s.find_last_of("/\\");
    if (pos != std::string::npos) s.erase(0, pos + 1);

    const auto dot = s.find_last_of('.');
    if (dot != std::string::npos && dot != 0) s.erase(dot);

    return s;
}

static const uint8_t colormap[64][3] = {
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

}  // namespace details

}  // namespace CECCO

#endif
