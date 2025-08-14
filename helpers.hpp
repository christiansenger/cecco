/**
 * @file helpers.hpp
 * @brief Utility functions and mathematical helpers
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
 * This header file provides essential utility functions and mathematical helpers. It supports:
 *
 * - **Thread-safe random number generation**: Configurable deterministic/hardware seeding with
 *   per-thread isolation using the singleton RNG class
 * - **Mathematical functions**: Extended Euclidean algorithm, modular inverse, factorial,
 *   binomial coefficients, and primality testing
 * - **Square-and-multiply** for exponentiation
 * - **Double-and-add** for multiplication
 * - **High-performance caching**: Template-based Cache class with compile-time type safety
 *   and O(1) access via std::variant and std::array
 * - **Utility functions**: Find maxima in vectors, constexpr floor function, divisibility testing
 */

#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "InfInt.hpp"

namespace CECCO {

/**
 * @brief Thread-safe random number generator with configurable seeding
 *
 * Singleton class providing thread-local std::mt19937 generators with centralized
 * seed management. Supports both deterministic seeding (for testing/reproducibility)
 * and hardware-based seeding (for cryptographic applications).
 *
 * @section Thread_Model
 * - Each thread gets its own std::mt19937 generator (thread_local storage)
 * - Atomic seed management ensures consistent seeding across all threads
 * - Lock-free design using std::atomic operations and thread-local storage
 *
 * @section Seeding_Behavior
 * - **Deterministic mode**: All threads use seed XOR thread_id for reproducibility
 * - **Hardware mode**: Each thread uses independent hardware entropy via std::random_device
 * - **Reseeding**: Threads automatically reseed when global seed changes
 *
 * @note This class cannot be instantiated directly (all members are static, no constructors are generated). Use the
 * static interface.
 */
class RNG {
   public:
    /**
     * @brief Get thread-local random number generator
     * @return Reference to thread-local std::mt19937 generator
     *
     * Returns a reference to this thread's std::mt19937 generator. The generator
     * is initialized on first access and automatically reseeded when the global
     * seed configuration changes.
     */
    static std::mt19937& get() {
        static thread_local std::mt19937 generator{get_initial_seed()};
        if (should_reseed()) generator.seed(get_current_seed());
        return generator;
    }

    /**
     * @brief Set deterministic seed for all threads
     * @param seed Base seed value (combined with thread ID)
     *
     * Configures all thread generators to use a deterministic seed based on
     * the provided value XOR each thread's ID. This ensures reproducible
     * results across runs while maintaining thread isolation.
     */
    static void seed(uint32_t seed) {
        use_deterministic_seed.store(true);
        deterministic_seed_value.store(seed);
        reseed_generation.fetch_add(1);  // Signal all threads to reseed
    }

    /**
     * @brief Enable hardware-based random seeding
     *
     * Configures all thread generators to use independent hardware entropy
     * via std::random_device. Each thread gets its own entropy source.
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
            auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
            return deterministic_seed_value.load() ^ static_cast<uint32_t>(tid);
        } else {
            static thread_local std::random_device rd;
            return rd();
        }
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
 * @brief Thread-safe convenience function for random number generation
 * @return Reference to thread-local random number generator
 *
 * Returns a reference to the current thread's std::mt19937 generator. Provides a simple interface while maintaining
 * full thread safety.
 *
 * @note Equivalent to RNG::get() but with a shorter name for convenience
 */
inline std::mt19937& gen() { return RNG::get(); }

/**
 * @brief Find all indices of maximum elements in a vector
 * @tparam T Element type (must support operator< for comparison)
 * @param v Input vector to search
 * @return Vector of indices where maximum elements occur (empty if input is empty)
 *
 * Returns a vector containing the indices of all elements that have the maximum value.
 * If multiple elements share the maximum value, all their indices are returned.
 *
 * @note Time complexity: O(n) with two passes over the data
 * @note Space complexity: O(k) where k is the number of maximum elements
 *
 * @code{.cpp}
 * std::vector<int> data = {1, 3, 2, 3, 1};
 * auto indices = find_maxima(data);  // Returns {1, 3} (indices of value 3)
 * @endcode
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
 * @brief Primality test using trial division
 * @tparam T Unsigned integer type
 * @param a Number to test for primality
 * @return true if a is prime, false otherwise
 *
 * Tests primality using optimized trial division. Only tests odd divisors
 * up to √a for efficiency. Returns false for a ≤ 1 and even numbers > 2.
 *
 * @code{.cpp}
 * bool p1 = is_prime(17);  // Returns true
 * bool p2 = is_prime(15);  // Returns false (3 * 5 == 15)
 * @endcode
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
 * @brief Extended Euclidean algorithm for computing GCD and Bézout coefficients
 * @tparam T Signed integral type
 * @param a First integer
 * @param b Second integer
 * @param s Pointer to store Bézout coefficient for a (optional)
 * @param t Pointer to store Bézout coefficient for b (optional)
 * @return Greatest common divisor of a and b
 *
 * Computes gcd(a,b) using the Euclidean algorithm. If s and t are provided,
 * also computes Bézout coefficients such that: a*s + b*t = gcd(a,b)
 *
 * @code{.cpp}
 * auto gcd = GCD(48, 18);        // Returns 6
 * int s, t;
 * auto gcd = GCD(48, 18, &s, &t); // Returns 6, sets s=1, t=-2 (48*1 + 18*(-2) = 6)
 * @endcode
 */
template <class T>
constexpr T GCD(T a, T b, T* s = nullptr, T* t = nullptr) noexcept {
    static_assert((std::is_integral<T>::value && std::is_signed<T>::value) || std::is_same_v<T, InfInt>, "GCD requires signed integral type or InfInt");
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
 * @brief Modular multiplicative inverse using extended Euclidean algorithm
 * @tparam p Prime modulus (must be prime for correct results)
 * @tparam T Signed integer type supporting arithmetic operations
 * @param a Element to invert
 * @return Modular inverse a^(-1) mod p
 *
 * Computes the multiplicative inverse of a modulo p by leveraging the extended
 * Euclidean algorithm. Returns x such that (a * x) ≡ 1 (mod p).
 *
 * @code{.cpp}
 * auto inv = modinv<7>(3);   // Returns 5 (since 3*5 ≡ 1 mod 7)
 * @endcode
 */
template <uint16_t p, class T>
constexpr T modinv(T a) noexcept {
    static_assert(is_prime(p), "p is not a prime");
    T s, t;
    GCD<T>(std::move(a), T(p), &s, &t);  // don't actually need the gcd
    T result = s % T(p);
    return result < 0 ? result + T(p) : result;
}

/**
 * @brief Factorial function a!
 * @tparam T Integer type supporting arithmetic operations
 * @param n Non-negative integer
 * @return Factorial a! = a * (a-1) * ... * 2 * 1
 *
 * Computes the factorial of a using iterative multiplication.
 * Returns 1 for a = 0 and a = 1 (by mathematical convention).
 *
 * @code{.cpp}
 * auto fact5 = fac<int>(5);  // Returns 120
 * auto fact0 = fac<int>(0);  // Returns 1
 * auto big_fact = fac<InfInt>(100);  // Arbitrary precision factorial
 * @endcode
 */
template <class T>
T fac(T a) noexcept {
    T res = 1;
    while (a > 1) {
        res *= a;
        --a;
    }
    return res;
}

/**
 * @brief Binomial coefficient C(n,k) = n choose k
 * @tparam T Integer type supporting arithmetic operations
 * @param n Total number of items
 * @param k Number of items to choose
 * @return Binomial coefficient C(n,k) = n! / (k! * (n-k)!)
 *
 * Computes binomial coefficients using the multiplicative formula to avoid
 * computing large factorials. Uses symmetry C(n, k) = C(n, n-k) for efficiency.
 *
 * @note InfInt specialization available for arbitrary precision
 *
 * @code{.cpp}
 * auto coeff = bin(10, 3);              // Returns 120
 * auto large_coeff = bin<InfInt>(100, 50); // Arbitrary precision
 * @endcode
 */
template <class T>
T bin(const T& n, T k) noexcept {
    if (k > n) return 0;
    if (k == 0 || n == k) return 1;
    if (k > n - k) k = n - k;  // symmetry
    T res = 1;
    for (T i = 1; i <= k; ++i) res = res * (n - k + i) / i;
    return res;
}

/**
 * @brief Binomial coefficient specialization for arbitrary precision integers
 * @param n Total number of items
 * @param k Number of items to choose
 * @return Binomial coefficient C(n,k) using arbitrary precision arithmetic
 *
 * Specialization of bin() for InfInt that can handle arbitrarily large values without overflow. Uses
 * numerator/denominator approach for improved performance (does not perform expensive divisions in each iteration).
 *
 * @note InfInt operations are significantly slower than native integer types
 * @note Suitable for combinatorics problems requiring exact large integer results
 */
template <>
InfInt bin(const InfInt& n, InfInt k) noexcept {
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
 * @brief Fast exponentiation using square-and-multiply algorithm
 * @tparam T Numeric type supporting multiplication
 * @param b Base value
 * @param e Exponent (can be negative for types supporting division)
 * @return b^e computed efficiently
 *
 * Computes b^e using the binary exponentiation algorithm with O(log e) complexity.
 * For negative exponents, computes (1/b)^|e| if T supports division.
 *
 * @note For negative exponents, T must support division (1/b)
 * @note Returns T(1) for e = 0 regardless of base value
 *
 * @code{.cpp}
 * auto result = sqm(2, 10);            // Returns 1024 (2^10)
 * auto big_pow = sqm<InfInt>(3, 100);  // Large exponentiation
 * auto inv_pow = sqm(2.0, -3);         // Returns 0.125 (2^-3)
 * @endcode
 */
template <class T>
constexpr T sqm(T b, int e) noexcept {
    static_assert(std::is_integral<decltype(e)>::value, "exponent must be integral type");
    if (e == 0) {
        return T(1);
    }
    if (e < 0) {
        b = T(1) / b;
        e = -e;
    }
    // square and multiply
    T temp(1);
    unsigned int exp = static_cast<unsigned int>(e);
    while (exp > 0) {
        if (exp & 1) temp *= b;
        b *= b;
        exp >>= 1;
    }
    return temp;
}

/**
 * @brief Fast scalar multiplication using double-and-add algorithm
 * @tparam T Numeric type supporting addition
 * @param b Base value (multiplicand)
 * @param m Multiplier (can be negative)
 * @return b * m computed efficiently using repeated doubling
 *
 * Computes b * m using the binary multiplication algorithm with O(log |m|) complexity.
 * For negative multipliers, computes (-b) * |m|.
 *
 * @note Returns T(0) for m = 0 regardless of base value
 *
 * @code{.cpp}
 * auto result = daa(7, 3);                  // Returns 21 (7*3)
 * auto large_mult = daa<InfInt>(123, 456);  // Large multiplication
 * auto neg_mult = daa(5, -4);               // Returns -20 (5*(-4))
 * @endcode
 */
template <class T>
constexpr T daa(T b, int m) noexcept {
    static_assert(std::is_integral<decltype(m)>::value, "multiplicand must be integral type");
    if (m == 0) {
        return T(0);
    }
    if (m < 0) {
        b = -b;
        m = -m;
    }
    // double and add
    T temp(0);
    unsigned int um = static_cast<unsigned int>(m);
    while (um > 0) {
        if (um & 1) temp += b;
        b += b;
        um >>= 1;
    }
    return temp;
}

namespace details {

/**
 * @brief Constexpr-compatible floor function
 * @param x Floating-point value
 * @return Floor of x (largest integer ≤ x)
 *
 * Constexpr-compatible alternative to std::floor for compile-time evaluation. Handles both positive and negative
 * numbers correctly.
 *
 * @note Returns double (not integer) to match std::floor behavior
 *
 * @code{.cpp}
 * constexpr auto f1 = floor_constexpr(3.7);   // Returns 3.0
 * constexpr auto f2 = floor_constexpr(-2.3);  // Returns -3.0
 * @endcode
 */
constexpr double floor_constexpr(double x) {
    long int i = static_cast<long int>(x);
    return (x < 0 && x != i) ? i - 1 : i;
}

/**
 * @brief Cache entry type specification for compile-time type-safe caching
 * @tparam ID Unique compile-time identifier for this cache entry
 * @tparam T Value type to be cached for this ID
 *
 * Template structure used to associate compile-time IDs with their corresponding
 * types in the Cache template. Each entry maps an ID to a specific type.
 *
 * @note Used in conjunction with Cache<ENTRIES...> for type-safe heterogeneous caching
 * @note ID must be unique within a given Cache instance
 *
 * @code{.cpp}
 * using VectorEntry = CacheEntry<0, std::vector<int>>;
 * using DoubleEntry = CacheEntry<1, double>;
 * Cache<VectorEntry, DoubleEntry> cache;
 * @endcode
 */
template <auto ID, typename T>
struct CacheEntry {
    static constexpr auto id = ID;
    using type = T;
};

/**
 * @brief High-performance heterogeneous cache with compile-time type safety
 * @tparam ENTRIES Pack of CacheEntry types specifying ID-to-type mappings
 *
 * Template-based cache providing O(1) access to heterogeneous values using
 * compile-time IDs. Uses std::variant and std::array for efficient storage
 * with full type safety verified at compile time.
 *
 * @section Design_Features
 * - **Compile-time type safety**: Each ID maps to exactly one type
 * - **O(1) access**: Direct array indexing based on compile-time IDs
 * - **Heterogeneous storage**: Different types can be cached together
 * - **Memory efficient**: Fixed-size array based on maximum ID
 * - **Lazy evaluation**: Values computed only when first requested
 *
 * @warning **Not thread-safe** for concurrent writes. Use external synchronization
 * if multiple threads may call set(), invalidate(), or operator() simultaneously.
 *
 * @code{.cpp}
 * using Entry1 = CacheEntry<0, std::vector<int>>;
 * using Entry2 = CacheEntry<1, double>;
 * using Entry3 = CacheEntry<5, std::string>;  // IDs need not be consecutive
 * Cache<Entry1, Entry2, Entry3> cache;
 *
 * // Set values
 * cache.set<0>(std::vector<int>{1, 2, 3});
 * cache.set<1>(3.14159);
 *
 * // Get or compute values
 * auto& vec = cache.get_or_compute<0>([]() { return std::vector<int>{4, 5, 6}; });
 * auto& pi = cache.get_or_compute<1>([]() { return compute_pi(); });
 *
 * // Check and invalidate
 * if (cache.is_set<0>()) cache.invalidate<0>();
 * @endcode
 */
template <typename... ENTRIES>
class Cache {
   private:
    // Calculate maximum ID at compile time
    static constexpr auto max_id = std::max({ENTRIES::id...});

    // Create variant type from all entry types
    using VariantType = std::variant<std::monostate, typename ENTRIES::type...>;

    // Fixed-size array for O(1) access
    mutable std::array<VariantType, max_id + 1> cache_data{};

    // Compile-time ID -> Type lookup using simple recursion
    template <auto ID, typename First, typename... Rest>
    struct type_finder_impl {
        using type =
            std::conditional_t<First::id == ID, typename First::type, typename type_finder_impl<ID, Rest...>::type>;
    };

    template <auto ID, typename Last>
    struct type_finder_impl<ID, Last> {
        static_assert(Last::id == ID,
                      "Cache ID not found in CacheEntry list - check that all cache IDs are properly defined");
        using type = typename Last::type;
    };

    template <auto ID>
    using type_for_id_t = typename type_finder_impl<ID, ENTRIES...>::type;

   public:
    // Default constructor - initializes all cache slots to empty
    Cache() { cache_data.fill(std::monostate{}); }

    // Check if specific ID is cached
    template <auto ID>
    bool is_set() const {
        static_assert(ID <= max_id, "Cache ID out of bounds");
        return !std::holds_alternative<std::monostate>(cache_data[ID]);
    }

    // Set value for specific ID with automatic type conversion
    template <auto ID, typename TYPE>
    auto set(TYPE&& value) const {
        static_assert(ID <= max_id, "Cache ID out of bounds");
        using ExpectedType = type_for_id_t<ID>;

        auto old_it = cache_data.begin() + ID;
        cache_data[ID] = static_cast<ExpectedType>(std::forward<TYPE>(value));
        return std::make_pair(old_it, true);
    }

    // Invalidate specific ID
    template <auto ID>
    bool invalidate() const noexcept {
        static_assert(ID <= max_id, "Cache ID out of bounds");
        bool was_set = is_set<ID>();
        cache_data[ID] = std::monostate{};
        return was_set;
    }

    // Invalidate all entries
    bool invalidate() const noexcept {
        bool had_any = std::any_of(cache_data.begin(), cache_data.end(),
                                   [](const auto& entry) { return !std::holds_alternative<std::monostate>(entry); });
        cache_data.fill(std::monostate{});
        return had_any;
    }

    // Get or compute with lambda
    template <auto ID>
    const auto& operator()(auto&& calculate_func) const {
        static_assert(ID <= max_id, "Cache ID out of bounds");
        using ReturnType = type_for_id_t<ID>;

        auto& cached = cache_data[ID];
        if (std::holds_alternative<std::monostate>(cached)) {
            cached = calculate_func();
        }
        return std::get<ReturnType>(cached);
    }

    // Alternative syntax for cleaner usage
    template <auto ID>
    const auto& get_or_compute(auto&& calculate_func) const {
        return operator()<ID>(std::forward<decltype(calculate_func)>(calculate_func));
    }
};

}  // namespace details

}  // namespace CECCO

#endif