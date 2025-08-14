#include <iostream>

// Use ECC library, compile with -std=c++20 -I../
// Adapt -Ipath_to_hpp_files if necessary, append -O3 for performance
#include "include/ecc.hpp"
using namespace ECC;

int main(void) {
    // Start with any finite field - main example here is characteristic 2 but library works for arbitrary
    // characteristic!
    using F2 = Fp<2>;
    // Find a monic irreducible polynomial of degree m with coefficients from F2 (as shown in
    // find_irreducible_polynomials.hpp), take its coefficient vector of length m + 1, e.g., (1, 0, 1, 1). Then define
    // degree m extension of F2:
    using F8 = Ext<F2, {1, 0, 1, 1}>;
    // Then use F8 just "normally" (arithmetics, orders, (un)erase, etc.) as in prime_fields.cpp
    std::cout << "Show textual info about F8:" << std::endl;
    std::cout << F8::get_info() << std::endl << std::endl;

    // Process can be repeated: Find a monic irreducible polynomial of degree m' with coefficients from F8, take its
    // coefficient vector of length m + 1, e.g., (6, 2, 1). Then define degree m' extension of F8:
    using F64_a = Ext<F8, {6, 2, 1}>;
    std::cout << "Show textual info about F64_a:" << std::endl;
    std::cout << F64_a::get_info() << std::endl << std::endl;

    // Field operations are based on pre-calculated LUTs. They can be displayed for debugging purposes:
    F8::show_tables();
    std::cout << std::endl;

    // By default, LUTs are calculated at runtime
    // using F4 = Ext<F2, {1, 1, 1}>
    // is identical to
    // using F4 = Ext<F2, {1, 1, 1}, LutMode::RunTime>;
    // Optionally, LUTs can also be calculated at compile time and embedded in the executable:
    using F4 = Ext<F2, {1, 1, 1}, LutMode::CompileTime>;
    // Compile time LUT calculation leads to slightly improved runtime performance and results in zero program startup
    // time. Downside: Compile-time calculation can exceed recursion limits of the compiler and can use huge amounts of
    // memory. Compiler limits can be tweaked with -fconstexpr-depth=4294967295 -fconstexpr-steps=4294967295 (clang++)
    // or -fconstexpr-ops-limit=4294967295 (g++).
    // Recommendation: Use compile-time LUTs for small fields (up to ~150 elements).

    // ** Compile-time LUT calculation is only possible if the base field also uses compile-time LUTs! **
    using F64_b = Ext<F4, {3, 1, 2, 1}, LutMode::CompileTime>;  // possible since F4 is also CompileTime
    std::cout << "Show textual info about F64_b:" << std::endl;
    std::cout << F64_b::get_info() << std::endl;

    // F64_a and F64_b are isomorphic, we can transition between the two:
    Isomorphism<F64_a, F64_b> phi{};
    auto a = F64_a().randomize();  // random element of F64_a, type: F64_a
    std::cout << "Element in F64_a: " << a << std::endl;
    auto b = phi(a);  // representation of a in F64_b, type: F64_b
    std::cout << "Element in F64_b: " << b << std::endl;

    // Following two lines are equivalent: Isomorphism<F64_b, F64_a> is automatically instantiated as the inverse of
    // Isomorphism<F64_a, F64_b>!
    auto c = phi.inverse()(b);  // type: F64_a
    // auto c = Isomorphism<F64_b, F64_a>()(b);

    std::cout << "... back in F64_a: " << c << std::endl;
    assert(a == c);
    std::cout << std::endl;

    // Yet another isomorphic field with 64 elements
    using F64_c = Ext<F2, {1, 0, 0, 1, 0, 0, 1}, LutMode::CompileTime>;
    std::cout << "Show textual info about F64_c:" << std::endl;
    std::cout << F64_c::get_info() << std::endl << std::endl;

    // At this point: Three "field towers" (all with characteristic 2)
    // F2 -> F8 -> F64_a
    // F2 -> F4 -> F64_b
    // F2 -> F64_c
    // Basically a tree with prime field F2 as root

    // Can merge vertices of isomorphic fields in order to get a true finite field lattice with intersections:
    using F64 = Iso<F64_c, F64_b, F64_a>;  // once we have F64: no reason to use F64_a,... any more
    std::cout << "Show textual info about F64:" << std::endl;
    std::cout << F64::get_info() << std::endl;
    // F64 is now a superfield of F2, F4, F8, F64_a, F64_b, F64_c, and trivially by itself
    static_assert(SubfieldOf<F64, F2>);
    static_assert(SubfieldOf<F64, F4>);
    static_assert(SubfieldOf<F64, F8>);
    static_assert(SubfieldOf<F64, F64_a>);
    static_assert(SubfieldOf<F64, F64_b>);
    static_assert(SubfieldOf<F64, F64_c>);
    static_assert(SubfieldOf<F64, F64>);
    // Other Superfield relations
    static_assert(SubfieldOf<F64_a, F64_a>);
    static_assert(SubfieldOf<F64_a, F2>);
    static_assert(SubfieldOf<F64_a, F8>);
    static_assert(SubfieldOf<F64_b, F64_b>);
    static_assert(SubfieldOf<F64_b, F2>);
    static_assert(SubfieldOf<F64_b, F4>);
    static_assert(SubfieldOf<F8, F8>);
    static_assert(SubfieldOf<F8, F2>);
    static_assert(SubfieldOf<F4, F4>);
    static_assert(SubfieldOf<F4, F2>);
    // Superfield non-relations
    static_assert(!SubfieldOf<F8, F4>);
    static_assert(!SubfieldOf<F64_a, F4>);
    static_assert(!SubfieldOf<F64_b, F8>);

    // Sub-/Superfield casts: Every subfield element can be safely "up-cast" to a superfield element
    // 0 is always mapped to 0 and 1 to 1 of superfield
    auto d = F2().randomize();
    std::cout << "d: " << d << std::endl;
    F8 e(d);
    std::cout << "e: " << e << std::endl;
    F64 f(e);
    std::cout << "f: " << f << std::endl;
    F64 g(d);
    std::cout << "g: " << g << std::endl;

    // More interesting if subfield has more elements
    auto h = F8().randomize();
    std::cout << "h: " << h << std::endl;
    F64 i(h);
    std::cout << "i: " << i << std::endl;
    // In general: h will "look" different from i but still the following always succeeds:
    std::cout << "Checking if h == F8(i): h=" << h << ", i=" << i << std::endl;
    // F8(i) is a "down-cast" from F64 to its subfield F8
    assert(h == F8(i));

    // Downcasts can fail! First, check the labels of all elements of F8 when up-cast to F64:
    std::cout << "Elements of F8 after casting them to superfield F64: " << std::endl;
    for (size_t i = 0; i < F8::get_q(); ++i) std::cout << F64(F8(i)) << " ";
    std::cout << std::endl;
    // Downcasting any of them to F8 certainly works:
    std::cout << "6 in F64 coincides with " << F8(F64(F8(3))) << " in F8" << std::endl;
    // However, downcasting some j from F64 that is not an embedded element of F8 will throw
    F64 j(43);
    std::cout << "j: " << j << std::endl;
    try {
        std::cout << F8(j) << std::endl;
    } catch (std::invalid_argument& e) {
        std::cout << "down-casting " << j << " from F64 to F8 is not possible: " << e.what() << std::endl;
    }

    // We can construct everything that is mathematically possible but anything that is not constructed does not exist,
    // for example we could have constructed F16 in two different ways but we didn't --- so F16 cannot be used in this
    // demo. We can do "wild" cross-casts (automatically through largest common subfield) within our constructed
    // lattice:
    auto k = F2().randomize();
    F4 l(k);
    F64 m(l);
    F8 n(m);
    F2 o(n);
    assert(k == o);

    auto p = F4().randomize();
    F64 q(p);
    try {
        F8 r(q);
        std::cout << "Up-casting " << p << " from F4 to F64 and then down-casting it to F8 gives " << r << std::endl;
    } catch (std::invalid_argument& e) {
        std::cout << "down-casting " << q << " from F64 to F8 is not possible: " << e.what() << std::endl;
    }

    // Any Ext or Iso element can be expanded into a vector over any subfield:
    std::cout << "j from F64 (see above) as vector over F2: " << j.as_vector<F2>() << std::endl;
    std::cout << "j from F64 (see above) as vector over F4: " << j.as_vector<F4>() << std::endl;
    std::cout << "j from F64 (see above) as vector over F8: " << j.as_vector<F8>() << std::endl;

    // Original element can always be recovered from such vectors
    assert(j == F64(j.as_vector<F2>()));
    assert(j == F64(j.as_vector<F4>()));
    assert(j == F64(j.as_vector<F8>()));

    // Exploding into subfield vectors is compatible with addition
    assert(i + j == F64(i.as_vector<F2>() + j.as_vector<F2>()));
    assert(i + j == F64(i.as_vector<F4>() + j.as_vector<F4>()));
    assert(i + j == F64(i.as_vector<F8>() + j.as_vector<F8>()));

    // Isomorphisms between same-size fields are compatible with addition and multiplication
    {
        auto x = F64_a().randomize();
        auto y = F64_a().randomize();
        assert((Isomorphism<F64_a, F64_a>()(x + y)) ==
               (Isomorphism<F64_a, F64_a>()(x) + Isomorphism<F64_a, F64_a>()(y)));
        assert((Isomorphism<F64_a, F64_a>()(x * y)) ==
               (Isomorphism<F64_a, F64_a>()(x) * Isomorphism<F64_a, F64_a>()(y)));
        assert((Isomorphism<F64_a, F64_b>()(x + y)) ==
               (Isomorphism<F64_a, F64_b>()(x) + Isomorphism<F64_a, F64_b>()(y)));
        assert((Isomorphism<F64_a, F64_b>()(x * y)) ==
               (Isomorphism<F64_a, F64_b>()(x) * Isomorphism<F64_a, F64_b>()(y)));
        assert((Isomorphism<F64_a, F64_c>()(x + y)) ==
               (Isomorphism<F64_a, F64_c>()(x) + Isomorphism<F64_a, F64_c>()(y)));
        assert((Isomorphism<F64_a, F64_c>()(x * y)) ==
               (Isomorphism<F64_a, F64_c>()(x) * Isomorphism<F64_a, F64_c>()(y)));
        assert((Isomorphism<F64_a, F64>()(x + y)) == (Isomorphism<F64_a, F64>()(x) + Isomorphism<F64_a, F64>()(y)));
        assert((Isomorphism<F64_a, F64>()(x * y)) == (Isomorphism<F64_a, F64>()(x) * Isomorphism<F64_a, F64>()(y)));
    }
    {
        auto x = F64_b().randomize();
        auto y = F64_b().randomize();
        assert((Isomorphism<F64_b, F64_a>()(x + y)) ==
               (Isomorphism<F64_b, F64_a>()(x) + Isomorphism<F64_b, F64_a>()(y)));
        assert((Isomorphism<F64_b, F64_a>()(x * y)) ==
               (Isomorphism<F64_b, F64_a>()(x) * Isomorphism<F64_b, F64_a>()(y)));
        assert((Isomorphism<F64_b, F64_b>()(x + y)) ==
               (Isomorphism<F64_b, F64_b>()(x) + Isomorphism<F64_b, F64_b>()(y)));
        assert((Isomorphism<F64_b, F64_b>()(x * y)) ==
               (Isomorphism<F64_b, F64_b>()(x) * Isomorphism<F64_b, F64_b>()(y)));
        assert((Isomorphism<F64_b, F64_c>()(x + y)) ==
               (Isomorphism<F64_b, F64_c>()(x) + Isomorphism<F64_b, F64_c>()(y)));
        assert((Isomorphism<F64_b, F64_c>()(x * y)) ==
               (Isomorphism<F64_b, F64_c>()(x) * Isomorphism<F64_b, F64_c>()(y)));
        assert((Isomorphism<F64_b, F64>()(x + y)) == (Isomorphism<F64_b, F64>()(x) + Isomorphism<F64_b, F64>()(y)));
        assert((Isomorphism<F64_b, F64>()(x * y)) == (Isomorphism<F64_b, F64>()(x) * Isomorphism<F64_b, F64>()(y)));
    }
    {
        auto x = F64_c().randomize();
        auto y = F64_c().randomize();
        assert((Isomorphism<F64_c, F64_a>()(x + y)) ==
               (Isomorphism<F64_c, F64_a>()(x) + Isomorphism<F64_c, F64_a>()(y)));
        assert((Isomorphism<F64_c, F64_a>()(x * y)) ==
               (Isomorphism<F64_c, F64_a>()(x) * Isomorphism<F64_c, F64_a>()(y)));
        assert((Isomorphism<F64_c, F64_b>()(x + y)) ==
               (Isomorphism<F64_c, F64_b>()(x) + Isomorphism<F64_c, F64_b>()(y)));
        assert((Isomorphism<F64_c, F64_b>()(x * y)) ==
               (Isomorphism<F64_c, F64_b>()(x) * Isomorphism<F64_c, F64_b>()(y)));
        assert((Isomorphism<F64_c, F64_c>()(x + y)) ==
               (Isomorphism<F64_c, F64_c>()(x) + Isomorphism<F64_c, F64_c>()(y)));
        assert((Isomorphism<F64_c, F64_c>()(x * y)) ==
               (Isomorphism<F64_c, F64_c>()(x) * Isomorphism<F64_c, F64_c>()(y)));
        assert((Isomorphism<F64_c, F64>()(x + y)) == (Isomorphism<F64_c, F64>()(x) + Isomorphism<F64_c, F64>()(y)));
        assert((Isomorphism<F64_c, F64>()(x * y)) == (Isomorphism<F64_c, F64>()(x) * Isomorphism<F64_c, F64>()(y)));
    }
    {
        auto x = F64().randomize();
        auto y = F64().randomize();
        assert((Isomorphism<F64, F64_a>()(x + y)) == (Isomorphism<F64, F64_a>()(x) + Isomorphism<F64, F64_a>()(y)));
        assert((Isomorphism<F64, F64_a>()(x * y)) == (Isomorphism<F64, F64_a>()(x) * Isomorphism<F64, F64_a>()(y)));
        assert((Isomorphism<F64, F64_b>()(x + y)) == (Isomorphism<F64, F64_b>()(x) + Isomorphism<F64, F64_b>()(y)));
        assert((Isomorphism<F64, F64_b>()(x * y)) == (Isomorphism<F64, F64_b>()(x) * Isomorphism<F64, F64_b>()(y)));
        assert((Isomorphism<F64, F64_c>()(x + y)) == (Isomorphism<F64, F64_c>()(x) + Isomorphism<F64, F64_c>()(y)));
        assert((Isomorphism<F64, F64_c>()(x * y)) == (Isomorphism<F64, F64_c>()(x) * Isomorphism<F64, F64_c>()(y)));
        assert((Isomorphism<F64, F64>()(x + y)) == (Isomorphism<F64, F64>()(x) + Isomorphism<F64, F64>()(y)));
        assert((Isomorphism<F64, F64>()(x * y)) == (Isomorphism<F64, F64>()(x) * Isomorphism<F64, F64>()(y)));
    }

    // We're not restricted to characteristic 2
    using F3 = Fp<3>;
    using F9 = Ext<F3, {2, 2, 1}, LutMode::CompileTime>;
    using F27 = Ext<F3, {1, 2, 0, 1}, LutMode::CompileTime>;
    using F81_a = Ext<F3, {2, 1, 0, 0, 1}, LutMode::CompileTime>;
    using F81_b = Ext<F9, {6, 0, 1}, LutMode::CompileTime>;
    using F81 = Iso<F81_a, F81_b>;
    using F243 = Ext<F3, {2, 0, 1, 2, 1, 1}>;
    using F729_a = Ext<F3, {2, 1, 2, 0, 1, 0, 1}>;
    using F729_b = Ext<F9, {7, 0, 6, 1}>;
    using F729_c = Ext<F27, {14, 20, 1}>;
    using F729 = Iso<F729_a, F729_b, F729_c>;

    auto s = F9().randomize();
    std::cout << "k: " << s << std::endl;
    F729 t(s);
    std::cout << "t: " << t << std::endl;
    // In general: t will "look" different from s but still the following always succeeds:
    std::cout << "Checking if s == F8(t): s=" << s << ", t=" << t << std::endl;
    // F9(k) is a "down-cast" from F729 to its subfield F9
    assert(s == F9(t));

    auto u = F729().randomize();
    std::cout << u << " (from F729) as vector with components from subfield F27: " << u.as_vector<F27>() << std::endl;
    auto v = F729().randomize();
    assert(u + v == F729(u.as_vector<F27>() + v.as_vector<F27>()));

    /*
     * See https://christiansenger.github.io/ecc/classECC_1_1Ext.html and
     * https://christiansenger.github.io/ecc/classECC_1_1Iso.html for documentation
     */

    return 0;
}