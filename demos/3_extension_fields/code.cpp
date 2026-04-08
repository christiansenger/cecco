#include <iostream>

#include "cecco.hpp"
using namespace CECCO;

int main(void) {
    using F2 = Fp<2>;
    using F8 = Ext<F2, {1, 0, 1, 1}>;
    std::cout << "Show textual info about F8:" << std::endl;
    std::cout << F8::get_info() << std::endl << std::endl;

    using F64_a = Ext<F8, {6, 2, 1}>;
    std::cout << "Show textual info about F64_a:" << std::endl;
    std::cout << F64_a::get_info() << std::endl << std::endl;

    F8::show_tables();
    std::cout << std::endl;

    using F4 = Ext<F2, {1, 1, 1}, LutMode::CompileTime>;

    using F64_b = Ext<F4, {3, 1, 2, 1}, LutMode::CompileTime>;
    std::cout << "Show textual info about F64_b:" << std::endl;
    std::cout << F64_b::get_info() << std::endl;

    auto phi = Isomorphism<F64_a, F64_b>();
    auto a = F64_a().randomize();  // random element of F64_a, type: F64_a
    std::cout << "Random element in F64_a: " << a << std::endl;
    auto b = phi(a);  // representation of a in F64_b, type: F64_b
    std::cout << "Same element in F64_b: " << b << std::endl;

    auto c = phi.inverse()(b);  // type: F64_a
    // auto c = Isomorphism<F64_b, F64_a>()(b);

    std::cout << "... back in F64_a: " << c << std::endl;
    assert(a == c);
    std::cout << std::endl;

    using F64_c = Ext<F2, {1, 0, 0, 1, 0, 0, 1}>;
    std::cout << "Show textual info about F64_c:" << std::endl;
    std::cout << F64_c::get_info() << std::endl << std::endl;

    using F64 = Iso<F64_c, F64_b, F64_a>;
    std::cout << "Show textual info about F64:" << std::endl;
    std::cout << F64::get_info() << std::endl;

    static_assert(SubfieldOf<F64, F2>);
    static_assert(SubfieldOf<F64, F4>);
    static_assert(SubfieldOf<F64, F8>);
    static_assert(SubfieldOf<F64, F64_a>);
    static_assert(SubfieldOf<F64, F64_b>);
    static_assert(SubfieldOf<F64, F64_c>);
    static_assert(SubfieldOf<F64, F64>);
    // other superfield relations
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
    // superfield non-relations
    static_assert(!SubfieldOf<F8, F4>);
    static_assert(!SubfieldOf<F64_a, F4>);
    static_assert(!SubfieldOf<F64_b, F8>);

    auto d = F2().randomize();
    std::cout << "d: " << d << std::endl;
    F8 e(d);
    std::cout << "e: " << e << std::endl;
    F64 f(e);
    std::cout << "f: " << f << std::endl;
    F64 g(d);
    std::cout << "g: " << g << std::endl;

    auto h = F8().randomize();
    std::cout << "h: " << h << std::endl;
    F64 i(h);
    std::cout << "i: " << i << std::endl;
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

    std::cout << "j from F64 (see above) as vector over F2: " << j.as_vector<F2>() << std::endl;
    std::cout << "j from F64 (see above) as vector over F4: " << j.as_vector<F4>() << std::endl;
    std::cout << "j from F64 (see above) as vector over F8: " << j.as_vector<F8>() << std::endl;

    assert(j == F64(j.as_vector<F2>()));
    assert(j == F64(j.as_vector<F4>()));
    assert(j == F64(j.as_vector<F8>()));

    assert(i + j == F64(i.as_vector<F2>() + j.as_vector<F2>()));
    assert(i + j == F64(i.as_vector<F4>() + j.as_vector<F4>()));
    assert(i + j == F64(i.as_vector<F8>() + j.as_vector<F8>()));

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

    using F3 = Fp<3>;
    using F9 = Ext<F3, {2, 2, 1}, LutMode::CompileTime>;
    using F27 = Ext<F3, {1, 2, 0, 1}, LutMode::CompileTime>;
    using F81_a = Ext<F3, {2, 1, 0, 0, 1}>;
    using F81_b = Ext<F9, {6, 0, 1}>;
    using F81 = Iso<F81_a, F81_b>;
    using F243 = Ext<F3, {2, 0, 1, 2, 1, 1}>;
    using F729_a = Ext<F3, {2, 1, 2, 0, 1, 0, 1}>;
    using F729_b = Ext<F9, {7, 0, 6, 1}>;
    using F729_c = Ext<F27, {14, 20, 1}>;
    using F729 = Iso<F729_a, F729_b, F729_c>;

    auto s = F9().randomize();
    std::cout << "s: " << s << std::endl;
    F729 t(s);
    std::cout << "t: " << t << std::endl;
    std::cout << "Checking if s == F9(t): s=" << s << ", t=" << t << std::endl;
    assert(s == F9(t));

    auto u = F729().randomize();
    std::cout << u << " (from F729) as vector with components from subfield F27: " << u.as_vector<F27>() << std::endl;
    auto v = F729().randomize();
    assert(u + v == F729(u.as_vector<F27>() + v.as_vector<F27>()));

    return 0;
}