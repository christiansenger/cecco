/**
 * @file trellises.hpp
 * @brief Trellis representation for code decoding
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.1.1
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
 * Trellis storage and workspaces used by Viterbi and BCJR decoding. A trellis stores vertices
 * in layers and edges between adjacent layers. Edge labels are field elements. The class also
 * provides trellis products, segment merging, text output, and TikZ export for finite fields.
 */

#ifndef TRELLISES_HPP
#define TRELLISES_HPP

#include "fields.hpp"
#include "vectors.hpp"

/*
// transitive
#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "field_concepts_traits.hpp"
#include "helpers.hpp"
*/

namespace CECCO {

namespace details {

template <FieldType T>
struct Vertex {
    explicit Vertex(uint32_t id) : id(id) {}

    uint32_t id;
};

template <FieldType T>
struct Edge {
    Edge(uint32_t from_id, uint32_t to_id, T value) : from_id(from_id), to_id(to_id), value(value) {}

    uint32_t from_id;
    uint32_t to_id;
    T value;
};

}  // namespace details

// Invariant: within each layer, V[s][i].id == i. The Viterbi / BCJR data plane
// indexes path_costs, alpha, beta, backptrs by edge from_id/to_id directly,
// so ids must equal positions. add_edge preserves this as long as new to_ids
// are introduced in increasing order (0, 1, 2, ...); all in-tree constructors
// (row_trellis, operator*, merge_segments) do.
template <FieldType T>
struct Trellis {
    Trellis() : V(1) { V[0].emplace_back(details::Vertex<T>(0)); }

    void add_edge(size_t segment, uint32_t id_from, uint32_t id_to, T value) {
        if (segment >= E.size()) {
            V.resize(segment + 2);
            E.resize(segment + 1);
        }

        auto& Vf = V[segment];
        if (std::ranges::find_if(Vf, [id_from](const auto& v) { return v.id == id_from; }) == Vf.cend())
            throw std::invalid_argument("Start node id " + std::to_string(id_from) + " not found in V[" +
                                        std::to_string(segment) + "]!");

        auto& Vt = V[segment + 1];
        if (std::ranges::find_if(Vt, [id_to](const auto& v) { return v.id == id_to; }) == Vt.cend()) {
            if (id_to != Vt.size())
                throw std::invalid_argument("New sink id " + std::to_string(id_to) + " in V[" +
                                            std::to_string(segment + 1) +
                                            "] must equal its position (id==index invariant)!");
            Vt.emplace_back(id_to);
        }

        auto& Es = E[segment];
        if (std::ranges::any_of(Es, [&](const auto& e) { return e.from_id == id_from && e.to_id == id_to; }))
            throw std::invalid_argument("Edge (" + std::to_string(id_from) + " -> " + std::to_string(id_to) +
                                        ") already exists in E[" + std::to_string(segment) + "]!");
        Es.emplace_back(id_from, id_to, value);
    }

    Trellis operator*(const Trellis& other) const {
        if (E.size() != other.E.size()) throw std::invalid_argument("Trellises must have the same number of segments!");

        const size_t num_segments = E.size();
        std::vector<std::unordered_map<uint64_t, uint32_t>> vmap(num_segments + 1);

        auto get_or_add = [&](size_t s, uint32_t a, uint32_t b) -> uint32_t {
            const uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
            auto [it, inserted] = vmap[s].try_emplace(key, static_cast<uint32_t>(vmap[s].size()));
            return it->second;
        };

        get_or_add(0, 0, 0);

        Trellis result;

        for (size_t s = 0; s < num_segments; ++s) {
            for (const auto& e1 : E[s]) {
                for (const auto& e2 : other.E[s]) {
                    result.add_edge(s, get_or_add(s, e1.from_id, e2.from_id), get_or_add(s + 1, e1.to_id, e2.to_id),
                                    e1.value + e2.value);
                }
            }
        }

        return result;
    }

    size_t get_maximum_depth() const noexcept {
        size_t max = 0;
        for (const auto& seg : V)
            if (seg.size() > max) max = seg.size();
        return max;
    }

    template <typename cost_t>
        requires std::integral<cost_t> || std::floating_point<cost_t>
    struct Viterbi_Workspace {
        static constexpr bool is_soft = std::is_floating_point_v<cost_t>;
        static constexpr cost_t init =
            is_soft ? std::numeric_limits<cost_t>::infinity() : std::numeric_limits<cost_t>::max();

        explicit Viterbi_Workspace(const Trellis& tr) {
            const size_t M = tr.get_maximum_depth();
            path_costs_prev.resize(M);
            path_costs_curr.resize(M);
            tie_counts.resize(M);
            backptrs.reserve(tr.V.size());
            for (size_t s = 0; s < tr.V.size(); ++s) backptrs.emplace_back(tr.V[s].size(), nullptr);
            edge_costs.reserve(tr.E.size());
            for (size_t s = 0; s < tr.E.size(); ++s) edge_costs.emplace_back(tr.E[s].size());
        }

        void calculate_edge_costs(const Trellis& tr, const Vector<T>& r)
            requires std::integral<cost_t>
        {
            if (r.get_n() != tr.E.size())
                throw std::invalid_argument("Vector length must match number of trellis segments!");
            for (size_t s = 0; s < tr.E.size(); ++s)
                for (size_t j = 0; j < tr.E[s].size(); ++j)
#ifdef CECCO_ERASURE_SUPPORT
                    edge_costs[s][j] =
                        r[s].is_erased() ? cost_t{0} : ((tr.E[s][j].value != r[s]) ? cost_t{1} : cost_t{0});
#else
                    edge_costs[s][j] = (tr.E[s][j].value != r[s]) ? cost_t{1} : cost_t{0};
#endif
        }

        void calculate_edge_costs(const Trellis& tr, const Vector<double>& llrs)
            requires std::floating_point<cost_t> && std::is_same_v<T, Fp<2>>
        {
            if (llrs.get_n() != tr.E.size())
                throw std::invalid_argument("Vector length must match number of trellis segments!");
            for (size_t s = 0; s < tr.E.size(); ++s)
                for (size_t j = 0; j < tr.E[s].size(); ++j)
                    edge_costs[s][j] = (tr.E[s][j].value == T(0)) ? cost_t{0} : llrs[s];
        }

        std::vector<cost_t> path_costs_prev;
        std::vector<cost_t> path_costs_curr;
        std::vector<std::vector<const details::Edge<T>*>> backptrs;
        std::vector<uint16_t> tie_counts;
        std::vector<std::vector<cost_t>> edge_costs;
        std::optional<std::variant<Vector<T>, Vector<double>>> v;
    };

    struct BCJR_Workspace {
        static constexpr bool is_soft = true;

        explicit BCJR_Workspace(const Trellis& tr) {
            alpha.reserve(tr.V.size());
            beta.reserve(tr.V.size());
            for (size_t s = 0; s < tr.V.size(); ++s) {
                alpha.emplace_back(tr.V[s].size(), -std::numeric_limits<double>::infinity());
                beta.emplace_back(tr.V[s].size(), -std::numeric_limits<double>::infinity());
            }
            edge_costs.reserve(tr.E.size());
            for (size_t s = 0; s < tr.E.size(); ++s) edge_costs.emplace_back(tr.E[s].size());
        }

        void calculate_edge_costs(const Trellis& tr, const Vector<double>& llrs)
            requires std::is_same_v<T, Fp<2>>
        {
            if (llrs.get_n() != tr.E.size())
                throw std::invalid_argument("Vector length must match number of trellis segments!");
            for (size_t s = 0; s < tr.E.size(); ++s)
                for (size_t j = 0; j < tr.E[s].size(); ++j)
                    edge_costs[s][j] = (tr.E[s][j].value != T(0)) ? llrs[s] : 0.0;
        }

        std::vector<std::vector<double>> alpha;
        std::vector<std::vector<double>> beta;
        std::vector<std::vector<double>> edge_costs;
        std::optional<std::variant<Vector<T>, Vector<double>>> v;
    };

    template <typename WS>
    std::ostream& print(std::ostream& os, const WS* ws) const {
        if (E.empty()) return os;
        for (size_t i = 0; i < E.size(); ++i) {
            if (E[i].empty()) return os;
            for (size_t j = 0; j < E[i].size(); ++j) {
                os << "(" << E[i][j].from_id << "--" << E[i][j].value;
                if (ws) os << "[" << ws->edge_costs[i][j] << "]";
                os << "--" << E[i][j].to_id << ")";
                if (j < E[i].size() - 1) os << ", ";
            }
            if (i != E.size() - 1) os << std::endl;
        }
        return os;
    }

    std::ostream& print(std::ostream& os) const { return print<Viterbi_Workspace<uint16_t>>(os, nullptr); }

    friend std::ostream& operator<<(std::ostream& os, const Trellis& Tr) { return Tr.print(os); }

    template <typename WS>
    void tikz_header(std::ostream& file) const {
        const double arrow_scale = std::min(1.4 / E.size() * 9.0, 1.4);
        const double vertex_size = std::min(5.0 / E.size() * 9.0, 5.0);

        file << R"(% required in preamble:
% \usepackage{amsfonts}
% \usepackage{bm}
% \usepackage{tikz}
% \usetikzlibrary{arrows.meta, backgrounds, calc, positioning}

\tikzset{>={Stealth[scale=)"
             << arrow_scale << R"(]}}
\tikzstyle{trellisvertex}=[circle, draw=black, outer sep=0pt, inner sep=0pt, minimum size=)"
             << vertex_size << R"(pt, left color=gray!80, right color=gray!20])";

        if constexpr (WS::is_soft) {
            file << R"(
\tikzstyle{trellisvertexprev}=[trellisvertex, inner sep=1pt, left color=blue!80, right color=blue!20, text width=4ex, align=center]
\tikzstyle{trellisvertexcurr}=[trellisvertex, inner sep=1pt, left color=green!80, right color=green!20, text width=4ex, align=center])";
        } else {
            file << R"(
\tikzstyle{trellisvertexprev}=[trellisvertex, inner sep=2pt, left color=blue!80, right color=blue!20]
\tikzstyle{trellisvertexcurr}=[trellisvertex, inner sep=2pt, left color=green!80, right color=green!20])";
        }

        file << R"(
\tikzstyle{trellisarrow}=[draw, ->, fill=black]
\tikzstyle{trellisarrowone}=[trellisarrow, fill=red, draw=red]
\tikzstyle{trellisarrowzero}=[trellisarrow, densely dashed, fill=black, draw=black]
\tikzstyle{trellisedgelabel}=[below, sloped]
\tikzstyle{trellispath}=[double distance=.075cm, thick, line join=round, cap=round])";
    }

    template <typename WS>
    void tikz_picture(std::ostream& file, const WS* ws, size_t frontier) const
        requires FiniteFieldType<T> && (T::get_size() <= 64)
    {
        file << "\n\n\\begin{tikzpicture}[x=\\linewidth/" << E.size() << ", y=1cm]";

        for (size_t s = 0; s < V.size(); ++s) {
            for (size_t i = 0; i < V[s].size(); ++i) {
                const char* style = "trellisvertex";
                bool labeled = false;
                if (ws) {
                    if constexpr (std::is_same_v<WS, BCJR_Workspace>) {
                        style = "trellisvertexprev";
                        labeled = true;
                    } else {
                        if (s == frontier) {
                            style = "trellisvertexcurr";
                            labeled = true;
                        } else if (frontier > 0 && s == frontier - 1) {
                            style = "trellisvertexprev";
                            labeled = true;
                        }
                    }
                }
                file << "\n    \\node[" << style << "] (" << s << "_" << V[s][i].id << ") at (" << s << ", "
                     << -static_cast<int>(V[s][i].id) << ") {\\tiny$";
                if (labeled) {
                    if constexpr (WS::is_soft) {
                        file << std::fixed << std::setprecision(2);
                        if constexpr (std::is_same_v<WS, BCJR_Workspace>) {
                            file << ws->alpha[s][i] << "\\mid " << ws->beta[s][i];
                        } else {
                            file << ((s == frontier) ? ws->path_costs_prev[i] : ws->path_costs_curr[i]);
                        }
                    } else {
                        file << ((s == frontier) ? ws->path_costs_prev[i] : ws->path_costs_curr[i]);
                    }
                }
                file << "$};";
            }
        }

        for (size_t s = 0; s < E.size(); ++s) {
            for (size_t j = 0; j < E[s].size(); ++j) {
                const auto& e = E[s][j];
                const auto label = e.value.get_label();
                if (label == 0) {
                    file << "\n    \\path[trellisarrowzero]";
                } else if (label == T::get_size() - 1) {
                    file << "\n    \\path[trellisarrowone]";
                } else {
                    const size_t a = 63 - 63 * static_cast<double>(label) / (T::get_size() - 1);
                    const uint8_t r = details::colormap[a][0];
                    const uint8_t g = details::colormap[a][1];
                    const uint8_t b = details::colormap[a][2];
                    file << "\n    \\definecolor{color}{RGB}{" << static_cast<int>(r) << ", " << static_cast<int>(g)
                         << ", " << static_cast<int>(b) << "}\\path[trellisarrow, draw=color, fill=color]";
                }
                file << " (" << s << "_" << e.from_id << ")";
                if (ws) {
                    file << " edge[trellisedgelabel] node[black] {\\tiny$";
                    if constexpr (WS::is_soft) file << std::fixed << std::setprecision(2);
                    file << ws->edge_costs[s][j] << "$}";
                } else {
                    file << " --";
                }
                file << " (" << s + 1 << "_" << e.to_id << ");";
            }
        }

        file << "\n    \\begin{scope}[on background layer]";
        const size_t maxdepth = get_maximum_depth();
        for (size_t s = 0; s < V.size(); ++s) {
            file << "\n        \\draw [dotted, shorten >=-5mm] (" << s << "_0) to (" << s << ", "
                 << -static_cast<int>(maxdepth) + 1 << ");";
        }

        if constexpr (!std::is_same_v<WS, BCJR_Workspace>) {
            if (ws && frontier > 0) {
                std::vector<size_t> path;
                path.reserve(frontier + 1);
                for (size_t i = 0; i < V[frontier].size(); ++i) {
                    path.clear();
                    size_t v = i;
                    path.push_back(v);
                    for (size_t s = frontier; s > 0; --s) {
                        v = ws->backptrs[s][v]->from_id;
                        path.push_back(v);
                    }
                    file << "\n        \\draw[trellispath, green, double=green!15] (" << frontier - 1 << "_" << path[1]
                         << ") -- (" << frontier << "_" << path[0] << ");";
                    if (path.size() > 2) {
                        file << "\n        \\draw[trellispath, blue, double=blue!15]";
                        for (size_t k = 1; k < path.size(); ++k) {
                            file << " (" << frontier - k << "_" << path[k] << ")";
                            if (k + 1 < path.size()) file << " --";
                        }
                        file << ";";
                    }
                }
            }
        }

        file << "\n    \\end{scope}"
             << "\n    \\node[node distance=.0cm, below left=of 0_0] {$\\mathfrak{s}$};"
             << "\n    \\node[node distance=.0cm, below right=of " << E.size() << "_0] {$\\mathfrak{t}$};";

        if constexpr (requires { ws->v; }) {
            if (ws && ws->v.has_value()) {
                file << "\n    \\node[anchor=east] at ($(0_0)+(0,.5)$) {\\small$\\bm{v}=$};";
                for (size_t s = 0; s < E.size(); ++s) {
                    file << "\n    \\node at ($(" << s << "_0)!0.5!(" << s + 1 << "_0)+(0,.5)$) {\\small$";
                    if (s == 0) file << "(";
                    std::visit([&](const auto& w) { file << w[s]; }, *(ws->v));
                    file << (s + 1 == E.size() ? ")" : ",") << "$};";
                }
            }
        }

        file << "\n\\end{tikzpicture}\n";
    }

    template <typename WS>
    void export_as_tikz(const std::string& filename, const WS* ws) const
        requires FiniteFieldType<T> && (T::get_size() <= 64)
    {
        std::ofstream file;
        file.open(filename);
        tikz_header<WS>(file);
        tikz_picture(file, ws, V.size() - 1);
        file.close();
    }

    void export_as_tikz(const std::string& filename) const
        requires FiniteFieldType<T> && (T::get_size() <= 64)
    {
        export_as_tikz<Viterbi_Workspace<uint16_t>>(filename, nullptr);
    }

    template <FiniteFieldType U>
        requires SubfieldOf<U, T>
    Trellis<U> merge_segments() const {
        constexpr size_t m = details::degree_over_prime_v<U> / details::degree_over_prime_v<T>;
        const size_t n = E.size();
        const size_t full_groups = n / m;

        Trellis<U> result;
        size_t seg = 0;

        for (size_t g = 0; g < full_groups; ++g) {
            std::vector<std::vector<const details::Edge<T>*>> paths;

            for (const auto& e : E[g * m]) paths.push_back({&e});

            for (size_t step = 1; step < m; ++step) {
                std::vector<std::vector<const details::Edge<T>*>> next;
                for (const auto& path : paths)
                    for (const auto& e : E[g * m + step])
                        if (e.from_id == path.back()->to_id) {
                            auto ext = path;
                            ext.push_back(&e);
                            next.push_back(std::move(ext));
                        }
                paths = std::move(next);
            }

            for (const auto& path : paths) {
                Vector<T> v(m);
                for (size_t i = 0; i < m; ++i) v.set_component(i, path[i]->value);
                result.add_edge(seg, path.front()->from_id, path.back()->to_id, U(v));
            }
            ++seg;
        }

        for (size_t s = full_groups * m; s < n; ++s) {
            for (const auto& e : E[s]) {
                Vector<T> v(m, T(0));
                v.set_component(0, e.value);
                result.add_edge(seg, e.from_id, e.to_id, U(v));
            }
            ++seg;
        }

        return result;
    }

    std::vector<std::vector<details::Vertex<T>>> V;
    std::vector<std::vector<details::Edge<T>>> E;
};

}  // namespace CECCO

#endif
