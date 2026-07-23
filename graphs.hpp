/**
 * @file graphs.hpp
 * @brief Trellises and Tanner graphs
 * @author Christian Senger <senger@inue.uni-stuttgart.de>
 * @version 2.3.1
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
 * Layered trellises and parity-check Tanner graphs used by the Viterbi, BCJR, and
 * belief-propagation decoders in `codes.hpp`.
 */

#ifndef GRAPHS_HPP
#define GRAPHS_HPP

#include <memory>
#include <variant>

#include "fields.hpp"
#include "vectors.hpp"

/*
// transitive
#include <algorithm>
#include <array>
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
#include <vector>

#include "field_concepts_traits.hpp"
#include "helpers.hpp"
*/

namespace CECCO {

namespace details {

/**
 * @brief Vertex in one trellis layer
 *
 * The `id` is also the position within its layer. Decoding workspaces index
 * arrays by this value.
 */
template <FieldType T>
struct Vertex {
    /// @brief Construct a vertex with layer-local id `id`
    explicit Vertex(uint32_t id) noexcept : id(id) {}

    /// @brief Layer-local vertex id
    uint32_t id;
};

/**
 * @brief Edge between two adjacent trellis layers
 *
 * @tparam T Field type used for edge labels
 */
template <FieldType T>
struct Edge {
    /**
     * @brief Construct an edge `from_id --value--> to_id`
     *
     * @param from_id Source vertex id in layer `s`
     * @param to_id Target vertex id in layer `s + 1`
     * @param value Field label carried by the edge
     */
    Edge(uint32_t from_id, uint32_t to_id, T value) : from_id(from_id), to_id(to_id), value(value) {}

    /// @brief Source vertex id
    uint32_t from_id;
    /// @brief Target vertex id
    uint32_t to_id;
    /// @brief Edge label
    T value;
};

/**
 * @brief Edge from a check node to a variable node in a Tanner graph
 *
 * @tparam T Field type used for edge labels
 */
template <FieldType T>
struct CheckEdge {
    /**
     * @brief Construct an edge to variable node `var_id` with label `value`
     *
     * @param var_id Incident variable-node id
     * @param value Parity-check coefficient carried by the edge
     */
    CheckEdge(uint32_t var_id, T value) : var_id(var_id), value(value) {}

    /// @brief Incident variable-node id
    uint32_t var_id;
    /// @brief Parity-check coefficient (label)
    T value;
};

/**
 * @brief Convert one column of symbol-level LLRs to trellis costs
 * @tparam T Finite field of the trellis edge labels
 * @param llrs Matrix whose entry `(a - 1, s)` is `ln(P(r_s | 0) / P(r_s | a))`
 * @param s Symbol index
 * @return Costs indexed by field-element label, with cost zero for symbol 0
 *
 * Lower cost means greater likelihood. Non-finite LLRs are clamped to ±1e6.
 */
template <FiniteFieldType T>
std::array<double, T::get_size()> symbol_costs_from_llrs(const Matrix<double>& llrs, size_t s) {
    constexpr size_t q = T::get_size();
    constexpr double L_MAX = 1.0e6;

    std::array<double, q> costs;
    costs[0] = 0.0;
    for (size_t a = 1; a < q; ++a) costs[a] = std::clamp(llrs(a - 1, s), -L_MAX, L_MAX);
    return costs;
}

inline double max_star(double a, double b) noexcept {
    constexpr double neg_inf = -std::numeric_limits<double>::infinity();
    if (a == neg_inf) return b;
    if (b == neg_inf) return a;
    const double d = std::abs(a - b);
    return std::max(a, b) + ((d > 500.0) ? 0.0 : std::log(1.0 + std::exp(-d)));
}

}  // namespace details

template <FieldType T>
struct Trellis;
template <FieldType T>
std::ostream& operator<<(std::ostream& os, const Trellis<T>& Tr);
template <FieldType T>
struct TannerGraph;

/**
 * @brief Layered trellis with field-labelled edges
 * @tparam T Field type used for edge labels
 *
 * Segment `s` connects layers `s` and `s + 1`. Vertices are stored in @ref V with
 * the invariant `V[s][i].id == i`; decoding workspaces use these indices for path metrics.
 *
 * Supports trellis products, segment merging, Viterbi and BCJR workspaces, and
 * text and TikZ export.
 */
template <FieldType T>
struct Trellis {
    /** @name Construction
     * @{
     */

    /// @brief Construct the one-vertex source trellis
    Trellis();

    /**
     * @brief Add one edge in segment `segment`
     *
     * @param segment Segment index; connects layer `segment` to `segment + 1`
     * @param id_from Source vertex id in layer `segment`
     * @param id_to Target vertex id in layer `segment + 1`
     * @param value Edge label
     * @throws std::invalid_argument if `id_from` is missing, `id_to` breaks the id/index invariant,
     *         or the edge already exists
     */
    void add_edge(size_t segment, uint32_t id_from, uint32_t id_to, T value);

    /**
     * @brief Add one edge in segment `segment`, allowing a parallel edge
     *
     * @param segment Segment index; connects layer `segment` to `segment + 1`
     * @param id_from Source vertex id in layer `segment`
     * @param id_to Target vertex id in layer `segment + 1`
     * @param value Edge label
     * @throws std::invalid_argument if `id_from` is missing or `id_to` breaks the id/index invariant
     *
     * Unlike @ref add_edge this does not reject an existing `(id_from, id_to)` edge; use it only
     * where several labels legitimately connect the same two vertices (e.g. the LC-OSD local trellis).
     */
    void add_parallel_edge(size_t segment, uint32_t id_from, uint32_t id_to, T value);

    /**
     * @brief Product trellis with componentwise added edge labels
     *
     * @param other Trellis with the same number of segments
     * @return Trellis whose vertices are pairs of vertices from the operands
     * @throws std::invalid_argument if the numbers of segments differ
     */
    Trellis operator*(const Trellis& other) const;

    /** @} */

    /** @name Information
     * @{
     */

    /// @brief Maximum number of vertices in any layer
    size_t get_maximum_depth() const noexcept;

    /** @} */

    /** @name Decoding Workspaces
     * @{
     */

    /**
     * @brief Workspace for Viterbi decoding on this trellis
     *
     * @tparam cost_t Integral type for hard-decision costs or floating-point type for soft costs
     *
     * Stores path metrics for two adjacent layers, edge costs, tie counts, and backpointers.
     * Construct it from the trellis before calling the Viterbi routines in `codes.hpp`.
     */
    template <typename cost_t>
        requires std::integral<cost_t> || std::floating_point<cost_t>
    struct Viterbi_Workspace {
        /// @brief True for floating-point soft metrics
        static constexpr bool is_soft = std::is_floating_point_v<cost_t>;
        /// @brief Initial path cost used for unreachable vertices
        static constexpr cost_t init =
            is_soft ? std::numeric_limits<cost_t>::infinity() : std::numeric_limits<cost_t>::max();

        /**
         * @brief Allocate metric arrays for `tr`
         *
         * @param tr Trellis whose layer and edge sizes define the workspace
         */
        explicit Viterbi_Workspace(const Trellis& tr);

        /**
         * @brief Set hard-decision edge costs from received word `r`
         *
         * @param tr Trellis whose edges define candidate symbols
         * @param r Received word; length must equal the number of segments
         * @throws std::invalid_argument if `r.get_n() != tr.E.size()`
         *
         * With erasure support, erased received symbols give cost 0 on all outgoing labels.
         */
        void calculate_edge_costs(const Trellis& tr, const Vector<T>& r)
            requires std::integral<cost_t>;

        /**
         * @brief Set soft edge costs from symbol-level LLRs
         *
         * @param trellis Trellis whose edges define candidate symbols
         * @param llrs Symbol-level LLR matrix with q−1 rows and one column per segment
         * @throws std::invalid_argument if the LLR matrix has the wrong number of rows or columns
         *
         * Edge costs are the symbol costs of @ref CECCO::details::symbol_costs_from_llrs: cost 0
         * on 0-labelled edges and L_s(a) on edges labelled by symbol a.
         */
        void calculate_edge_costs(const Trellis& trellis, const Matrix<double>& llrs)
            requires std::floating_point<cost_t> && FiniteFieldType<T>;

        /// @brief Path costs from the previous layer
        std::vector<cost_t> path_costs_prev;
        /// @brief Path costs for the current layer
        std::vector<cost_t> path_costs_curr;
        /// @brief Backpointer selected for each vertex and layer
        std::vector<std::vector<const details::Edge<T>*>> backptrs;
        /// @brief Number of equal-cost paths seen for randomized tie-breaking
        std::vector<uint16_t> tie_counts;
        /// @brief Edge cost for each segment and edge index
        std::vector<std::vector<cost_t>> edge_costs;
        /// @brief Optional received word (hard symbols or LLR matrix) shown in TikZ output
        std::optional<std::variant<Vector<T>, Matrix<double>>> v;
    };

    /**
     * @brief Workspace for list-output Viterbi decoding on this trellis
     *
     * @tparam cost_t Integral type for hard-decision costs or floating-point type for soft costs
     *
     * Stores edge costs by segment and, for the priority-first path enumeration, the backward
     * cost-to-go from every vertex to the final layer. Construct it from the trellis before calling
     * the list Viterbi routine in `codes.hpp`.
     */
    template <typename cost_t>
        requires std::integral<cost_t> || std::floating_point<cost_t>
    struct ListViterbi_Workspace {
        /// @brief True for floating-point soft metrics
        static constexpr bool is_soft = std::is_floating_point_v<cost_t>;
        /// @brief Cost of an unreachable vertex or an infeasible path
        static constexpr cost_t init =
            is_soft ? std::numeric_limits<cost_t>::infinity() : std::numeric_limits<cost_t>::max();

        /**
         * @brief Allocate metric arrays for `tr`
         *
         * @param tr Trellis whose layer and edge sizes define the workspace
         */
        explicit ListViterbi_Workspace(const Trellis& tr);

        /**
         * @brief Set hard-decision edge costs from received word `r`
         *
         * @param tr Trellis whose edges define candidate symbols
         * @param r Received word; length must equal the number of segments
         * @throws std::invalid_argument if `r.get_n() != tr.E.size()`
         */
        void calculate_edge_costs(const Trellis& tr, const Vector<T>& r)
            requires std::integral<cost_t>;

        /**
         * @brief Set soft edge costs from symbol-level LLRs
         *
         * @param trellis Trellis whose edges define candidate symbols
         * @param llrs Symbol-level LLR matrix with q−1 rows and one column per segment
         * @throws std::invalid_argument if the LLR matrix has the wrong number of rows or columns
         */
        void calculate_edge_costs(const Trellis& trellis, const Matrix<double>& llrs)
            requires std::floating_point<cost_t> && FiniteFieldType<T>;

        /// @brief Edge cost for each segment and edge index
        std::vector<std::vector<cost_t>> edge_costs;
        /// @brief Minimum cost from each vertex (by layer and id) to any final-layer vertex
        std::vector<std::vector<cost_t>> cost_to_go;
    };

    /**
     * @brief Workspace for BCJR forward-backward decoding
     *
     * Stores α and β metrics by layer and edge costs by segment. It is used by
     * `LinearCode<T>::dec_BCJR`.
     */
    struct BCJR_Workspace {
        /// @brief BCJR uses floating-point soft metrics
        static constexpr bool is_soft = true;

        /**
         * @brief Allocate metric arrays for `tr`
         *
         * @param tr Trellis whose layer and edge sizes define the workspace
         */
        explicit BCJR_Workspace(const Trellis& tr);

        /**
         * @brief Set BCJR edge costs from symbol-level LLRs
         *
         * @param trellis Trellis whose edges define candidate symbols
         * @param llrs Symbol-level LLR matrix with q−1 rows and one column per segment
         * @throws std::invalid_argument if the LLR matrix has the wrong number of rows or columns
         *
         * Edge costs are the symbol costs of @ref CECCO::details::symbol_costs_from_llrs.
         */
        void calculate_edge_costs(const Trellis& trellis, const Matrix<double>& llrs)
            requires FiniteFieldType<T>;

        /// @brief Forward metrics by layer and vertex id
        std::vector<std::vector<double>> alpha;
        /// @brief Backward metrics by layer and vertex id
        std::vector<std::vector<double>> beta;
        /// @brief Edge cost for each segment and edge index
        std::vector<std::vector<double>> edge_costs;
        /// @brief Optional received word (hard symbols or LLR matrix) shown in TikZ output
        std::optional<std::variant<Vector<T>, Matrix<double>>> v;
    };

    /** @} */

    /** @name Text and TikZ Output
     * @{
     */

    /**
     * @brief Write a segment-by-segment text representation
     *
     * @param os Output stream
     * @param ws Optional workspace; if non-null, edge costs are printed with edge labels
     * @return Output stream
     */
    template <typename WS>
    std::ostream& print(std::ostream& os, const WS* ws) const;

    /**
     * @brief Write a segment-by-segment text representation without edge costs
     *
     * @param os Output stream
     * @return Output stream
     */
    std::ostream& print(std::ostream& os) const;

    /// @brief Stream output via @ref print
    friend std::ostream& operator<< <>(std::ostream& os, const Trellis& Tr);

    /**
     * @brief Write TikZ style definitions
     *
     * @param file Output stream
     */
    template <typename WS>
    void tikz_header(std::ostream& file) const;

    /**
     * @brief Write one TikZ picture of the trellis
     *
     * @param file Output stream
     * @param ws Optional decoding workspace; if non-null, metrics and edge costs are shown
     * @param frontier Viterbi frontier layer to highlight
     *
     * Available only for finite fields of size at most 64.
     */
    template <typename WS>
    void tikz_picture(std::ostream& file, const WS* ws, size_t frontier) const
        requires FiniteFieldType<T> && (T::get_size() <= 64);

    /**
     * @brief Export a standalone TikZ fragment with workspace metrics
     *
     * @param filename Output filename
     * @param ws Optional decoding workspace; if non-null, metrics and edge costs are shown
     *
     * Available only for finite fields of size at most 64.
     */
    template <typename WS>
    void export_as_tikz(const std::string& filename, const WS* ws) const
        requires FiniteFieldType<T> && (T::get_size() <= 64);

    /**
     * @brief Export a standalone TikZ fragment without workspace metrics
     *
     * @param filename Output filename
     *
     * Available only for finite fields of size at most 64.
     */
    void export_as_tikz(const std::string& filename) const
        requires FiniteFieldType<T> && (T::get_size() <= 64);

    /** @} */

    /** @name Field Conversion
     * @{
     */

    /**
     * @brief Merge groups of base-field segments into extension-field labels
     *
     * @tparam U Extension field with `T ⊆ U`
     * @return Trellis over `U`
     *
     * Consecutive groups of `[U:T]` segments are replaced by one segment whose labels
     * are the corresponding `U` elements. A final incomplete group is zero-padded.
     */
    template <FiniteFieldType U>
        requires SubfieldOf<U, T>
    Trellis<U> merge_segments() const;

    /** @} */

    /// @brief Vertices by layer; `V[s][i].id == i`
    std::vector<std::vector<details::Vertex<T>>> V;
    /// @brief Edges by segment; `E[s]` connects `V[s]` to `V[s + 1]`
    std::vector<std::vector<details::Edge<T>>> E;
};

template <FieldType T>
Trellis<T>::Trellis() : V(1) {
    V[0].emplace_back(details::Vertex<T>(0));
}

template <FieldType T>
void Trellis<T>::add_edge(size_t segment, uint32_t id_from, uint32_t id_to, T value) {
    if (segment < E.size() &&
        std::ranges::any_of(E[segment], [&](const auto& e) { return e.from_id == id_from && e.to_id == id_to; }))
        throw std::invalid_argument("Edge (" + std::to_string(id_from) + " -> " + std::to_string(id_to) +
                                    ") already exists in E[" + std::to_string(segment) + "]!");
    add_parallel_edge(segment, id_from, id_to, value);
}

template <FieldType T>
void Trellis<T>::add_parallel_edge(size_t segment, uint32_t id_from, uint32_t id_to, T value) {
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

    E[segment].emplace_back(id_from, id_to, value);
}

template <FieldType T>
Trellis<T> Trellis<T>::operator*(const Trellis& other) const {
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
        for (auto e1 = E[s].cbegin(); e1 != E[s].cend(); ++e1) {
            for (auto e2 = other.E[s].cbegin(); e2 != other.E[s].cend(); ++e2) {
                result.add_parallel_edge(s, get_or_add(s, e1->from_id, e2->from_id),
                                         get_or_add(s + 1, e1->to_id, e2->to_id), e1->value + e2->value);
            }
        }
    }

    return result;
}

template <FieldType T>
size_t Trellis<T>::get_maximum_depth() const noexcept {
    size_t max = 0;
    for (size_t s = 0; s < V.size(); ++s)
        if (V[s].size() > max) max = V[s].size();
    return max;
}

template <FieldType T>
template <typename cost_t>
    requires std::integral<cost_t> || std::floating_point<cost_t>
Trellis<T>::Viterbi_Workspace<cost_t>::Viterbi_Workspace(const Trellis& tr) {
    const size_t M = tr.get_maximum_depth();
    path_costs_prev.resize(M);
    path_costs_curr.resize(M);
    tie_counts.resize(M);
    backptrs.reserve(tr.V.size());
    for (size_t s = 0; s < tr.V.size(); ++s) backptrs.emplace_back(tr.V[s].size(), nullptr);
    edge_costs.reserve(tr.E.size());
    for (size_t s = 0; s < tr.E.size(); ++s) edge_costs.emplace_back(tr.E[s].size());
}

template <FieldType T>
template <typename cost_t>
    requires std::integral<cost_t> ||
             std::floating_point<cost_t>
             void Trellis<T>::Viterbi_Workspace<cost_t>::calculate_edge_costs(const Trellis& tr, const Vector<T>& r)
                 requires std::integral<cost_t>
{
    if (r.get_n() != tr.E.size()) throw std::invalid_argument("Vector length must match number of trellis segments!");
    for (size_t s = 0; s < tr.E.size(); ++s)
        for (size_t j = 0; j < tr.E[s].size(); ++j)
#ifdef CECCO_ERASURE_SUPPORT
            edge_costs[s][j] = r[s].is_erased() ? cost_t{0} : ((tr.E[s][j].value != r[s]) ? cost_t{1} : cost_t{0});
#else
            edge_costs[s][j] = (tr.E[s][j].value != r[s]) ? cost_t{1} : cost_t{0};
#endif
}

template <FieldType T>
template <typename cost_t>
    requires std::integral<cost_t> ||
             std::floating_point<cost_t>
             void Trellis<T>::Viterbi_Workspace<cost_t>::calculate_edge_costs(const Trellis& trellis,
                                                                              const Matrix<double>& llrs)
                 requires std::floating_point<cost_t> && FiniteFieldType<T>
{
    if (llrs.get_m() != T::get_size() - 1 || llrs.get_n() != trellis.E.size())
        throw std::invalid_argument("LLR matrix must have q-1 rows and one column per trellis segment!");
    for (size_t s = 0; s < trellis.E.size(); ++s) {
        const auto symbol_costs = details::symbol_costs_from_llrs<T>(llrs, s);
        for (size_t j = 0; j < trellis.E[s].size(); ++j)
            edge_costs[s][j] = symbol_costs[trellis.E[s][j].value.get_label()];
    }
}

template <FieldType T>
template <typename cost_t>
    requires std::integral<cost_t> || std::floating_point<cost_t>
Trellis<T>::ListViterbi_Workspace<cost_t>::ListViterbi_Workspace(const Trellis& tr) {
    edge_costs.reserve(tr.E.size());
    for (size_t s = 0; s < tr.E.size(); ++s) edge_costs.emplace_back(tr.E[s].size());
    cost_to_go.reserve(tr.V.size());
    for (size_t s = 0; s < tr.V.size(); ++s) cost_to_go.emplace_back(tr.V[s].size(), init);
}

template <FieldType T>
template <typename cost_t>
    requires std::integral<cost_t> ||
             std::floating_point<cost_t>
             void Trellis<T>::ListViterbi_Workspace<cost_t>::calculate_edge_costs(const Trellis& tr, const Vector<T>& r)
                 requires std::integral<cost_t>
{
    if (r.get_n() != tr.E.size()) throw std::invalid_argument("Vector length must match number of trellis segments!");
    for (size_t s = 0; s < tr.E.size(); ++s)
        for (size_t j = 0; j < tr.E[s].size(); ++j)
#ifdef CECCO_ERASURE_SUPPORT
            edge_costs[s][j] = r[s].is_erased() ? cost_t{0} : ((tr.E[s][j].value != r[s]) ? cost_t{1} : cost_t{0});
#else
            edge_costs[s][j] = (tr.E[s][j].value != r[s]) ? cost_t{1} : cost_t{0};
#endif
}

template <FieldType T>
template <typename cost_t>
    requires std::integral<cost_t> ||
             std::floating_point<cost_t>
             void Trellis<T>::ListViterbi_Workspace<cost_t>::calculate_edge_costs(const Trellis& trellis,
                                                                                  const Matrix<double>& llrs)
                 requires std::floating_point<cost_t> && FiniteFieldType<T>
{
    if (llrs.get_m() != T::get_size() - 1 || llrs.get_n() != trellis.E.size())
        throw std::invalid_argument("LLR matrix must have q-1 rows and one column per trellis segment!");
    for (size_t s = 0; s < trellis.E.size(); ++s) {
        const auto symbol_costs = details::symbol_costs_from_llrs<T>(llrs, s);
        for (size_t j = 0; j < trellis.E[s].size(); ++j)
            edge_costs[s][j] = symbol_costs[trellis.E[s][j].value.get_label()];
    }
}

template <FieldType T>
Trellis<T>::BCJR_Workspace::BCJR_Workspace(const Trellis& tr) {
    alpha.reserve(tr.V.size());
    beta.reserve(tr.V.size());
    for (size_t s = 0; s < tr.V.size(); ++s) {
        alpha.emplace_back(tr.V[s].size(), -std::numeric_limits<double>::infinity());
        beta.emplace_back(tr.V[s].size(), -std::numeric_limits<double>::infinity());
    }
    edge_costs.reserve(tr.E.size());
    for (size_t s = 0; s < tr.E.size(); ++s) edge_costs.emplace_back(tr.E[s].size());
}

template <FieldType T>
void Trellis<T>::BCJR_Workspace::calculate_edge_costs(const Trellis& trellis, const Matrix<double>& llrs)
    requires FiniteFieldType<T>
{
    if (llrs.get_m() != T::get_size() - 1 || llrs.get_n() != trellis.E.size())
        throw std::invalid_argument("LLR matrix must have q-1 rows and one column per trellis segment!");
    for (size_t s = 0; s < trellis.E.size(); ++s) {
        const auto symbol_costs = details::symbol_costs_from_llrs<T>(llrs, s);
        for (size_t j = 0; j < trellis.E[s].size(); ++j)
            edge_costs[s][j] = symbol_costs[trellis.E[s][j].value.get_label()];
    }
}

template <FieldType T>
template <typename WS>
std::ostream& Trellis<T>::print(std::ostream& os, const WS* ws) const {
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

template <FieldType T>
std::ostream& Trellis<T>::print(std::ostream& os) const {
    return this->template print<Viterbi_Workspace<uint16_t>>(os, nullptr);
}

template <FieldType T>
std::ostream& operator<<(std::ostream& os, const Trellis<T>& Tr) {
    return Tr.print(os);
}

template <FieldType T>
template <typename WS>
void Trellis<T>::tikz_header(std::ostream& file) const {
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
\tikzstyle{trellisvertexprev}=[trellisvertex, font=\tiny, inner sep=1pt, left color=blue!80, right color=blue!20, text width=4ex, align=center]
\tikzstyle{trellisvertexcurr}=[trellisvertex, font=\tiny, inner sep=1pt, left color=green!80, right color=green!20, text width=4ex, align=center])";
    } else {
        file << R"(
\tikzstyle{trellisvertexprev}=[trellisvertex, font=\footnotesize, inner sep=2pt, left color=blue!80, right color=blue!20]
\tikzstyle{trellisvertexcurr}=[trellisvertex, font=\footnotesize, inner sep=2pt, left color=green!80, right color=green!20])";
    }

    if constexpr (std::is_same_v<WS, BCJR_Workspace>) {
        file << R"(
\tikzstyle{trellisvertexsplit}=[trellisvertexprev, inner sep=1pt, circle split, font=\tiny, inner sep=1pt])";
    } else {
    }

    file << R"(
\tikzstyle{trellisarrow}=[draw, ->, fill=black]
\tikzstyle{trellisarrowone}=[trellisarrow, fill=red, draw=red]
\tikzstyle{trellisarrowzero}=[trellisarrow, densely dashed, fill=black, draw=black]
\tikzstyle{trellisedgelabel}=[below, sloped]
\tikzstyle{trellispath}=[double distance=.075cm, thick, line join=round, cap=round])";
}

template <FieldType T>
template <typename WS>
void Trellis<T>::tikz_picture(std::ostream& file, const WS* ws, size_t frontier) const
    requires FiniteFieldType<T> && (T::get_size() <= 64)
{
    file << "\n\n\\begin{tikzpicture}[x=\\linewidth/" << E.size() << ", y=1cm]";

    for (size_t s = 0; s < V.size(); ++s) {
        for (size_t i = 0; i < V[s].size(); ++i) {
            const char* style = "trellisvertex";
            bool labeled = false;
            if (ws) {
                if constexpr (std::is_same_v<WS, BCJR_Workspace>) {
                    style = "trellisvertexsplit";
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
            if (!labeled) {
                file << "\n    \\node[" << style << "] (" << s << "_" << V[s][i].id << ") at (" << s << ", "
                     << -static_cast<int>(V[s][i].id) << ") {};";
            } else {
                if constexpr (WS::is_soft) {
                    file << std::fixed << std::setprecision(2);
                    if constexpr (std::is_same_v<WS, BCJR_Workspace>) {
                        file << "\n    \\node[" << style << "] (" << s << "_" << V[s][i].id << ") at (" << s << ", "
                             << -static_cast<int>(V[s][i].id) << ") {$" << ws->alpha[s][i] << "$\\nodepart{lower} $"
                             << ws->beta[s][i] << "$};";
                    } else {
                        file << "\n    \\node[" << style << "] (" << s << "_" << V[s][i].id << ") at (" << s << ", "
                             << -static_cast<int>(V[s][i].id) << ") {$"
                             << ((s == frontier) ? ws->path_costs_prev[i] : ws->path_costs_curr[i]) << "$};";
                    }
                } else {
                    file << "\n    \\node[" << style << "] (" << s << "_" << V[s][i].id << ") at (" << s << ", "
                         << -static_cast<int>(V[s][i].id) << ") {$"
                         << ((s == frontier) ? ws->path_costs_prev[i] : ws->path_costs_curr[i]) << "$};";
                }
            }
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
                file << "\n    \\definecolor{color}{RGB}{" << static_cast<int>(r) << ", " << static_cast<int>(g) << ", "
                     << static_cast<int>(b) << "}\\path[trellisarrow, draw=color, fill=color]";
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
                bool complete = true;
                for (size_t s = frontier; s > 0; --s) {
                    const auto* edge = ws->backptrs[s][v];
                    if (edge == nullptr) {  // no survivor recorded (yet), nothing to draw
                        complete = false;
                        break;
                    }
                    v = edge->from_id;
                    path.push_back(v);
                }
                if (!complete) continue;
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
                std::visit(
                    [&](const auto& w) {
                        if constexpr (std::is_same_v<std::decay_t<decltype(w)>, Matrix<double>>)
                            file << w(0, s);
                        else
                            file << w[s];
                    },
                    *(ws->v));
                file << (s + 1 == E.size() ? ")" : ",") << "$};";
            }
        }
    }

    file << "\n\\end{tikzpicture}\n";
}

template <FieldType T>
template <typename WS>
void Trellis<T>::export_as_tikz(const std::string& filename, const WS* ws) const
    requires FiniteFieldType<T> && (T::get_size() <= 64)
{
    std::ofstream file;
    file.open(filename);
    tikz_header<WS>(file);
    tikz_picture(file, ws, V.size() - 1);
    file.close();
}

template <FieldType T>
void Trellis<T>::export_as_tikz(const std::string& filename) const
    requires FiniteFieldType<T> && (T::get_size() <= 64)
{
    export_as_tikz<Viterbi_Workspace<uint16_t>>(filename, nullptr);
}

template <FieldType T>
template <FiniteFieldType U>
    requires SubfieldOf<U, T>
Trellis<U> Trellis<T>::merge_segments() const {
    constexpr size_t m = details::degree_over_prime_v<U> / details::degree_over_prime_v<T>;
    const size_t n = E.size();
    const size_t full_groups = n / m;
    const size_t tail = n - full_groups * m;
    const size_t segments = full_groups + (tail != 0 ? 1 : 0);

    Trellis<U> result;

    // add_parallel_edge accepts a new sink id only in ascending order, which paths cannot guarantee
    result.V.resize(segments + 1);
    result.E.resize(segments);
    for (size_t s = 1; s <= segments; ++s) {
        const size_t src = std::min(s * m, n);
        result.V[s].reserve(V[src].size());
        for (size_t i = 0; i < V[src].size(); ++i) result.V[s].emplace_back(static_cast<uint32_t>(i));
    }

    for (size_t g = 0; g < segments; ++g) {
        const size_t width = (g < full_groups) ? m : tail;
        std::vector<std::vector<const details::Edge<T>*>> paths;

        for (auto e = E[g * m].cbegin(); e != E[g * m].cend(); ++e) paths.push_back({std::to_address(e)});

        for (size_t step = 1; step < width; ++step) {
            std::vector<std::vector<const details::Edge<T>*>> next;
            for (auto path = paths.cbegin(); path != paths.cend(); ++path)
                for (auto e = E[g * m + step].cbegin(); e != E[g * m + step].cend(); ++e)
                    if (e->from_id == path->back()->to_id) {
                        auto ext = *path;
                        ext.push_back(std::to_address(e));
                        next.push_back(std::move(ext));
                    }
            paths = std::move(next);
        }

        for (auto path = paths.cbegin(); path != paths.cend(); ++path) {
            Vector<T> v(m);
            for (size_t i = 0; i < width; ++i) v.set_component(i, (*path)[i]->value);
            result.add_parallel_edge(g, path->front()->from_id, path->back()->to_id, U(v));
        }
    }

    return result;
}

/**
 * @brief Tanner graph of a parity-check matrix
 * @tparam T Field type used for edge labels
 *
 * Contains `n` variable nodes and one node per parity-check equation. An edge
 * `(check, variable, value)` represents a nonzero parity-check coefficient.
 * @ref BP_Workspace provides sum-product decoding state.
 */
template <FieldType T>
struct TannerGraph {
    /// @brief Construct a Tanner graph with `n` variable nodes and no checks
    explicit TannerGraph(size_t n) : n(n) {}

    /**
     * @brief Add a check-to-variable edge `check --value--> var_id`
     *
     * @param check Check-node index; checks are created on demand
     * @param var_id Incident variable-node id
     * @param value Parity-check coefficient carried by the edge
     * @throws std::invalid_argument if `var_id` is not a variable node of this graph
     */
    void add_edge(size_t check, uint32_t var_id, T value) {
        if (var_id >= n)
            throw std::invalid_argument("Variable node id " + std::to_string(var_id) +
                                        " out of range for Tanner graph with " + std::to_string(n) + " variable nodes");
        if (check >= checks.size()) checks.resize(check + 1);
        checks[check].emplace_back(var_id, value);
    }

    /**
     * @brief Workspace for sum-product (belief propagation) decoding
     *
     * Stores variable-to-check and check-to-variable messages, the intrinsic (channel) costs, and the
     * running posterior beliefs, each a length-q array per node. It is used by `LinearCode<T>::dec_BP`.
     */
    struct BP_Workspace {
        /// @brief Number of field symbols (message and belief vector length)
        static constexpr size_t q = T::get_size();

        /**
         * @brief Allocate message and belief arrays for `g`
         *
         * @param g Tanner graph whose variable- and check-node degrees define the workspace
         */
        explicit BP_Workspace(const TannerGraph& g) : intrinsic(g.n), posterior(g.n), var_edges(g.n) {
            m_vc.reserve(g.checks.size());
            m_cv.reserve(g.checks.size());
            for (size_t i = 0; i < g.checks.size(); ++i) {
                m_vc.emplace_back(g.checks[i].size());
                m_cv.emplace_back(g.checks[i].size());
                for (size_t e = 0; e < g.checks[i].size(); ++e) var_edges[g.checks[i][e].var_id].emplace_back(i, e);
            }
        }

        /**
         * @brief Set intrinsic (channel) costs from symbol-level LLRs
         *
         * @param g Tanner graph providing the variable-node count
         * @param llrs Symbol-level LLR matrix with q−1 rows and one column per variable node
         *
         * Intrinsic costs are the negated symbol costs of @ref CECCO::details::symbol_costs_from_llrs.
         */
        void calculate_intrinsic(const TannerGraph& g, const Matrix<double>& llrs)
            requires FiniteFieldType<T>
        {
            for (size_t j = 0; j < g.n; ++j) {
                const auto costs = details::symbol_costs_from_llrs<T>(llrs, j);
                for (size_t a = 0; a < q; ++a) intrinsic[j][a] = -costs[a];
            }
        }

        /// @brief Variable-to-check messages by check and incident edge
        std::vector<std::vector<std::array<double, q>>> m_vc;
        /// @brief Check-to-variable messages by check and incident edge
        std::vector<std::vector<std::array<double, q>>> m_cv;
        /// @brief Intrinsic (channel) cost per variable node and symbol
        std::vector<std::array<double, q>> intrinsic;
        /// @brief Posterior belief per variable node and symbol
        std::vector<std::array<double, q>> posterior;
        /// @brief Incident (check, edge) pairs for each variable node
        std::vector<std::vector<std::pair<size_t, size_t>>> var_edges;
        /// @brief Optional received LLR matrix shown in TikZ output
        std::optional<Matrix<double>> v;
    };

    /**
     * @brief Write TikZ style definitions
     *
     * @param file Output stream
     */
    void tikz_header(std::ostream& file) const {
        file << R"(% required in preamble:
% \usepackage{amsfonts}
% \usepackage{bm}
% \usepackage{tikz}
% \usetikzlibrary{calc, positioning}

\tikzstyle{tannervariable}=[circle, draw=black, inner sep=0pt, minimum size=6pt, left color=gray!80, right color=gray!20]
\tikzstyle{tannervariablesoft}=[circle, draw=black, font=\tiny, inner sep=1pt, left color=green!80, right color=green!20, text width=4ex, align=center]
\tikzstyle{tannercheck}=[rectangle, draw=black, inner sep=0pt, minimum size=7pt, fill=gray!40]
\tikzstyle{tanneredge}=[draw])";
    }

    /**
     * @brief Write one TikZ picture of the Tanner graph
     *
     * @param file Output stream
     * @param ws Optional decoding workspace; if non-null, posterior beliefs and decisions are shown
     *
     * Available only for finite fields of size at most 64.
     */
    void tikz_picture(std::ostream& file, const BP_Workspace* ws) const
        requires FiniteFieldType<T> && (T::get_size() <= 64)
    {
        const size_t redundancy = checks.size();
        const double width = (n > 1) ? static_cast<double>(n - 1) : 1.0;
        file << "\n\n\\begin{tikzpicture}[x=\\linewidth/" << width << ", y=1cm]";

        for (size_t j = 0; j < n; ++j) {
            if constexpr (T::get_size() == 2) {
                if (ws) {
                    file << "\n    \\node[tannervariablesoft] (v" << j << ") at (" << j << ", 0) {$" << std::fixed
                         << std::setprecision(2) << (ws->posterior[j][0] - ws->posterior[j][1]) << "$};";
                    const size_t dec = (ws->posterior[j][1] > ws->posterior[j][0]) ? 1 : 0;
                    file << "\n    \\node[anchor=north, font=\\tiny] at ($(v" << j << ")-(0,0.3)$) {$" << dec << "$};";
                    continue;
                }
            }
            file << "\n    \\node[tannervariable] (v" << j << ") at (" << j << ", 0) {};";
        }

        for (size_t i = 0; i < redundancy; ++i) {
            const double x = (redundancy > 1) ? static_cast<double>(i) * (n - 1) / (redundancy - 1) : (n - 1) / 2.0;
            file << "\n    \\node[tannercheck] (c" << i << ") at (" << x << ", 3) {};";
        }

        for (size_t i = 0; i < redundancy; ++i) {
            for (size_t e = 0; e < checks[i].size(); ++e) {
                const auto label = checks[i][e].value.get_label();
                if (label == T::get_size() - 1) {
                    file << "\n    \\path[tanneredge, draw=red]";
                } else {
                    const size_t a = 63 - 63 * static_cast<double>(label) / (T::get_size() - 1);
                    file << "\n    \\definecolor{color}{RGB}{" << static_cast<int>(details::colormap[a][0]) << ", "
                         << static_cast<int>(details::colormap[a][1]) << ", "
                         << static_cast<int>(details::colormap[a][2]) << "}\\path[tanneredge, draw=color]";
                }
                file << " (v" << checks[i][e].var_id << ") -- (c" << i << ");";
            }
        }

        if constexpr (T::get_size() == 2) {
            if (ws && ws->v.has_value()) {
                for (size_t j = 0; j < n; ++j) {
                    file << "\n    \\node[anchor=south, font=\\tiny] at ($(v" << j << ")+(0,0.3)$) {$" << std::fixed
                         << std::setprecision(2) << (*(ws->v))(0, j) << "$};";
                }
            }
        }

        file << "\n\\end{tikzpicture}\n";
    }

    /**
     * @brief Export a standalone TikZ fragment with workspace beliefs
     *
     * @param filename Output filename
     * @param ws Optional decoding workspace; if non-null, posterior beliefs and decisions are shown
     *
     * Available only for finite fields of size at most 64.
     */
    void export_as_tikz(const std::string& filename, const BP_Workspace* ws) const
        requires FiniteFieldType<T> && (T::get_size() <= 64)
    {
        std::ofstream file;
        file.open(filename);
        tikz_header(file);
        tikz_picture(file, ws);
        file.close();
    }

    /**
     * @brief Export a standalone TikZ fragment without workspace beliefs
     *
     * @param filename Output filename
     *
     * Available only for finite fields of size at most 64.
     */
    void export_as_tikz(const std::string& filename) const
        requires FiniteFieldType<T> && (T::get_size() <= 64)
    {
        export_as_tikz(filename, nullptr);
    }

    /// @brief Number of variable nodes
    size_t n;
    /// @brief Incident edges by check node; `checks[i]` lists the variables in parity check `i`
    std::vector<std::vector<details::CheckEdge<T>>> checks;
};

}  // namespace CECCO

#endif
