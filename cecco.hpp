/*
 * @copyright
 * Copyright (c) 2026, Christian Senger <senger@inue.uni-stuttgart.de>
 *
 * Licensed for noncommercial use only, including academic teaching, research, and personal non-profit purposes.
 * Commercial use is prohibited without a separate commercial license. See the [LICENSE](../../LICENSE) file in the
 * repository root for full terms and how to request a commercial license.
 */

#ifndef CECCO_HPP
#define CECCO_HPP

/**
 * @namespace CECCO
 * @brief Provides a framework for error correcting codes
 */
namespace CECCO {
/**
 * @namespace CECCO::details
 * @brief Contains implementation details not to be exposed to the user. Functions and classes here may change without
 * notice.
 */
namespace details {}
}  // namespace CECCO

#include "blocks.hpp"
#include "codes.hpp"
/*
// transitive
#include "code_bounds.hpp"
#include "field_concepts_traits.hpp"
#include "fields.hpp"
#include "helpers.hpp"
#include "matrices.hpp"
#include "polynomials.hpp"
#include "graphs.hpp"
#include "vectors.hpp"
*/

#endif
