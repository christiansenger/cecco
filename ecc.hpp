/*
 * @copyright
 * Copyright (c) 2025, Christian Senger <senger@inue.uni-stuttgart.de>
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

#include "codes.hpp"
// #include "field_concepts_traits.hpp" // transitive through codes.hpp
//  #include "helpers.hpp" // transitive through codes.hpp
//  #include "fields.hpp" // transitive through codes.hpp
//  #include "blocks.hpp" // transitive through codes.hpp
//  #include "vectors.hpp" // transitive through codes.hpp
//  #include "polynomials.hpp" // transitive through codes.hpp
//  #include "matrices.hpp" // transitive through codes.hpp

#endif