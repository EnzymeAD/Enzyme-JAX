//===- ISLUtils.cpp - ISL Polyhedral & Affine Utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Polymer/Support/ISLUtils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "isl/aff.h"
#include "isl/constraint.h"
#include "isl/ctx.h"
#include "isl/local_space.h"
#include "isl/map.h"
#include "isl/mat.h"
#include "isl/set.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/val.h"

#include <map>
#include <tuple>
#include <utility>
#include <vector>

#define DEBUG_TYPE "polymer-isl-utils"

using namespace mlir;

namespace {

struct SpaceCompare {
  bool operator()(const isl::space &lhs, const isl::space &rhs) const {
    return lhs.get() < rhs.get();
  }
};

struct VertexMaps {
  std::vector<isl::map> in;
  std::vector<isl::map> intra;
  std::vector<isl::map> out;
};

static std::pair<isl::union_map, bool> path_contract(isl::union_map umap) {
  if (umap.is_null())
    return {umap, false};

  std::map<isl::space, VertexMaps, SpaceCompare> spaces;

  auto findSpace = [&](isl::space toFind) -> isl::space {
    for (const auto &kv : spaces) {
      if (toFind.is_equal(kv.first))
        return kv.first;
    }
    spaces.insert({toFind, {}});
    return toFind;
  };

  umap.foreach_map([&](isl::map map) {
    isl::space domainSpace = findSpace(map.domain().get_space());
    isl::space rangeSpace = findSpace(map.range().get_space());
    if (domainSpace.is_equal(rangeSpace)) {
      spaces[domainSpace].intra.push_back(map);
    } else {
      spaces[domainSpace].out.push_back(map);
      spaces[rangeSpace].in.push_back(map);
    }
    return isl::stat::ok();
  });

  bool foundSpaceToRemove = false;
  isl::space contractedSpace;
  isl::union_map contracted =
      isl::manage(isl_union_map_empty(umap.get_space().release()));

  for (auto &[k, v] : spaces) {
    if (v.in.empty() || v.out.empty())
      continue;

    isl::space joint = k.map_from_set();
    isl::map identity = isl::map::identity(joint);
    isl::map closure = identity;
    isl::map rclosure = identity;

    if (!v.intra.empty()) {
      isl::map intraUnion = v.intra[0];
      for (size_t idx = 1; idx < v.intra.size(); ++idx)
        intraUnion = intraUnion.unite(v.intra[idx]);

      isl::map rintra = intraUnion.reverse();
      isl_bool exact = isl_bool_false;
      isl::map tc =
          isl::manage(isl_map_transitive_closure(intraUnion.copy(), &exact));
      isl::map rtc =
          isl::manage(isl_map_transitive_closure(rintra.copy(), &exact));
      closure = tc.unite(identity);
      rclosure = rtc.unite(identity);
    }

    bool allIntraCovered = true;
    for (const auto &intra : v.intra) {
      (void)intra;
      bool exerciserExists = false;
      for (const auto &in : v.in) {
        for (const auto &out : v.out) {
          isl::set inComp = in.range().apply(closure).unite(in.range());
          isl::map rout = out.reverse();
          isl::set outComp = rout.range().apply(rclosure).unite(rout.range());
          if (inComp.is_equal(outComp)) {
            exerciserExists = true;
            break;
          }
        }
        if (exerciserExists)
          break;
      }
      allIntraCovered = allIntraCovered && exerciserExists;
    }

    if (!allIntraCovered)
      continue;

    contractedSpace = k;
    foundSpaceToRemove = true;
    for (const auto &in : v.in) {
      for (const auto &out : v.out) {
        isl::map comp = in.apply_range(closure).apply_range(out);
        contracted = contracted.unite(comp.to_union_map());
      }
    }
    break;
  }

  if (!foundSpaceToRemove)
    return {umap, false};

  umap.foreach_map([&](isl::map map) {
    isl::space domainSpace = map.domain().get_space();
    isl::space rangeSpace = map.range().get_space();
    if (domainSpace.is_equal(contractedSpace) ||
        rangeSpace.is_equal(contractedSpace))
      return isl::stat::ok();
    contracted = contracted.unite(map.to_union_map());
    return isl::stat::ok();
  });

  return {contracted, true};
}

static bool extractInt64(__isl_take isl_val *v, int64_t &out) {
  if (!v)
    return false;
  if (!isl_val_is_int(v)) {
    isl_val_free(v);
    return false;
  }
  out = isl_val_get_num_si(v);
  isl_val_free(v);
  return true;
}

} // namespace

namespace polymer {

isl::union_map get_maximal_paths(isl::union_map umap) {
  isl::union_map contracted = umap;
  bool didIt = true;
  while (didIt)
    std::tie(contracted, didIt) = path_contract(contracted);
  return contracted;
}

mlir::AffineExpr islAffToAffineExpr(__isl_keep isl_aff *aff,
                                    mlir::MLIRContext *ctx) {
  if (!aff || isl_aff_is_nan(aff) != isl_bool_false)
    return {};

  isl_val *denomVal = isl_aff_get_denominator_val(aff);
  int64_t denom = 1;
  if (!extractInt64(isl_val_copy(denomVal), denom) || denom <= 0) {
    isl_val_free(denomVal);
    return {};
  }

  auto getScaledCoeff = [&](isl_val *raw, int64_t &coeffOut) -> bool {
    raw = isl_val_mul(raw, isl_val_copy(denomVal));
    return extractInt64(raw, coeffOut);
  };

  int64_t cst = 0;
  if (!getScaledCoeff(isl_aff_get_constant_val(aff), cst)) {
    isl_val_free(denomVal);
    return {};
  }

  AffineExpr expr = getAffineConstantExpr(cst, ctx);

  isl_size nIn = isl_aff_dim(aff, isl_dim_in);
  isl_size nParam = isl_aff_dim(aff, isl_dim_param);
  isl_size nDiv = isl_aff_dim(aff, isl_dim_div);
  if (nIn < 0 || nParam < 0 || nDiv < 0) {
    isl_val_free(denomVal);
    return {};
  }

  for (int i = 0; i < nIn; ++i) {
    int64_t coeff = 0;
    if (!getScaledCoeff(isl_aff_get_coefficient_val(aff, isl_dim_in, i),
                        coeff)) {
      isl_val_free(denomVal);
      return {};
    }
    if (coeff != 0)
      expr = expr + coeff * getAffineDimExpr(i, ctx);
  }

  for (int p = 0; p < nParam; ++p) {
    int64_t coeff = 0;
    if (!getScaledCoeff(isl_aff_get_coefficient_val(aff, isl_dim_param, p),
                        coeff)) {
      isl_val_free(denomVal);
      return {};
    }
    if (coeff != 0)
      expr = expr + coeff * getAffineSymbolExpr(p, ctx);
  }

  for (int d = 0; d < nDiv; ++d) {
    int64_t coeff = 0;
    if (!getScaledCoeff(isl_aff_get_coefficient_val(aff, isl_dim_div, d),
                        coeff)) {
      isl_val_free(denomVal);
      return {};
    }
    if (coeff == 0)
      continue;
    isl_aff *divAff = isl_aff_get_div(aff, d);
    AffineExpr divExpr = islAffToAffineExpr(divAff, ctx);
    isl_aff_free(divAff);
    if (!divExpr) {
      isl_val_free(denomVal);
      return {};
    }
    expr = expr + coeff * divExpr;
  }

  isl_val_free(denomVal);
  if (denom != 1)
    expr = expr.floorDiv(denom);
  return expr;
}

mlir::AffineMap islMultiAffToAffineMap(__isl_keep isl_multi_aff *maff,
                                       mlir::MLIRContext *ctx) {
  if (!maff)
    return {};
  isl_size nIn = isl_multi_aff_dim(maff, isl_dim_in);
  isl_size nParam = isl_multi_aff_dim(maff, isl_dim_param);
  isl_size nOut = isl_multi_aff_dim(maff, isl_dim_out);
  if (nIn < 0 || nParam < 0 || nOut < 0)
    return {};

  SmallVector<AffineExpr, 4> results;
  results.reserve(nOut);
  for (int i = 0; i < nOut; ++i) {
    isl_aff *aff = isl_multi_aff_get_aff(maff, i);
    AffineExpr expr = islAffToAffineExpr(aff, ctx);
    isl_aff_free(aff);
    if (!expr)
      return {};
    results.push_back(expr);
  }
  return AffineMap::get(nIn, nParam, results, ctx);
}

mlir::IntegerSet islBasicSetToIntegerSet(__isl_keep isl_basic_set *bset,
                                         mlir::MLIRContext *ctx) {
  if (!bset)
    return {};
  isl_size nDim = isl_basic_set_dim(bset, isl_dim_set);
  isl_size nSym = isl_basic_set_dim(bset, isl_dim_param);
  if (nDim < 0 || nSym < 0)
    return {};

  isl_constraint_list *cList = isl_basic_set_get_constraint_list(bset);
  if (!cList)
    return {};
  isl_size nCst = isl_constraint_list_n_constraint(cList);
  if (nCst <= 0) {
    isl_constraint_list_free(cList);
    return {};
  }

  SmallVector<AffineExpr, 8> constraints;
  SmallVector<bool, 8> eqFlags;
  bool ok = true;

  for (int i = 0; i < nCst && ok; ++i) {
    isl_constraint *c = isl_constraint_list_get_constraint(cList, i);
    bool isEq = isl_constraint_is_equality(c) == isl_bool_true;

    int64_t cst = 0;
    if (!extractInt64(isl_constraint_get_constant_val(c), cst)) {
      isl_constraint_free(c);
      ok = false;
      break;
    }
    AffineExpr expr = getAffineConstantExpr(cst, ctx);

    for (int d = 0; d < nDim && ok; ++d) {
      int64_t coeff = 0;
      if (!extractInt64(
              isl_constraint_get_coefficient_val(c, isl_dim_set, d), coeff)) {
        ok = false;
        break;
      }
      if (coeff != 0)
        expr = expr + coeff * getAffineDimExpr(d, ctx);
    }

    for (int s = 0; s < nSym && ok; ++s) {
      int64_t coeff = 0;
      if (!extractInt64(
              isl_constraint_get_coefficient_val(c, isl_dim_param, s), coeff)) {
        ok = false;
        break;
      }
      if (coeff != 0)
        expr = expr + coeff * getAffineSymbolExpr(s, ctx);
    }

    isl_size nDiv = isl_constraint_dim(c, isl_dim_div);
    for (int dv = 0; dv < nDiv && ok; ++dv) {
      int64_t coeff = 0;
      if (!extractInt64(
              isl_constraint_get_coefficient_val(c, isl_dim_div, dv), coeff)) {
        ok = false;
        break;
      }
      if (coeff == 0)
        continue;
      isl_aff *divAff = isl_constraint_get_div(c, dv);
      AffineExpr divExpr = islAffToAffineExpr(divAff, ctx);
      isl_aff_free(divAff);
      if (!divExpr) {
        ok = false;
        break;
      }
      expr = expr + coeff * divExpr;
    }

    isl_constraint_free(c);
    if (ok) {
      constraints.push_back(expr);
      eqFlags.push_back(isEq);
    }
  }

  isl_constraint_list_free(cList);
  if (!ok || constraints.empty())
    return {};
  return IntegerSet::get(nDim, nSym, constraints, eqFlags);
}

} // namespace polymer
