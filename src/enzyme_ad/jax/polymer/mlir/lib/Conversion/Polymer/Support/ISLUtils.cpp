//===- ISLUtils.cpp - ISL Polyhedral & Affine Utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Polymer/Support/ISLUtils.h"

#include "isl/map.h"
#include "isl/union_map.h"

#include <map>
#include <tuple>
#include <utility>
#include <vector>

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

} // namespace

namespace polymer {

isl::union_map get_maximal_paths(isl::union_map umap) {
  isl::union_map contracted = umap;
  bool didIt = true;
  while (didIt)
    std::tie(contracted, didIt) = path_contract(contracted);
  return contracted;
}

} // namespace polymer
