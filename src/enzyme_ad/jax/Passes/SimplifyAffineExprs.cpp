
#include "Passes.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include <isl/aff.h>
#include <isl/constraint.h>
#include <isl/ctx.h>
#include <isl/local_space.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/space_type.h>
#include <isl/val.h>

#define DEBUG_TYPE "llvm-to-memref-access"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SIMPLIFYAFFINEEXPRSPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace affine;

isl_mat *createConstraintRows(isl_ctx *ctx,
                              affine::FlatAffineValueConstraints &cst,
                              bool isEq) {
  unsigned numRows = isEq ? cst.getNumEqualities() : cst.getNumInequalities();
  unsigned numDimIds = cst.getNumDimVars();
  unsigned numLocalIds = cst.getNumLocalVars();
  unsigned numSymbolIds = cst.getNumSymbolVars();

  LLVM_DEBUG(llvm::dbgs() << "createConstraintRows " << numRows << " "
                          << numDimIds << " " << numLocalIds << " "
                          << numSymbolIds << "\n");

  unsigned numCols = cst.getNumCols();
  isl_mat *mat = isl_mat_alloc(ctx, numRows, numCols);

  for (unsigned i = 0; i < numRows; i++) {
    // Get the row based on isEq.
    auto row = isEq ? cst.getEquality(i) : cst.getInequality(i);

    assert(row.size() == numCols);

    // Dims stay at the same positions.
    for (unsigned j = 0; j < numDimIds; j++)
      mat = isl_mat_set_element_si(mat, i, j, (int64_t)row[j]);
    // Output local ids before symbols.
    for (unsigned j = 0; j < numLocalIds; j++)
      mat = isl_mat_set_element_si(mat, i, j + numDimIds,
                                   (int64_t)row[j + numDimIds + numSymbolIds]);
    // Output symbols in the end.
    for (unsigned j = 0; j < numSymbolIds; j++)
      mat = isl_mat_set_element_si(mat, i, j + numDimIds + numLocalIds,
                                   (int64_t)row[j + numDimIds]);
    // Finally outputs the constant.
    mat =
        isl_mat_set_element_si(mat, i, numCols - 1, (int64_t)row[numCols - 1]);
  }
  return mat;
}

std::tuple<isl_set *, FlatAffineValueConstraints> getDomain(isl_ctx *ctx,
                                                            Operation *op) {
  // Extract the affine for/if ops enclosing the caller and insert them into the
  // enclosingOps list.
  using EnclosingOpList = llvm::SmallVector<mlir::Operation *, 8>;
  EnclosingOpList enclosingOps;
  affine::getEnclosingAffineOps(*op, &enclosingOps);

  // The domain constraints can then be collected from the enclosing ops.
  mlir::affine::FlatAffineValueConstraints cst;
  auto res = succeeded(getIndexSet(enclosingOps, &cst));
  assert(res);

  // Symbol values, which could be a BlockArgument, or the result of DimOp or
  // IndexCastOp, or even an affine.apply. Here we limit the cases to be either
  // BlockArgument or IndexCastOp, and if it is an IndexCastOp, the cast source
  // should be a top-level BlockArgument.
  SmallVector<mlir::Value, 8> symValues;
  llvm::DenseMap<mlir::Value, mlir::Value> symMap;
  cst.getValues(cst.getNumDimVars(), cst.getNumDimAndSymbolVars(), &symValues);
  SmallVector<int64_t, 8> eqs, inEqs;
  isl_mat *eqMat = createConstraintRows(ctx, cst, /*isEq=*/true);
  isl_mat *ineqMat = createConstraintRows(ctx, cst, /*isEq=*/false);
  LLVM_DEBUG({
    llvm::dbgs() << "Adding domain relation\n";
    llvm::dbgs() << " ISL eq mat:\n";
    isl_mat_dump(eqMat);
    llvm::dbgs() << " ISL ineq mat:\n";
    isl_mat_dump(ineqMat);
    llvm::dbgs() << "\n";
  });

  isl_space *space =
      isl_space_set_alloc(ctx, cst.getNumSymbolVars(), cst.getNumDimVars());
  LLVM_DEBUG(llvm::dbgs() << "space: ");
  LLVM_DEBUG(isl_space_dump(space));
  return {isl_set_from_basic_set(isl_basic_set_from_constraint_matrices(
              space, eqMat, ineqMat, isl_dim_set, isl_dim_div, isl_dim_param,
              isl_dim_cst)),
          cst};
}

isl_aff *getIslAff(AffineExpr expr, isl_local_space *ls, isl_ctx *ctx,
                   llvm::MapVector<unsigned, unsigned> posMap) {
  if (auto bo = dyn_cast<AffineBinaryOpExpr>(expr)) {
    isl_aff *lhs = getIslAff(bo.getLHS(), ls, ctx, posMap);
    isl_aff *rhs = getIslAff(bo.getRHS(), ls, ctx, posMap);
    switch (bo.getKind()) {
    case mlir::AffineExprKind::Add:
      return isl_aff_add(lhs, rhs);
    case mlir::AffineExprKind::CeilDiv:
      return isl_aff_ceil(isl_aff_div(lhs, rhs));
    case mlir::AffineExprKind::FloorDiv:
      return isl_aff_floor(isl_aff_div(lhs, rhs));
    case mlir::AffineExprKind::Mod:
      return isl_aff_mod_val(lhs, isl_aff_get_constant_val(rhs));
    case mlir::AffineExprKind::Mul:
      return isl_aff_mul(lhs, rhs);
    default:
      LLVM_DEBUG(llvm::dbgs()
                 << "Unhandled kind " << (unsigned)bo.getKind() << "\n");
      return nullptr;
    }
  } else if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
    return isl_aff_val_on_domain(isl_local_space_copy(ls),
                                 isl_val_int_from_si(ctx, c.getValue()));
  } else if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
    unsigned pos = posMap[dim.getPosition()];
    return isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, pos);
  } else if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) {
    unsigned pos = posMap[dim.getPosition()];
    return isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_param, pos);
  }
  LLVM_DEBUG(llvm::dbgs() << "Unhandled expr " << expr << "\n");
  return nullptr;
}

struct SimplifyAffineExprsPass
    : public enzyme::impl::SimplifyAffineExprsPassBase<
          SimplifyAffineExprsPass> {
  using SimplifyAffineExprsPassBase::SimplifyAffineExprsPassBase;
  void runOnOperation() override {
    isl_ctx *ctx = isl_ctx_alloc();

    Operation *op = getOperation();

    op->walk([&](AffineLoadOp load) {
      LLVM_DEBUG(llvm::dbgs() << "Got domain\n");
      auto [domain, cst] = getDomain(ctx, load);
      isl_set_dump(domain);
      AffineMap map = load.getMap();
      AffineValueMap avm(map, load.getOperands(), {});
      llvm::MapVector<unsigned, unsigned> posMap;
      for (unsigned i = 0; i < cst.getNumVars(); i++) {
        for (unsigned origDim = 0; origDim < map.getNumInputs(); origDim++) {
          Value dim = avm.getOperand(origDim);
          if (cst.getValue(i) == dim) {
            posMap[origDim] = i;
            break;
          }
        }
      }

      isl_space *space =
          isl_space_set_alloc(ctx, cst.getNumSymbolVars(), cst.getNumDimVars());
      for (unsigned i = 0; i < map.getNumResults(); i++) {
        AffineExpr expr = map.getResult(i);
        LLVM_DEBUG(llvm::dbgs() << "Handling AffineExpr\n" << expr << "\n");
        isl_local_space *ls = isl_local_space_from_space(isl_space_copy(space));
        LLVM_DEBUG(llvm::dbgs() << "Got aff\n");
        isl_aff *aff = getIslAff(expr, ls, ctx, posMap);
        isl_aff_dump(aff);
        ls = isl_local_space_free(ls);
        LLVM_DEBUG(llvm::dbgs() << "Gisted aff\n");
        aff = isl_aff_gist(aff, isl_set_copy(domain));
        isl_aff_dump(aff);
      }
      space = isl_space_free(space);
      domain = isl_set_free(domain);
    });
  }
};
