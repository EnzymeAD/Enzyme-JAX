
#include "AffineUtils.h"
#include "Passes.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "llvm/ADT/ScopeExit.h"

#include <isl/aff.h>
#include <isl/aff_type.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/constraint.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/local_space.h>
#include <isl/map.h>
#include <isl/map_type.h>
#include <isl/mat.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/space_type.h>
#include <isl/val.h>
#include <optional>

extern "C" {
#include <isl_ast_build_expr.h>
}

#define DEBUG_TYPE "simplify-affine-exprs"

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

static LogicalResult addAffineIfOpDomain(AffineIfOp ifOp, bool isElse,
                                         FlatAffineValueConstraints *domain) {
  IntegerSet set = ifOp.getIntegerSet();
  // Canonicalize set and operands to ensure unique values for
  // FlatAffineValueConstraints below and for early simplification.
  SmallVector<Value> operands(ifOp.getOperands());
  canonicalizeSetAndOperands(&set, &operands);

  // Create the base constraints from the integer set attached to ifOp. This
  // fails for semi-affine sets, which cannot be flattened.
  FailureOr<FlatAffineValueConstraints> cst =
      FlatAffineValueConstraints::create(set, operands);
  if (failed(cst)) {
    LLVM_DEBUG(llvm::dbgs()
               << "semi-affine integer sets in 'affine.if' not supported\n");
    return failure();
  }

  if (!isElse) {
    domain->mergeAndAlignVarsWithOther(0, &*cst);
    domain->append(*cst);
    return success();
  }

  presburger::PresburgerRelation pr(*cst);
  pr = pr.complement();
  if (pr.getNumDisjuncts() > 1) {
    // TODO: we can turn the domain into a PresburgerSet that supports
    // disjunctions, and update the ISL lowering to handle that correctly.
    LLVM_DEBUG(llvm::dbgs()
               << "disjunctive conditions in 'else' not yet supported\n");
    return failure();
  }

  FlatLinearValueConstraints flvc(
      presburger::IntegerPolyhedron(pr.getDisjunct(0)), cst->getMaybeValues());

  domain->mergeAndAlignVarsWithOther(0, &flvc);
  domain->append(flvc);
  return success();
}

static LogicalResult getIndexSetEx(ArrayRef<Operation *> ops,
                                   ArrayRef<bool> isElse,
                                   FlatAffineValueConstraints *domain,
                                   bool allowFail = false) {
  assert(ops.size() == isElse.size() &&
         "expected co-indexed ops and isElse arrays");
  SmallVector<Value> indices;
  SmallVector<Operation *> loopOps;
  size_t numDims = 0;
  for (Operation *op : ops) {
    if (!isa<AffineForOp, AffineIfOp, AffineParallelOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "getIndexSet only handles affine.for/if/"
                                 "parallel ops");
      return failure();
    }
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      loopOps.push_back(forOp);
      // An AffineForOp retains only 1 induction variable.
      numDims += 1;
    } else if (AffineParallelOp parallelOp = dyn_cast<AffineParallelOp>(op)) {
      loopOps.push_back(parallelOp);
      numDims += parallelOp.getNumDims();
    }
  }
  extractInductionVars(loopOps, indices);
  // Reset while associating Values in 'indices' to the domain.
  *domain = FlatAffineValueConstraints(numDims, /*numSymbols=*/0,
                                       /*numLocals=*/0, indices);
  for (auto &&[op, complement] : llvm::zip(ops, isElse)) {
    // Add constraints from forOp's bounds.
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      if (failed(domain->addAffineForOpDomain(forOp)))
        return failure();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      if (failed(addAffineIfOpDomain(ifOp, complement, domain)) && !allowFail)
        return failure();
    } else if (auto parallelOp = dyn_cast<AffineParallelOp>(op))
      if (failed(domain->addAffineParallelOpDomain(parallelOp)))
        return failure();
  }
  return success();
}

std::tuple<isl_set *, FlatAffineValueConstraints>
getDomain(isl_ctx *ctx, Operation *op, bool overApproximationAllowed = false) {
  // Extract the affine for/if ops enclosing the caller and insert them into the
  // enclosingOps list.
  using EnclosingOpList = llvm::SmallVector<mlir::Operation *, 8>;
  EnclosingOpList enclosingOps;
  affine::getEnclosingAffineOps(*op, &enclosingOps);
  SmallVector<bool> isElse;
  for (auto enclosing : enclosingOps) {
    if (auto ifOp = dyn_cast<AffineIfOp>(enclosing)) {
      if (ifOp.getElseRegion().isAncestor(op->getParentRegion())) {
        isElse.push_back(true);
        continue;
      }
    }
    isElse.push_back(false);
  }
  // The domain constraints can then be collected from the enclosing ops.
  mlir::affine::FlatAffineValueConstraints cst;
  auto res = succeeded(
      getIndexSetEx(enclosingOps, isElse, &cst, overApproximationAllowed));
  if (!res)
    return {nullptr, FlatAffineValueConstraints()};

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

using PosMapTy = llvm::MapVector<unsigned, unsigned>;

struct AffineExprToIslAffConverter {
  PosMapTy dimPosMap;
  PosMapTy symPosMap;
  isl_local_space *ls;
  isl_ctx *ctx;

  isl_aff *getIslAff(AffineExpr expr) {
    if (auto bo = dyn_cast<AffineBinaryOpExpr>(expr)) {
      isl_aff *lhs = getIslAff(bo.getLHS());
      isl_aff *rhs = getIslAff(bo.getRHS());
      switch (bo.getKind()) {
      case mlir::AffineExprKind::Add:
        return isl_aff_add(lhs, rhs);
      case mlir::AffineExprKind::CeilDiv:
        return isl_aff_ceil(isl_aff_div(lhs, rhs));
      case mlir::AffineExprKind::FloorDiv:
        return isl_aff_floor(isl_aff_div(lhs, rhs));
      case mlir::AffineExprKind::Mod: {
        if (isl_aff_is_cst(rhs) == isl_bool_true) {
          isl_aff *r = isl_aff_mod_val(lhs, isl_aff_get_constant_val(rhs));
          isl_aff_free(rhs);
          return r;
        } else {
          isl_aff_free(lhs);
          isl_aff_free(rhs);
          return nullptr;
        }
      }
      case mlir::AffineExprKind::Mul:
        return isl_aff_mul(lhs, rhs);
      default:
        LLVM_DEBUG(llvm::dbgs()
                   << "Unhandled kind " << (unsigned)bo.getKind() << "\n");
        isl_aff_free(lhs);
        isl_aff_free(rhs);
        return nullptr;
      }
    } else if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
      return isl_aff_val_on_domain(isl_local_space_copy(ls),
                                   isl_val_int_from_si(ctx, c.getValue()));
    } else if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      unsigned pos = dimPosMap[dim.getPosition()];
      return isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, pos);
    } else if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) {
      unsigned pos = symPosMap[sym.getPosition()];
      return isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_param,
                                   pos);
    }
    LLVM_DEBUG(llvm::dbgs() << "Unhandled expr " << expr << "\n");
    return nullptr;
  }
};

AffineExpr internalAdd(AffineExpr LHS, AffineExpr RHS, bool allownegate = true);

// Decompose `term` into (body, coeff) such that term == body * coeff.
static std::pair<AffineExpr, int64_t> decomposeMulByConst(AffineExpr term) {
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(term))
    if (bin.getKind() == AffineExprKind::Mul)
      if (auto cst = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        return {bin.getLHS(), cst.getValue()};
  return {term, 1};
}

// Fold the implicit remainder c*e + (-c*k)*(e floordiv k) to (e mod k) * c.
// This is a pure integer identity, valid for any e and constants c, k > 1.
static std::optional<AffineExpr> tryFoldImplicitMod(AffineExpr A,
                                                    AffineExpr B) {
  for (int i = 0; i < 2; i++) {
    auto [e, c] = decomposeMulByConst(i == 0 ? A : B);
    auto [divBody, divCoeff] = decomposeMulByConst(i == 0 ? B : A);
    auto div = dyn_cast<AffineBinaryOpExpr>(divBody);
    if (!div || div.getKind() != AffineExprKind::FloorDiv)
      continue;
    auto kCst = dyn_cast<AffineConstantExpr>(div.getRHS());
    if (!kCst || kCst.getValue() < 2)
      continue;
    if (div.getLHS() != e)
      continue;
    if (divCoeff != -c * kCst.getValue())
      continue;
    return (e % kCst.getValue()) * c;
  }
  return std::nullopt;
}

AffineExpr commonAddWithMul(AffineExpr LHS, AffineExpr RHS,
                            bool allownegate = true) {
  if (auto folded = tryFoldImplicitMod(LHS, RHS))
    return *folded;
  auto lhsD = llvm::DynamicAPInt(LHS.getLargestKnownDivisor());
  auto rhsD = llvm::DynamicAPInt(RHS.getLargestKnownDivisor());
  auto gcd = llvm::int64fromDynamicAPInt(llvm::gcd(abs(lhsD), abs(rhsD)));
  SmallVector<int64_t, 2> vals;

  if (gcd != 1)
    vals.push_back(gcd);
  bool negate = false;
  for (auto v : {LHS, RHS})
    if (auto bin = dyn_cast<AffineBinaryOpExpr>(v)) {
      if (auto cst1 = dyn_cast<AffineConstantExpr>(bin.getLHS()))
        if (cst1.getValue() < 0)
          negate = true;
      if (auto cst2 = dyn_cast<AffineConstantExpr>(bin.getRHS()))
        if (cst2.getValue() < 0)
          negate = true;
    }
  if (negate && allownegate)
    vals.push_back(-gcd);

  for (auto val : vals) {
    auto LHSg = val == -1 ? (LHS * val) : LHS.floorDiv(val);
    auto RHSg = val == -1 ? (RHS * val) : RHS.floorDiv(val);
    auto add = internalAdd(LHSg, RHSg, val != -1);
    auto add2 = dyn_cast<AffineBinaryOpExpr>(add);
    if (!add2)
      return add * val;
    if (add2.getKind() != AffineExprKind::Add)
      return add * val;
    if (!((add2.getLHS() == LHSg && add2.getRHS() == RHSg) ||
          (add2.getRHS() == LHSg && add2.getLHS() == RHSg)))
      return add * val;
  }

  return LHS + RHS;
}

bool affineCmp(AffineExpr lhs, AffineExpr rhs) {
  if (isa<AffineConstantExpr>(lhs) && !isa<AffineConstantExpr>(rhs))
    return true;

  if (!isa<AffineConstantExpr>(lhs) && isa<AffineConstantExpr>(rhs))
    return false;

  if (auto L = dyn_cast<AffineConstantExpr>(lhs))
    if (auto R = dyn_cast<AffineConstantExpr>(rhs))
      return L.getValue() < R.getValue();

  if (isa<AffineSymbolExpr>(lhs) && !isa<AffineSymbolExpr>(rhs))
    return true;

  if (!isa<AffineSymbolExpr>(lhs) && isa<AffineSymbolExpr>(rhs))
    return false;

  if (auto L = dyn_cast<AffineSymbolExpr>(lhs))
    if (auto R = dyn_cast<AffineSymbolExpr>(rhs))
      return L.getPosition() < R.getPosition();

  if (isa<AffineDimExpr>(lhs) && !isa<AffineDimExpr>(rhs))
    return true;

  if (!isa<AffineDimExpr>(lhs) && isa<AffineDimExpr>(rhs))
    return false;

  if (auto L = dyn_cast<AffineDimExpr>(lhs))
    if (auto R = dyn_cast<AffineDimExpr>(rhs))
      return L.getPosition() < R.getPosition();

  auto L = cast<AffineBinaryOpExpr>(lhs);
  auto R = cast<AffineBinaryOpExpr>(rhs);
  if (affineCmp(L.getLHS(), R.getLHS()))
    return true;
  if (affineCmp(R.getLHS(), L.getLHS()))
    return false;

  if (affineCmp(L.getRHS(), R.getRHS()))
    return true;
  if (affineCmp(R.getRHS(), L.getRHS()))
    return false;
  return false;
}

SmallVector<AffineExpr> getSumOperands(AffineExpr expr) {
  SmallVector<AffineExpr> todo = {expr};
  SmallVector<AffineExpr> base;
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (auto Add = dyn_cast<AffineBinaryOpExpr>(cur))
      if (Add.getKind() == AffineExprKind::Add) {
        todo.push_back(Add.getLHS());
        todo.push_back(Add.getRHS());
        continue;
      }
    base.push_back(cur);
  }
  return base;
}

AffineExpr sortSum(AffineExpr expr) {
  auto Add = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!Add)
    return expr;
  auto exprs = getSumOperands(Add);
  llvm::sort(exprs, affineCmp);
  auto res = exprs[0];
  for (int i = 1; i < exprs.size(); i++)
    res = res + exprs[i];
  return res;
}

AffineExpr internalAdd(AffineExpr LHS, AffineExpr RHS, bool allownegate) {
  SmallVector<AffineExpr> base[2] = {getSumOperands(LHS), getSumOperands(RHS)};
  if (base[0].size() == 1 && base[1].size() == 1)
    return commonAddWithMul(LHS, RHS, allownegate);

  llvm::sort(base[0], affineCmp);
  llvm::sort(base[1], affineCmp);

  for (int i = 0; i < base[0].size(); i++)
    for (int j = 0; j < base[1].size(); j++) {
      auto fuse = commonAddWithMul(base[0][i], base[1][j]);
      bool simplified = false;
      if (auto Add = dyn_cast<AffineBinaryOpExpr>(fuse)) {
        if (Add.getLHS() == base[0][i] && Add.getRHS() == base[1][j])
          simplified = true;
        if (Add.getRHS() == base[0][i] && Add.getLHS() == base[1][j])
          simplified = true;
      }
      if (!simplified) {
        for (int i2 = 0; i2 < base[0].size(); i2++) {
          if (i != i2)
            fuse = commonAddWithMul(fuse, base[0][i2]);
        }
        for (int j2 = 0; j2 < base[1].size(); j2++) {
          if (j != j2)
            fuse = commonAddWithMul(fuse, base[1][j2]);
        }
        return fuse;
      }
    }
  return commonAddWithMul(LHS, RHS, allownegate);
}

AffineExpr mlir::enzyme::recreateExpr(AffineExpr expr) {
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    auto lhs = recreateExpr(bin.getLHS());
    auto rhs = recreateExpr(bin.getRHS());

    switch (bin.getKind()) {
    case AffineExprKind::Add:
      return internalAdd(lhs, rhs);
    case AffineExprKind::Mul:
      return sortSum(lhs) * sortSum(rhs);
    case AffineExprKind::Mod: {
      rhs = sortSum(rhs);
      SmallVector<AffineExpr> toMod;
      if (auto cst = dyn_cast<AffineConstantExpr>(rhs)) {
        for (auto expr : getSumOperands(lhs)) {
          if (!expr.isMultipleOf(cst.getValue()))
            toMod.push_back(expr);
        }
      } else {
        toMod.push_back(sortSum(lhs));
      }
      llvm::sort(toMod, affineCmp);
      AffineExpr out = getAffineConstantExpr(0, expr.getContext());
      for (auto expr : toMod)
        out = out + expr;
      out = out % rhs;
      return out;
    }
    case AffineExprKind::FloorDiv: {
      rhs = sortSum(rhs);
      SmallVector<AffineExpr> toDivide;
      SmallVector<AffineExpr> alreadyDivided;
      if (auto cst = dyn_cast<AffineConstantExpr>(rhs)) {
        for (auto expr : getSumOperands(lhs)) {
          if (expr.isMultipleOf(cst.getValue())) {
            alreadyDivided.push_back(expr.floorDiv(cst));
          } else if (auto cst2 = dyn_cast<AffineConstantExpr>(expr)) {
            if (cst2.getValue() > 0 && cst.getValue() > 0 &&
                cst2.getValue() > cst.getValue()) {
              toDivide.push_back(expr % rhs);
              alreadyDivided.push_back(expr.floorDiv(rhs));
            } else {
              toDivide.push_back(expr);
            }
          } else
            toDivide.push_back(expr);
        }
      } else {
        toDivide.push_back(sortSum(lhs));
      }
      llvm::sort(toDivide, affineCmp);
      AffineExpr out = getAffineConstantExpr(0, expr.getContext());
      for (auto expr : toDivide)
        out = out + expr;
      out = out.floorDiv(rhs);
      alreadyDivided.push_back(out);
      out = getAffineConstantExpr(0, expr.getContext());
      llvm::sort(alreadyDivided, affineCmp);
      for (auto expr : alreadyDivided)
        out = out + expr;
      return out;
    }
    default:
      return expr;
    }
  }
  return expr;
}

IntegerSet mlir::enzyme::recreateExpr(IntegerSet map) {
  SmallVector<AffineExpr> exprs;
  for (auto expr : map.getConstraints()) {
    auto expr2 = sortSum(recreateExpr(expr));
    exprs.push_back(expr2);
  }
  return IntegerSet::get(map.getNumDims(), map.getNumSymbols(), exprs,
                         map.getEqFlags());
}

AffineMap mlir::enzyme::recreateExpr(AffineMap map) {
  SmallVector<AffineExpr> exprs;
  for (auto expr : map.getResults()) {
    auto expr2 = sortSum(recreateExpr(expr));
    exprs.push_back(expr2);
  }
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), exprs,
                        map.getContext());
}

struct IslToAffineExprConverter {
  MLIRContext *mlirContext;
  unsigned symOffset;
  PosMapTy dimPosMap;
  PosMapTy symPosMap;

  AffineExpr createOpBin(__isl_take isl_ast_expr *Expr) {
    AffineExpr LHS, RHS, Res;
    isl_ast_op_type OpType;
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "isl ast expression not of type isl_ast_op");
    assert(isl_ast_expr_get_op_n_arg(Expr) == 2 &&
           "not a binary isl ast expression");

    OpType = isl_ast_expr_get_op_type(Expr);

    LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
    RHS = create(isl_ast_expr_get_op_arg(Expr, 1));

    isl_ast_expr_free(Expr);

    if (!LHS || !RHS) {
      return nullptr;
    }

    if (OpType == isl_ast_op_sub) {
      RHS = -1 * RHS;
      OpType = isl_ast_op_add;
    }
    Res = nullptr;
    switch (OpType) {
    default:
    case isl_ast_op_sub:
      llvm_unreachable("This is no binary isl ast expression");
    case isl_ast_op_add:
      Res = internalAdd(LHS, RHS);
      break;
    case isl_ast_op_mul:
      Res = (LHS * RHS);
      /*
      if (auto bin = dyn_cast<AffineBinaryOpExpr>(LHS)) {
        if (bin.getKind() == AffineExprKind::FloorDiv && bin.getRHS() == RHS) {
          Res = bin.getLHS() - (bin.getLHS() % RHS);
        }
      }
      */
      break;
    case isl_ast_op_div:
    case isl_ast_op_pdiv_q: // Dividend is non-negative
    case isl_ast_op_fdiv_q: // Round towards -infty
      if (RHS.isSymbolicOrConstant())
        Res = LHS.floorDiv(RHS);
      break;
    case isl_ast_op_pdiv_r: // Dividend is non-negative
    case isl_ast_op_zdiv_r: // Result only compared against zero
      if (RHS.isSymbolicOrConstant())
        Res = LHS % RHS;
      break;
    }
    return Res;
  }

  AffineExpr createOpUnary(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_minus &&
           "Unsupported unary operation");

    AffineExpr V = create(isl_ast_expr_get_op_arg(Expr, 0));

    isl_ast_expr_free(Expr);
    return -V;
  }

  AffineExpr createOp(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "Expression not of type isl_ast_expr_op");
    switch (isl_ast_expr_get_op_type(Expr)) {
    case isl_ast_op_error:
    case isl_ast_op_cond:
    case isl_ast_op_call:
    case isl_ast_op_member:
      break;
    case isl_ast_op_access:
      break;
    case isl_ast_op_max:
    case isl_ast_op_min:
      break;
    case isl_ast_op_add:
    case isl_ast_op_sub:
    case isl_ast_op_mul:
    case isl_ast_op_div:
    case isl_ast_op_fdiv_q: // Round towards -infty
    case isl_ast_op_pdiv_q: // Dividend is non-negative
    case isl_ast_op_pdiv_r: // Dividend is non-negative
    case isl_ast_op_zdiv_r: // Result only compared against zero
      return createOpBin(Expr);
    case isl_ast_op_minus:
      return createOpUnary(Expr);
    case isl_ast_op_select:
      break;
    case isl_ast_op_and:
    case isl_ast_op_or:
      break;
    case isl_ast_op_and_then:
    case isl_ast_op_or_else:
      break;
    case isl_ast_op_eq:
    case isl_ast_op_le:
    case isl_ast_op_lt:
    case isl_ast_op_ge:
    case isl_ast_op_gt:
      break;
    case isl_ast_op_address_of:
      break;
    }
    isl_ast_expr_free(Expr);
    return nullptr;
  }

  APInt APIntFromVal(__isl_take isl_val *Val) {
    uint64_t *Data;
    int NumChunks;
    const static int ChunkSize = sizeof(uint64_t);

    assert(isl_val_is_int(Val) && "Only integers can be converted to APInt");

    NumChunks = isl_val_n_abs_num_chunks(Val, ChunkSize);
    Data = (uint64_t *)malloc(NumChunks * ChunkSize);
    isl_val_get_abs_num_chunks(Val, ChunkSize, Data);
    int NumBits = CHAR_BIT * ChunkSize * NumChunks;
    APInt A(NumBits, NumChunks, Data);

    // As isl provides only an interface to obtain data that describes the
    // absolute value of an isl_val, A at this point always contains a positive
    // number. In case Val was originally negative, we expand the size of A by
    // one and negate the value (in two's complement representation). As a
    // result, the new value in A corresponds now with Val.
    if (isl_val_is_neg(Val)) {
      A = A.zext(A.getBitWidth() + 1);
      A = -A;
    }

    // isl may represent small numbers with more than the minimal number of
    // bits. We truncate the APInt to the minimal number of bits needed to
    // represent the signed value it contains, to ensure that the bitwidth is
    // always minimal.
    if (A.getSignificantBits() < A.getBitWidth())
      A = A.trunc(A.getSignificantBits());

    free(Data);
    isl_val_free(Val);
    return A;
  }

  AffineExpr createInt(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_int &&
           "Expression not of type isl_ast_expr_int");
    isl_val *Val;
    APInt APValue;
    Val = isl_ast_expr_get_val(Expr);
    APValue = APIntFromVal(Val);

    AffineExpr V = getAffineConstantExpr(APValue.getSExtValue(), mlirContext);
    isl_ast_expr_free(Expr);
    return V;
  }

  AffineExpr createId(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_id &&
           "Expression not of type isl_ast_expr_ident");

    isl_id *Id;
    AffineExpr V;

    Id = isl_ast_expr_get_id(Expr);

    unsigned id = (uintptr_t)isl_id_get_user(Id);
    id = id - 1;
    if (id < symOffset)
      V = getAffineDimExpr(dimPosMap[id], mlirContext);
    else
      V = getAffineSymbolExpr(symPosMap[id - symOffset], mlirContext);

    isl_id_free(Id);
    isl_ast_expr_free(Expr);

    return V;
  }

  AffineExpr create(__isl_take isl_ast_expr *Expr) {
    switch (isl_ast_expr_get_type(Expr)) {
    case isl_ast_expr_error:
      break;
    case isl_ast_expr_op:
      return createOp(Expr);
    case isl_ast_expr_int:
      return createInt(Expr);
    case isl_ast_expr_id:
      return createId(Expr);
    }
    isl_ast_expr_free(Expr);
    return nullptr;
  }
};

namespace mlir {
AffineValueMap getAVM(Operation *op) {
  if (auto cop = dyn_cast<AffineLoadOp>(op))
    return AffineValueMap(cop.getMap(), cop.getMapOperands(), {});
  else if (auto cop = dyn_cast<AffineStoreOp>(op))
    return AffineValueMap(cop.getMap(), cop.getMapOperands(), {});
  else if (auto cop = dyn_cast<AffineVectorLoadOp>(op))
    return AffineValueMap(cop.getMap(), cop.getMapOperands(), {});
  else if (auto cop = dyn_cast<AffineVectorStoreOp>(op))
    return AffineValueMap(cop.getMap(), cop.getMapOperands(), {});
  llvm_unreachable("Called with non affine op");
}
} // namespace mlir

isl_set *IslAnalysis::getMemrefShape(MemRefType ty) {
  // TODO we can support params in some cases
  if (!ty.hasStaticShape())
    return nullptr;
  isl_space *space = isl_space_set_alloc(ctx, 0, ty.getRank());
  isl_multi_aff *ma =
      isl_multi_aff_identity_on_domain_space(isl_space_copy(space));
  isl_set *set = isl_set_universe(isl_space_copy(space));
  for (unsigned i = 0; i < ty.getRank(); i++) {
    isl_aff *dim = isl_multi_aff_get_at(ma, i);
    isl_aff *lb = isl_aff_val_on_domain_space(isl_space_copy(space),
                                              isl_val_int_from_si(ctx, 0));
    isl_aff *ub = isl_aff_val_on_domain_space(
        isl_space_copy(space), isl_val_int_from_si(ctx, ty.getDimSize(i)));

    set = isl_set_intersect(set, isl_aff_ge_set(isl_aff_copy(dim), lb));
    set = isl_set_intersect(set, isl_aff_lt_set(dim, ub));
  }
  isl_space_free(space);
  isl_multi_aff_free(ma);

  return set;
}

isl_map *IslAnalysis::getAccessMap(mlir::Operation *op) {
  auto exprs = getAffExprs(op);
  if (!exprs)
    return nullptr;
  if (exprs->size() == 0)
    return nullptr;
  isl_aff_list *list = isl_aff_list_alloc(ctx, exprs->size());
  isl_space *domain = isl_space_domain(isl_aff_get_space((*exprs)[0]));
  isl_space *range = isl_space_set_alloc(ctx, 0, exprs->size());
  isl_space *space = isl_space_map_from_domain_and_range(domain, range);
  for (auto aff : *exprs) {
#ifndef NDEBUG
    isl_space *affSpace = isl_aff_get_space(aff);
    assert(isl_space_dim(affSpace, isl_dim_param) == 0 &&
           "only no-parameter aff supported currently");
    isl_space_free(affSpace);
#endif
    list = isl_aff_list_add(list, aff);
  }
  isl_multi_aff *maff = isl_multi_aff_from_aff_list(space, list);
  return isl_map_from_multi_aff(maff);
}

std::optional<SmallVector<isl_aff *>>
IslAnalysis::getAffExprs(Operation *op, AffineValueMap avm) {
  LLVM_DEBUG(llvm::dbgs() << "Got domain\n");
  auto [domain, cst] = ::getDomain(ctx, op, true);
  if (!domain)
    return std::nullopt;
  LLVM_DEBUG(isl_set_dump(domain));
  LLVM_DEBUG(cst.dump());
  AffineMap map = avm.getAffineMap();

  LLVM_DEBUG(llvm::dbgs() << "Mapping dims:\n");
  PosMapTy dimPosMap;
  PosMapTy dimPosMapReverse;
  for (unsigned i = 0; i < cst.getNumDimVars(); i++) {
    Value cstVal = cst.getValue(i);
    LLVM_DEBUG(llvm::dbgs() << "cstVal " << cstVal << "\n");
    for (unsigned origDim = 0; origDim < map.getNumDims(); origDim++) {
      Value dim = avm.getOperand(origDim);
      LLVM_DEBUG(llvm::dbgs() << "dim " << dim << "\n");
      if (cstVal == dim) {
        LLVM_DEBUG(llvm::dbgs() << origDim << " <--> " << i << "\n");
        dimPosMap[origDim] = i;
        dimPosMapReverse[i] = origDim;
        break;
      }
    }
  }

  if (avm.getNumSymbols() != 0 || cst.getNumSymbolVars() != 0) {
    // TODO While the fact that all dims from the map _must_ appear in the cst,
    // this is not the case for symbols. We do not handle that case correctly
    // currently, thus we abort early.
    domain = isl_set_free(domain);
    return std::nullopt;
  }

  LLVM_DEBUG(llvm::dbgs() << "Mapping syms:\n");
  PosMapTy symPosMap;
  PosMapTy symPosMapReverse;
  for (unsigned i = 0; i < cst.getNumSymbolVars(); i++) {
    for (unsigned origSym = 0; origSym < map.getNumSymbols(); origSym++) {
      Value dim = avm.getOperand(origSym + map.getNumDims());
      if (cst.getValue(i + cst.getNumDimVars()) == dim) {
        LLVM_DEBUG(llvm::dbgs() << origSym << " <--> " << i << "\n");
        symPosMap[origSym] = i;
        symPosMapReverse[i] = origSym;
        break;
      }
    }
  }

  isl_space *space =
      isl_space_set_alloc(ctx, cst.getNumSymbolVars(), cst.getNumDimVars());
  for (unsigned i = 0; i < cst.getNumDimVars(); i++) {
    isl_id *id = isl_id_alloc(ctx, "dim", (void *)(size_t)(i + 1));
    space = isl_space_set_dim_id(space, isl_dim_set, i, id);
  }
  unsigned symOffset = cst.getNumDimVars();
  for (unsigned i = 0; i < cst.getNumSymbolVars(); i++) {
    isl_id *id = isl_id_alloc(ctx, "sym", (void *)(size_t)(symOffset + i + 1));
    space = isl_space_set_dim_id(space, isl_dim_set, i, id);
  }

  isl_local_space *ls = isl_local_space_from_space(isl_space_copy(space));
  space = isl_space_free(space);
  AffineExprToIslAffConverter m2i{dimPosMap, symPosMap, ls, ctx};
  SmallVector<isl_aff *> affVec;
  for (unsigned i = 0; i < map.getNumResults(); i++) {
    AffineExpr mlirExpr = map.getResult(i);
    LLVM_DEBUG(llvm::dbgs() << "Handling AffineExpr\n" << mlirExpr << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Got aff\n");
    isl_aff *aff = m2i.getIslAff(mlirExpr);
    affVec.push_back(aff);
  }
  ls = isl_local_space_free(ls);
  domain = isl_set_free(domain);

  return affVec;
}

isl_set *IslAnalysis::getDomain(Operation *op) {
  auto [domain, cst] = ::getDomain(ctx, op);

  return domain;
}

std::tuple<isl_set *, FlatAffineValueConstraints>
IslAnalysis::getDomainAndValueConstraints(Operation *op) {
  return ::getDomain(ctx, op);
}

std::optional<SmallVector<isl_aff *>> IslAnalysis::getAffExprs(Operation *op) {
  return getAffExprs(op, getAVM(op));
}

IslAnalysis::IslAnalysis() {
  ctx = isl_ctx_alloc();
  [[maybe_unused]] isl_stat r =
      isl_options_set_ast_build_exploit_nested_bounds(ctx, 1);
  assert(r == isl_stat_ok);
}

IslAnalysis::~IslAnalysis() { isl_ctx_free(ctx); }

std::optional<AffineMap> handleAffineValueMap(IslAnalysis &islAnalysis,
                                              AffineValueMap avm,
                                              isl_set *domain,
                                              FlatAffineValueConstraints cst) {
  isl_ctx *ctx = islAnalysis.getCtx();
  AffineMap map = avm.getAffineMap();
  LLVM_DEBUG(llvm::dbgs() << "Mapping dims:\n");
  PosMapTy dimPosMap;
  PosMapTy dimPosMapReverse;
  for (unsigned i = 0; i < cst.getNumDimVars(); i++) {
    Value cstVal = cst.getValue(i);
    LLVM_DEBUG(llvm::dbgs() << "cstVal " << cstVal << "\n");
    for (unsigned origDim = 0; origDim < map.getNumDims(); origDim++) {
      Value dim = avm.getOperand(origDim);
      LLVM_DEBUG(llvm::dbgs() << "dim " << dim << "\n");
      if (cstVal == dim) {
        LLVM_DEBUG(llvm::dbgs() << origDim << " <--> " << i << "\n");
        dimPosMap[origDim] = i;
        dimPosMapReverse[i] = origDim;
        break;
      }
    }
  }

  if (avm.getNumSymbols() != 0 || cst.getNumSymbolVars() != 0) {
    // TODO While the fact that all dims from the map _must_ appear in the cst,
    // this is not the case for symbols. We do not handle that case correctly
    // currently, thus we abort early.
    domain = isl_set_free(domain);
    return {};
  }

  bool changed = false;

  LLVM_DEBUG(llvm::dbgs() << "Mapping syms:\n");
  PosMapTy symPosMap;
  PosMapTy symPosMapReverse;
  for (unsigned i = 0; i < cst.getNumSymbolVars(); i++) {
    for (unsigned origSym = 0; origSym < map.getNumSymbols(); origSym++) {
      Value dim = avm.getOperand(origSym + map.getNumDims());
      if (cst.getValue(i + cst.getNumDimVars()) == dim) {
        LLVM_DEBUG(llvm::dbgs() << origSym << " <--> " << i << "\n");
        symPosMap[origSym] = i;
        symPosMapReverse[i] = origSym;
        break;
      }
    }
  }

  isl_space *space =
      isl_space_set_alloc(ctx, cst.getNumSymbolVars(), cst.getNumDimVars());
  for (unsigned i = 0; i < cst.getNumDimVars(); i++) {
    isl_id *id = isl_id_alloc(ctx, "dim", (void *)(size_t)(i + 1));
    space = isl_space_set_dim_id(space, isl_dim_set, i, id);
  }
  unsigned symOffset = cst.getNumDimVars();
  for (unsigned i = 0; i < cst.getNumSymbolVars(); i++) {
    isl_id *id = isl_id_alloc(ctx, "sym", (void *)(size_t)(symOffset + i + 1));
    space = isl_space_set_dim_id(space, isl_dim_set, i, id);
  }

  isl_ast_build *build =
      isl_ast_build_from_context(isl_set_universe(isl_space_copy(space)));
  isl_local_space *ls = isl_local_space_from_space(isl_space_copy(space));
  space = isl_space_free(space);
  AffineExprToIslAffConverter m2i{dimPosMap, symPosMap, ls, ctx};
  IslToAffineExprConverter i2m{map.getContext(), symOffset, dimPosMapReverse,
                               symPosMapReverse};
  SmallVector<AffineExpr> newExprs;
  for (unsigned i = 0; i < map.getNumResults(); i++) {
    AffineExpr mlirExpr = map.getResult(i);
    LLVM_DEBUG(llvm::dbgs() << "Handling AffineExpr\n" << mlirExpr << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Got aff\n");
    isl_aff *aff = m2i.getIslAff(mlirExpr);
    LLVM_DEBUG(isl_aff_dump(aff));
    aff = isl_aff_gist(aff, isl_set_copy(domain));
    LLVM_DEBUG(llvm::dbgs() << "Gisted aff\n");
    LLVM_DEBUG(isl_aff_dump(aff));
    isl_ast_expr *expr = isl_ast_expr_from_aff(aff, build);
    LLVM_DEBUG(llvm::dbgs() << "ast expr\n");
    LLVM_DEBUG(isl_ast_expr_dump(expr));
    LLVM_DEBUG(llvm::dbgs() << "Back to AffineExpr\n");
    AffineExpr newMlirExpr = i2m.create(expr);
    LLVM_DEBUG(llvm::dbgs() << newMlirExpr << "\n");
    newExprs.push_back(newMlirExpr);
    if (mlirExpr != newMlirExpr)
      changed = true;
  }
  ls = isl_local_space_free(ls);
  domain = isl_set_free(domain);
  build = isl_ast_build_free(build);

  if (!changed)
    return std::nullopt;

  AffineMap newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                    newExprs, map.getContext());
  newMap = mlir::enzyme::recreateExpr(newMap);

  if (map == newMap)
    return {};

  return newMap;
}

template <typename T>
LogicalResult handleAffineAccessOp(IslAnalysis &islAnalysis, T access) {
  isl_ctx *ctx = islAnalysis.getCtx();
  LLVM_DEBUG(llvm::dbgs() << "Got domain\n");
  auto [domain, cst] = ::getDomain(ctx, access, true);
  if (!domain)
    return failure();
  LLVM_DEBUG(isl_set_dump(domain));
  LLVM_DEBUG(cst.dump());
  AffineMap map = access.getMap();
  AffineValueMap avm(map, access.getMapOperands(), {});

  auto newMap = handleAffineValueMap(islAnalysis, avm, domain, cst);
  if (!newMap)
    return failure();

  access.setMap(*newMap);
  return success();
}

// Simplifies the condition of `ifOp` against its domain and rebuilds its
// constraints in the pass's canonical form, which applies even where the domain
// is unavailable.
LogicalResult handleAffineIfOp(IslAnalysis &islAnalysis, AffineIfOp ifOp) {
  isl_ctx *ctx = islAnalysis.getCtx();
  IntegerSet set = ifOp.getCondition();
  IntegerSet newSet = set;
  LLVM_DEBUG(llvm::dbgs() << "Got domain\n");
  auto [domain, cst] = ::getDomain(ctx, ifOp, true);
  if (domain) {
    LLVM_DEBUG(isl_set_dump(domain));
    LLVM_DEBUG(cst.dump());
    auto csts = set.getConstraints();
    AffineMap map = AffineMap::get(set.getNumDims(), set.getNumSymbols(), csts,
                                   ifOp.getContext());
    AffineValueMap avm(map, ifOp.getOperands(), {});
    if (auto newMap = handleAffineValueMap(islAnalysis, avm, domain, cst))
      newSet = IntegerSet::get(set.getNumDims(), set.getNumSymbols(),
                               newMap->getResults(), set.getEqFlags());
  }
  newSet = mlir::enzyme::recreateExpr(newSet);
  if (newSet == set)
    return failure();
  ifOp.setCondition(newSet);
  return success();
}

// Drop bound expressions another expression makes redundant: an upper bound
// min(a, b) where b >= a for every value of the dims and symbols is min(a),
// and a lower bound max is its dual. Two bound operands often say the same
// thing through different arithmetic -- a grid counted in blocks against the
// size it was computed from -- so the map is first composed through the
// arithmetic defining its symbol operands, down to shared base values, before
// asking isl whether the comparison can ever fail. The composition is
// affine-cfg's, and reads the arithmetic the way affine-cfg read it when it
// wrote the bounds: casts preserve the value, sums and products do not wrap,
// and signed division is floordiv.

// Returns `map` with every expression another expression of its group
// subsumes dropped, with the groups' new sizes; std::nullopt when all stay.
static std::optional<std::pair<AffineMap, SmallVector<int32_t>>>
pruneBoundMap(__isl_keep isl_ctx *ctx, AffineMap map, ValueRange operands,
              DenseIntElementsAttr groups, bool isUpper, Region *scope) {
  AffineMap expanded = map;
  SmallVector<Value> expandedOperands(operands);
  if (!fully2ComposeAffineMapAndOperands(&expanded, &expandedOperands, scope,
                                         /*throughSymbols=*/true))
    return std::nullopt;

  unsigned numDims = expanded.getNumDims();
  unsigned numSyms = expanded.getNumSymbols();
  isl_space *space = isl_space_set_alloc(ctx, numSyms, numDims);
  for (unsigned i : llvm::seq(numDims))
    space =
        isl_space_set_dim_id(space, isl_dim_set, i,
                             isl_id_alloc(ctx, "dim", (void *)(size_t)(i + 1)));
  for (unsigned i : llvm::seq(numSyms))
    space =
        isl_space_set_dim_id(space, isl_dim_param, i,
                             isl_id_alloc(ctx, "sym", (void *)(size_t)(i + 1)));
  AffineExprToIslAffConverter conv{
      {}, {}, isl_local_space_from_space(space), ctx};
  for (unsigned i : llvm::seq(numDims))
    conv.dimPosMap[i] = i;
  for (unsigned i : llvm::seq(numSyms))
    conv.symPosMap[i] = i;

  SmallVector<AffineExpr> keptExprs;
  SmallVector<int32_t> keptGroups;
  bool changed = false;
  unsigned start = 0;
  for (unsigned size : groups.getValues<uint32_t>()) {
    ArrayRef<AffineExpr> group = map.getResults().slice(start, size);
    SmallVector<isl_aff *> affs;
    for (AffineExpr expr : expanded.getResults().slice(start, size))
      affs.push_back(conv.getIslAff(expr));
    start += size;
    SmallVector<bool> kept(size, true);
    for (auto [j, affJ] : llvm::enumerate(affs)) {
      if (!affJ)
        continue;
      for (auto [i, affI] : llvm::enumerate(affs)) {
        if (i == j || !kept[i] || !affI)
          continue;
        // Upper: j is redundant where e_j >= e_i everywhere, i.e. the set
        // e_j < e_i is empty. Lower: the dual.
        isl_set *fails =
            isUpper ? isl_aff_lt_set(isl_aff_copy(affJ), isl_aff_copy(affI))
                    : isl_aff_gt_set(isl_aff_copy(affJ), isl_aff_copy(affI));
        bool redundant = isl_set_is_empty(fails) == isl_bool_true;
        isl_set_free(fails);
        if (redundant) {
          kept[j] = false;
          break;
        }
      }
    }
    for (isl_aff *aff : affs)
      isl_aff_free(aff);
    unsigned keptCount = 0;
    for (auto [j, expr] : llvm::enumerate(group)) {
      if (!kept[j])
        continue;
      keptExprs.push_back(expr);
      keptCount++;
    }
    keptGroups.push_back(keptCount);
    changed |= keptCount != size;
  }
  isl_local_space_free(conv.ls);
  if (!changed)
    return std::nullopt;
  return std::make_pair(AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                       keptExprs, map.getContext()),
                        std::move(keptGroups));
}

LogicalResult pruneParallelBounds(IslAnalysis &islAnalysis,
                                  affine::AffineParallelOp op,
                                  RewriterBase &rewriter) {
  Region *scope = getLocalAffineScope(op);
  if (!scope)
    return failure();
  isl_ctx *ctx = islAnalysis.getCtx();
  auto lower =
      pruneBoundMap(ctx, op.getLowerBoundsMap(), op.getLowerBoundsOperands(),
                    op.getLowerBoundsGroups(), /*isUpper=*/false, scope);
  auto upper =
      pruneBoundMap(ctx, op.getUpperBoundsMap(), op.getUpperBoundsOperands(),
                    op.getUpperBoundsGroups(), /*isUpper=*/true, scope);
  if (!lower && !upper)
    return failure();
  rewriter.modifyOpInPlace(op, [&] {
    if (lower) {
      op.setLowerBoundsMapAttr(AffineMapAttr::get(lower->first));
      op.setLowerBoundsGroupsAttr(rewriter.getI32TensorAttr(lower->second));
    }
    if (upper) {
      op.setUpperBoundsMapAttr(AffineMapAttr::get(upper->first));
      op.setUpperBoundsGroupsAttr(rewriter.getI32TensorAttr(upper->second));
    }
  });
  return success();
}

// The domain of the affine loop and conditional structure around an
// operation, for the folds below.
//
// A domain is the `getDomain` set above (the enclosing affine.for/parallel
// bounds and affine.if integer sets, conditions that do not convert dropped):
// one set dimension per induction variable, one parameter per symbol the
// bounds and conditions depend on. On top of that, every symbol is composed
// through the arithmetic defining it, as `pruneBoundMap` composes bound
// operands, and equated with the composition; the values composition stops at
// are further parameters. A queried value composes the same way, so
// `blockIdx + gridDim` compared against an extent whose bound shares
// `gridDim`'s base meets the bound on `blockIdx` in the same parameters.
//
// Built per query, as the access simplifications above build theirs: the
// patterns run under drivers with and without a listener.
namespace {
class LoopDomain {
public:
  // The domain of the operations in `op`'s block; false when no affine op
  // encloses the block or its structure does not convert.
  LoopDomain(isl_ctx *ctx, Operation *op);
  ~LoopDomain() { isl_set_free(set); }
  LoopDomain(const LoopDomain &) = delete;
  LoopDomain &operator=(const LoopDomain &) = delete;
  explicit operator bool() const { return set != nullptr; }

  // The results of `map` on `operands` as expressions over the dimensions and
  // parameters of the domain, each operand composed through the arithmetic
  // defining it; a value composition stops at that is not an induction
  // variable or a known parameter becomes a new parameter. A value keyed in
  // `substitution` reads as the value it maps to. Fails when the composition
  // could not run to completion.
  FailureOr<SmallVector<AffineExpr>>
  compose(AffineMap map, ValueRange operands,
          const DenseMap<Value, Value> *substitution = nullptr);

  // `exprs` over the dimensions and parameters of the domain as isl functions.
  // Fails when some expression has no affine form (a modulo by a
  // non-constant). The caller frees the functions.
  FailureOr<SmallVector<isl_aff *>> getAffs(ArrayRef<AffineExpr> exprs);

  // `values` composed as `compose` describes, as isl functions.
  FailureOr<SmallVector<isl_aff *>>
  getAffs(ArrayRef<Value> values,
          const DenseMap<Value, Value> *substitution = nullptr) {
    SmallVector<AffineExpr> exprs;
    for (Value value : values) {
      FailureOr<AffineExpr> expr = compose(value, substitution);
      if (failed(expr))
        return failure();
      exprs.push_back(*expr);
    }
    return getAffs(exprs);
  }

  // Whether `other` has no point in the domain.
  bool emptyOnDomain(__isl_take isl_set *other) const {
    isl_set *inDomain = isl_set_intersect(isl_set_copy(set), other);
    bool empty = isl_set_is_empty(inDomain) == isl_bool_true;
    isl_set_free(inDomain);
    return empty;
  }

  // Whether `aff` is non-negative on every point of the domain.
  bool nonNegOnDomain(__isl_keep isl_aff *aff) const {
    isl_aff *zero = isl_aff_val_on_domain(isl_aff_get_domain_local_space(aff),
                                          isl_val_zero(ctx));
    return emptyOnDomain(isl_aff_lt_set(isl_aff_copy(aff), zero));
  }

  // Set dimensions are the induction variables, parameters the symbols and
  // the values compositions stopped at.
  isl_set *set = nullptr;

private:
  // `value` composed as `compose` describes.
  FailureOr<AffineExpr> compose(Value value,
                                const DenseMap<Value, Value> *substitution) {
    AffineMap map =
        AffineMap::get(0, 1, getAffineSymbolExpr(0, value.getContext()));
    FailureOr<SmallVector<AffineExpr>> exprs =
        compose(map, value, substitution);
    if (failed(exprs))
      return failure();
    return (*exprs)[0];
  }

  isl_ctx *ctx;
  // Induction variable to set dimension.
  DenseMap<Value, unsigned> dimPos;
  // Symbol or leaf value to parameter.
  DenseMap<Value, unsigned> paramPos;
  // The affine scope compositions stay within.
  Region *scope = nullptr;
};

LoopDomain::LoopDomain(isl_ctx *ctx, Operation *op) : ctx(ctx) {
  SmallVector<Operation *> enclosing;
  affine::getEnclosingAffineOps(*op, &enclosing);
  scope = getLocalAffineScope(op);
  if (enclosing.empty() || !scope)
    return;
  auto [domain, cst] = ::getDomain(ctx, op, /*overApproximationAllowed=*/true);
  if (!domain)
    return;
  set = domain;

  unsigned numDims = cst.getNumDimVars();
  for (unsigned pos : llvm::seq(numDims))
    dimPos[cst.getValue(pos)] = pos;
  for (auto [pos, symbol] :
       llvm::enumerate(cst.getMaybeValues(presburger::VarKind::Symbol)))
    if (symbol)
      paramPos[*symbol] = pos;

  // Tie each symbol to what it composes to. A symbol composition stops at, or
  // fails on, is its own parameter and gets no tie. Compose everything before
  // converting: compositions add parameters, and the conversion works in the
  // final space.
  SmallVector<std::pair<Value, unsigned>> symbols(paramPos.begin(),
                                                  paramPos.end());
  SmallVector<std::pair<unsigned, AffineExpr>> symbolExprs;
  for (auto [symbol, pos] : symbols)
    if (FailureOr<AffineExpr> expr = compose(symbol, /*substitution=*/nullptr);
        succeeded(expr))
      symbolExprs.emplace_back(pos, *expr);
  isl_local_space *localSpace =
      isl_local_space_from_space(isl_set_get_space(set));
  llvm::scope_exit freeLocalSpace([&] { isl_local_space_free(localSpace); });
  AffineExprToIslAffConverter converter{{}, {}, localSpace, ctx};
  for (unsigned pos : llvm::seq(numDims))
    converter.dimPosMap[pos] = pos;
  for (unsigned pos : llvm::seq(isl_set_dim(set, isl_dim_param)))
    converter.symPosMap[pos] = pos;
  for (auto [pos, expr] : symbolExprs) {
    if (expr == getAffineSymbolExpr(pos, expr.getContext()))
      continue;
    isl_aff *composedAff = converter.getIslAff(expr);
    if (!composedAff)
      continue;
    isl_aff *symbolAff = isl_aff_var_on_domain(isl_local_space_copy(localSpace),
                                               isl_dim_param, pos);
    set = isl_set_intersect(set, isl_aff_eq_set(symbolAff, composedAff));
  }
}

FailureOr<SmallVector<AffineExpr>>
LoopDomain::compose(AffineMap map, ValueRange operands,
                    const DenseMap<Value, Value> *substitution) {
  MLIRContext *mlirCtx = map.getContext();
  SmallVector<Value> composedOperands(operands);
  // Without a rewriter the composition stops at an operand it would have to
  // hoist (see composeAffineMapAndOperands).
  if (!fully2ComposeAffineMapAndOperands(&map, &composedOperands, scope,
                                         /*throughSymbols=*/true))
    return failure();
  SmallVector<AffineExpr> dimReplacements, symbolReplacements;
  for (auto [index, operand] : llvm::enumerate(composedOperands)) {
    AffineExpr replacement;
    auto dim = dimPos.find(operand);
    if (dim != dimPos.end()) {
      replacement = getAffineDimExpr(dim->second, mlirCtx);
    } else if (substitution && substitution->contains(operand)) {
      FailureOr<AffineExpr> composed =
          compose(substitution->lookup(operand), substitution);
      if (failed(composed))
        return failure();
      replacement = *composed;
    } else {
      auto param = paramPos.find(operand);
      if (param == paramPos.end()) {
        unsigned pos = isl_set_dim(set, isl_dim_param);
        set = isl_set_add_dims(set, isl_dim_param, 1);
        param = paramPos.try_emplace(operand, pos).first;
      }
      replacement = getAffineSymbolExpr(param->second, mlirCtx);
    }
    (index < map.getNumDims() ? dimReplacements : symbolReplacements)
        .push_back(replacement);
  }
  SmallVector<AffineExpr> exprs;
  for (AffineExpr result : map.getResults())
    exprs.push_back(
        result.replaceDimsAndSymbols(dimReplacements, symbolReplacements));
  return exprs;
}

FailureOr<SmallVector<isl_aff *>>
LoopDomain::getAffs(ArrayRef<AffineExpr> exprs) {
  isl_local_space *localSpace =
      isl_local_space_from_space(isl_set_get_space(set));
  llvm::scope_exit freeLocalSpace([&] { isl_local_space_free(localSpace); });
  AffineExprToIslAffConverter converter{{}, {}, localSpace, ctx};
  for (unsigned pos : llvm::seq(isl_set_dim(set, isl_dim_set)))
    converter.dimPosMap[pos] = pos;
  for (unsigned pos : llvm::seq(isl_set_dim(set, isl_dim_param)))
    converter.symPosMap[pos] = pos;
  SmallVector<isl_aff *> affs;
  for (AffineExpr expr : exprs)
    affs.push_back(converter.getIslAff(expr));
  if (llvm::is_contained(affs, nullptr)) {
    for (isl_aff *aff : affs)
      isl_aff_free(aff);
    return failure();
  }
  return affs;
}

// The points where `lhs pred rhs` holds; isl compares integers, so an
// unsigned predicate fails.
FailureOr<isl_set *> comparisonSet(arith::CmpIPredicate pred,
                                   __isl_take isl_aff *lhs,
                                   __isl_take isl_aff *rhs) {
  using Pred = arith::CmpIPredicate;
  switch (pred) {
  case Pred::eq:
    return isl_aff_eq_set(lhs, rhs);
  case Pred::ne:
    return isl_aff_ne_set(lhs, rhs);
  case Pred::slt:
    return isl_aff_lt_set(lhs, rhs);
  case Pred::sle:
    return isl_aff_le_set(lhs, rhs);
  case Pred::sgt:
    return isl_aff_gt_set(lhs, rhs);
  case Pred::sge:
    return isl_aff_ge_set(lhs, rhs);
  case Pred::ult:
  case Pred::ule:
  case Pred::ugt:
  case Pred::uge:
    break;
  }
  isl_aff_free(lhs);
  isl_aff_free(rhs);
  return failure();
}

// Collapse a min/max whose order the enclosing loop bounds already decide
// (maxsi(tid + 2, 2) under tid >= 0), so a loop bound built from it is
// affine and the loop raises.
template <typename MinMaxOp>
struct FoldMinMaxUsingLoopBounds : public OpRewritePattern<MinMaxOp> {
  IslAnalysis &islAnalysis;
  FoldMinMaxUsingLoopBounds(MLIRContext &context, IslAnalysis &islAnalysis)
      : OpRewritePattern<MinMaxOp>(&context), islAnalysis(islAnalysis) {}

  LogicalResult matchAndRewrite(MinMaxOp op,
                                PatternRewriter &rewriter) const override {
    bool isMax = std::is_same_v<MinMaxOp, arith::MaxSIOp> ||
                 std::is_same_v<MinMaxOp, arith::MaxUIOp>;
    bool isUnsigned = std::is_same_v<MinMaxOp, arith::MaxUIOp> ||
                      std::is_same_v<MinMaxOp, arith::MinUIOp>;
    Value lhs = op.getLhs(), rhs = op.getRhs();
    LoopDomain domain(islAnalysis.getCtx(), op);
    if (!domain)
      return failure();
    FailureOr<SmallVector<isl_aff *>> affs = domain.getAffs({lhs, rhs});
    if (failed(affs))
      return failure();
    isl_aff *lhsAff = (*affs)[0], *rhsAff = (*affs)[1];
    llvm::scope_exit freeAffs([&] {
      isl_aff_free(lhsAff);
      isl_aff_free(rhsAff);
    });
    if (isUnsigned &&
        !(domain.nonNegOnDomain(lhsAff) && domain.nonNegOnDomain(rhsAff)))
      return failure();
    bool lhsGeRhs = domain.emptyOnDomain(
        isl_aff_lt_set(isl_aff_copy(lhsAff), isl_aff_copy(rhsAff)));
    bool rhsGeLhs =
        !lhsGeRhs && domain.emptyOnDomain(isl_aff_lt_set(isl_aff_copy(rhsAff),
                                                         isl_aff_copy(lhsAff)));
    if (!lhsGeRhs && !rhsGeLhs)
      return failure();
    rewriter.replaceOp(op, (lhsGeRhs == isMax) ? lhs : rhs);
    return success();
  }
};

// The value `cmp` has on every point of `domain`, its operands read through
// `substitution`; fails when the domain does not decide it. Values are
// mathematical integers (no wraparound); an unsigned predicate is only decided
// when both sides are provably non-negative on the domain.
FailureOr<bool>
decideOnDomain(LoopDomain &domain, arith::CmpIOp cmp,
               const DenseMap<Value, Value> *substitution = nullptr) {
  arith::CmpIPredicate pred = cmp.getPredicate();
  FailureOr<SmallVector<isl_aff *>> affs =
      domain.getAffs({cmp.getLhs(), cmp.getRhs()}, substitution);
  if (failed(affs))
    return failure();
  isl_aff *lhsAff = (*affs)[0], *rhsAff = (*affs)[1];
  llvm::scope_exit freeAffs([&] {
    isl_aff_free(lhsAff);
    isl_aff_free(rhsAff);
  });
  if (mlir::enzyme::isUnsignedPredicate(pred)) {
    if (!(domain.nonNegOnDomain(lhsAff) && domain.nonNegOnDomain(rhsAff)))
      return failure();
    pred = mlir::enzyme::signedPredicate(pred);
  }

  FailureOr<isl_set *> fails = comparisonSet(
      arith::invertPredicate(pred), isl_aff_copy(lhsAff), isl_aff_copy(rhsAff));
  FailureOr<isl_set *> holds =
      comparisonSet(pred, isl_aff_copy(lhsAff), isl_aff_copy(rhsAff));
  if (failed(fails) || failed(holds)) {
    if (succeeded(fails))
      isl_set_free(*fails);
    if (succeeded(holds))
      isl_set_free(*holds);
    return failure();
  }
  if (domain.emptyOnDomain(*fails)) {
    isl_set_free(*holds);
    return true;
  }
  if (domain.emptyOnDomain(*holds))
    return false;
  return failure();
}

// Fold an integer comparison to a constant when the enclosing affine loop
// bounds already decide it — e.g. a peeled grid-stride residual's guard
// (blockIdx + gridDim compared against an extent sharing gridDim's base) or
// a thread-id test under a constant-extent axis.
struct FoldCmpUsingLoopBounds : public OpRewritePattern<arith::CmpIOp> {
  IslAnalysis &islAnalysis;
  FoldCmpUsingLoopBounds(MLIRContext &context, IslAnalysis &islAnalysis)
      : OpRewritePattern<arith::CmpIOp>(&context), islAnalysis(islAnalysis) {}

  LogicalResult matchAndRewrite(arith::CmpIOp cmp,
                                PatternRewriter &rewriter) const override {
    LoopDomain domain(islAnalysis.getCtx(), cmp);
    if (!domain)
      return failure();
    FailureOr<bool> decided = decideOnDomain(domain, cmp);
    if (failed(decided))
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        cmp, rewriter.getBoolAttr(*decided));
    return success();
  }
};

// An scf.while's condition is evaluated first with the inits as the before
// arguments, and only evaluated again if it held. So when the enclosing loop
// bounds show it false on the inits it is false whenever evaluated, and the
// while is one execution of its before region (InlineNeverLoopingWhile). This
// is how a rotated do-while remainder loop that the propagated block size
// made single-shot disappears.
struct FoldNeverLoopingSCFWhileCondition
    : public OpRewritePattern<scf::WhileOp> {
  IslAnalysis &islAnalysis;
  FoldNeverLoopingSCFWhileCondition(MLIRContext &context,
                                    IslAnalysis &islAnalysis)
      : OpRewritePattern<scf::WhileOp>(&context), islAnalysis(islAnalysis) {}

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp condOp = whileOp.getConditionOp();
    auto cmp = condOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      return failure();
    LoopDomain domain(islAnalysis.getCtx(), whileOp);
    if (!domain)
      return failure();
    DenseMap<Value, Value> initOfArg;
    for (auto [arg, init] :
         llvm::zip(whileOp.getBeforeArguments(), whileOp.getInits()))
      initOfArg[arg] = init;
    FailureOr<bool> decided = decideOnDomain(domain, cmp, &initOfArg);
    // Holding on the first evaluation says nothing about the next.
    if (failed(decided) || *decided)
      return failure();
    rewriter.setInsertionPoint(condOp);
    Value falseValue = arith::ConstantOp::create(rewriter, condOp.getLoc(),
                                                 rewriter.getBoolAttr(false));
    rewriter.modifyOpInPlace(
        condOp, [&] { condOp.getConditionMutable().assign(falseValue); });
    return success();
  }
};

} // namespace

struct SimplifyAffineExprsPass
    : public enzyme::impl::SimplifyAffineExprsPassBase<
          SimplifyAffineExprsPass> {
  using SimplifyAffineExprsPassBase::SimplifyAffineExprsPassBase;
  void runOnOperation() override {
    IslAnalysis ia;

    Operation *op = getOperation();
    // Constants stay where they are.
    RewritePatternSet patterns(op->getContext());
    populateAffineExprSimplificationPatterns(ia, patterns);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
      signalPassFailure();
  }
};

template <class T>
struct SimplifyAccessAffineExprs : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  IslAnalysis &islAnalysis;
  SimplifyAccessAffineExprs(MLIRContext &context, IslAnalysis &islAnalysis)
      : OpRewritePattern<T>(&context), islAnalysis(islAnalysis) {}
  LogicalResult matchAndRewrite(T access,
                                PatternRewriter &rewriter) const override {
    return handleAffineAccessOp(islAnalysis, access);
  }
};

struct SimplifyIfAffineExprs : public OpRewritePattern<AffineIfOp> {
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;
  IslAnalysis &islAnalysis;
  SimplifyIfAffineExprs(MLIRContext &context, IslAnalysis &islAnalysis)
      : OpRewritePattern<AffineIfOp>(&context), islAnalysis(islAnalysis) {}
  LogicalResult matchAndRewrite(AffineIfOp op,
                                PatternRewriter &rewriter) const override {
    return handleAffineIfOp(islAnalysis, op);
  }
};

struct PruneParallelBounds : public OpRewritePattern<AffineParallelOp> {
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;
  IslAnalysis &islAnalysis;
  PruneParallelBounds(MLIRContext &context, IslAnalysis &islAnalysis)
      : OpRewritePattern<AffineParallelOp>(&context), islAnalysis(islAnalysis) {
  }
  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    return pruneParallelBounds(islAnalysis, op, rewriter);
  }
};

void mlir::populateAffineExprSimplificationPatterns(
    IslAnalysis &islAnalysis, RewritePatternSet &patterns) {
  // clang-format off
  patterns.insert<
    SimplifyAccessAffineExprs<affine::AffineLoadOp>,
    SimplifyAccessAffineExprs<affine::AffineStoreOp>,
    SimplifyAccessAffineExprs<affine::AffineVectorLoadOp>,
    SimplifyAccessAffineExprs<affine::AffineVectorStoreOp>,
    SimplifyIfAffineExprs,
    PruneParallelBounds,
    FoldCmpUsingLoopBounds,
    FoldMinMaxUsingLoopBounds<arith::MaxSIOp>,
    FoldMinMaxUsingLoopBounds<arith::MinSIOp>,
    FoldMinMaxUsingLoopBounds<arith::MaxUIOp>,
    FoldMinMaxUsingLoopBounds<arith::MinUIOp>,
    FoldNeverLoopingSCFWhileCondition
  >(*patterns.getContext(), islAnalysis);
  // clang-format on
  mlir::enzyme::populateInlineNeverLoopingWhilePattern(patterns);
}
