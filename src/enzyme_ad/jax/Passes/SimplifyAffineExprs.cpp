
#include "Passes.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IntegerSet.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include <isl/aff.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/constraint.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/local_space.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/space_type.h>
#include <isl/val.h>

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

AffineExpr commonAddWithMul(AffineExpr LHS, AffineExpr RHS,
                            bool allownegate = true) {
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

AffineExpr internalAdd(AffineExpr LHS, AffineExpr RHS, bool allownegate) {
  SmallVector<AffineExpr> todo[2];
  todo[0] = {LHS};
  todo[1] = {RHS};
  SmallVector<AffineExpr> base[2];
  for (int i = 0; i < 2; i++)
    while (!todo[i].empty()) {
      auto cur = todo[i].pop_back_val();
      if (auto Add = dyn_cast<AffineBinaryOpExpr>(cur))
        if (Add.getKind() == AffineExprKind::Add) {
          todo[i].push_back(Add.getLHS());
          todo[i].push_back(Add.getRHS());
          continue;
        }
      base[i].push_back(cur);
    }
  if (base[0].size() == 1 && base[1].size() == 1)
    return commonAddWithMul(LHS, RHS, allownegate);
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

AffineExpr recreateExpr(AffineExpr expr) {
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr)) {
    auto lhs = recreateExpr(bin.getLHS());
    auto rhs = recreateExpr(bin.getRHS());

    switch (bin.getKind()) {
    case AffineExprKind::Add:
      return internalAdd(lhs, rhs);
    case AffineExprKind::Mul:
      return lhs * rhs;
    case AffineExprKind::Mod:
      return lhs % rhs;
    case AffineExprKind::FloorDiv:
      return lhs.floorDiv(rhs);
    case AffineExprKind::CeilDiv:
      return lhs.ceilDiv(rhs);
    default:
      return expr;
    }
  }
  return expr;
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

template <typename T> void handleAffineOp(isl_ctx *ctx, T load) {
  LLVM_DEBUG(llvm::dbgs() << "Got domain\n");
  auto [domain, cst] = getDomain(ctx, load);
  LLVM_DEBUG(isl_set_dump(domain));
  LLVM_DEBUG(cst.dump());
  AffineMap map = load.getMap();
  AffineValueMap avm(map, load.getMapOperands(), {});

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
    return;
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

  isl_ast_build *build =
      isl_ast_build_from_context(isl_set_universe(isl_space_copy(space)));
  isl_local_space *ls = isl_local_space_from_space(isl_space_copy(space));
  space = isl_space_free(space);
  AffineExprToIslAffConverter m2i{dimPosMap, symPosMap, ls, ctx};
  IslToAffineExprConverter i2m{load->getContext(), symOffset, dimPosMapReverse,
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
  }
  ls = isl_local_space_free(ls);
  domain = isl_set_free(domain);
  build = isl_ast_build_free(build);

  AffineMap newMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                    newExprs, load->getContext());
  load.setMap(newMap);
}

struct SimplifyAffineExprsPass
    : public enzyme::impl::SimplifyAffineExprsPassBase<
          SimplifyAffineExprsPass> {
  using SimplifyAffineExprsPassBase::SimplifyAffineExprsPassBase;
  void runOnOperation() override {
    isl_ctx *ctx = isl_ctx_alloc();
    auto r = isl_options_set_ast_build_exploit_nested_bounds(ctx, 1);
    if (r != isl_stat_ok) {
      signalPassFailure();
      return;
    }

    Operation *op = getOperation();
    op->walk([&](Operation *op) {
      if (auto cop = dyn_cast<AffineLoadOp>(op))
        handleAffineOp(ctx, cop);
      else if (auto cop = dyn_cast<AffineStoreOp>(op))
        handleAffineOp(ctx, cop);
      else if (auto cop = dyn_cast<AffineVectorLoadOp>(op))
        handleAffineOp(ctx, cop);
      else if (auto cop = dyn_cast<AffineVectorStoreOp>(op))
        handleAffineOp(ctx, cop);
    });
    isl_ctx_free(ctx);

    op->walk([=](AffineIfOp affineOp) {
      auto map = affineOp.getIntegerSet();
      bool changed = false;
      SmallVector<AffineExpr> exprs;
      for (auto expr : map.getConstraints()) {
        auto expr2 = recreateExpr(expr);
        changed |= (expr != expr2);
        exprs.push_back(expr2);
      }
      if (changed)
        affineOp.setIntegerSet(IntegerSet::get(
            map.getNumDims(), map.getNumSymbols(), exprs, map.getEqFlags()));
    });
  }
};
