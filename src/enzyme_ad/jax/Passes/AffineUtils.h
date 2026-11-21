#ifndef ENZYME_JAX_PASSES_AFFINEUTILS_H_
#define ENZYME_JAX_PASSES_AFFINEUTILS_H_

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "llvm/ADT/SmallVector.h"

#include <isl/aff.h>
#include <isl/set.h>

namespace mlir {

mlir::affine::AffineValueMap getAVM(mlir::Operation *op);

/// The number of loops surrounding `op` within its affine scope: each
/// affine.for one, each affine.parallel one per dimension. These are the loops
/// the affine dependence analysis puts in an access's domain
/// (getEnclosingAffineOps stops at the scope), so `depth + 1` is the depth at
/// which to ask it about two accesses in the same iteration of them; the loops
/// around the scope, as around a kernel launched in a loop of the host, are
/// not counted, unlike by mlir::affine::getNestingDepth.
unsigned getNestingDepthInAffineScope(mlir::Operation *op);

class IslAnalysis {
public:
  std::optional<llvm::SmallVector<isl_aff *>>
  getAffExprs(mlir::Operation *op, mlir::affine::AffineValueMap avm);

  std::optional<llvm::SmallVector<isl_aff *>> getAffExprs(mlir::Operation *op);

  isl_map *getAccessMap(mlir::Operation *op);

  isl_set *getDomain(Operation *op);

  std::tuple<isl_set *, mlir::affine::FlatAffineValueConstraints>
  getDomainAndValueConstraints(Operation *op);

  isl_set *getMemrefShape(MemRefType ty);

  ~IslAnalysis();
  IslAnalysis();

  isl_ctx *getCtx() { return ctx; }

private:
  isl_ctx *ctx;
};

template <typename T> class IslScopeFree {
public:
  T obj;
  IslScopeFree(T obj) : obj(obj) {}
  ~IslScopeFree() { isl_set_free(obj); }
};

void populateAffineExprSimplificationPatterns(IslAnalysis &islAnalysis,
                                              RewritePatternSet &patterns);

} // namespace mlir

#endif // ENZYME_JAX_PASSES_AFFINEUTILS_H_
