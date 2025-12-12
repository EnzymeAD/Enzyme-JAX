//===- IslScop.cc -----------------------------------------------*- C++ -*-===//

#include "mlir/Conversion/Polymer/Support/IslScop.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Conversion/Polymer/Support/ScatteringUtils.h"
#include "mlir/Conversion/Polymer/Support/ScopStmt.h"
#include "mlir/Conversion/Polymer/Target/ISL.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"

#include "isl/aff_type.h"
#include "isl/ast.h"
#include "isl/ast_type.h"
#include "isl/ctx.h"
#include "isl/id_to_id.h"
#include "isl/isl-noexceptions.h"
#include "isl/map_type.h"
#include "isl/printer.h"
#include "isl/space_type.h"
#include <isl/aff.h>
#include <isl/ast_build.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/map.h>
#include <isl/mat.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>
#include <memory>

using namespace polymer;
using namespace mlir;

using llvm::dbgs;
using llvm::errs;
using llvm::formatv;

#define DEBUG_TYPE "islscop"

static void replace(std::string &str, StringRef find, StringRef replace) {
  size_t pos = 0;
  while ((pos = str.find(find, pos)) != std::string::npos) {
    str.replace(pos, find.size(), replace);
    pos += replace.size();
  }
}

namespace polymer {
void makeIslCompatible(std::string &str) {
  replace(str, ".", "_");
  replace(str, "\"", "_");
  replace(str, " ", "__");
  replace(str, "=>", "TO");
  replace(str, "+", "_");
}
} // namespace polymer
using polymer::makeIslCompatible;

IslScop::IslScop(Operation *op) {
  IslCtx.reset(isl_ctx_alloc(), isl_ctx_free);
  root = op;
}

IslScop::~IslScop() { isl_schedule_free(schedule); }

void IslScop::addContextRelation(affine::FlatAffineValueConstraints cst) {
  // Project out the dim IDs in the context with only the symbol IDs left.
  SmallVector<mlir::Value, 8> dimValues;
  cst.getValues(0, cst.getNumDimVars(), &dimValues);
  for (mlir::Value dimValue : dimValues)
    cst.projectOut(dimValue);
  if (cst.getNumDimAndSymbolVars() > 0)
    cst.removeIndependentConstraints(0, cst.getNumDimAndSymbolVars());
}

namespace {
struct IslStr {
  char *s;
  char *str() { return s; }
  IslStr(char *s) : s(s) {}
  ~IslStr() { free(s); }
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, IslStr s) {
  return os << s.str();
}
} // namespace

inline void islAssert(const isl_size &size) { assert(size != isl_size_error); }
inline unsigned unsignedFromIslSize(const isl_size &size) {
  islAssert(size);
  return static_cast<unsigned>(size);
}

#define POLYMER_ISL_DEBUG(S, X)                                                \
  LLVM_DEBUG({                                                                 \
    llvm::dbgs() << S;                                                         \
    X;                                                                         \
    llvm::dbgs() << "\n";                                                      \
  })

static __isl_give isl_union_pw_multi_aff *
mapToDimensionUPMA(__isl_take isl_union_set *uset, unsigned N) {
  assert(!isl_union_set_is_empty(uset));
  N += 1;

  isl_union_pw_multi_aff *res =
      isl_union_pw_multi_aff_empty(isl_union_set_get_space(uset));
  isl_set_list *bsetlist = isl_union_set_get_set_list(uset);
  for (unsigned i = 0; i < unsignedFromIslSize(isl_set_list_size(bsetlist));
       i++) {
    isl_set *set = isl_set_list_get_at(bsetlist, i);
    unsigned Dim = unsignedFromIslSize(isl_set_dim(set, isl_dim_set));
    assert(Dim >= N);
    auto pma = isl_pw_multi_aff_project_out_map(isl_set_get_space(set),
                                                isl_dim_set, N, Dim - N);
    isl_set_free(set);

    if (N > 1)
      pma = isl_pw_multi_aff_drop_dims(pma, isl_dim_out, 0, N - 1);
    res = isl_union_pw_multi_aff_add_pw_multi_aff(res, pma);
  }

  isl_set_list_free(bsetlist);
  isl_union_set_free(uset);

  return res;
}

static __isl_give isl_multi_union_pw_aff *
mapToDimensionMUPA(__isl_take isl_union_set *uset, unsigned N) {
  return isl_multi_union_pw_aff_from_union_pw_multi_aff(
      mapToDimensionUPMA(uset, N));
}

static bool isMark(isl_id *id, StringRef mark) {
  return mark.str() == isl_id_get_name(id);
}

static constexpr char parallelLoopMark[] = "parallel";
static isl_id *getParallelLoopMark(isl_ctx *ctx) {
  isl_id *loopMark = isl_id_alloc(ctx, parallelLoopMark, nullptr);
  return loopMark;
}
static bool isParallelLoopMark(isl_id *id) {
  return std::string(parallelLoopMark) == isl_id_get_name(id);
}

static isl_schedule *markPermutable(isl_schedule *schedule) {
  isl_schedule_node *node = isl_schedule_get_root(schedule);
  schedule = isl_schedule_free(schedule);
  node = isl_schedule_node_first_child(node);
  node = isl_schedule_node_band_set_permutable(node, 1);
  // node = isl_schedule_node_insert_mark(node, getParallelLoopMark(ctx));
  schedule = isl_schedule_node_get_schedule(node);
  isl_schedule_node_free(node);
  return schedule;
}

isl_schedule *
IslScop::buildParallelSchedule(affine::AffineParallelOp parallelOp,
                               unsigned depth) {
  assert(parallelOp->getNumResults() == 0 && "no parallel reductions");
  isl_schedule *schedule =
      buildLoopSchedule(parallelOp, depth, parallelOp.getNumDims(), true);
  return schedule;
}

template <typename T>
isl_schedule *IslScop::buildLoopSchedule(T loopOp, unsigned depth,
                                         unsigned numDims, bool permutable) {
  SmallVector<Operation *> body = getSequenceScheduleOpList(loopOp.getBody());

  isl_schedule *child = buildSequenceSchedule(body, depth + numDims);
  POLYMER_ISL_DEBUG("CHILD:\n", isl_schedule_dump(child));
  isl_schedule *schedule = child;
  for (unsigned dim = 0; dim < numDims; dim++) {
    isl_union_set *domain = isl_schedule_get_domain(schedule);
    POLYMER_ISL_DEBUG("MUPA dom: ", isl_union_set_dump(domain));
    isl_multi_union_pw_aff *mupa =
        mapToDimensionMUPA(domain, depth + numDims - dim - 1);
    std::string name = ("L" + std::to_string(loopId++) + "." +
                        loopOp->getName().getStringRef().str());
    makeIslCompatible(name);
    mupa =
        isl_multi_union_pw_aff_set_tuple_name(mupa, isl_dim_set, name.c_str());
    POLYMER_ISL_DEBUG("MUPA: ", isl_multi_union_pw_aff_dump(mupa));
    schedule = isl_schedule_insert_partial_schedule(schedule, mupa);
    if (permutable)
      schedule = markPermutable(schedule);
  }

  POLYMER_ISL_DEBUG("Created loop schedule:\n", isl_schedule_dump(schedule));

  return schedule;
}

isl_schedule *IslScop::buildForSchedule(affine::AffineForOp forOp,
                                        unsigned depth) {
  isl_schedule *schedule = buildLoopSchedule(forOp, depth, 1, false);
  return schedule;
}

StringRef getStmtName(Operation *op) {
  return cast<StringAttr>(op->getAttr("polymer.stmt.name")).getValue();
}

isl_schedule *IslScop::buildLeafSchedule(Operation *op) {
  // TODO check that we are really calling a statement
  auto &stmt = getIslStmt(op);
  isl_schedule *schedule = isl_schedule_from_domain(
      isl_union_set_from_set(isl_set_copy(stmt.islDomain)));
  LLVM_DEBUG({
    llvm::errs() << "Created leaf schedule:\n";
    isl_schedule_dump(schedule);
    llvm::errs() << "\n";
  });
  return schedule;
}

SmallVector<Operation *> IslScop::getSequenceScheduleOpList(Operation *begin,
                                                            Operation *end) {
  SmallVector<Operation *> ops;
  for (auto op = begin; op != end; op = op->getNextNode()) {
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
      auto thenOps = getSequenceScheduleOpList(ifOp.getThenBlock());
      if (ifOp.hasElse()) {
        auto elseOps = getSequenceScheduleOpList(ifOp.getElseBlock());
        ops.insert(ops.end(), elseOps.begin(), elseOps.end());
      }
      ops.insert(ops.end(), thenOps.begin(), thenOps.end());
    } else {
      ops.push_back(op);
    }
  }
  return ops;
}

SmallVector<Operation *> IslScop::getSequenceScheduleOpList(Block *block) {
  return getSequenceScheduleOpList(&block->front(), nullptr);
}

isl_schedule *IslScop::buildSequenceSchedule(SmallVector<Operation *> ops,
                                             unsigned depth) {
  auto buildOpSchedule = [&](Operation *op) {
    if (op->getAttr("polymer.stmt.name")) {
      return buildLeafSchedule(op);
    } else if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      return buildForSchedule(forOp, depth);
    } else if (auto parallelOp = dyn_cast<affine::AffineParallelOp>(op)) {
      return buildParallelSchedule(parallelOp, depth);
    } else if (auto alloca = dyn_cast<memref::AllocaOp>(op)) {
      return (isl_schedule *)nullptr;
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      llvm_unreachable("??");
      // return buildLeafSchedule(callOp);
    } else {
      assert(isMemoryEffectFree(op));
      return (isl_schedule *)nullptr;
    }
  };

  auto len = ops.size();
  if (len == 1)
    return buildOpSchedule(ops[0]);

  isl_schedule *schedule = nullptr;
  for (auto curOp : ops) {
    isl_schedule *child = buildOpSchedule(curOp);
    if (!child)
      continue;
    if (!schedule)
      schedule = child;
    else
      schedule = isl_schedule_sequence(schedule, child);
  }
  assert(schedule);

  LLVM_DEBUG({
    llvm::errs() << "Created sequence schedule:\n";
    isl_schedule_dump(schedule);
    llvm::errs() << "\n";
  });

  return schedule;
}

ScopStmt &IslScop::getIslStmt(llvm::StringRef name) {
  for (auto &stmt : stmts)
    if (name == stmt.name)
      return stmt;
  llvm_unreachable("stmt not found");
}
ScopStmt &IslScop::getIslStmt(Operation *op) {
  for (auto &stmt : stmts)
    if (op == stmt.op)
      return stmt;
  llvm_unreachable("stmt not found");
}

void IslScop::dumpAccesses(llvm::raw_ostream &os) {
  auto o = [&os](unsigned n) -> llvm::raw_ostream & {
    return os << std::string(n, ' ');
  };
  isl_union_set *domain = isl_schedule_get_domain(schedule);
  o(0) << "domain: \"" << IslStr(isl_union_set_to_str(domain)) << "\"\n";
  domain = isl_union_set_free(domain);
  o(0) << "accesses:\n";
  for (auto &stmt : *this) {
    o(2) << "- " << stmt.name << ":"
         << "\n";
    for (MemoryAccess *MA : stmt) {
      std::string type;
      if (MA->isRead())
        type = "read";
      else if (MA->isMustWrite())
        type = "must_write";
      else if (MA->isMayWrite())
        type = "may_write";
      else if (MA->isKill())
        type = "kill";
      assert(type != "");
      o(8) << "- " << type << " " << '"'
           << IslStr(isl_map_to_str(MA->AccessRelation.get())) << '"' << "\n";
    }
  }
}
void IslScop::dumpSchedule(llvm::raw_ostream &os) {
  LLVM_DEBUG(llvm::errs() << "Dumping islexternal\n");
  LLVM_DEBUG(llvm::errs() << "Schedule:\n\n");
  LLVM_DEBUG(isl_schedule_dump(schedule));
  LLVM_DEBUG(llvm::errs() << "\n");

  os << IslStr(isl_schedule_to_str(schedule)) << "\n";
}

isl_space *IslScop::setupSpace(isl_space *space,
                               affine::FlatAffineValueConstraints &cst,
                               isl::id tupleId) {
  for (unsigned i = 0; i < cst.getNumSymbolVars(); i++) {
    Value val =
        cst.getValue(cst.getVarKindOffset(presburger::VarKind::Symbol) + i);
    isl::id id = valueTable[val];
    space = isl_space_set_dim_id(space, isl_dim_param, i, id.copy());
  }
  if (!tupleId.is_null())
    space = isl_space_set_tuple_id(space, isl_dim_set, tupleId.copy());
  return space;
}

// adapted from `pet_scop_set_independent`
void IslScop::addIndependences() {
  if (stmts.empty())
    return;
  // FIXME we need the param space here - perhaps in the future we may not have
  // all the params on all stmts
  this->independence = isl::manage(
      isl_union_map_empty(isl_set_get_space(stmts.front().islDomain)));
  for (auto &stmt : *this) {
    isl_set *domain = isl_set_copy(stmt.islDomain);
    unsigned totalDims = unsignedFromIslSize(isl_set_dim(domain, isl_dim_set));
    LLVM_DEBUG(dbgs() << "Adding independence for stmt with domain ";
               isl_set_dump(domain));
    for (unsigned dim = 0; dim < totalDims;) {
      unsigned bandStart = dim, bandEnd = dim;
      affine::AffineParallelOp parOp = nullptr;
      for (; bandEnd < totalDims; bandEnd++) {
        auto curPar = dyn_cast<affine::AffineParallelOp>(stmt.getMlirDomain()
                                                             ->getValue(bandEnd)
                                                             .getParentBlock()
                                                             ->getParentOp());
        if (!parOp)
          parOp = curPar;
        else if (parOp != curPar)
          break;
      }
      if (bandStart == bandEnd) {
        dim++;
        continue;
      } else {
        dim = bandEnd;
      }

      // TODO collect all arrays in the dim and add that info to the scop
      isl_union_set *local = nullptr;

      isl_space *space;
      isl_map *map;
      isl_union_map *independence;
      isl_union_pw_multi_aff *proj;

      assert(domain);

      space = isl_space_map_from_set(isl_set_get_space(domain));
      map = isl_map_universe(space);
      for (unsigned i = 0; i < bandStart; ++i)
        map = isl_map_equate(map, isl_dim_in, i, isl_dim_out, i);
      for (unsigned i = bandEnd; i < totalDims; ++i)
        map = isl_map_equate(map, isl_dim_in, i, isl_dim_out, i);
      isl_map *eq = isl_map_copy(map);
      for (unsigned i = bandStart; i < bandEnd; ++i)
        eq = isl_map_equate(eq, isl_dim_in, i, isl_dim_out, i);
      map = isl_map_subtract(map, eq);

      independence = isl_union_map_from_map(map);
      space = isl_space_params(isl_set_get_space(domain));
      {
        proj = isl_union_pw_multi_aff_empty(space);
        isl_space *space;
        isl_multi_aff *ma;
        isl_pw_multi_aff *pma;
        space = isl_set_get_space(stmt.getDomain().release());
        int dim;
        dim = isl_space_dim(space, isl_dim_set);
        ma = isl_multi_aff_project_out_map(space, isl_dim_set, dim,
                                           totalDims - dim);
        ma = isl_multi_aff_set_tuple_id(ma, isl_dim_out,
                                        isl_set_get_tuple_id(domain));
        pma = isl_pw_multi_aff_from_multi_aff(ma);
        proj = isl_union_pw_multi_aff_add_pw_multi_aff(proj, pma);
      }
      // proj = outer_projection(scop, space, dim);
      independence = isl_union_map_preimage_domain_union_pw_multi_aff(
          independence, isl_union_pw_multi_aff_copy(proj));
      independence =
          isl_union_map_preimage_range_union_pw_multi_aff(independence, proj);

      LLVM_DEBUG(dbgs() << "Independence for band " << bandStart << " "
                        << bandEnd << " ";
                 isl_union_map_dump(independence));

      this->independence = this->independence.unite(isl::manage(independence));
    }
  }
  LLVM_DEBUG(dbgs() << "Independence "; polly::dumpIslObj(this->independence));
}

void IslScop::addDomainRelation(ScopStmt &stmt,
                                affine::FlatAffineValueConstraints &cst) {
  SmallVector<int64_t, 8> eqs, inEqs;
  isl_mat *eqMat = createConstraintRows(cst, /*isEq=*/true);
  isl_mat *ineqMat = createConstraintRows(cst, /*isEq=*/false);
  LLVM_DEBUG({
    llvm::errs() << "Adding domain relation\n";
    llvm::errs() << " ISL eq mat:\n";
    isl_mat_dump(eqMat);
    llvm::errs() << " ISL ineq mat:\n";
    isl_mat_dump(ineqMat);
    llvm::errs() << "\n";
  });

  isl_space *space = isl_space_set_alloc(getIslCtx(), cst.getNumSymbolVars(),
                                         cst.getNumDimVars());
  space = setupSpace(space, cst, stmt.id);
  LLVM_DEBUG(llvm::errs() << "space: ");
  LLVM_DEBUG(isl_space_dump(space));
  stmt.islDomain =
      isl_set_from_basic_set(isl_basic_set_from_constraint_matrices(
          space, eqMat, ineqMat, isl_dim_set, isl_dim_div, isl_dim_param,
          isl_dim_cst));
  LLVM_DEBUG(llvm::errs() << "bset: ");
  LLVM_DEBUG(isl_set_dump(stmt.islDomain));
  assert((int)cst.getNumDimVars() == isl_set_dim(stmt.islDomain, isl_dim_set));
}

LogicalResult
IslScop::addAccessRelation(ScopStmt &stmt, MemoryAccess::AccessType type,
                           mlir::Value memref, affine::AffineValueMap &vMap,
                           bool universe,
                           affine::FlatAffineValueConstraints &domain) {
  affine::FlatAffineValueConstraints cst;
  isl_map *map = nullptr;

  unsigned rank = 0;
  if (auto ty = dyn_cast<MemRefType>(memref.getType()))
    rank = ty.getRank();

  assert(universe || vMap.getNumResults() == rank);

  isl_space *as =
      isl_space_set_alloc(getIslCtx(), domain.getNumSymbolVars(), rank);
  as = setupSpace(as, domain, isl::id());
  ScopArrayInfo &ai = getOrAddArray(isl::manage(as), memref);
  isl::id arrayId = ai.id;
  isl::space arraySpace = ai.space;

  if (universe) {
    isl_set *range = isl_space_universe_set(arraySpace.copy());
    map = isl_map_from_domain_and_range(isl_set_copy(stmt.islDomain), range);
  } else if (createAccessRelationConstraints(vMap, cst, domain).failed()) {
    LLVM_DEBUG(llvm::dbgs() << "createAccessRelationConstraints failed\n");

    // Conservatively act on the entire array
    isl_set *range = isl_space_universe_set(arraySpace.copy());
    map = isl_map_from_domain_and_range(isl_set_copy(stmt.islDomain), range);

    if (type == MemoryAccess::AccessType::MUST_WRITE) {
      // If we could not get exact relation, we need to downgrade to a may write
      type = MemoryAccess::AccessType::MAY_WRITE;
    } else if (type == MemoryAccess::AccessType::KILL) {
      // May kills are useless
      return failure();
    }
  } else {
    isl_space *space = isl_space_alloc(
        getIslCtx(), cst.getNumSymbolVars(),
        cst.getNumDimVars() - vMap.getNumResults(), vMap.getNumResults());
    space = setupSpace(space, cst, valueTable[memref]);

    isl_mat *eqMat = createConstraintRows(cst, /*isEq=*/true);
    isl_mat *ineqMat = createConstraintRows(cst, /*isEq=*/false);

    LLVM_DEBUG({
      llvm::errs() << "Adding access relation\n";
      dbgs() << "Resolved MLIR access constraints:\n";
      cst.dump();
      llvm::errs() << " ISL eq mat:\n";
      isl_mat_dump(eqMat);
      llvm::errs() << " ISL ineq mat:\n";
      isl_mat_dump(ineqMat);
      llvm::errs() << "\n";
    });

    assert(cst.getNumInequalities() == 0);
    isl_basic_map *bmap;
    bmap = isl_basic_map_from_constraint_matrices(
        space, eqMat, ineqMat, isl_dim_out, isl_dim_in, isl_dim_div,
        isl_dim_param, isl_dim_cst);
    map = isl_map_from_basic_map(bmap);
  }

  isl::id stmtId = stmt.getDomain().get_tuple_id();
  assert(stmtId.get() == stmt.id.get());
  static constexpr unsigned ArrayLen = 5;
  static const std::array<std::string, ArrayLen> TypeStrings = {
      "", "_Read", "_Write", "_MayWrite", "_Kill"};
  static_assert(ArrayLen == MemoryAccess::AccessType_LAST + 1);
  std::string Access = TypeStrings[type] + llvm::utostr(stmt.size());
  Access = stmt.getName() + Access;
  makeIslCompatible(Access);
  isl::id accessId =
      isl::id::alloc(getIslCtx(), Access, memref.getAsOpaquePointer());

  map = isl_map_set_tuple_id(map, isl_dim_out, arrayId.copy());
  map = isl_map_set_tuple_id(map, isl_dim_in, stmtId.copy());
  POLYMER_ISL_DEBUG("Created relation: ", isl_map_dump(map));
  stmt.memoryAccesses.push_back(new MemoryAccess{
      accessId, isl::manage(map), MemoryAccess::MT_Array, type, &ai});

  return success();
}

void IslScop::initializeSymbolTable(Operation *f,
                                    affine::FlatAffineValueConstraints *cst) {
  symbolTable.clear();

  // Setup the symbol table.
  for (unsigned i = 0; i < cst->getNumDimAndSymbolVars(); i++) {
    Value val = cst->getValue(i);
    std::string sym;
    switch (cst->getVarKindAt(i)) {
    case presburger::VarKind::Domain:
      sym = "I";
      break;
    case presburger::VarKind::Local:
      sym = "O";
      break;
    case presburger::VarKind::Symbol:
      sym = "P";
      break;
    case presburger::VarKind::Range:
      sym = "R";
      break;
    }
    sym += std::to_string(i - cst->getVarKindOffset(cst->getVarKindAt(i)));
    // symbolTable.insert(std::make_pair(sym, val));
    // valueTable.insert(std::make_pair(val, sym));
    makeIslCompatible(sym);
    valueTable[val] = isl::manage(
        isl_id_alloc(IslCtx.get(), sym.c_str(), val.getAsOpaquePointer()));
  }
  for (const auto &it : memRefIdMap) {
    std::string sym(formatv("A{0}", it.second));
    symbolTable.insert(std::make_pair(sym, it.first));
    valueTable.insert(std::make_pair(
        it.first, isl::manage(isl_id_alloc(IslCtx.get(), sym.c_str(),
                                           it.first.getAsOpaquePointer()))));
  }
  // constants
  unsigned numConstants = 0;
  for (mlir::Value arg : f->getRegion(0).begin()->getArguments()) {
    if (valueTable.find(arg) == valueTable.end()) {
      std::string sym(formatv("C{0}", numConstants++));
      symbolTable.insert(std::make_pair(sym, arg));
      valueTable.insert(std::make_pair(
          arg, isl::manage(isl_id_alloc(IslCtx.get(), sym.c_str(),
                                        arg.getAsOpaquePointer()))));
    }
  }
}

isl_mat *IslScop::createConstraintRows(affine::FlatAffineValueConstraints &cst,
                                       bool isEq) {
  unsigned numRows = isEq ? cst.getNumEqualities() : cst.getNumInequalities();
  unsigned numDimIds = cst.getNumDimVars();
  unsigned numLocalIds = cst.getNumLocalVars();
  unsigned numSymbolIds = cst.getNumSymbolVars();

  LLVM_DEBUG(llvm::errs() << "createConstraintRows " << numRows << " "
                          << numDimIds << " " << numLocalIds << " "
                          << numSymbolIds << "\n");

  unsigned numCols = cst.getNumCols();
  isl_mat *mat = isl_mat_alloc(getIslCtx(), numRows, numCols);

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

LogicalResult IslScop::createAccessRelationConstraints(
    mlir::affine::AffineValueMap &vMap,
    mlir::affine::FlatAffineValueConstraints &cst,
    mlir::affine::FlatAffineValueConstraints &domain) {
  cst = mlir::affine::FlatAffineValueConstraints();
  cst.mergeAndAlignVarsWithOther(0, &domain);

  LLVM_DEBUG({
    dbgs() << "Building access relation.\n"
           << " + Domain:\n";
    domain.dump();
  });

  SmallVector<mlir::Value, 8> idValues;
  domain.getValues(0, domain.getNumDimAndSymbolVars(), &idValues);
  llvm::SetVector<mlir::Value> idValueSet;
  for (auto val : idValues)
    idValueSet.insert(val);

  for (auto operand : vMap.getOperands())
    if (!idValueSet.contains(operand)) {
      llvm::errs() << "Operand missing: ";
      operand.dump();
    }

  // The results of the affine value map, which are the access addresses, will
  // be placed to the leftmost of all columns.
  return cst.composeMap(&vMap);
}

IslScop::SymbolTable *IslScop::getSymbolTable() { return &symbolTable; }

IslScop::ValueTable *IslScop::getValueTable() { return &valueTable; }

IslScop::MemRefToId *IslScop::getMemRefIdMap() { return &memRefIdMap; }

namespace polymer {
class IslMLIRBuilder {
public:
  // To be initialized by user
  OpBuilder &b;
  IRMapping funcMapping;
  IslScop &scop;
  unsigned integerBitWidth = 64;
  // To be initialized by user end

  Location loc = b.getUnknownLoc();
  using IDToValueTy = llvm::MapVector<isl_id *, Value>;
  IDToValueTy IDToValue{};

  // How many to create
  unsigned outerParallelBands = 1;
  unsigned outerParallelBandsCreated = 0;

  struct IndexedArray {
    MemrefValue base, indexed;
  };
  IndexedArray createIndexedArray(__isl_take isl_ast_expr *Expr) {
    if (isl_ast_expr_get_op_type(Expr) != isl_ast_op_call) {
      llvm_unreachable("unexpected op type");
    }
    POLYMER_ISL_DEBUG("Building Call:\n", isl_ast_expr_dump(Expr));
    isl_ast_expr *CalleeExpr = isl_ast_expr_get_op_arg(Expr, 0);
    isl_id *Id = isl_ast_expr_get_id(CalleeExpr);
    const char *CalleeName = isl_id_get_name(Id);

    POLYMER_ISL_DEBUG("Will be expanding array ", isl_id_dump(Id));
    ScopArrayInfo *array = scop.getArray(Id);
    assert(array && "Non existent array");
    LLVM_DEBUG(llvm::dbgs() << array->val << "\n");

    auto originalArray = cast<MemrefValue>(array->val);
    auto expandedArray = cast<MemrefValue>(funcMapping.lookup(array->val));
    unsigned originalRank = originalArray.getType().getRank();
    unsigned expandedRank = expandedArray.getType().getRank();
    unsigned expandedDims = expandedRank - originalRank;

    if (expandedRank == originalRank)
      return {originalArray, expandedArray};

    auto one = b.getIndexAttr(1);
    auto zero = b.getIndexAttr(0);
    unsigned numAvailableIndices = isl_ast_expr_get_op_n_arg(Expr) - 1;

    // Indexing for each array will be generated from the root of the schedule
    // tree. However, additional dimensions for the memref will only be added
    // after the scope the array is allocated, so there is a discrepancy here
    // and we need to drop these unneeded leading indices.
    unsigned toDrop = numAvailableIndices - expandedDims;
    // indices.erase(indices.begin(), std::next(indices.begin(), toDrop));
    // strides.erase(strides.begin(), std::next(strides.begin(), toDrop));
    // sizes.erase(sizes.begin(), std::next(sizes.begin(), toDrop));

    // Generate indexing into the expandedDims
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    for (unsigned i = 0; i < numAvailableIndices - toDrop; ++i) {
      isl_ast_expr *SubExpr = isl_ast_expr_get_op_arg(Expr, i + 1 + toDrop);
      Value V = create(SubExpr);
      convertToIndex(V);
      offsets.push_back(V);
      auto stride = b.getIndexAttr(1);
      uint64_t curSize = expandedArray.getType().getShape()[i];
      strides.push_back(stride);
      sizes.push_back(one);
    }

    // Generate indexing into the original memref rank.
    for (unsigned i = 0; i < originalRank; i++) {
      offsets.push_back(zero);
      auto curSize = originalArray.getType().getShape()[i];
      assert(curSize != ShapedType::kDynamic &&
             "dynamic sized array expansion currently unsupported");
      auto sizeAttr = b.getIndexAttr(curSize);
      sizes.push_back(sizeAttr);
      strides.push_back(b.getIndexAttr(1));
    }

    auto inferredType =
        cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            originalArray.getType().getShape(),
            cast<MemRefType>(expandedArray.getType()), offsets, sizes,
            strides));
    LLVM_DEBUG(llvm::dbgs() << "Inferred " << inferredType << "\n");

    auto subview = b.create<memref::SubViewOp>(
        array->val.getLoc(), cast<MemRefType>(inferredType), expandedArray,
        offsets, /*sizes=*/sizes, /*strides=*/strides);

    return {cast<MemrefValue>(originalArray),
            cast<MemrefValue>(subview.getResult())};
  }

  Value createOp(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "Expression not of type isl_ast_expr_op");
    switch (isl_ast_expr_get_op_type(Expr)) {
    case isl_ast_op_error:
    case isl_ast_op_cond:
    case isl_ast_op_call:
    case isl_ast_op_member:
      llvm_unreachable("Unsupported isl ast expression");
    case isl_ast_op_access:
      return createOpAccess(Expr);
    case isl_ast_op_max:
    case isl_ast_op_min:
      return createOpNAry(Expr);
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
      return createOpSelect(Expr);
    case isl_ast_op_and:
    case isl_ast_op_or:
      return createOpBoolean(Expr);
    case isl_ast_op_and_then:
    case isl_ast_op_or_else:
      return createOpBooleanConditional(Expr);
    case isl_ast_op_eq:
    case isl_ast_op_le:
    case isl_ast_op_lt:
    case isl_ast_op_ge:
    case isl_ast_op_gt:
      return createOpICmp(Expr);
    case isl_ast_op_address_of:
      return createOpAddressOf(Expr);
    }

    llvm_unreachable("Unsupported isl_ast_expr_op kind.");
  }

  Value createOpAddressOf(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createOpUnary(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createOpAccess(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }

  Value createMul(Value LHS, Value RHS, std::string Name = "") {
    return b.create<arith::MulIOp>(loc, LHS, RHS);
  }
  Value createSub(Value LHS, Value RHS, std::string Name = "") {
    return b.create<arith::SubIOp>(loc, LHS, RHS);
  }
  Value createAdd(Value LHS, Value RHS, std::string Name = "") {
    return b.create<arith::AddIOp>(loc, LHS, RHS);
  }

  Value createOpBin(__isl_take isl_ast_expr *Expr) {
    Value LHS, RHS, Res;
    Type MaxType;
    isl_ast_op_type OpType;

    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "isl ast expression not of type isl_ast_op");
    assert(isl_ast_expr_get_op_n_arg(Expr) == 2 &&
           "not a binary isl ast expression");

    OpType = isl_ast_expr_get_op_type(Expr);

    LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
    RHS = create(isl_ast_expr_get_op_arg(Expr, 1));

    MaxType = convertToMaxWidth(LHS, RHS);

    switch (OpType) {
    default:
      llvm_unreachable("This is no binary isl ast expression");
    case isl_ast_op_add:
      Res = createAdd(LHS, RHS);
      break;
    case isl_ast_op_sub:
      Res = createSub(LHS, RHS);
      break;
    case isl_ast_op_mul:
      Res = createMul(LHS, RHS);
      break;
    case isl_ast_op_div:
      Res = b.create<arith::DivSIOp>(loc, LHS, RHS);
      break;
    case isl_ast_op_pdiv_q: // Dividend is non-negative
      Res = b.create<arith::DivUIOp>(loc, LHS, RHS);
      break;
    case isl_ast_op_fdiv_q: { // Round towards -infty
      // if (auto Const = dyn_cast<arith::ConstantIntOp>(RHS)) {
      //   auto &Val = Const.getValue();
      //   if (Val.isPowerOf2() && Val.isNonNegative()) {
      //     Res = b.create<arith::ShRSIOp>(loc, LHS, Val.ceilLogBase2());
      //     break;
      //   }
      // }

      // TODO: Review code and check that this calculation does not yield
      //       incorrect overflow in some edge cases.
      //
      // floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
      Value One = b.create<arith::ConstantIntOp>(loc, 1, MaxType);
      Value Zero = b.create<arith::ConstantIntOp>(loc, 0, MaxType);
      Value Sum1 = createSub(LHS, RHS, "pexp.fdiv_q.0");
      Value Sum2 = createAdd(Sum1, One, "pexp.fdiv_q.1");
      Value isNegative =
          b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, LHS, Zero);
      Value Dividend = b.create<arith::SelectOp>(loc, isNegative, Sum2, LHS);
      Res = b.create<arith::DivSIOp>(loc, Dividend, RHS);
      break;
    }
    case isl_ast_op_pdiv_r: // Dividend is non-negative
      Res = b.create<arith::RemUIOp>(loc, LHS, RHS);
      break;
    case isl_ast_op_zdiv_r: // Result only compared against zero
      Res = b.create<arith::RemSIOp>(loc, LHS, RHS);
      break;
    }
    isl_ast_expr_free(Expr);
    return Res;
  }

  Value createOpNAry(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "isl ast expression not of type isl_ast_op");
    assert(isl_ast_expr_get_op_n_arg(Expr) >= 2 &&
           "We need at least two operands in an n-ary operation");

    std::function<Value(Value, Value)> Aggregate;
    switch (isl_ast_expr_get_op_type(Expr)) {
    default:
      llvm_unreachable("This is not a an n-ary isl ast expression");
    case isl_ast_op_max:
      Aggregate = [&](Value x, Value y) {
        return b.create<arith::MaxSIOp>(loc, x, y);
      };
      break;
    case isl_ast_op_min:
      Aggregate = [&](Value x, Value y) {
        return b.create<arith::MinSIOp>(loc, x, y);
      };
      break;
    }

    Value V = create(isl_ast_expr_get_op_arg(Expr, 0));

    for (int i = 1; i < isl_ast_expr_get_op_n_arg(Expr); ++i) {
      Value OpV = create(isl_ast_expr_get_op_arg(Expr, i));
      convertToMaxWidth(V, OpV);
      V = Aggregate(OpV, V);
    }

    isl_ast_expr_free(Expr);
    return V;
  }
  Value createOpSelect(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createOpICmp(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "Expected an isl_ast_expr_op expression");

    Value LHS, RHS, Res;

    auto *Op0 = isl_ast_expr_get_op_arg(Expr, 0);
    auto *Op1 = isl_ast_expr_get_op_arg(Expr, 1);
    bool HasNonAddressOfOperand =
        isl_ast_expr_get_type(Op0) != isl_ast_expr_op ||
        isl_ast_expr_get_type(Op1) != isl_ast_expr_op ||
        isl_ast_expr_get_op_type(Op0) != isl_ast_op_address_of ||
        isl_ast_expr_get_op_type(Op1) != isl_ast_op_address_of;

    // TODO not sure if we would ever get pointers here
    bool UseUnsignedCmp = !HasNonAddressOfOperand;

    LHS = create(Op0);
    RHS = create(Op1);

    if (LHS.getType() != RHS.getType()) {
      convertToMaxWidth(LHS, RHS);
    }

    isl_ast_op_type OpType = isl_ast_expr_get_op_type(Expr);
    assert(OpType >= isl_ast_op_eq && OpType <= isl_ast_op_gt &&
           "Unsupported ICmp isl ast expression");
    static_assert(isl_ast_op_eq + 4 == isl_ast_op_gt,
                  "Isl ast op type interface changed");

    arith::CmpIPredicate Predicates[5][2] = {
        {arith::CmpIPredicate::eq, arith::CmpIPredicate::eq},
        {arith::CmpIPredicate::sle, arith::CmpIPredicate::ule},
        {arith::CmpIPredicate::slt, arith::CmpIPredicate::ult},
        {arith::CmpIPredicate::sge, arith::CmpIPredicate::uge},
        {arith::CmpIPredicate::sgt, arith::CmpIPredicate::ugt},
    };

    Res = b.create<arith::CmpIOp>(
        loc, Predicates[OpType - isl_ast_op_eq][UseUnsignedCmp], LHS, RHS);

    isl_ast_expr_free(Expr);
    return Res;
  }

  Value createOpBoolean(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
           "Expected an isl_ast_expr_op expression");

    Value LHS, RHS, Res;
    isl_ast_op_type OpType;

    OpType = isl_ast_expr_get_op_type(Expr);

    assert((OpType == isl_ast_op_and || OpType == isl_ast_op_or) &&
           "Unsupported isl_ast_op_type");

    LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
    RHS = create(isl_ast_expr_get_op_arg(Expr, 1));

    // Even though the isl pretty printer prints the expressions as 'exp && exp'
    // or 'exp || exp', we actually code generate the bitwise expressions
    // 'exp & exp' or 'exp | exp'. This forces the evaluation of both branches,
    // but it is, due to the use of i1 types, otherwise equivalent. The reason
    // to go for bitwise operations is, that we assume the reduced control flow
    // will outweigh the overhead introduced by evaluating unneeded expressions.
    // The isl code generation currently does not take advantage of the fact
    // that the expression after an '||' or '&&' is in some cases not evaluated.
    // Evaluating it anyways does not cause any undefined behaviour.
    //
    // TODO: Document in isl itself, that the unconditionally evaluating the
    // second part of '||' or '&&' expressions is safe.
    if (!(LHS.getType().isInteger() &&
          LHS.getType().getIntOrFloatBitWidth() == 1))
      llvm_unreachable("unhandled");
    if (!(RHS.getType().isInteger() &&
          RHS.getType().getIntOrFloatBitWidth() == 1))
      llvm_unreachable("unhandled");
    // RHS = Builder.CreateIsNotNull(RHS);

    switch (OpType) {
    default:
      llvm_unreachable("Unsupported boolean expression");
    case isl_ast_op_and:
      Res = b.create<arith::AndIOp>(LHS.getLoc(), LHS, RHS);
      break;
    case isl_ast_op_or:
      Res = b.create<arith::OrIOp>(LHS.getLoc(), LHS, RHS);
      break;
    }

    isl_ast_expr_free(Expr);
    return Res;
  }
  Value createOpBooleanConditional(__isl_take isl_ast_expr *Expr) {
    llvm_unreachable("unimplemented");
  }
  Value createId(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_id &&
           "Expression not of type isl_ast_expr_ident");

    isl_id *Id;
    Value V;

    Id = isl_ast_expr_get_id(Expr);

    assert(IDToValue.count(Id) && "Identifier not found");

    V = IDToValue[Id];
    assert(V && "Unknown parameter id found");

    isl_id_free(Id);
    isl_ast_expr_free(Expr);

    return V;
  }
  IntegerType getMaxType() {
    return IntegerType::get(b.getContext(), integerBitWidth);
  }
  IntegerType getType(__isl_keep isl_ast_expr *Expr) {
    // XXX: We assume i64 is large enough. This is often true, but in general
    //      incorrect. Also, on 32bit architectures, it would be beneficial to
    //      use a smaller type. We can and should directly derive this
    //      information during code generation.
    return getMaxType();
  }
  Value createInt(__isl_take isl_ast_expr *Expr) {
    assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_int &&
           "Expression not of type isl_ast_expr_int");
    isl_val *Val;
    Value V;
    APInt APValue;
    IntegerType T;

    Val = isl_ast_expr_get_val(Expr);
    APValue = polly::APIntFromVal(Val);

    auto BitWidth = APValue.getBitWidth();
    if (BitWidth <= 64)
      T = getType(Expr);
    else
      T = b.getIntegerType(BitWidth);

    APValue = APValue.sext(T.getWidth());
    V = b.create<arith::ConstantIntOp>(loc, APValue.getSExtValue(), T);

    isl_ast_expr_free(Expr);
    return V;
  }
  Value _create(__isl_take isl_ast_expr *Expr) {
    switch (isl_ast_expr_get_type(Expr)) {
    case isl_ast_expr_error:
      llvm_unreachable("Code generation error");
    case isl_ast_expr_op:
      return createOp(Expr);
    case isl_ast_expr_id:
      return createId(Expr);
    case isl_ast_expr_int:
      return createInt(Expr);
    }
    llvm_unreachable("Unexpected enum value");
  }
  Value create(__isl_take isl_ast_expr *Expr) {
    LLVM_DEBUG(dbgs() << "Creating Expr\n"; polly::dumpIslObj(Expr););
    Value created = _create(Expr);
    LLVM_DEBUG(dbgs() << "Done\n"; created.dump(););
    return created;
  }

  void createUser(__isl_keep isl_ast_node *User) {
    POLYMER_ISL_DEBUG("Building User:\n", isl_ast_node_dump(User));

    isl_ast_expr *Expr = isl_ast_node_user_get_expr(User);
    if (isl_ast_expr_get_op_type(Expr) != isl_ast_op_call) {
      llvm_unreachable("unexpected op type");
    }
    isl_ast_expr *CalleeExpr = isl_ast_expr_get_op_arg(Expr, 0);
    isl_id *Id = isl_ast_expr_get_id(CalleeExpr);
    const char *CalleeName = isl_id_get_name(Id);

    SmallVector<Value> ivs;
    SmallVector<Value> arrayArgs;
    IRMapping arrayMapping;
    bool arraysStarted = false;
    for (int i = 0; i < isl_ast_expr_get_op_n_arg(Expr) - 1; ++i) {
      isl_ast_expr *SubExpr = isl_ast_expr_get_op_arg(Expr, i + 1);
      if (isl_ast_expr_get_type(SubExpr) == isl_ast_expr_op &&
          isl_ast_expr_get_op_type(SubExpr) == isl_ast_op_call)
        arraysStarted = true;
      if (arraysStarted) {
        IndexedArray ia = createIndexedArray(SubExpr);
        arrayMapping.map(ia.base, ia.indexed);
        arrayArgs.push_back(ia.indexed);
      } else {
        Value V = create(SubExpr);
        ivs.push_back(V);
      }
    }

    ScopStmt &stmt = scop.getIslStmt(CalleeName);

    Operation *origCaller = stmt.getOperation();
    llvm::DenseSet<Value> origArgs;
    origCaller->walk([&](Operation *op) {
      for (auto &opr : op->getOpOperands())
        if (opr.get().getParentRegion()->getParentOp()->isProperAncestor(
                origCaller))
          origArgs.insert(opr.get());
    });
    // The remapping for each statement is different so we need to construct a
    // new mapping for each one.
    IRMapping stmtMapping = funcMapping;
    for (Value origArg : origArgs) {
      auto ba = dyn_cast<BlockArgument>(origArg);
      if (ba) {
        Operation *owner = ba.getOwner()->getParentOp();
        if (isa<affine::AffineForOp, affine::AffineParallelOp>(owner)) {
          SmallVector<Operation *> enclosing;
          stmt.getEnclosingAffineOps(enclosing);
          unsigned ivId = 0;
          for (auto *op : enclosing) {
            if (isa<affine::AffineIfOp>(op)) {
              continue;
            } else if (auto par = dyn_cast<affine::AffineParallelOp>(op)) {
              if (owner == op) {
                ivId += ba.getArgNumber();
                assert(par.getIVs()[0].getArgNumber() == 0 &&
                       "Make sure the IVs start at arg #0");
                break;
              }
              ivId += par.getNumDims();
            } else if (isa<affine::AffineForOp>(op)) {
              if (owner == op)
                break;
              ivId++;
            } else {
              llvm_unreachable("non-affine enclosing op");
            }
          }
          Value arg = ivs[ivId];
          if (arg.getType() != origArg.getType()) {
            // This can only happen to index types as we may have replaced them
            // with the target system width
            assert(isa<IndexType>(origArg.getType()));
            arg = b.create<arith::IndexCastOp>(loc, origArg.getType(), arg);
          }
          stmtMapping.map(origArg, arg);
        } else {
          llvm_unreachable("unexpected");
        }
      } else {
        assert(funcMapping.contains(origArg));
        auto indexed = arrayMapping.lookupOrNull(origArg);
        if (indexed) {
          stmtMapping.map(origArg, indexed);
          LLVM_DEBUG(llvm::dbgs()
                     << "Remapping original array " << origArg
                     << " to indexed expanded " << indexed << "\n");
        }
      }
    }
    if (getenv("ISL_SCOP_GENERATE_TEST_STMT_CALLS")) {
      SmallVector<Value> args = ivs;
      args.insert(args.end(), arrayArgs.begin(), arrayArgs.end());
      b.create<func::CallOp>(origCaller->getLoc(), CalleeName, TypeRange{},
                             args);
    } else {
      Operation *newStmt = b.clone(*origCaller, stmtMapping);

      funcMapping.map(origCaller, newStmt);
      for (unsigned i = 0, e = origCaller->getNumResults(); i != e; ++i)
        funcMapping.map(origCaller->getResult(i), newStmt->getResult(i));
    }

    isl_ast_expr_free(Expr);
    isl_ast_node_free(User);
    isl_ast_expr_free(CalleeExpr);
    isl_id_free(Id);
  }

  void createMark(__isl_take isl_ast_node *Node) {
    POLYMER_ISL_DEBUG("Building Mark:\n", isl_ast_node_dump(Node));

    auto *Id = isl_ast_node_mark_get_id(Node);
    auto *Child = isl_ast_node_mark_get_node(Node);
    isl_ast_node_free(Node);

    // TODO this needs to check for "permutable" instead
    if (isParallelLoopMark(Id)) {
      assert(isl_ast_node_get_type(Child) == isl_ast_node_for);
      createFor(Child);
    } else if (isMark(Id, gridParallelMark)) {
      assert(isl_ast_node_get_type(Child) == isl_ast_node_for);
      auto nMembers = (uintptr_t)isl::manage_copy(Id).get_user();
      auto pop = createParallel(Child, nMembers);
      pop->setAttr("gpu.par.grid", UnitAttr::get(pop->getContext()));
    } else if (isMark(Id, allocateArrayMark)) {
      AllocateArrayMarkInfo *info =
          (AllocateArrayMarkInfo *)isl::manage_copy(Id).get_user();
      isl::union_set &toAllocate = info->allocate;
      auto &toExpand = info->expand;
      for (auto [arrayId, expand] : info->expand) {
        assert(
            llvm::any_of(expand, [&](unsigned expand) { return expand > 1; }));
        if (toAllocate
                .intersect(isl::set::universe(toAllocate.get_space())
                               .set_tuple_id(isl::manage_copy(arrayId))
                               .to_union_set())
                .is_empty()) {
          llvm_unreachable("Expansion with no allocation unsupported");
        }
      }
      auto r = toAllocate.foreach_set([&](isl::set set) {
        isl::id arrayId = set.get_tuple_id();
        memref::AllocaOp allocaOp =
            Value::getFromOpaquePointer(arrayId.get_user())
                .getDefiningOp<memref::AllocaOp>();
        auto found = toExpand.find(arrayId.get());
        if (found == toExpand.end()) {
          b.clone(*allocaOp, funcMapping);
        } else {
          auto expand = found->second;
          MemRefType ty = allocaOp.getMemref().getType();
          ArrayRef<int64_t> shapeArray = ty.getShape();
          // Prepend expanded dimensions
          SmallVector<int64_t> shape(expand.begin(), expand.end());
          shape.insert(shape.end(), shapeArray.begin(), shapeArray.end());
          auto layout = ty.getLayout();
          // TODO need to add layout for the new dimensions
          layout = {};
          MemRefType newType = MemRefType::get(shape, ty.getElementType(),
                                               layout, ty.getMemorySpace());
          auto newAllocaOp = b.create<memref::AllocaOp>(
              allocaOp->getLoc(), newType, allocaOp.getDynamicSizes(),
              allocaOp.getSymbolOperands(), allocaOp.getAlignmentAttr());
          funcMapping.map(allocaOp.getResult(), newAllocaOp.getResult());
        }
        return isl::stat::ok();
      });
      if (r.is_error())
        llvm_unreachable("error?");
      create(Child);
    } else if (isMark(Id, asyncWaitGroupMark)) {
      // TODO maybe implemnet this as a generic callback so that we don't need
      // to put gpu/nvidia/etc intrinsic specific stuff here
      if (!getenv("GPU_AFFINE_OPT_DISABLE_CONVERT_TO_ASYNC")) {
        AsyncWaitGroupInfo *info =
            (AsyncWaitGroupInfo *)isl::manage_copy(Id).get_user();
        b.create<NVVM::CpAsyncWaitGroupOp>(b.getUnknownLoc(), info->num);
      }
      create(Child);
    } else {
      llvm_unreachable("Unknown mark");
    }

    isl_id_free(Id);
  }

  void createIf(__isl_take isl_ast_node *If) {
    POLYMER_ISL_DEBUG("Building If:\n", isl_ast_node_dump(If));
    isl_ast_expr *Cond = isl_ast_node_if_get_cond(If);

    Value Predicate = create(Cond);

    bool hasElse = isl_ast_node_if_has_else(If);
    auto ifOp = b.create<scf::IfOp>(loc, TypeRange(), Predicate,
                                    /*addThenBlock=*/true,
                                    /*addElseBlock=*/hasElse);

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    b.create<scf::YieldOp>(loc);
    b.setInsertionPointToStart(&ifOp.getThenRegion().front());
    create(isl_ast_node_if_get_then(If));

    if (hasElse) {
      b.setInsertionPointToStart(&ifOp.getElseRegion().front());
      b.create<scf::YieldOp>(loc);
      b.setInsertionPointToStart(&ifOp.getElseRegion().front());
      create(isl_ast_node_if_get_else(If));
    }

    isl_ast_node_free(If);
  }

  void createBlock(__isl_keep isl_ast_node *Block) {
    POLYMER_ISL_DEBUG("Building Block:\n", isl_ast_node_dump(Block));
    isl_ast_node_list *List = isl_ast_node_block_get_children(Block);

    for (int i = 0; i < isl_ast_node_list_n_ast_node(List); ++i)
      create(isl_ast_node_list_get_ast_node(List, i));

    isl_ast_node_free(Block);
    isl_ast_node_list_free(List);
  }

  isl::ast_expr getUpperBound(isl::ast_node_for For,
                              arith::CmpIPredicate &Predicate) {
    isl::ast_expr Cond = For.cond();
    isl::ast_expr Iterator = For.iterator();
    // The isl code generation can generate arbitrary expressions to check if
    // the upper bound of a loop is reached, but it provides an option to
    // enforce 'atomic' upper bounds. An 'atomic upper bound is always of the
    // form iv <= expr, where expr is an (arbitrary) expression not containing
    // iv.
    //
    // We currently only support atomic upper bounds for ease of codegen
    //
    // This is needed for parallel loops but maybe we can weaken the requirement
    // for sequential loops if needed
    assert(isl_ast_expr_get_type(Cond.get()) == isl_ast_expr_op &&
           "conditional expression is not an atomic upper bound");

    isl_ast_op_type OpType = isl_ast_expr_get_op_type(Cond.get());

    switch (OpType) {
    case isl_ast_op_le:
      Predicate = arith::CmpIPredicate::sle;
      break;
    case isl_ast_op_lt:
      Predicate = arith::CmpIPredicate::slt;
      break;
    default:
      llvm_unreachable("Unexpected comparison type in loop condition");
    }

    isl::ast_expr Arg0 = Cond.get_op_arg(0);

    assert(isl_ast_expr_get_type(Arg0.get()) == isl_ast_expr_id &&
           "conditional expression is not an atomic upper bound");

    isl::id UBID = Arg0.get_id();

    assert(isl_ast_expr_get_type(Iterator.get()) == isl_ast_expr_id &&
           "Could not get the iterator");

    isl::id IteratorID = Iterator.get_id();

    assert(UBID.get() == IteratorID.get() &&
           "conditional expression is not an atomic upper bound");

    return Cond.get_op_arg(1);
  }

  template <class... Ts>
  void convertToIndex(Ts &&...args) {
    SmallVector<Value *> Args({&args...});
    for (unsigned I = 0; I < Args.size(); I++) {
      Type Ty = Args[I]->getType();
      if (!Ty.isa<IndexType>()) {
        *Args[I] =
            b.create<arith::IndexCastOp>(loc, b.getIndexType(), *Args[I]);
      }
    }
  }

  template <class... Ts>
  Type convertToMaxWidth(Ts &&...args) {
    SmallVector<Value *> Args({&args...});
    if (llvm::all_of(Args,
                     [&](Value *V) { return V->getType().isa<IndexType>(); }))
      return Args[0]->getType();
    Type MaxTypeI = Args[0]->getType();
    IntegerType MaxType;
    if (MaxTypeI.isa<IndexType>())
      // TODO This is temporary and we should get the target system index here
      MaxType = getMaxType();
    else
      MaxType = MaxTypeI.cast<IntegerType>();
    unsigned MaxWidth = MaxType.getWidth();
    for (unsigned I = 0; I < Args.size(); I++) {
      Type Ty = Args[I]->getType();
      if (Ty.isa<IndexType>())
        // TODO This is temporary and we should get the target system index here
        Ty = getMaxType();
      if (Ty.cast<IntegerType>().getWidth() > MaxWidth) {
        MaxType = Ty.cast<IntegerType>();
        MaxWidth = MaxType.getWidth();
      }
    }
    for (unsigned I = 0; I < Args.size(); I++) {
      Type Ty = Args[I]->getType();
      if (Ty.isa<IndexType>()) {
        *Args[I] = b.create<arith::IndexCastOp>(loc, MaxType, *Args[I]);
      } else if (Ty != MaxType) {
        *Args[I] = b.create<arith::ExtSIOp>(loc, MaxType, *Args[I]);
      }
    }
    return MaxType;
  }

  scf::ParallelOp createParallel(__isl_take isl_ast_node *For,
                                 unsigned nLoops) {
    SmallVector<Value> LBs;
    SmallVector<Value> Incs;
    SmallVector<isl_id *> Iterators;
    SmallVector<Value> UBs;
    isl_ast_node *Body;
    while (For && nLoops > 0) {
      POLYMER_ISL_DEBUG("Building Parallel:\n", isl_ast_node_dump(For));
      Body = isl_ast_node_for_get_body(For);
      isl_ast_expr *Init = isl_ast_node_for_get_init(For);
      isl_ast_expr *Inc = isl_ast_node_for_get_inc(For);
      isl_ast_expr *Iterator = isl_ast_node_for_get_iterator(For);
      isl_id *IteratorId = isl_ast_expr_get_id(Iterator);
      Iterators.push_back(IteratorId);
      Iterator = isl_ast_expr_free(Iterator);
      arith::CmpIPredicate Predicate;
      isl_ast_expr *UB =
          getUpperBound(isl::manage_copy(For).as<isl::ast_node_for>(),
                        Predicate)
              .release();

      Value ValueLB = create(Init);
      Value ValueUB = create(UB);
      Value ValueInc = create(Inc);
      convertToMaxWidth(ValueLB, ValueUB, ValueInc);

      if (Predicate == arith::CmpIPredicate::sle)
        ValueUB = b.create<arith::AddIOp>(
            loc, ValueUB,
            b.create<arith::ConstantIntOp>(loc, 1, ValueUB.getType()));

      convertToIndex(ValueLB, ValueUB, ValueInc);

      LBs.push_back(ValueLB);
      UBs.push_back(ValueUB);
      Incs.push_back(ValueInc);

      switch (isl_ast_node_get_type(Body)) {
      case isl_ast_node_for:
        For = Body;
        Body = isl_ast_node_free(Body);
        break;
      default:
        For = nullptr;
      }
      nLoops -= 1;
    }

    assert(nLoops == 0 && "Not enough nested loops");

    auto pop = b.create<scf::ParallelOp>(loc, LBs, UBs, Incs);

    for (unsigned i = 0; i < pop.getNumLoops(); i++) {
      IDToValue[Iterators[i]] = pop.getInductionVars()[i];
      Iterators[i] = isl_id_free(Iterators[i]);
    }

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(pop.getBody());
    create(Body);

    isl_ast_node_free(For);

    return pop;
  }

  void createFor(__isl_take isl_ast_node *For) {
    using ForOpTy = scf::ForOp;
    POLYMER_ISL_DEBUG("Building For:\n", isl_ast_node_dump(For));
    isl_ast_node *Body = isl_ast_node_for_get_body(For);
    isl_ast_expr *Init = isl_ast_node_for_get_init(For);
    isl_ast_expr *Inc = isl_ast_node_for_get_inc(For);
    isl_ast_expr *Iterator = isl_ast_node_for_get_iterator(For);
    isl_id *IteratorID = isl_ast_expr_get_id(Iterator);
    arith::CmpIPredicate Predicate;
    isl_ast_expr *UB =
        getUpperBound(isl::manage_copy(For).as<isl::ast_node_for>(), Predicate)
            .release();

    Value ValueLB = create(Init);
    Value ValueUB = create(UB);
    Value ValueInc = create(Inc);
    convertToMaxWidth(ValueLB, ValueUB, ValueInc);

    if (Predicate == arith::CmpIPredicate::sle)
      ValueUB = b.create<arith::AddIOp>(
          loc, ValueUB,
          b.create<arith::ConstantIntOp>(loc, 1, ValueUB.getType()));

    // scf::ParallelOp only supports index as bounds
    if constexpr (std::is_same<ForOpTy, scf::ParallelOp>::value) {
      convertToIndex(ValueLB, ValueUB, ValueInc);
    }

    auto forOp = b.create<ForOpTy>(loc, ValueLB, ValueUB, ValueInc);

    IDToValue[IteratorID] = forOp.getInductionVar();

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(forOp.getBody());
    create(Body);

    isl_ast_expr_free(Iterator);
    isl_id_free(IteratorID);
    isl_ast_node_free(For);
  }

  void create(__isl_take isl_ast_node *Node) {
    switch (isl_ast_node_get_type(Node)) {
    case isl_ast_node_error:
      llvm_unreachable("code generation error");
    case isl_ast_node_mark:
      createMark(Node);
      return;
    case isl_ast_node_for:
      createFor(Node);
      return;
    case isl_ast_node_if:
      createIf(Node);
      return;
    case isl_ast_node_user:
      createUser(Node);
      return;
    case isl_ast_node_block:
      createBlock(Node);
      return;
    }

    llvm_unreachable("Unknown isl_ast_node type");
  }

  void mapParams(__isl_take isl_union_set *domain) {
    isl_space *space = isl_union_set_get_space(domain);

    int nparams = isl_space_dim(space, isl_dim_param);
    for (int i = 0; i < nparams; i++) {
      isl_id *Id = isl_space_get_dim_id(space, isl_dim_param, i);
      const char *paramName = isl_id_get_name(Id);
      Value V = Value::getFromOpaquePointer(isl_id_get_user(Id));
      IDToValue[Id] = funcMapping.lookup(V);
      isl_id_free(Id);
    }
    isl_space_free(space);
    isl_union_set_free(domain);
  }
};
} // namespace polymer

static void setIslOptions(isl_ctx *ctx) {
  isl_stat stat;
#define check_res(code)                                                        \
  do {                                                                         \
    stat = code;                                                               \
    assert(stat == isl_stat_ok);                                               \
  } while (0)
  check_res(isl_options_set_ast_build_atomic_upper_bound(ctx, 1));
  check_res(isl_options_set_ast_build_exploit_nested_bounds(ctx, 1));
  check_res(isl_options_set_ast_build_group_coscheduled(ctx, 1));
  check_res(isl_options_set_ast_build_allow_else(ctx, 1));
#undef check_res
}

void IslScop::cleanup(Operation *func) {
  func->walk([](affine::AffineStoreVar op) { op->erase(); });
}

IslScop::ApplyScheduleRes
IslScop::applySchedule(__isl_take isl_schedule *newSchedule,
                       __isl_take isl_union_map *lrs, Operation *originalFunc,
                       unsigned integerBitWidth) {
  IRMapping oldToNewMapping;
  OpBuilder moduleBuilder(originalFunc);
  Operation *f =
      moduleBuilder.cloneWithoutRegions(*originalFunc, oldToNewMapping);

  assert(originalFunc->getNumRegions() == 1);
  assert(originalFunc->getRegion(0).getBlocks().size() == 1);

  {
    // TODO need to place these at their appropriate scope (maybe use isl
    // marks?) - currently we are working around this in GPUAffineOpt using
    // MoveAllocas
    OpBuilder b(f->getContext());
    b.createBlock(&f->getRegion(0), f->getRegion(0).begin(),
                  originalFunc->getRegion(0).front().getArgumentTypes(),
                  originalFunc->getRegion(0).front().getArgumentLocs());

    oldToNewMapping.map(originalFunc->getRegion(0).front().getArguments(),
                        f->getRegion(0).front().getArguments());

    for (auto &op : originalFunc->getRegion(0).front().getOperations()) {
      if (isa<affine::AffineDialect>(op.getDialect())) {
        assert(op.getNumResults() == 0);
      } else {
        b.clone(op, oldToNewMapping);
      }
    }
  }

  // TODO we also need to allocate new arrays which may have been introduced,
  // see polly::NodeBuilder::allocateNewArrays, buildAliasScopes

  OpBuilder b(f->getRegion(0).front().getTerminator());

  LLVM_DEBUG({
    llvm::dbgs() << "Applying new schedule to scop:\n";
    isl_schedule_dump(newSchedule);
  });
  isl_union_set *domain = isl_schedule_get_domain(newSchedule);
  setIslOptions(getIslCtx());
  isl_ast_build *build = isl_ast_build_alloc(getIslCtx());
  build = isl_ast_build_set_live_range_span(build, lrs);
  IslMLIRBuilder bc = {b, oldToNewMapping, *this, integerBitWidth};
  isl_ast_node *node =
      isl_ast_build_node_from_schedule(build, isl_schedule_copy(newSchedule));
  LLVM_DEBUG({
    llvm::dbgs() << "New AST:\n";
    isl_ast_node_dump(node);
  });

  bc.mapParams(domain);
  bc.create(node);
  LLVM_DEBUG(llvm::dbgs() << *f << "\n");

// FIXME we are getting some double frees/invalid read/writes due to these...
#if 0
  isl_ast_build_free(build);
  isl_schedule_free(newSchedule);
#endif

  return IslScop::ApplyScheduleRes{f, oldToNewMapping};
}

// TODO this takes the union of the write effects in the operations we rescope
// to. instead, what should happen is we should do flow analysis to see what
// memory effects live-out and live-in, i.e. not care about memory effects
// that are not observable outside the rescoped op.
void IslScop::rescopeStatements(
    std::function<bool(Operation *op)> shouldRescope,
    std::function<bool(Operation *op)> isValidAsyncCopy) {

  unsigned rescopedNum = 0;

  root->walk([&](Operation *rescopeOp) {
    if (!shouldRescope(rescopeOp))
      return WalkResult::advance();

    LLVM_DEBUG(llvm::dbgs() << "Handling rescope\n" << *rescopeOp << "\n");

    std::string newStmtName = "RS" + std::to_string(rescopedNum++) + "." +
                              rescopeOp->getName().getStringRef().str();
    makeIslCompatible(newStmtName);
    // FIXME this only works because we do not assign statement names to the
    // affine.parallel we rescope to.
    ScopStmt &newStmt = stmts.emplace_back(rescopeOp, this, newStmtName,
                                           isValidAsyncCopy(rescopeOp));
    affine::FlatAffineValueConstraints domain = *newStmt.getMlirDomain();
    addDomainRelation(newStmt, domain);

    unsigned depth = domain.getNumDimVars();
    // FIXME wrong
    LLVM_DEBUG(llvm::dbgs() << "Depth " << depth << "\n");

    for (auto it = stmts.begin(); it != stmts.end();) {
      polymer::ScopStmt &stmt = *it;

      if (&stmt == &newStmt) {
        it++;
        continue;
      }

      Operation *nested = stmt.getOperation();
      if (!rescopeOp->isAncestor(nested)) {
        it++;
        continue;
      }

      assert(rescopeOp != nested);
      isl::set domain = stmt.getDomain();
      LLVM_DEBUG(llvm::dbgs() << "nested stmt " << *nested << "\n");
      LLVM_DEBUG(isl_set_dump(domain.get()));
      isl::map domMap = isl::map::from_domain(domain);
      LLVM_DEBUG(isl_map_dump(domMap.get()));

      for (MemoryAccess *ma : stmt) {
        Value memref = ma->getScopArrayInfo()->val;
        // FIXME this is wrong for iter args of for loops for example but we
        // do not rescope operations where this would be problematic
        Operation *scope = memref.getParentBlock()->getParentOp();
        // If the scope of the memref is inside the rescoped op then we do not
        // need to care about accesses to it
        if (rescopeOp->isAncestor(scope))
          continue;

        isl::map accRel = ma->getAccessRelation();
        LLVM_DEBUG(isl_map_dump(accRel.get()));
        isl::map stmtToRescoped = domMap;
        stmtToRescoped = stmtToRescoped.project_out(isl::dim::in, 0, depth);
        auto nDims = stmtToRescoped.space().dim(isl::dim::in);
        assert(!nDims.is_error());
        stmtToRescoped = isl::manage(isl_map_insert_dims(
            stmtToRescoped.release(), isl_dim_in, 0, depth));
        stmtToRescoped = stmtToRescoped.add_dims(isl::dim::out, depth);
        for (unsigned i = 0; i < depth; i++) {
          auto cst = isl::constraint::alloc_equality(
              isl::local_space(stmtToRescoped.get_space()));
          cst = cst.set_coefficient_si(isl::dim::in, i, 1);
          cst = cst.set_coefficient_si(isl::dim::out, i, -1);
          stmtToRescoped = stmtToRescoped.add_constraint(cst);
        }
        stmtToRescoped = stmtToRescoped.set_tuple_id(
            isl::dim::in, accRel.get_tuple_id(isl::dim::in));
        stmtToRescoped = isl::manage(isl_map_set_tuple_id(
            stmtToRescoped.release(), isl_dim_out, newStmt.id.copy()));
        LLVM_DEBUG(isl_map_dump(stmtToRescoped.get()));
        auto rescopedAccRel = accRel.apply_domain(stmtToRescoped);
        LLVM_DEBUG(dbgs() << "Computed combined acc rel for rescoped: ";);
        LLVM_DEBUG(isl_map_dump(rescopedAccRel.get()));
        newStmt.memoryAccesses.push_back(
            new MemoryAccess{ma->Id, rescopedAccRel, ma->Kind, ma->AccType,
                             ma->getScopArrayInfo()});
      }

      it = stmts.erase(it);
    }

    // no nested resopes
    return WalkResult::skip();
  });

  addIndependences();
}

namespace polymer {

/// Build IslScop from a given FuncOp.
std::unique_ptr<IslScop> IslScopBuilder::build(Operation *f) {

  /// Context constraints.
  affine::FlatAffineValueConstraints ctx;

  // Initialize a new Scop per FuncOp. The osl_scop object within it will be
  // created. It doesn't contain any fields, and this may incur some problems,
  // which the validate function won't discover, e.g., no context will cause
  // segfault when printing scop. Please don't just return this object.
  auto scop = std::make_unique<IslScop>(f);

  // Find all caller/callee pairs in which the callee has the attribute of
  // name SCOP_STMT_ATTR_NAME.
  IRMapping redirectMap;
  gatherStmts(f, redirectMap, *scop);

  // Build context in it.
  buildScopContext(f, scop.get(), ctx);

  scop->initializeSymbolTable(f, &ctx);

  for (auto &stmt : scop->stmts) {
    LLVM_DEBUG({
      dbgs() << "Adding relations to statement: \n";
      stmt.getOperation()->dump();
    });

    // Collet the domain
    affine::FlatAffineValueConstraints domain = *stmt.getMlirDomain();

    LLVM_DEBUG({
      dbgs() << "Domain:\n";
      domain.dump();
    });

    Operation *op = stmt.getOperation();

    LLVM_DEBUG({
      dbgs() << "op:\n";
      op->dump();
    });

    scop->addDomainRelation(stmt, domain);

    {
      // FIXME remapping of ataddr op to the llvm pointer does not work
      // because we get the information about the array rank from the memref
      // value type, temp fix
      if (isa<memref::AtAddrOp>(op))
        continue;

      LLVM_DEBUG(dbgs() << "Creating access relation for: " << *op << '\n');
      auto needsMemEffects = [&](Value memref) {
        if (affine::isValidDim(memref) || affine::isValidSymbol(memref))
          return false;

        if (auto *op = memref.getDefiningOp())
          if (op->hasTrait<OpTrait::ConstantLike>())
            return false;

        return true;
      };
      auto addLoad = [&](Value memref, affine::AffineValueMap map) {
        if (needsMemEffects(memref))
          (void)scop->addAccessRelation(stmt, polymer::MemoryAccess::READ,
                                        redirectMap.lookupOrDefault(memref),
                                        map, false, domain);
      };
      auto addMayStore = [&](Value memref, affine::AffineValueMap map) {
        if (needsMemEffects(memref))
          (void)scop->addAccessRelation(stmt, polymer::MemoryAccess::MAY_WRITE,
                                        redirectMap.lookupOrDefault(memref),
                                        map, false, domain);
      };
      auto addMustStore = [&](Value memref, affine::AffineValueMap map) {
        if (needsMemEffects(memref))
          (void)scop->addAccessRelation(stmt, polymer::MemoryAccess::MUST_WRITE,
                                        redirectMap.lookupOrDefault(memref),
                                        map, false, domain);
      };
      auto addKill = [&](Value memref, affine::AffineValueMap map,
                         bool universe) {
        auto redirected = redirectMap.lookupOrDefault(memref);
        if (redirected.getParentBlock()->getParentOp() == f)
          return;
        if (needsMemEffects(memref))
          (void)scop->addAccessRelation(stmt, polymer::MemoryAccess::KILL,
                                        redirected, map, universe, domain);
      };
      bool needToLoadOperands = true;
      bool needToStoreResults = true;
      auto unitMap = AffineMap::get(op->getContext());
      affine::AffineValueMap unitVMap(unitMap, ValueRange{}, ValueRange{});
      if (!isMemoryEffectFree(op)) {
        // TODO FIXME NEED TO PUT THE VECTOR SIZE INTO THE RELATION FOR
        // affine.vector_{store,load}
        if (isa<mlir::affine::AffineReadOpInterface>(op) ||
            isa<mlir::affine::AffineWriteOpInterface>(op)) {

          affine::AffineValueMap vMap;
          mlir::Value memref;

          AffineMap map;
          SmallVector<Value, 4> indices;
          if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
            memref = loadOp.getMemRef();
            llvm::append_range(indices, loadOp.getMapOperands());
            map = loadOp.getAffineMap();
            vMap.reset(map, indices);
            addLoad(memref, vMap);
            addMustStore(loadOp.getValue(), unitVMap);
          } else {
            assert(isa<affine::AffineWriteOpInterface>(op) &&
                   "Affine read/write op expected");
            auto storeOp = cast<affine::AffineWriteOpInterface>(op);
            memref = storeOp.getMemRef();
            llvm::append_range(indices, storeOp.getMapOperands());
            map = cast<affine::AffineWriteOpInterface>(op).getAffineMap();
            vMap.reset(map, indices);
            addMustStore(memref, vMap);
            addLoad(storeOp.getValueToStore(), unitVMap);
          }
          needToLoadOperands = false;
          needToStoreResults = false;
        } else {
          assert((isa<memref::AllocOp, memref::AllocaOp>(op)));
          needToLoadOperands = false;
          needToStoreResults = false;
        }
      } else if (auto storeVar = dyn_cast<affine::AffineStoreVar>(op)) {
        assert(storeVar->getNumOperands() == 2);
        Value val = storeVar->getOperand(0);
        Value addr = storeVar->getOperand(1);
        addLoad(val, unitVMap);
        addMustStore(addr, unitVMap);
        needToLoadOperands = false;
        needToStoreResults = false;
      } else if (auto yield = dyn_cast<affine::AffineYieldOp>(op)) {
        for (auto [res, opr] :
             llvm::zip(ValueRange(yield->getParentOp()->getResults()),
                       ValueRange(yield->getOperands()))) {
          addMustStore(res, unitVMap);
          addLoad(opr, unitVMap);
        }
        needToLoadOperands = false;
        needToStoreResults = false;
      }

      if (op->getBlock()->getTerminator() == op)
        for (auto &toKill : op->getBlock()->without_terminator())
          for (auto res : toKill.getResults())
            addKill(res, {}, true);

      if (llvm::all_of(op->getOpResults(),
                       [&](Value v) { return redirectMap.contains(v); }))
        continue;

      if (needToStoreResults)
        for (auto res : op->getResults())
          addMustStore(res, unitVMap);
      if (needToLoadOperands)
        for (auto opr : op->getOperands())
          addLoad(opr, unitVMap);
    }
  }

  scop->addIndependences();

  return scop;
}

static void createForIterArgAccesses(affine::AffineForOp forOp,
                                     IRMapping &map) {
  OpBuilder builder(forOp);
  for (auto [init, res] : llvm::zip(ValueRange(forOp.getInitsMutable()),
                                    ValueRange(forOp.getResults())))
    builder.create<affine::AffineStoreVar>(
        forOp.getLoc(), ValueRange{init, res},
        builder.getStringAttr("for.iv.init"));
  map.map(forOp.getRegionIterArgs(), forOp.getResults());
}

void IslScopBuilder::gatherStmts(Operation *f, IRMapping &map,
                                 IslScop &S) const {
  f->walk(
      [&](affine::AffineForOp forOp) { createForIterArgAccesses(forOp, map); });
  // f->walk([&](memref::AtAddrOp atAddr) {
  //   map.map(atAddr.getResult(), atAddr.getAddr());
  // });
  unsigned stmtId = 0;
  f->walk([&](mlir::Operation *op) {
    if (isa<affine::AffineForOp, affine::AffineIfOp, affine::AffineParallelOp>(
            op))
      return;
    if (op == f)
      return;
    std::string calleeName = "S" + std::to_string(stmtId++) + "." +
                             op->getName().getStringRef().str();
    S.stmts.emplace_back(op, &S, calleeName.c_str());
  });
}

void IslScopBuilder::buildScopContext(
    Operation *f, IslScop *scop,
    affine::FlatAffineValueConstraints &ctx) const {
  LLVM_DEBUG(dbgs() << "--- Building SCoP context ...\n");

  // First initialize the symbols of the ctx by the order of arg number.
  // This simply aims to make mergeAndAlignVarsWithOthers work.
  SmallVector<Value> symbols;
  auto insertSyms = [&](auto syms) {
    for (Value sym : syms) {
      // Find the insertion position.
      auto it = symbols.begin();
      while (it != symbols.end()) {
        auto lhs = it->getAsOpaquePointer();
        auto rhs = sym.getAsOpaquePointer();
        if (lhs >= rhs)
          break;
        ++it;
      }
      if (it == symbols.end() || *it != sym)
        symbols.insert(it, sym);
    }
  };
  for (auto &stmt : scop->stmts) {
    auto domain = stmt.getMlirDomain();
    SmallVector<Value> syms;
    domain->getValues(domain->getNumDimVars(), domain->getNumDimAndSymbolVars(),
                      &syms);

    insertSyms(syms);
  }
  f->walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<affine::AffineReadOpInterface>(op)) {
      insertSyms(loadOp.getMapOperands().drop_front(
          loadOp.getAffineMap().getNumDims()));
    } else if (auto storeOp = dyn_cast<affine::AffineWriteOpInterface>(op)) {
      insertSyms(storeOp.getMapOperands().drop_front(
          storeOp.getAffineMap().getNumDims()));
    }
  });

  ctx = affine::FlatAffineValueConstraints(/*numDims=*/0,
                                           /*numSymbols=*/symbols.size());
  ctx.setValues(0, symbols.size(), symbols);

  // Union with the domains of all Scop statements. We first merge and align
  // the IDs of the context and the domain of the scop statement, and then
  // append the constraints from the domain to the context. Note that we don't
  // want to mess up with the original domain at this point. Trivial redundant
  // constraints will be removed.
  for (auto &stmt : scop->stmts) {
    affine::FlatAffineValueConstraints *domain = stmt.getMlirDomain();
    affine::FlatAffineValueConstraints cst(*domain);

    LLVM_DEBUG(dbgs() << "Statement:\n");
    LLVM_DEBUG(stmt.getOperation()->dump());
    LLVM_DEBUG(dbgs() << "Target domain: \n");
    LLVM_DEBUG(domain->dump());

    LLVM_DEBUG({
      dbgs() << "Domain values: \n";
      SmallVector<Value> values;
      domain->getValues(0, domain->getNumDimAndSymbolVars(), &values);
      for (Value value : values)
        dbgs() << " * " << value << '\n';
    });

    ctx.mergeAndAlignVarsWithOther(0, &cst);
    ctx.append(cst);
    ctx.removeRedundantConstraints();

    LLVM_DEBUG(dbgs() << "Updated context: \n");
    LLVM_DEBUG(ctx.dump());

    LLVM_DEBUG({
      dbgs() << "Context values: \n";
      SmallVector<Value> values;
      ctx.getValues(0, ctx.getNumDimAndSymbolVars(), &values);
      for (Value value : values)
        dbgs() << " * " << value << '\n';
    });
  }

  // Then, create the single context relation in scop.
  scop->addContextRelation(ctx);

  // Finally, given that ctx has all the parameters in it, we will make sure
  // that each domain is aligned with them, i.e., every domain has the same
  // parameter columns (Values & order).
  SmallVector<mlir::Value, 8> symValues;
  ctx.getValues(ctx.getNumDimVars(), ctx.getNumDimAndSymbolVars(), &symValues);

  // Add and align domain SYMBOL columns.
  for (auto &stmt : scop->stmts) {
    affine::FlatAffineValueConstraints *domain = stmt.getMlirDomain();
    // For any symbol missing in the domain, add them directly to the end.
    for (unsigned i = 0; i < ctx.getNumSymbolVars(); ++i) {
      unsigned pos;
      if (!domain->findVar(symValues[i], &pos)) // insert to the back
        domain->appendSymbolVar(symValues[i]);
      else
        LLVM_DEBUG(dbgs() << "Found " << symValues[i] << '\n');
    }

    // Then do the aligning.
    LLVM_DEBUG(domain->dump());
    for (unsigned i = 0; i < ctx.getNumSymbolVars(); i++) {
      mlir::Value sym = symValues[i];
      unsigned pos;
      domain->findVar(sym, &pos);

      unsigned posAsCtx = i + domain->getNumDimVars();
      LLVM_DEBUG(dbgs() << "Swapping " << posAsCtx << " " << pos << "\n");
      if (pos != posAsCtx)
        domain->swapVar(posAsCtx, pos);
    }

    // for (unsigned i = 0; i < ctx.getNumSymbolVars(); i++) {
    //   mlir::Value sym = symValues[i];
    //   unsigned pos;
    //   // If the symbol can be found in the domain, we put it in the same
    //   // position as the ctx.
    //   if (domain->findVar(sym, &pos)) {
    //     if (pos != i + domain->getNumDimVars())
    //       domain->swapVar(i + domain->getNumDimVars(), pos);
    //   } else {
    //     domain->insertSymbolId(i, sym);
    //   }
    // }
  }
}

std::unique_ptr<IslScop> createIslFromFuncOp(Operation *f) {
  return IslScopBuilder().build(f);
}

} // namespace polymer
