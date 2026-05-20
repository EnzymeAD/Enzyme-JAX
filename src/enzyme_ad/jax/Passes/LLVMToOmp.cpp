//===- LLVMToOmp.cpp - Convert LLVM dialect OMP runtime → OMP dialect ----===//
//
// Converts OpenMP runtime calls (omp_* and __kmpc_*) in the LLVM dialect to
// structured OpenMP dialect operations.  Aligned with OpenMPOps.td.
//
// ── Design ────────────────────────────────────────────────────────────────
//   Phase 1 — module walk: structural conversions that cross function
//             boundaries (fork_call→parallel, paired calls, wsloop, tasks,
//             atomics).
//   Phase 2 — greedy pattern rewrite: leaf replacements (barrier, flush,
//             taskwait, taskyield, cancel, …).
//
// ── API notes vs OpenMPOps.td ─────────────────────────────────────────────
//
//  BUILDER FORMS
//  Ops that have an explicit OpBuilder<(ins CArg<"const *Operands &">:$clauses)>
//  use the struct-based form.  Atomic ops do NOT — they use auto-generated
//  positional builders (arguments from !con(explicit-args, clausesArgs)):
//
//    AtomicReadOp  (loc, x_src, v_dst, element_type_attr, hint, memory_order)
//    AtomicWriteOp (loc, x_dst, expr, hint, memory_order)
//    AtomicUpdateOp(loc, x_ptr, atomic_control, hint, memory_order)
//    AtomicCaptureOp(loc, hint, memory_order)
//
//  REGION / BLOCK CREATION
//    WsloopOp    — [NoTerminator, SingleBlock]: region.front() auto-created;
//                  never add omp.terminator inside wsloop.
//    AtomicUpdateOp — SingleBlockImplicitTerminator<"YieldOp"> (implies
//                  SingleBlock): block auto-created with implicit empty YieldOp.
//    AtomicCaptureOp— SingleBlockImplicitTerminator<"TerminatorOp"> (implies
//                  SingleBlock): ONE block, implicit TerminatorOp at end;
//                  contains nested atomic sub-ops (update+read, etc.).
//    All other ops (ParallelOp, TeamsOp, SingleOp, MasterOp, CriticalOp,
//    OrderedRegionOp, TaskgroupOp, TaskOp, LoopNestOp …) — create block
//    explicitly.
//
//  ARGUMENT SEMANTICS
//    AtomicReadOp : x = SOURCE ptr, v = DESTINATION ptr.
//    AtomicWriteOp: x = DESTINATION ptr, expr = value to write.
//    YieldOp with value: b.create<omp::YieldOp>(loc, ValueRange{val}).
//
//  NOWAIT
//    Supported on WsloopOp (via WsloopOperands.nowait = UnitAttr) and
//    SingleOp (via SingleOperands.nowait = UnitAttr).
//    Detection: if a __kmpc_barrier call immediately follows the
//    end-of-construct call in the same block, the construct is synchronising
//    (no nowait).  If absent, set nowait = true.
//
// ── Assumptions ────────────────────────────────────────────────────────────
//   A. Outlined fns follow Clang naming and are called from exactly one fork.
//   B. Push calls precede the fork in the same block.
//   C. Paired begin/end calls are in the same block.
//   D. Static wsloop init/fini in the same block.
//   E. Dynamic wsloop triple in the same block (run before loop transforms).
//   F. GPU device modules (nvptx64/amdgcn) rejected upfront.
//
// ── Unsupported ────────────────────────────────────────────────────────────
//   __kmpc_reduce: needs mem2reg + struct layout analysis → emitWarning.
//   Task depend clauses: needs kmp_depend_info_t layout → dropped + warning.
//   GPU __kmpc_target_init/deinit → emitError (hard reject).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::enzyme {
#define GEN_PASS_DEF_LLVMTOOMP
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

#define DEBUG_TYPE "llvm-to-omp"

#define SW(s, p) ((s).starts_with(p))

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::omp;

//===----------------------------------------------------------------------===//
// §1  Utilities
//===----------------------------------------------------------------------===//
namespace {

static StringRef getCalleeName(LLVM::CallOp op) {
  if (auto c = op.getCallee()) return *c;
  return {};
}
static bool isCallTo(LLVM::CallOp op, StringRef n) {
  return getCalleeName(op) == n;
}
static bool calleeStartsWith(LLVM::CallOp op, StringRef p) {
  return SW(getCalleeName(op), p);
}

static LLVM::LLVMFuncOp resolveOutlinedFn(ModuleOp mod, LLVM::CallOp call,
                                           unsigned idx = 2) {
  if (call.getNumOperands() <= idx) return {};
  if (auto a = call.getOperand(idx).getDefiningOp<LLVM::AddressOfOp>())
    return mod.lookupSymbol<LLVM::LLVMFuncOp>(a.getGlobalName());
  return {};
}


static std::optional<int64_t> getConstInt(Value v) {
  if (!v) return {};
  if (auto c = v.getDefiningOp<LLVM::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) return ia.getInt();
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) return c.value();
  return {};
}

/// Like getConstInt but also handles `arith.constant N : index`
/// (arith::ConstantOp whose value attribute carries an index integer).
static std::optional<int64_t> getConstIndex(Value v) {
  if (!v) return {};
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return getConstInt(v);
}

/// Forward-scan block of `after` for the first call to `callee`.
static LLVM::CallOp findNextCallTo(Operation *after, StringRef callee) {
  // First try same-block scan (fast path, covers simple cases).
  bool found = false;
  for (Operation &op : *after->getBlock()) {
    if (!found) { found = (&op == after); continue; }
    if (auto c = dyn_cast<LLVM::CallOp>(&op))
      if (getCalleeName(c) == callee) return c;
  }

  // Cross-block BFS: follow successor edges.
  // Stop at function boundaries or if we visit too many blocks.
  SmallPtrSet<Block *, 16> visited;
  SmallVector<Block *, 8> worklist;

  // Seed with successors of after's block.
  for (Block *succ : after->getBlock()->getSuccessors())
    worklist.push_back(succ);

  while (!worklist.empty()) {
    Block *bb = worklist.pop_back_val();
    if (!visited.insert(bb).second) continue;

    // Scan this block for the target call.
    for (Operation &op : *bb) {
      if (auto c = dyn_cast<LLVM::CallOp>(&op))
        if (getCalleeName(c) == callee) return c;
    }

    // Continue to successors.
    for (Block *succ : bb->getSuccessors())
      worklist.push_back(succ);
  }
  return {};
}

/// Find the matching end call for `start`, correctly handling nesting.
/// `beginCallee` is the name of the opening call (same as start's callee).
/// `endCallee` is the name of the closing call.
/// Searches only within the same block as `start`.
static LLVM::CallOp findMatchingEnd(LLVM::CallOp start,
                                    StringRef beginCallee,
                                    StringRef endCallee) {
  int depth = 0;
  bool past = false;
  for (Operation &op : *start->getBlock()) {
    if (!past) { past = (&op == start.getOperation()); continue; }
    auto c = dyn_cast<LLVM::CallOp>(&op);
    if (!c) continue;
    StringRef n = getCalleeName(c);
    if (n == beginCallee) { ++depth; continue; }
    if (n == endCallee) {
      if (depth == 0) return c;
      --depth;
    }
  }
  return {};
}

/// Like findMatchingEnd but matches the begin by prefix rather than
/// exact name. Used for wsloop init/fini where the type suffix varies.
static LLVM::CallOp findMatchingEndByPrefix(LLVM::CallOp start,
                                             StringRef beginPrefix,
                                             StringRef endCallee) {
  int depth = 0;
  bool past = false;
  for (Operation &op : *start->getBlock()) {
    if (!past) { past = (&op == start.getOperation()); continue; }
    auto c = dyn_cast<LLVM::CallOp>(&op);
    if (!c) continue;
    StringRef n = getCalleeName(c);
    if (n.starts_with(beginPrefix)) { ++depth; continue; }
    if (n == endCallee) {
      if (depth == 0) return c;
      --depth;
    }
  }
  return {};
}

/// All ops strictly between begin and end in the same block.
static SmallVector<Operation *> opsBetween(Operation *begin, Operation *end) {
  SmallVector<Operation *> r;
  bool in = false;
  for (Operation &op : *begin->getBlock()) {
    if (&op == begin) { in = true; continue; }
    if (&op == end)   break;
    if (in)            r.push_back(&op);
  }
  return r;
}

/// Move ops before the last op of `tgt` (the terminator).
static void moveBeforeTerminator(SmallVectorImpl<Operation *> &ops, Block *tgt) {
  Operation *term = &tgt->back();
  for (Operation *op : ops) op->moveBefore(term);
}

//===----------------------------------------------------------------------===//
// §2  Nowait detection
//
// Clang emits __kmpc_barrier after constructs without the nowait clause.
// If the very next call after `endOp` in the same block is a barrier,
// the construct is synchronising (not nowait).  Otherwise → nowait.
//===----------------------------------------------------------------------===//

/// Returns true if NO implicit barrier follows `endOp` (nowait semantics).
/// If eraseBarrier and a barrier is found, erase it (so Phase-2 BarrierPat
/// won't double-convert it).
static bool detectNowait(Operation *endOp, bool eraseBarrier = false) {
  bool found = false;
  for (Operation &op : *endOp->getBlock()) {
    if (!found) { found = (&op == endOp); continue; }
    if (auto c = dyn_cast<LLVM::CallOp>(&op)) {
      bool isB = (getCalleeName(c) == "__kmpc_barrier" ||
                  getCalleeName(c) == "__kmpc_cancel_barrier");
      if (isB && eraseBarrier) c.erase();
      return !isB;
    }
    return true; // non-call op before any barrier → nowait
  }
  return true;
}

// The fini lives inside a scf.if then-block.
// The barrier lives in that scf.if's PARENT block.
// Walk up: fini->block->parentOp (the scf.if) -> its block (parallel body).
// static bool detectNowaitAcrossScfIf(LLVM::CallOp fini, bool eraseBarrier) {
//   // Try same-block first (the direct case, e.g. no scf.if wrapper)
//   if (!detectNowait(fini, /*eraseBarrier=*/false)) {
//     // barrier found in same block
//     if (eraseBarrier) {
//       // find and erase it
//       bool past = false;
//       for (Operation &op : *fini->getBlock()) {
//         if (!past) { past = (&op == fini.getOperation()); continue; }
//         if (auto c = dyn_cast<LLVM::CallOp>(&op)) {
//           if (getCalleeName(c) == "__kmpc_barrier" ||
//               getCalleeName(c) == "__kmpc_cancel_barrier") {
//             c.erase(); break;
//           }
//         }
//       }
//     }
//     return false; // not nowait
//   }

//   // Not found in same block — check parent block after the enclosing scf.if.
//   Operation *parentOp = fini->getBlock()->getParentOp(); // the scf.if
//   if (!parentOp) return true; // no parent, assume nowait
//   Block *parentBlock = parentOp->getBlock();
//   if (!parentBlock) return true;

//   bool past = false;
//   for (Operation &op : *parentBlock) {
//     if (&op == parentOp) { past = true; continue; }
//     if (!past) continue;
//     auto c = dyn_cast<LLVM::CallOp>(&op);
//     if (!c) break; // stop at first non-call (e.g. omp.terminator)
//     StringRef n = getCalleeName(c);
//     if (n == "__kmpc_barrier" || n == "__kmpc_cancel_barrier") {
//       if (eraseBarrier) c.erase();
//       return false; // barrier found → NOT nowait
//     }
//     break;
//   }
//   return true; // no barrier found at either level → nowait
// }



//===----------------------------------------------------------------------===//
// §3  Outlined-function inliner
//
// Drops args 0,1 (global_tid*, bound_tid*), maps rest to `caps`, clones
// all blocks, converts llvm.return → omp.terminator.
//===----------------------------------------------------------------------===//

static LogicalResult inlineOutlinedBody(LLVM::LLVMFuncOp fn, Region &dst,
                                         OpBuilder &b, ValueRange caps) {
  if (fn.getBody().empty())
    return fn.emitError("LLVMToOmp: outlined fn '") << fn.getName()
           << "' has empty body";
  IRMapping map;
  Block &entry = fn.getBody().front();

  // Map arg0 (global_tid*) and arg1 (bound_tid*) to undef so that
  // cloned ops don't hold live references into the source function's
  // block arguments after fn.erase().  These are OMP bookkeeping ptrs
  // unused by the structured dialect; any __kmpc_barrier calls that
  // consume the loaded gtid are handled by BarrierPat in Phase 2.
  for (unsigned i = 0; i < 2 && i < entry.getNumArguments(); ++i) {
    BlockArgument arg = entry.getArgument(i);
    map.map(arg, b.create<LLVM::UndefOp>(fn.getLoc(), arg.getType()));
  }

  // Map captured args (index 2 onwards) to caller-supplied values.
  for (unsigned i = 0, e = caps.size(); i < e; ++i) {
    unsigned ai = i + 2;
    if (ai < entry.getNumArguments())
      map.map(entry.getArgument(ai), caps[i]);
  }
  SmallVector<Block *> dblks;
  bool first = true;
  for (Block &sb : fn.getBody()) {
    Block *db;
    if (first) { db = &dst.front(); first = false; }
    else {
      db = b.createBlock(&dst);
      for (BlockArgument a : sb.getArguments())
        map.map(a, db->addArgument(a.getType(), a.getLoc()));
    }
    map.map(&sb, db);
    dblks.push_back(db);
  }
  unsigned idx = 0;
  for (Block &sb : fn.getBody()) {
    b.setInsertionPointToEnd(dblks[idx++]);
    for (Operation &op : sb)
      isa<LLVM::ReturnOp>(op) ? (void)b.create<omp::TerminatorOp>(op.getLoc())
                               : (void)b.clone(op, map);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// §4  Schedule mapping
//===----------------------------------------------------------------------===//

static omp::ClauseScheduleKind kmpSched(int64_t s) {
  // Mask off OpenMP 5.0 monotonic/nonmonotonic modifier bits (29, 30)
  // before matching against base schedule constants.
  s &= ~((int64_t)0x60000000);
  switch (s) {
  case 33: case 34: return omp::ClauseScheduleKind::Static;
  case 35:          return omp::ClauseScheduleKind::Dynamic;
  case 36:          return omp::ClauseScheduleKind::Guided;
  case 38:          return omp::ClauseScheduleKind::Auto;
  default:          return omp::ClauseScheduleKind::Static;
  }
}

//===----------------------------------------------------------------------===//
// §5  Atomic name parser
//===----------------------------------------------------------------------===//

struct AtomicDesc {
  enum class Op  { Unknown, Read, Write, Update, Capture };
  enum class Bin { Unknown, Add, Sub, Mul, Div, And, Or, Xor, Shl, Shr };
  Type elemType;
  Op   op  = Op::Unknown;
  Bin  bin = Bin::Unknown;
};

static AtomicDesc parseAtomicName(StringRef name, MLIRContext *ctx) {
  AtomicDesc d;
  StringRef r = name.drop_front(strlen("__kmpc_atomic_"));
  // Longest-match on type prefix (float10/16 before float1).
  struct TE { const char *p; unsigned n; Type (*mk)(MLIRContext*); };
  static const TE T[] = {
    {"fixed1_",7,[](MLIRContext*c){return(Type)IntegerType::get(c,8);}},
    {"fixed2_",7,[](MLIRContext*c){return(Type)IntegerType::get(c,16);}},
    {"fixed4_",7,[](MLIRContext*c){return(Type)IntegerType::get(c,32);}},
    {"fixed8_",7,[](MLIRContext*c){return(Type)IntegerType::get(c,64);}},
    {"float4_",7,[](MLIRContext*c){return(Type)Float32Type::get(c);}},
    {"float8_",7,[](MLIRContext*c){return(Type)Float64Type::get(c);}},
    {"float10_",8,[](MLIRContext*c){return(Type)Float80Type::get(c);}},
    {"float16_",8,[](MLIRContext*c){return(Type)Float128Type::get(c);}},
  };
  for (auto &t : T) {
    if (SW(r, t.p)) { d.elemType = t.mk(ctx); r = r.drop_front(t.n); break; }
  }
  if (!d.elemType) return d;
  if      (r == "rd")           d.op = AtomicDesc::Op::Read;
  else if (r == "wr")           d.op = AtomicDesc::Op::Write;
  else if (SW(r, "capture"))    d.op = AtomicDesc::Op::Capture;
  else {
    d.op = AtomicDesc::Op::Update;
    if      (SW(r,"add"))  d.bin = AtomicDesc::Bin::Add;
    else if (SW(r,"sub"))  d.bin = AtomicDesc::Bin::Sub;
    else if (SW(r,"mul"))  d.bin = AtomicDesc::Bin::Mul;
    else if (SW(r,"div"))  d.bin = AtomicDesc::Bin::Div;
    else if (SW(r,"andb")||SW(r,"andl")) d.bin = AtomicDesc::Bin::And;
    else if (SW(r,"orb") ||SW(r,"orl"))  d.bin = AtomicDesc::Bin::Or;
    else if (SW(r,"xor")) d.bin = AtomicDesc::Bin::Xor;
    else if (SW(r,"shl")) d.bin = AtomicDesc::Bin::Shl;
    else if (SW(r,"shr")) d.bin = AtomicDesc::Bin::Shr;
    else                   d.op  = AtomicDesc::Op::Unknown;
  }
  return d;
}

static Value emitBinop(OpBuilder &b, Location loc,
                       AtomicDesc::Bin bin,
                       Value lhs, Value rhs) {
  bool fp = isa<FloatType>(lhs.getType());

  switch (bin) {
  case AtomicDesc::Bin::Add:
    return fp
      ? b.create<LLVM::FAddOp>(loc, lhs, rhs).getResult()
      : b.create<LLVM::AddOp>(loc, lhs, rhs).getResult();

  case AtomicDesc::Bin::Sub:
    return fp
      ? b.create<LLVM::FSubOp>(loc, lhs, rhs).getResult()
      : b.create<LLVM::SubOp>(loc, lhs, rhs).getResult();

  case AtomicDesc::Bin::Mul:
    return fp
      ? b.create<LLVM::FMulOp>(loc, lhs, rhs).getResult()
      : b.create<LLVM::MulOp>(loc, lhs, rhs).getResult();

  case AtomicDesc::Bin::Div:
    return fp
      ? b.create<LLVM::FDivOp>(loc, lhs, rhs).getResult()
      : b.create<LLVM::SDivOp>(loc, lhs, rhs).getResult();

  case AtomicDesc::Bin::And:
    return b.create<LLVM::AndOp>(loc, lhs, rhs).getResult();

  case AtomicDesc::Bin::Or:
    return b.create<LLVM::OrOp>(loc, lhs, rhs).getResult();

  case AtomicDesc::Bin::Xor:
    return b.create<LLVM::XOrOp>(loc, lhs, rhs).getResult();

  case AtomicDesc::Bin::Shl:
    return b.create<LLVM::ShlOp>(loc, lhs, rhs).getResult();

  case AtomicDesc::Bin::Shr:
    return b.create<LLVM::LShrOp>(loc, lhs, rhs).getResult();

  default:
    return {};
  }
}

//===----------------------------------------------------------------------===//
// §6  Pending-clause accumulator  (__kmpc_push_* before fork)
//===----------------------------------------------------------------------===//

struct PendingClauses {
  Value numThreads;
  bool  hasProcBind = false;
  omp::ClauseProcBindKind procBind = omp::ClauseProcBindKind::Primary;
  Value numTeams, /*, numTeamsUpper*/ threadLimit;
};

static PendingClauses collectPendingClauses(LLVM::CallOp anchor) {
  PendingClauses pc;
  for (Operation &op : *anchor->getBlock()) {
    if (&op == anchor.getOperation()) break;
    auto call = dyn_cast<LLVM::CallOp>(&op);
    if (!call) continue;
    StringRef n = getCalleeName(call);
    if (n == "__kmpc_push_num_threads" && call.getNumOperands() >= 3)
      pc.numThreads = call.getOperand(2);
    else if (n == "__kmpc_push_proc_bind" && call.getNumOperands() >= 3) {
      pc.hasProcBind = true;
      if (auto v = getConstInt(call.getOperand(2))) {
        switch (*v) {
        case 1: pc.procBind = omp::ClauseProcBindKind::Primary; break;
        case 2: pc.procBind = omp::ClauseProcBindKind::Close;   break;
        case 3: pc.procBind = omp::ClauseProcBindKind::Spread;  break;
        }
      }
    } else if (n == "__kmpc_push_num_teams" && call.getNumOperands() >= 4) {
      pc.numTeams    = call.getOperand(2);
      pc.threadLimit = call.getOperand(3);
    }
  }
  return pc; // no erase vector, no erase loop
}

//===----------------------------------------------------------------------===//
// §7  AtomicUpdateOp region helper
//
// td: traits = [SingleBlockImplicitTerminator<"YieldOp">]
//       → region auto-created with ONE block + empty YieldOp at end.
//     arguments include explicit $x and $atomic_control (OptionalAttr).
//     The block arg = current value of *x (element type) must be added.
//     The implicit empty YieldOp must be replaced with one yielding the result.
//
// YieldOp with a value: b.create<omp::YieldOp>(loc, ValueRange{result})
//   (the explicit no-arg builder is just a convenience; ODS also generates
//    the variadic-operand form used here)
//===----------------------------------------------------------------------===//

static LogicalResult buildAtomicUpdateRegion(OpBuilder &b, Location loc,
                                              omp::AtomicUpdateOp updOp,
                                              Type elemTy,
                                              AtomicDesc::Bin bin, Value rhs) {
  // SingleBlockImplicitTerminator<"YieldOp"> → block already exists.
  Block *blk = &updOp.getRegion().front();
  BlockArgument xCurr = blk->addArgument(elemTy, loc); // current value of *x
  b.setInsertionPoint(blk->getTerminator()); // before implicit YieldOp
  Value result = emitBinop(b, loc, bin, xCurr, rhs);
  if (!result) return failure();
  blk->getTerminator()->erase();           // remove implicit empty YieldOp
  b.setInsertionPointToEnd(blk);
  // YieldOp with value:  assemblyFormat = "('(' $results^ ':' type($results) ')')?"
  b.create<omp::YieldOp>(loc, ValueRange{result});
  return success();
}

//===----------------------------------------------------------------------===//
// §8  Pass definition
//===----------------------------------------------------------------------===//
namespace mlir::enzyme {
namespace {

struct LLVMToOMPPass : ::mlir::enzyme::impl::LLVMToOMPBase<LLVMToOMPPass> {
  using ::mlir::enzyme::impl::LLVMToOMPBase<LLVMToOMPPass>::LLVMToOMPBase;

  void runOnOperation() override;
private:
  void convertTeams    (ModuleOp);
  void convertParallel (ModuleOp);
  void convertPaired   (ModuleOp);
  void convertStaticWs (ModuleOp);
  void convertDynWs    (ModuleOp);
  void convertTasks    (ModuleOp);
  void convertTaskloop (ModuleOp);
  void convertAtomics  (ModuleOp);
  void convertReductions(ModuleOp);
  void applyLeafPatterns(ModuleOp);
};
//===----------------------------------------------------------------------===//
// §9  __kmpc_fork_teams → omp.teams
//
// td: TeamsOp — explicit builder OpBuilder<(ins CArg<"const TeamsOperands &">)>
//     No SingleBlock → create entry block explicitly.
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertTeams(ModuleOp mod) {
  SmallVector<LLVM::CallOp> calls;
  mod.walk([&](LLVM::CallOp c){
    if (isCallTo(c,"__kmpc_fork_teams")) calls.push_back(c);
  });
  for (LLVM::CallOp call : calls) {
    PendingClauses pc = collectPendingClauses(call);
    LLVM::LLVMFuncOp fn = resolveOutlinedFn(mod, call);
    if (!fn) {
      call.emitWarning("LLVMToOmp: cannot resolve outlined fn for "
                       "__kmpc_fork_teams — left as llvm.call"); continue;
    }
    SmallVector<Value> caps;
    for (unsigned i = 3; i < call.getNumOperands(); ++i)
      caps.push_back(call.getOperand(i));
    OpBuilder b(call); Location loc = call.getLoc();

    omp::TeamsOperands tOprs;

    tOprs.ifExpr = Value{};

    if (pc.numTeams)
      tOprs.numTeamsLower = pc.numTeams;

    if (pc.threadLimit)
      tOprs.threadLimitVars = ValueRange(pc.threadLimit);
    else
      tOprs.threadLimitVars = ValueRange{};

    // optional fields
    tOprs.allocateVars = ValueRange{};
    tOprs.allocatorVars = ValueRange{};
    tOprs.privateVars = ValueRange{};


    // TeamsOperands fields (from OpenMP_NumTeamsClause, OpenMP_ThreadLimitClause):
    //   numTeams (Value), threadLimit (Value)
    auto teamsOp = b.create<omp::TeamsOp>(loc, tOprs);
    b.createBlock(&teamsOp.getRegion()); // no SingleBlock → explicit
    b.setInsertionPointToStart(&teamsOp.getRegion().front());
    if (failed(inlineOutlinedBody(fn, teamsOp.getRegion(), b, caps))) {
      teamsOp.erase();
      call.emitError("LLVMToOmp: inlining fork_teams failed");
      signalPassFailure(); return;
    }
    call.erase();
    fn.erase();
    LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] fork_teams → omp.teams\n");
  }
}

//===----------------------------------------------------------------------===//
// §10  __kmpc_fork_call / __kmpc_parallel_51 → omp.parallel
//
// td: ParallelOp — explicit builder OpBuilder<(ins CArg<"const ParallelOperands &">)>
//     No SingleBlock → create entry block explicitly.
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertParallel(ModuleOp mod) {
  SmallVector<LLVM::CallOp> calls;
  mod.walk([&](LLVM::CallOp c){
    if (isCallTo(c,"__kmpc_fork_call") || isCallTo(c,"__kmpc_parallel_51"))
      calls.push_back(c);
  });
  for (LLVM::CallOp call : calls) {
    PendingClauses pc = collectPendingClauses(call);
    // __kmpc_parallel_51: ident(0) gtid(1) if_val(2) num_threads(3)
    //                     proc_bind(4) fn_ptr(5) wrapper_ptr(6) ...
    bool is51 = isCallTo(call,"__kmpc_parallel_51");
    unsigned fnIdx = is51 ? 5 : 2;
    if (is51 && call.getNumOperands() > 3) pc.numThreads = call.getOperand(3);

    LLVM::LLVMFuncOp fn = resolveOutlinedFn(mod, call, fnIdx);
    if (!fn) {
      call.emitWarning("LLVMToOmp: cannot resolve outlined fn for "
                       "fork_call — left as llvm.call"); continue;
    }
    SmallVector<Value> caps;
    for (unsigned i = fnIdx + 1; i < call.getNumOperands(); ++i)
      caps.push_back(call.getOperand(i));
    OpBuilder b(call); Location loc = call.getLoc();
    MLIRContext *ctx = mod.getContext();

    // ParallelOperands fields (from clauses):
    //   numThreads  (Value, OpenMP_NumThreadsClause)
    //   procBindKind (ClauseProcBindKindAttr, OpenMP_ProcBindClause)
    omp::ClauseProcBindKindAttr procBindKind;

    if (pc.hasProcBind) {
      procBindKind = omp::ClauseProcBindKindAttr::get(ctx, pc.procBind);
    }

    omp::ParallelOperands ops;

    ops.ifExpr = Value{};

    if (pc.numThreads)
      ops.numThreadsVars = ValueRange(pc.numThreads);
    else
      ops.numThreadsVars = ValueRange{};

    ops.procBindKind = procBindKind;

    ops.privateVars = ValueRange{};
    ops.allocateVars = ValueRange{};
    ops.allocatorVars = ValueRange{};
    ops.reductionVars = ValueRange{};

    auto parallelOp = b.create<omp::ParallelOp>(loc, ops);

    b.createBlock(&parallelOp.getRegion()); // no SingleBlock → explicit
    b.setInsertionPointToStart(&parallelOp.getRegion().front());
    if (failed(inlineOutlinedBody(fn, parallelOp.getRegion(), b, caps))) {
      parallelOp.erase();
      call.emitError("LLVMToOmp: inlining parallel outlined fn failed");
      signalPassFailure(); return;
    }

    // grab before erasing
    // Value fnPtr = call.getOperand(fnIdx);
    // auto addrOfOp = fnPtr.getDefiningOp<LLVM::AddressOfOp>();

    call.erase();
    fn.erase();

    // kill the now-broken symbol ref
    // if (addrOfOp && addrOfOp->use_empty())
    //     addrOfOp->erase();
    LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] fork_call → omp.parallel\n");
  }
}

//===----------------------------------------------------------------------===//
// §11  Paired begin/end structural calls
//
//   single    / end_single    → omp.single   (OpenMP_NowaitClause ✓)
//   master    / end_master    → omp.master   (no clause builder)
//   critical  / end_critical  → omp.critical (OptionalAttr<FlatSymbolRefAttr>)
//   ordered   / end_ordered   → omp.ordered.region (OpenMP_ParallelizationLevelClause)
//   taskgroup / end_taskgroup → omp.taskgroup (OpenMP_TaskReductionClause)
//   sections                  → warn + leave
//
// None of these have SingleBlock → create blocks explicitly.
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertPaired(ModuleOp mod) {


  // ---- shared helper for scf.if-guarded constructs (single, master) ----
  //
  // Both __kmpc_single and __kmpc_master return an i32 that gates the body:
  //   %r  = llvm.call @__kmpc_BEGIN(...)
  //   %ne = arith.cmpi ne, %r, 0
  //   scf.if %ne {
  //     <body>
  //     llvm.call @__kmpc_END(...)
  //   }
  //   [optional __kmpc_barrier → no nowait]
  //
  auto convertIfGuarded = [&](StringRef beginCallee, StringRef endCallee,
                              auto makeOmpOp) {
    SmallVector<LLVM::CallOp> starts;
    mod.walk([&](LLVM::CallOp c) {
      if (isCallTo(c, beginCallee)) starts.push_back(c);
    });
    for (LLVM::CallOp start : starts) {
      OpBuilder b(start);
      Location loc = start.getLoc();

      // Replace the i32 guard result so cmpi users don't dangle after erase.
      if (!start.getResults().empty()) {
        Value one = b.create<LLVM::ConstantOp>(
            loc, b.getI32Type(), b.getI32IntegerAttr(1));
        start.getResult().replaceAllUsesWith(one);
      }

      // Find the scf.if in the same block whose then-region holds endCallee.
      scf::IfOp ifOp;
      bool pastStart = false;
      for (Operation &op : *start->getBlock()) {
        if (&op == start.getOperation()) { pastStart = true; continue; }
        if (!pastStart) continue;
        auto sif = dyn_cast<scf::IfOp>(&op);
        if (!sif) continue;
        bool hasEnd = false;
        sif.getThenRegion().walk([&](LLVM::CallOp c) {
          if (isCallTo(c, endCallee)) hasEnd = true;
        });
        if (hasEnd) { ifOp = sif; break; }
      }
      if (!ifOp) {
        start.emitError("LLVMToOmp: '") << beginCallee
            << "' without matching '" << endCallee << "'";
        signalPassFailure(); return;
      }

      // Create the omp op just before the scf.if.
      b.setInsertionPoint(ifOp);
      Operation *ompOp = makeOmpOp(b, loc, ifOp);
      Block *body = b.createBlock(&ompOp->getRegion(0));
      b.setInsertionPointToStart(body);
      b.create<omp::TerminatorOp>(loc);

      // Move body ops (everything before endCallee in the then-block).
      LLVM::CallOp endCall;
      ifOp.getThenRegion().walk([&](LLVM::CallOp c) {
        if (isCallTo(c, endCallee)) endCall = c;
      });
      SmallVector<Operation *> toMove;
      for (Operation &op : ifOp.getThenRegion().front()) {
        if (&op == endCall.getOperation() || isa<scf::YieldOp>(&op)) break;
        toMove.push_back(&op);
      }
      moveBeforeTerminator(toMove, body);

      endCall.erase();
      start.erase();
      ifOp.erase();
    }
  };

  // ---- single ----
  // td: SingleOp has clauses=[..., OpenMP_NowaitClause, ...]
  //     Builder: OpBuilder<(ins CArg<"const SingleOperands &">:$clauses)>
  //     SingleOperands.nowait = UnitAttr (from OpenMP_NowaitClause)
  convertIfGuarded(
    "__kmpc_single", "__kmpc_end_single",
    [&](OpBuilder &b, Location loc, scf::IfOp ifOp) -> Operation * {
      // Barrier after the scf.if signals no-nowait.
      bool nw = detectNowait(ifOp, /*eraseBarrier=*/false);
      omp::SingleOperands sOps;
      if (nw) sOps.nowait = b.getUnitAttr();
      LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] single → omp.single"
                               << (nw ? " nowait" : "") << "\n");
      return b.create<omp::SingleOp>(loc, sOps).getOperation();
    });

  // ---- master ----
  // td: MasterOp — no clause builder, `$region attr-dict`. No explicit args.
  convertIfGuarded(
    "__kmpc_master", "__kmpc_end_master",
    [&](OpBuilder &b, Location loc, scf::IfOp) -> Operation * {
      LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] master → omp.master\n");
      return b.create<omp::MasterOp>(loc).getOperation();
    });

    // ---- masked ----
// td: MaskedOp — clauses=[OpenMP_FilterClause]
//     Builder: OpBuilder<(ins CArg<"const MaskedOperands &">:$clauses)>
//     MaskedOperands.filteredThreadId = Value (optional i32 filter thread ID)
//     __kmpc_masked(ident, gtid, filter) → i32 guard, same scf.if pattern as
//     master/single.  filter=0 is tid==master; filter>0 targets that specific tid.
  {
    SmallVector<LLVM::CallOp> starts;
    mod.walk([&](LLVM::CallOp c) {
      if (isCallTo(c, "__kmpc_masked")) starts.push_back(c);
    });
    for (LLVM::CallOp start : starts) {
      OpBuilder b(start);
      Location loc = start.getLoc();

      // filter value = arg 2 of __kmpc_masked(ident, gtid, filter)
      Value filterVal = start.getNumOperands() > 2 ? start.getOperand(2) : Value{};

      // Replace guard result with constant 1 so cmpi users don't dangle after erase.
      if (!start.getResults().empty()) {
        Value one = b.create<LLVM::ConstantOp>(
            loc, b.getI32Type(), b.getI32IntegerAttr(1));
        start.getResult().replaceAllUsesWith(one);
      }

      // Find the scf.if in the same block whose then-region holds __kmpc_end_masked.
      scf::IfOp ifOp;
      bool pastStart = false;
      for (Operation &op : *start->getBlock()) {
        if (&op == start.getOperation()) { pastStart = true; continue; }
        if (!pastStart) continue;
        auto sif = dyn_cast<scf::IfOp>(&op);
        if (!sif) continue;
        bool hasEnd = false;
        sif.getThenRegion().walk([&](LLVM::CallOp c) {
          if (isCallTo(c, "__kmpc_end_masked")) hasEnd = true;
        });
        if (hasEnd) { ifOp = sif; break; }
      }
      if (!ifOp) {
        start.emitError("LLVMToOmp: '__kmpc_masked' without matching "
                        "'__kmpc_end_masked'");
        signalPassFailure(); return;
      }

      // Build omp.masked before the scf.if.
      b.setInsertionPoint(ifOp);
      omp::MaskedOperands mOps;
      if (filterVal)
        mOps.filteredThreadId = filterVal;
      auto maskedOp = b.create<omp::MaskedOp>(loc, mOps);
      Block *body = b.createBlock(&maskedOp.getRegion());
      b.setInsertionPointToStart(body);
      b.create<omp::TerminatorOp>(loc);

      // Move body ops (everything before __kmpc_end_masked in the then-block).
      LLVM::CallOp endCall;
      ifOp.getThenRegion().walk([&](LLVM::CallOp c) {
        if (isCallTo(c, "__kmpc_end_masked")) endCall = c;
      });
      SmallVector<Operation *> toMove;
      for (Operation &op : ifOp.getThenRegion().front()) {
        if (&op == endCall.getOperation() || isa<scf::YieldOp>(&op)) break;
        toMove.push_back(&op);
      }
      moveBeforeTerminator(toMove, body);

      endCall.erase();
      start.erase();
      ifOp.erase();
      LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] masked → omp.masked"
                << (filterVal ? " (filter)" : "") << "\n");
    }
  }

  // ---- critical / critical_with_hint ----
  // td: CriticalOp — arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$name)
  //     assemblyFormat = [("(" $name^ ")")? $region attr-dict]
  {
    SmallVector<LLVM::CallOp> starts;
    mod.walk([&](LLVM::CallOp c){
      if (isCallTo(c,"__kmpc_critical") ||
          isCallTo(c,"__kmpc_critical_with_hint"))
        starts.push_back(c);
    });
    for (LLVM::CallOp start : starts) {
      LLVM::CallOp end = findMatchingEnd(start,
                                   "__kmpc_critical",
                                   "__kmpc_end_critical");
      if (!end) {
        start.emitError("LLVMToOmp: __kmpc_critical without matching end");
        signalPassFailure(); return;
      }
      OpBuilder b(start); Location loc = start.getLoc();
      // name = FlatSymbolRefAttr{} means unnamed (anonymous critical).
      // Could derive from the kmp_critical_name* global; left unnamed here.
      auto critOp = b.create<omp::CriticalOp>(loc, FlatSymbolRefAttr{});
      Block *body = b.createBlock(&critOp.getRegion());
      b.setInsertionPointToStart(body);
      b.create<omp::TerminatorOp>(loc);
      auto ops = opsBetween(start, end);
      moveBeforeTerminator(ops, body);
      start.erase(); end.erase();
      LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] critical → omp.critical\n");
    }
  }

  // ---- ordered ----
  // td: OrderedRegionOp — clauses=[OpenMP_ParallelizationLevelClause]
  //     Builder: OpBuilder<(ins CArg<"const OrderedRegionOperands &">:$clauses)>
  //     parallelized = UnitAttr when simd.  __kmpc_ordered is always non-simd.
  {
    SmallVector<LLVM::CallOp> starts;
    mod.walk([&](LLVM::CallOp c){
      if (isCallTo(c,"__kmpc_ordered")) starts.push_back(c);
    });
    for (LLVM::CallOp start : starts) {
      LLVM::CallOp end = findMatchingEnd(start,
                                   "__kmpc_ordered",
                                   "__kmpc_end_ordered");
      if (!end) {
        start.emitError("LLVMToOmp: __kmpc_ordered without matching end");
        signalPassFailure(); return;
      }
      OpBuilder b(start); Location loc = start.getLoc();
      omp::OrderedRegionOperands ordOps; // parallelized not set → non-simd
      auto ordOp = b.create<omp::OrderedRegionOp>(loc, ordOps);
      Block *body = b.createBlock(&ordOp.getRegion());
      b.setInsertionPointToStart(body);
      b.create<omp::TerminatorOp>(loc);
      auto ops = opsBetween(start, end);
      moveBeforeTerminator(ops, body);
      start.erase(); end.erase();
      LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] ordered → omp.ordered.region\n");
    }
  }

  // ---- taskgroup ----
  // td: TaskgroupOp — clauses=[OpenMP_AllocateClause, OpenMP_TaskReductionClause]
  //     Builder: OpBuilder<(ins CArg<"const TaskgroupOperands &">:$clauses)>
  //     Assembly: custom<TaskReductionRegion>($region, $task_reduction_vars,
  //               type($task_reduction_vars), $task_reduction_byref,
  //               $task_reduction_syms)
  //     NOTE: fields are task_reduction_vars/byref/syms — NOT reduction_vars.
  {
    SmallVector<LLVM::CallOp> starts;
    mod.walk([&](LLVM::CallOp c){
      if (isCallTo(c,"__kmpc_taskgroup")) starts.push_back(c);
    });
    for (LLVM::CallOp start : starts) {
      LLVM::CallOp end = findMatchingEnd(start,
                                   "__kmpc_taskgroup",
                                   "__kmpc_end_taskgroup");
      if (!end) {
        start.emitWarning("LLVMToOmp: __kmpc_taskgroup without matching end"
                          " — left as llvm.call"); continue;
      }
      OpBuilder b(start); Location loc = start.getLoc();
      omp::TaskgroupOperands tgOps; // task_reduction_* all empty
      auto tgOp = b.create<omp::TaskgroupOp>(loc, tgOps);
      Block *body = b.createBlock(&tgOp.getRegion());
      b.setInsertionPointToStart(body);
      b.create<omp::TerminatorOp>(loc);
      auto ops = opsBetween(start, end);
      moveBeforeTerminator(ops, body);
      start.erase(); end.erase();
      LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] taskgroup → omp.taskgroup\n");
    }
  }

  // ---- sections ----
  // Full recovery needs CFG analysis of the switch dispatch pattern.
  // Warn and leave as-is.
  mod.walk([&](LLVM::CallOp c){
    if (isCallTo(c,"__kmpc_sections_init") || isCallTo(c,"__kmpc_next_section"))
      c.emitWarning("LLVMToOmp: sections not yet converted — left as llvm.call");
  });
}

//===----------------------------------------------------------------------===//
// §12  Static work-sharing loop  (__kmpc_for_static_init_* / _fini)
//
// td: WsloopOp traits = [..., NoTerminator, ..., SingleBlock]
//       → region.front() already exists after build(); NEVER add omp.terminator.
//     WsloopOp clauses include OpenMP_NowaitClause and OpenMP_ScheduleClause.
//     WsloopOperands.nowait = UnitAttr (OpenMP_NowaitClause)
//     WsloopOperands.schedule = ClauseScheduleKindAttr (OpenMP_ScheduleClause)
//
// td: LoopNestOp traits = [RecursiveMemoryEffects, SameVariadicOperandSize]
//       → no SingleBlock; create block explicitly and add IV block arg.
//     LoopNestOperands.loopLowerBounds/loopUpperBounds/loopSteps
//       come from OpenMP_LoopRelatedClause.
//     IVs = region.getArguments() (entry block args).
//     Terminated by omp.yield.
//
// Nowait detection: is there a __kmpc_barrier immediately after fini?
//===----------------------------------------------------------------------===//

// //===----------------------------------------------------------------------===//
// Helper 1: trace pre-init stored value through pointer2memref + memref.store
//
// Clang stores lb/ub/stride into the alloca ptrs BEFORE the init call:
//   pointer2memref(ptr) → memref.store val, [0]
// We need `val` (the original full-range bound), not a load from ptr.
//===----------------------------------------------------------------------===//

static Value traceLastStoredValue(Value ptr, Operation *before) {
  Value result;
  Operation *latestStore = nullptr;
  for (Operation *user : ptr.getUsers()) {
    // Must be the pointer2memref bridge
    if (user->getName().getStringRef() != "enzymexla.pointer2memref") continue;
    if (user->getNumResults() != 1) continue;
    Value p2m = user->getResult(0);
    for (Operation *su : p2m.getUsers()) {
      auto st = dyn_cast<memref::StoreOp>(su);
      if (!st) continue;
      // Must be in the same block as `before` and precede it
      if (st->getBlock() != before->getBlock()) continue;
      if (!st->isBeforeInBlock(before)) continue;
      // Keep the most recent store (handles re-stores)
      if (!latestStore || latestStore->isBeforeInBlock(st)) {
        latestStore = st;
        result = st.getValueToStore();
      }
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Helper 2: locate the scf.while between init and fini
//
// Clang nests it inside an "lb <= ub" scf.if guard:
//   init
//   [ub-clamp ops]
//   scf.if (lb <= ub) {
//     scf.while (%i = lb_val) { body; condition } do { yield }
//   }
//   fini
//===----------------------------------------------------------------------===//

static scf::WhileOp findWhileOpBetween(Operation *init, Operation *fini) {
  scf::WhileOp result;
  bool inRange = false;
  for (Operation &op : *init->getBlock()) {
    if (&op == init) { inRange = true; continue; }
    if (&op == fini) break;
    if (!inRange) continue;
    // Direct while in the same block
    if (auto w = dyn_cast<scf::WhileOp>(&op)) { return w; }
    // While nested inside an scf.if guard
    if (auto sif = dyn_cast<scf::IfOp>(&op)) {
      sif.getThenRegion().walk([&](scf::WhileOp w) {
        if (!result) result = w;
      });
      if (result) return result;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Helper 3: clone only the per-iteration work from the while's before-block
//           into nestBlk, remapping the while IV → (possibly extended) nestIV.
//
// Clang's before-block mixes body work and loop-control in one block:
//
//   %28 = muli %iv, %iv           ← BODY
//   %29 = gep arr_ptr[%iv]        ← BODY
//   %30 = trunci %28              ← BODY  (feeds memref.store)
//   memref.store %30, p2m(%29)[0] ← BODY
//   %32 = addi %iv, 1             ← CONTROL (step)
//   %33 = trunci %32              ← CONTROL (for comparison)
//   %34 = cmpi ne, %ub1, %33      ← CONTROL (condition)
//   scf.condition(%34) %32        ← CONTROL (terminator)
//
// Strategy: backward-reachability from scf.condition marks control ops;
//           everything else is body and gets cloned.
//===----------------------------------------------------------------------===//

static void inlineWhileBodyIntoNest(OpBuilder &b, Location loc,
                                    Block *nestBlk,
                                    scf::WhileOp whileOp,
                                    BlockArgument nestIV) {
  Block *beforeBlk = &whileOp.getBefore().front();
  BlockArgument whileIV = beforeBlk->getArgument(0);

  // ── Backward reachability from scf.condition → identify control ops ──
  SmallPtrSet<Operation *, 8> controlOps;
  Operation *condOp = beforeBlk->getTerminator(); // scf.condition
  SmallVector<Value> worklist(condOp->getOperands().begin(),
                               condOp->getOperands().end());
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    Operation *def = v.getDefiningOp();
    // Stop at block arguments (whileIV) or values from outer scope
    if (!def || def->getBlock() != beforeBlk) continue;
    if (!controlOps.insert(def).second) continue; // already visited
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }

  // ── Build IRMapping: while IV → (possibly sign-extended) nestIV ──────
  IRMapping map;
  b.setInsertionPoint(nestBlk->getTerminator());
  Value mappedIV = nestIV;
  if (whileIV.getType() != nestIV.getType()) {
    // e.g. nestIV : i32  →  whileIV : i64  (Clang extends before looping)
    mappedIV = b.create<arith::ExtSIOp>(loc, whileIV.getType(), nestIV);
  }
  map.map(whileIV, mappedIV);

  // ── Clone body ops (non-control) before the omp.yield terminator ─────
  b.setInsertionPoint(nestBlk->getTerminator());
  for (Operation &op : beforeBlk->without_terminator()) {
    if (!controlOps.contains(&op))
      b.clone(op, map);
  }
}

//===----------------------------------------------------------------------===//
// Helper 4: find first scf.for between init and fini (possibly in scf.if)
// Used for the sections lowering where Clang emits scf.for instead of
// scf.while.
//===----------------------------------------------------------------------===//

static scf::ForOp findForOpBetween(Operation *init, Operation *fini) {
  scf::ForOp result;
  bool inRange = false;
  for (Operation &op : *init->getBlock()) {
    if (&op == init) { inRange = true; continue; }
    if (&op == fini) break;
    if (!inRange) continue;
    // Direct for in the same block
    if (auto f = dyn_cast<scf::ForOp>(&op)) return f;
    // For nested inside the lb<=ub guard scf.if
    if (auto sif = dyn_cast<scf::IfOp>(&op)) {
      sif.getThenRegion().walk([&](scf::ForOp f) {
        if (!result) result = f;
      });
      if (result) return result;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Helper 5: clone the scf.for body into nestBlk, remapping the for IV to
// nestIV.  Unlike the while case there are no mixed control ops — the entire
// body (minus the scf.yield terminator) is the per-iteration work.
//===----------------------------------------------------------------------===//

static void inlineForBodyIntoNest(OpBuilder &b, Location loc,
                                   Block *nestBlk,
                                   scf::ForOp forOp,
                                   BlockArgument nestIV) {
  Block *bodyBlk = forOp.getBody();
  Value forIV = forOp.getInductionVar();  // ← Value, not BlockArgument

  IRMapping map;
  b.setInsertionPoint(nestBlk->getTerminator());

  Value mappedIV = nestIV;
  if (forIV.getType() != nestIV.getType()) {
    if (isa<IndexType>(forIV.getType()))
      mappedIV = b.create<arith::IndexCastUIOp>(loc, forIV.getType(), nestIV);
    else
      mappedIV = b.create<arith::ExtSIOp>(loc, forIV.getType(), nestIV);
  }
  map.map(forIV, mappedIV);

  for (Operation &op : bodyBlk->without_terminator())
    b.clone(op, map);
}

//===----------------------------------------------------------------------===//
// Rewritten convertStaticWs
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertStaticWs(ModuleOp mod) {
  SmallVector<LLVM::CallOp> inits;
  mod.walk([&](LLVM::CallOp c) {
    if (calleeStartsWith(c, "__kmpc_for_static_init_")) inits.push_back(c);
  });

  for (LLVM::CallOp init : inits) {
    LLVM::CallOp fini = findMatchingEndByPrefix(
        init, "__kmpc_for_static_init_", "__kmpc_for_static_fini");
    if (!fini) {
      init.emitWarning("LLVMToOmp: for_static_init without fini — skipped");
      continue;
    }
    if (init->getBlock() != fini->getBlock()) {
      init.emitWarning(
          "LLVMToOmp: init/fini in different blocks (optimized CFG) — skipped");
      continue;
    }

    bool nw = detectNowait(fini, /*eraseBarrier=*/true);

    StringRef sfx = getCalleeName(init).drop_front(
        strlen("__kmpc_for_static_init_"));
    MLIRContext *ctx = mod.getContext();
    Type iterTy = SW(sfx, "8") ? (Type)IntegerType::get(ctx, 64)
                                : (Type)IntegerType::get(ctx, 32);
    
    // to my understanding the sched can only be static, 
    // since otherwise no __kmpc_for_static_init_ would have been generated
    omp::ClauseScheduleKind sched = omp::ClauseScheduleKind::Static;
    if (auto sv = getConstInt(init.getNumOperands() > 2
                                  ? init.getOperand(2) : Value{}))
      sched = kmpSched(*sv);
    Value chunk = init.getNumOperands() > 8 ? init.getOperand(8) : Value{};

    // Args: ident(0) gtid(1) sched(2) p_last(3) p_lb(4) p_ub(5) p_stride(6)
    Value lbPtr = init.getNumOperands() > 4 ? init.getOperand(4) : Value{};
    Value ubPtr = init.getNumOperands() > 5 ? init.getOperand(5) : Value{};
    Value stPtr = init.getNumOperands() > 6 ? init.getOperand(6) : Value{};
    if (!lbPtr || !ubPtr || !stPtr) {
      init.emitWarning("LLVMToOmp: for_static_init missing ptr args — skipped");
      continue;
    }

    // ── FIX 1: trace pre-init values through pointer2memref + memref.store ─
    // These are the ORIGINAL full-range bounds (before the runtime writes
    // per-thread chunks back into the same allocas).  omp.wsloop handles
    // the chunking itself via schedule(static).
    Value lb = traceLastStoredValue(lbPtr, init); // e.g. 0
    Value ub = traceLastStoredValue(ubPtr, init); // e.g. n-1  (inclusive)
    Value st = traceLastStoredValue(stPtr, init); // e.g. 1
    if (!lb || !ub || !st) {
      init.emitWarning(
          "LLVMToOmp: cannot trace wsloop bounds via pointer2memref — "
          "compile with -O0 or pre-run mem2reg");
      continue;
    }

    // ── FIX 2: find the loop body — scf.while for normal loops,
    //           scf.for for sections lowering (Clang uses index dispatch).
    scf::WhileOp whileOp = findWhileOpBetween(init, fini);
    scf::ForOp   forOp;
    if (!whileOp)
      forOp = findForOpBetween(init, fini);

    if (!whileOp && !forOp) {
      init.emitWarning(
          "LLVMToOmp: no loop body (scf.while or scf.for) found between "
          "init/fini — skipped");
      continue;
    }

    OpBuilder b(init);
    Location loc = init.getLoc();

    // ── Build omp.wsloop ────────────────────────────────────────────────
    omp::WsloopOperands wOps;
    wOps.scheduleKind = omp::ClauseScheduleKindAttr::get(ctx, sched);
    if (chunk) wOps.scheduleChunk = chunk;
    if (nw)    wOps.nowait        = b.getUnitAttr();

    auto wsOp = b.create<omp::WsloopOp>(loc, wOps);
    Block *wsBlk = b.createBlock(&wsOp.getRegion());
    b.setInsertionPointToStart(wsBlk);

    // ── Build omp.loop_nest ─────────────────────────────────────────────
    // The traced `ub` is inclusive (Clang stores n-1).
    // Use loopInclusive so the dialect treats it as [lb, ub] closed interval.
    omp::LoopNestOperands lnOps;
    lnOps.loopLowerBounds = {lb};
    lnOps.loopUpperBounds = {ub};
    lnOps.loopSteps       = {st};
    lnOps.loopInclusive   = b.getUnitAttr(); // ub is inclusive (n-1, not n)

    auto nestOp = b.create<omp::LoopNestOp>(loc, lnOps);
    Block *nestBlk = b.createBlock(&nestOp.getRegion());
    nestBlk->addArgument(iterTy, loc); // the IV, getIVs()[0]
    b.setInsertionPointToStart(nestBlk);
    b.create<omp::YieldOp>(loc); // placeholder; body inserted before this

    // ── FIX 3: inline the per-iteration body ────────────────────────────────
    if (whileOp)
      inlineWhileBodyIntoNest(b, loc, nestBlk, whileOp, nestBlk->getArgument(0));
    else
      inlineForBodyIntoNest(b, loc, nestBlk, forOp, nestBlk->getArgument(0));

    // ── Erase everything between init and fini (now superseded) ─────────
    // Reverse order so inner uses are dropped before outer defs.
    auto between = opsBetween(init, fini);
    for (Operation *op : llvm::reverse(between)) {
      // Some ops (like scf.while result) may be use-empty after clone;
      // others (like lifetime.start ops) have no results and can always go.
      op->dropAllUses();
      op->erase();
    }

    init.erase();
    fini.erase();

    LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] static wsloop (ptr2memref bounds)"
                             << (nw ? " nowait" : "") << "\n");
  }
}

//===----------------------------------------------------------------------===//
// §13  Dynamic work-sharing loop  (__kmpc_dispatch_init/next/fini)
//
// Same WsloopOp/LoopNestOp API as §12.
// Body ops are between dispatch_next and dispatch_fini.
// clang outputs __kmpc_for_static_init_* __kmpc_for_static_fini per default
// when using schedule (dynamic|runtime|guided|auto), it outputs get __kmpc_dispatch_init_*
// + __kmpc_dispatch_next_* +__kmpc_dispatch_next_* + __kmpc_dispatch_deinit, 
// with the third argument of __kmpc_dispatch_init_* being differently
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertDynWs(ModuleOp mod) {
  SmallVector<LLVM::CallOp> inits;

  mod.walk([&](LLVM::CallOp c) {
    if (calleeStartsWith(c, "__kmpc_dispatch_init_"))
      inits.push_back(c);
  });

  for (LLVM::CallOp init : inits) {
    // Match the corresponding dispatch deinit/fini.
    // Modern runtimes use __kmpc_dispatch_deinit while
    // older runtimes may still emit __kmpc_dispatch_fini_N.
    LLVM::CallOp deinit = findMatchingEndByPrefix(
        init,
        "__kmpc_dispatch_init_",
        "__kmpc_dispatch_deinit");

    if (!deinit) {
      StringRef sfx = getCalleeName(init).drop_front(
          strlen("__kmpc_dispatch_init_"));

      auto finiN = ("__kmpc_dispatch_fini_" + sfx).str();

      deinit = findMatchingEndByPrefix(
          init,
          "__kmpc_dispatch_init_",
          finiN);
    }

    if (!deinit) {
      init.emitWarning(
          "LLVMToOmp: dispatch_init without deinit — skipped");
      continue;
    }

    // Current implementation only supports the simple
    // single-block canonical lowering.
    if (init->getBlock() != deinit->getBlock()) {
      init.emitWarning(
          "LLVMToOmp: init/deinit in different blocks "
          "(optimized CFG) — skipped");
      continue;
    }

    bool nowait =
        detectNowait(deinit, /*eraseBarrier=*/false);

    StringRef sfx = getCalleeName(init).drop_front(
        strlen("__kmpc_dispatch_init_"));

    MLIRContext *ctx = mod.getContext();

    Type iterTy =
        SW(sfx, "8")
            ? (Type)IntegerType::get(ctx, 64)
            : (Type)IntegerType::get(ctx, 32);

    // OpenMP runtimes OR scheduling modifiers into the
    // high bits of the schedule value:
    //
    //   bit 29 -> monotonic
    //   bit 30 -> nonmonotonic
    //
    // Strip them before mapping to omp schedule kinds.
    omp::ClauseScheduleKind sched =
        omp::ClauseScheduleKind::Dynamic;

    if (auto sv = getConstInt(
            init.getNumOperands() > 2
                ? init.getOperand(2)
                : Value{})) {
      int64_t base = *sv & ~((int64_t)0x60000000);
      sched = kmpSched(base);
    }

    // dispatch_init arguments:
    //
    //   dispatch_init(ident, gtid, sched,
    //                 lb, ub, step, chunk)
    //
    // Bounds are passed directly as values.
    Value lb =
        init.getNumOperands() > 3
            ? init.getOperand(3)
            : Value{};

    Value ub =
        init.getNumOperands() > 4
            ? init.getOperand(4)
            : Value{};

    Value step =
        init.getNumOperands() > 5
            ? init.getOperand(5)
            : Value{};

    Value chunk =
        init.getNumOperands() > 6
            ? init.getOperand(6)
            : Value{};

    if (!lb || !ub || !step) {
      init.emitWarning(
          "LLVMToOmp: dispatch_init missing bounds — skipped");
      continue;
    }

    // The current canonical lowering emits a single
    // scf.while representing the actual iteration loop.
    scf::WhileOp iterWhile =
        findWhileOpBetween(init, deinit);

    if (!iterWhile) {
      init.emitWarning(
          "LLVMToOmp: no dispatch while found — skipped");
      continue;
    }

    OpBuilder b(init);
    Location loc = init.getLoc();

    // Create omp.wsloop carrying the recovered
    // scheduling information.
    omp::WsloopOperands wsOps;
    wsOps.scheduleKind =
        omp::ClauseScheduleKindAttr::get(ctx, sched);

    if (chunk)
      wsOps.scheduleChunk = chunk;

    if (nowait)
      wsOps.nowait = b.getUnitAttr();

    auto wsOp =
        b.create<omp::WsloopOp>(loc, wsOps);

    b.createBlock(&wsOp.getRegion());
    b.setInsertionPointToStart(
        &wsOp.getRegion().front());

    // dispatch_init upper bounds are inclusive.
    omp::LoopNestOperands lnOps;
    lnOps.loopLowerBounds = {lb};
    lnOps.loopUpperBounds = {ub};
    lnOps.loopSteps = {step};
    lnOps.loopInclusive = b.getUnitAttr();

    auto nestOp =
        b.create<omp::LoopNestOp>(loc, lnOps);

    Block *nestBlk =
        b.createBlock(&nestOp.getRegion());

    nestBlk->addArgument(iterTy, loc);

    b.setInsertionPointToStart(nestBlk);
    b.create<omp::YieldOp>(loc);

    // Clone the original loop body while remapping
    // the original induction variable to the new
    // omp.loop_nest IV.
    inlineWhileBodyIntoNest(
        b,
        loc,
        nestBlk,
        iterWhile,
        nestBlk->getArgument(0));

    // Remove the original runtime-dispatch lowering.
    SmallVector<Operation *> between =
        opsBetween(init, deinit);

    for (Operation *op : llvm::reverse(between)) {
      op->dropAllUses();
      op->erase();
    }

    init.erase();
    deinit.erase();

    LLVM_DEBUG(
        llvm::dbgs()
        << "[LLVMToOmp] dynamic wsloop -> omp.wsloop"
        << (nowait ? " nowait" : "")
        << "\n");
  }
}


//===----------------------------------------------------------------------===//
// §14  Tasks  (__kmpc_task_alloc + __kmpc_omp_task → omp.task)
//
// ── Calling conventions ──────────────────────────────────────────────────
//   Parallel/teams outlined fn: (global_tid*, bound_tid*, cap0, cap1, …)
//     → inlineOutlinedBody drops args 0,1; maps variadic tail as caps.
//   Task entry fn:              (i32 gtid, ptr kmp_task_t*)
//     → ONLY TWO args; captures live in kmp_task_t.shareds (ptr at [0]).
//
// ── Capture layout (Clang's kmp_task_t initialisation) ───────────────────
//   Between __kmpc_task_alloc and __kmpc_omp_task Clang emits:
//     td        = task_alloc(…)                          : !llvm.ptr
//     p2m_td    = pointer2memref(td)                     : memref<?x!llvm.ptr>
//     shareds   = memref.load(p2m_td)[0]                 : !llvm.ptr
//     p2m_sh    = pointer2memref(shareds)                : memref<?x!llvm.ptr>
//     memref.store cap_i, p2m_sh[i]
//   harvestTaskCaptures() traces this chain → {index→Value} pairs.
//
// ── Inlining ─────────────────────────────────────────────────────────────
//   inlineTaskBody() maps arg0→undef, arg1→undef, then pre-maps every
//   cap-load result  memref.load(p2m(shareds_ptr))[i]  to caps[i] in the
//   IRMapping before cloning begins.  Ops whose results are fully
//   pre-mapped are skipped during cloning; the dead p2m/load chain that
//   fed them is left for DCE.
//
// ── Erase order ──────────────────────────────────────────────────────────
//   1. taskCall.erase()           – removes the __kmpc_omp_task consumer of td
//   2. complete_if0 erased        – removes if0 completion call (also uses td)
//   3. eraseTaskSharedsInit()     – removes the shareds store chain (uses td)
//   4. alloc.erase()              – td is now use-empty
//   5. fn.erase()                 – outlined entry fn no longer needed
//
// td: TaskOp — explicit builder OpBuilder<(ins CArg<"const TaskOperands &">)>
//     No SingleBlock → create block explicitly.
//     TaskOperands.untied    = UnitAttr (OpenMP_UntiedClause)
//     TaskOperands.mergeable = UnitAttr (OpenMP_MergeableClause)
//===----------------------------------------------------------------------===//

/// Trace the kmp_task_t initialisation chain from `alloc`'s result and
/// collect the captured values as (index, Value) pairs.
///
/// Recognises the memref+pointer2memref pattern emitted by EnzymeXLA:
///   td        = __kmpc_task_alloc(…)
///   p2m_td    = pointer2memref(td)        : memref<?x!llvm.ptr>
///   shareds   = memref.load(p2m_td)[0]   : !llvm.ptr
///   p2m_sh    = pointer2memref(shareds)   : memref<?x!llvm.ptr>
///   memref.store cap_i, p2m_sh[i]
///
/// Also recognises a plain llvm.load for the first-level dereference
/// (shareds pointer extraction) as a fallback.
static SmallVector<std::pair<int64_t, Value>>
harvestTaskCaptures(LLVM::CallOp alloc) {
  SmallVector<std::pair<int64_t, Value>> result;
  Value td = alloc.getResult();

  // ─── Step 1: find shareds_ptr ─────────────────────────────────────────
  Value sharedsPtr;
  for (Operation *u : td.getUsers()) {
    // memref path: pointer2memref(td) → memref.load[0]
    if (u->getName().getStringRef() == "enzymexla.pointer2memref" &&
        u->getNumResults() == 1) {
      Value p2m = u->getResult(0);
      for (Operation *lu : p2m.getUsers()) {
        auto ld = dyn_cast<memref::LoadOp>(lu);
        if (!ld || ld.getIndices().size() != 1) continue;
        auto idx = getConstIndex(ld.getIndices()[0]);
        if (idx && *idx == 0) { sharedsPtr = ld.getResult(); break; }
      }
    }
    // LLVM path: llvm.load td → shareds_ptr  (non-memref lowering)
    if (!sharedsPtr)
      if (auto llvmLd = dyn_cast<LLVM::LoadOp>(u))
        sharedsPtr = llvmLd.getResult();
    if (sharedsPtr) break;
  }
  if (!sharedsPtr) return result;

  // ─── Step 2: collect stores into shareds ──────────────────────────────
  for (Operation *u : sharedsPtr.getUsers()) {
    if (u->getName().getStringRef() == "enzymexla.pointer2memref" &&
        u->getNumResults() == 1) {
      Value p2m = u->getResult(0);
      for (Operation *su : p2m.getUsers()) {
        auto st = dyn_cast<memref::StoreOp>(su);
        if (!st || st.getIndices().size() != 1) continue;
        auto idx = getConstIndex(st.getIndices()[0]);
        if (idx) result.emplace_back(*idx, st.getValueToStore());
      }
    }
  }

  llvm::sort(result, [](const auto &a, const auto &b) {
    return a.first < b.first;
  });
  return result;
}

/// Erase the shareds-initialisation chain anchored at alloc's result:
///   td → p2m_td → load[0] → p2m_sh → stores
/// Call AFTER taskCall.erase() so td's only remaining users are this chain.
static void eraseTaskSharedsInit(LLVM::CallOp alloc) {
  Value td = alloc.getResult();
  // Collect in inner-to-outer order: stores → p2m_sh → load → p2m_td
  // so each op is use_empty() by the time we erase it.
  SmallVector<Operation *> toErase;

  for (Operation *u : td.getUsers()) {
    if (u->getName().getStringRef() != "enzymexla.pointer2memref") continue;
    Value p2m_td = u->getResult(0);
    for (Operation *lu : p2m_td.getUsers()) {
      auto ld = dyn_cast<memref::LoadOp>(lu);
      if (!ld) continue;
      Value shareds = ld.getResult();
      for (Operation *su : shareds.getUsers()) {
        if (su->getName().getStringRef() != "enzymexla.pointer2memref") continue;
        Value p2m_sh = su->getResult(0);
        for (Operation *stu : p2m_sh.getUsers())
          if (isa<memref::StoreOp>(stu)) toErase.push_back(stu); // innermost
        toErase.push_back(su);   // p2m_sh
      }
      toErase.push_back(lu);     // load
    }
    toErase.push_back(u);        // p2m_td
  }

  for (Operation *op : toErase)
    if (op->use_empty()) op->erase();
}

/// Walk fn to pre-populate `map` with  capLoad.getResult() → caps[i].value
/// for the access chain:
///   pointer2memref(arg1)[0] → shareds_ptr
///   pointer2memref(shareds_ptr)[i] → cap_i   ← mapped here
static void buildTaskCaptureMap(LLVM::LLVMFuncOp fn,
                                 ArrayRef<std::pair<int64_t, Value>> caps,
                                 IRMapping &map) {
  if (fn.getBody().empty() || fn.getBody().front().getNumArguments() < 2)
    return;
  BlockArgument arg1 = fn.getBody().front().getArgument(1);

  // Find shareds_ptr inside the function body
  Value sharedsPtr;
  fn.walk([&](Operation *op) {
    if (sharedsPtr) return;
    if (op->getName().getStringRef() != "enzymexla.pointer2memref") return;
    if (op->getNumOperands() != 1 || op->getOperand(0) != arg1) return;
    if (op->getNumResults() != 1) return;
    Value p2m = op->getResult(0);
    for (Operation *u : p2m.getUsers()) {
      auto ld = dyn_cast<memref::LoadOp>(u);
      if (!ld || ld.getIndices().size() != 1) continue;
      auto idx = getConstIndex(ld.getIndices()[0]);
      if (idx && *idx == 0) { sharedsPtr = ld.getResult(); break; }
    }
  });
  if (!sharedsPtr) return;

  // Map each cap-load result to the corresponding harvested value
  fn.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "enzymexla.pointer2memref") return;
    if (op->getNumOperands() != 1 || op->getOperand(0) != sharedsPtr) return;
    if (op->getNumResults() != 1) return;
    Value p2m = op->getResult(0);
    for (Operation *u : p2m.getUsers()) {
      auto ld = dyn_cast<memref::LoadOp>(u);
      if (!ld || ld.getIndices().size() != 1) continue;
      auto idx = getConstIndex(ld.getIndices()[0]);
      if (!idx) continue;
      for (const auto &[capIdx, capVal] : caps)
        if (capIdx == *idx) { map.map(ld.getResult(), capVal); break; }
    }
  });
}

/// Inline a task-entry fn (i32 gtid, ptr kmp_task_t*) into `dst`.
/// arg0 and arg1 both map to undef; captured values are substituted via
/// buildTaskCaptureMap before cloning so downstream ops get the real values.
/// Ops whose results are all pre-mapped are skipped (dead intermediates
/// are left for DCE).
static LogicalResult inlineTaskBody(LLVM::LLVMFuncOp fn, Region &dst,
                                     OpBuilder &b,
                                     ArrayRef<std::pair<int64_t, Value>> caps) {
  if (fn.getBody().empty())
    return fn.emitError("LLVMToOmp: task entry fn '")
           << fn.getName() << "' has empty body";

  IRMapping map;
  Location loc = fn.getLoc();
  Block &entry = fn.getBody().front();

  // arg0=gtid (i32), arg1=kmp_task_t* — both are runtime internals.
  for (BlockArgument arg : entry.getArguments())
    map.map(arg, b.create<LLVM::UndefOp>(loc, arg.getType()));

  // Pre-map cap-load results AFTER arg mappings (buildTaskCaptureMap
  // compares operands against the original arg1 BlockArgument).
  buildTaskCaptureMap(fn, caps, map);

  if (caps.empty() && entry.getNumArguments() > 1 &&
      !entry.getArgument(1).use_empty())
    fn.emitWarning("LLVMToOmp: task captures not harvested (unrecognised "
                   "kmp_task_t access pattern) — shared variables will be "
                   "undef in the inlined task body");

  // Clone blocks — identical structure to inlineOutlinedBody
  SmallVector<Block *> dblks;
  bool first = true;
  for (Block &sb : fn.getBody()) {
    Block *db;
    if (first) { db = &dst.front(); first = false; }
    else {
      db = b.createBlock(&dst);
      for (BlockArgument a : sb.getArguments())
        map.map(a, db->addArgument(a.getType(), a.getLoc()));
    }
    map.map(&sb, db);
    dblks.push_back(db);
  }
  unsigned blkIdx = 0;
  for (Block &sb : fn.getBody()) {
    b.setInsertionPointToEnd(dblks[blkIdx++]);
    for (Operation &op : sb) {
      if (isa<LLVM::ReturnOp>(op)) {
        b.create<omp::TerminatorOp>(op.getLoc());
        continue;
      }
      // Skip ops whose results are all pre-mapped (the folded cap loads).
      // Their operand producers are cloned but left dead for DCE.
      if (!op.getResults().empty() &&
          llvm::all_of(op.getResults(),
                       [&](Value r) { return map.contains(r); }))
        continue;
      b.clone(op, map);
    }
  }
  return success();
}

void LLVMToOMPPass::convertTasks(ModuleOp mod) {
  SmallVector<LLVM::CallOp> allocs;
  mod.walk([&](LLVM::CallOp c){
    if (isCallTo(c,"__kmpc_omp_task_alloc")) allocs.push_back(c);
  });
  for (LLVM::CallOp alloc : allocs) {
    Value td = alloc.getResult();
    // Skip allocs whose result flows into __kmpc_taskloop.
    // Those are taskloop constructs handled by convertTaskloop(),
    // which runs after this function in runOnOperation().
    bool feedsTaskloop = false;
    for (Operation *u : td.getUsers())
      if (auto uc = dyn_cast<LLVM::CallOp>(u))
        if (isCallTo(uc, "__kmpc_taskloop")) { feedsTaskloop = true; break; }
    if (feedsTaskloop) continue;
    // ─────────────────────────────────────────────────────────────────────

    // ── Normal task: find the scheduling call that consumes `td`. ───────
    LLVM::CallOp taskCall;
    for (Operation *u : llvm::make_early_inc_range(td.getUsers())) {
      if (auto uc = dyn_cast<LLVM::CallOp>(u)) {
        StringRef n = getCalleeName(uc);
        // In convertTasks, extend the consumer search:
        if (n == "__kmpc_omp_task" || n == "__kmpc_omp_task_with_deps")
          { taskCall = uc; break; }
        if (n == "__kmpc_omp_task_begin_if0")
          { taskCall = uc; break; }
        if (n == "__kmpc_taskloop")
          { /* skip — handled by convertTaskloop */ taskCall = {}; break; }
      }
    }
    if (!taskCall) {
      alloc.emitWarning("LLVMToOmp: __kmpc_task_alloc without consumer — "
                        "left as llvm.call"); continue;
    }
    if (isCallTo(taskCall,"__kmpc_omp_task_with_deps"))
      taskCall.emitWarning("LLVMToOmp: task depend clauses dropped — "
                           "kmp_depend_info_t layout analysis needed");

    // Harvest captures BEFORE touching any IR.
    auto caps = harvestTaskCaptures(alloc);

    // alloc arg 2 = flags: bit0=untied, bit1=final, bit2=mergeable
    bool isUntied = false, isMergeable = false;
    if (auto fv = getConstInt(alloc.getNumOperands()>2
                                  ? alloc.getOperand(2) : Value{})) {
      isUntied    = *fv & 1;
      isMergeable = (*fv >> 2) & 1;
    }
    // alloc arg 5 = pointer to task-entry outlined function
    LLVM::LLVMFuncOp fn;
    if (alloc.getNumOperands() > 5)
      if (auto a = alloc.getOperand(5).getDefiningOp<LLVM::AddressOfOp>())
        fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(a.getGlobalName());

    OpBuilder b(alloc); Location loc = alloc.getLoc();
    omp::TaskOperands tOps;
    if (isUntied)    tOps.untied    = b.getUnitAttr();
    if (isMergeable) tOps.mergeable = b.getUnitAttr();

    auto taskOp = b.create<omp::TaskOp>(loc, tOps);
    b.createBlock(&taskOp.getRegion());
    b.setInsertionPointToStart(&taskOp.getRegion().front());

    if (fn) {
      if (failed(inlineTaskBody(fn, taskOp.getRegion(), b, caps))) {
        taskOp.erase();
        alloc.emitWarning("LLVMToOmp: failed to inline task body — skipped");
        continue;
      }
    } else {
      b.create<omp::TerminatorOp>(loc);
      alloc.emitWarning("LLVMToOmp: task entry fn unresolvable — "
                        "emitting empty omp.task");
    }

    // Erase order matters for SSA validity:
    // (1) taskCall holds a direct use of td — erase first.
    taskCall.erase();

    // (2) __kmpc_omp_task_complete_if0, if present, also uses td.
    {
      SmallVector<Operation *> completions;
      for (Operation *u : td.getUsers())
        if (auto uc = dyn_cast<LLVM::CallOp>(u))
          if (isCallTo(uc, "__kmpc_omp_task_complete_if0"))
            completions.push_back(u);
      for (Operation *op : completions) op->erase();
    }

    // (3) Erase the shareds init chain (p2m → load → p2m → stores).
    eraseTaskSharedsInit(alloc);

    // (4) td is now use-empty — safe to erase.
    alloc.erase();

    // (5) Remove the outlined entry function from the module.
    if (fn) fn.erase();

    LLVM_DEBUG({
      llvm::dbgs() << "[LLVMToOmp] task_alloc → omp.task";
      if (!caps.empty()) llvm::dbgs() << " (" << caps.size() << " cap(s))";
      llvm::dbgs() << "\n";
    });
  }
}

/// Inline the per-iteration body of a taskloop entry function into `destBlk`.
///
/// The Clang-emitted task-entry fn always has this shape:
///   lb_raw  = load arg1[5]           // i64, lower 32 bits = lb
///   ub      = load arg1[6]           // i64
///   lb_sext = sign_extend(lb_raw)    // shli 32 + shrsi 32
///   if ub >= lb_sext:
///     start = lb_sext + 1
///     end   = max(ub+1, start) + 1
///     for iv_for in [start, end, 1):
///       val = lb_sext + (iv_for - start)   // = iv_for - 1
///       <per-iteration work using val>
///
/// Strategy:
///   1. Pre-map lb_raw / lb_sext → lb, ub_raw → ub (taskloop call args).
///   2. Clone preamble ops (start/end computations) into destBlk.
///   3. Map iv_for → nestIV + 1  so  val = (nestIV+1) - 1 = nestIV.
///   4. Clone scf.for body ops (the actual per-iteration work).
static LogicalResult inlineTaskloopBody(
    LLVM::LLVMFuncOp fn, Block *destBlk, OpBuilder &b, Location loc,
    ArrayRef<std::pair<int64_t, Value>> caps,
    Value lb, Value ub, BlockArgument nestIV) {

  if (fn.getBody().empty())
    return fn.emitError("LLVMToOmp: taskloop entry fn '")
           << fn.getName() << "' has empty body";

  Block &entry = fn.getBody().front();

  // ── (1) Base mapping: fn args → undef + user captures ─────────────────
  IRMapping map;
  for (BlockArgument arg : entry.getArguments())
    map.map(arg, b.create<LLVM::UndefOp>(loc, arg.getType()));
  buildTaskCaptureMap(fn, caps, map);

  // ── (2) Pre-map lb/ub field accesses from the task struct (arg1) ───────
  // arg1[5] → raw lb (i64) → sign-extended chain → lb_sext   map all → lb
  // arg1[6] → ub                                              map → ub
  if (entry.getNumArguments() > 1) {
    BlockArgument arg1 = entry.getArgument(1);
    fn.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "enzymexla.pointer2memref") return;
      if (op->getNumOperands() != 1 || op->getOperand(0) != arg1) return;
      Value p2m = op->getResult(0);
      for (Operation *u : p2m.getUsers()) {
        auto ld = dyn_cast<memref::LoadOp>(u);
        if (!ld || ld.getIndices().size() != 1) continue;
        auto idx = getConstIndex(ld.getIndices()[0]);
        if (!idx) continue;
        if (*idx == 5) {
          map.map(ld.getResult(), lb);
          // sign-extension chain: shli 32 → shrsi 32
          for (Operation *u2 : ld.getResult().getUsers()) {
            auto shli = dyn_cast<arith::ShLIOp>(u2);
            if (!shli) continue;
            map.map(shli.getResult(), lb);
            for (Operation *u3 : shli.getResult().getUsers()) {
              auto shrsi = dyn_cast<arith::ShRSIOp>(u3);
              if (!shrsi) continue;
              map.map(shrsi.getResult(), lb);
            }
          }
        }
        if (*idx == 6) map.map(ld.getResult(), ub);
      }
    });
  }

  // ── (3) Locate scf.if (bounds guard) and scf.for (chunk loop) ─────────
  scf::IfOp  outerIf;
  scf::ForOp bodyFor;
  fn.walk([&](scf::IfOp  f) { if (!outerIf) outerIf = f; });
  fn.walk([&](scf::ForOp f) { if (!bodyFor) bodyFor = f; });

  if (!bodyFor) {
    fn.emitWarning("LLVMToOmp: taskloop fn has no scf.for — "
                   "falling back to full-body inline (suboptimal)");
    b.setInsertionPoint(destBlk->getTerminator());
    for (Block &sb : fn.getBody())
      for (Operation &op : sb) {
        if (isa<LLVM::ReturnOp>(op)) continue;
        if (!op.getResults().empty() &&
            llvm::all_of(op.getResults(),
                         [&](Value r) { return map.contains(r); }))
          continue;
        b.clone(op, map);
      }
    return success();
  }

  b.setInsertionPoint(destBlk->getTerminator());

  // ── (4) Clone preamble ops (before the scf.for, inside the scf.if) ────
  // These compute start = lb+1, end = ub+2, etc., needed by the for body.
  Block *preamble = outerIf ? &outerIf.getThenRegion().front() : &entry;
  for (Operation &op : *preamble) {
    if (&op == bodyFor.getOperation()) break;
    if (isa<scf::YieldOp>(op))         break;
    // skip ops whose results are already in the map (pre-mapped lb/ub chain)
    if (!op.getResults().empty() &&
        llvm::all_of(op.getResults(),
                     [&](Value r) { return map.contains(r); }))
      continue;
    b.clone(op, map);
  }

  // ── (5) Map scf.for IV → nestIV + 1 ────────────────────────────────────
  // The for body computes: val = lb + (iv_for - start) = iv_for - 1
  // Setting iv_for = nestIV + 1  →  val = nestIV  ✓
  Value one = b.create<arith::ConstantOp>(
      loc, b.getIntegerAttr(nestIV.getType(), 1));
  Value ivForVal = b.create<arith::AddIOp>(loc, nestIV, one);
  map.map(bodyFor.getInductionVar(), ivForVal);

  // ── (6) Clone scf.for body ops (the actual per-iteration work) ─────────
  for (Operation &op : bodyFor.getBody()->without_terminator()) {
    if (!op.getResults().empty() &&
        llvm::all_of(op.getResults(),
                     [&](Value r) { return map.contains(r); }))
      continue;
    b.clone(op, map);
  }

  return success();
}
void LLVMToOMPPass::convertTaskloop(ModuleOp mod) {
  SmallVector<LLVM::CallOp> calls;
  mod.walk([&](LLVM::CallOp c) {
    if (isCallTo(c, "__kmpc_taskloop")) calls.push_back(c);
  });

  for (LLVM::CallOp tlCall : calls) {
    // ── (A) Locate the kmp_task_t* from __kmpc_omp_task_alloc ───────────
    // __kmpc_taskloop(ident*, gtid, task*, if_val,
    //                 lb_ptr, ub_ptr, step,
    //                 sched, nogroup, grainsize_or_num_tasks, task_dup*)
    if (tlCall.getNumOperands() < 7) {
      tlCall.emitWarning("LLVMToOmp: __kmpc_taskloop has too few args — skipped");
      continue;
    }
    Value taskPtr = tlCall.getOperand(2);
    auto alloc = dyn_cast_or_null<LLVM::CallOp>(taskPtr.getDefiningOp());
    if (!alloc || !isCallTo(alloc, "__kmpc_omp_task_alloc")) {
      tlCall.emitWarning("LLVMToOmp: __kmpc_taskloop task ptr not from "
                         "__kmpc_omp_task_alloc — skipped");
      continue;
    }

    // ── (B) Extract loop bounds from call args ──────────────────────────
    Value lbPtr = tlCall.getOperand(4);
    Value ubPtr = tlCall.getOperand(5);
    Value step  = tlCall.getOperand(6);

    // sched: 0=default, 1=grainsize, 2=num_tasks
    int64_t schedMode = 0;
    if (auto sv = getConstInt(tlCall.getNumOperands() > 7
                                  ? tlCall.getOperand(7) : Value{}))
      schedMode = *sv;

    // nogroup: 1 = no implicit taskgroup
    bool isNogroup = false;
    if (auto ng = getConstInt(tlCall.getNumOperands() > 8
                                  ? tlCall.getOperand(8) : Value{}))
      isNogroup = (*ng != 0);

    Value grainOrNum = tlCall.getNumOperands() > 9
                           ? tlCall.getOperand(9) : Value{};

    // ── (C) Resolve task-entry outlined function ────────────────────────
    LLVM::LLVMFuncOp fn;
    if (alloc.getNumOperands() > 5)
      if (auto a = alloc.getOperand(5).getDefiningOp<LLVM::AddressOfOp>())
        fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(a.getGlobalName());
    if (!fn) {
      tlCall.emitWarning("LLVMToOmp: __kmpc_taskloop entry fn unresolvable "
                         "— skipped");
      continue;
    }

    // ── (D) Harvest captures and build bounds values ────────────────────
    auto caps = harvestTaskCaptures(alloc);

    // alloc arg 2 = flags: bit0=untied, bit2=mergeable
    bool isUntied = false, isMergeable = false;
    if (auto fv = getConstInt(alloc.getNumOperands() > 2
                                  ? alloc.getOperand(2) : Value{})) {
      isUntied    = (*fv & 1) != 0;
      isMergeable = ((*fv >> 2) & 1) != 0;
    }

    MLIRContext *ctx = mod.getContext();
    // Taskloop bounds are always i64 in the kmp_task_t struct.
    Type iterTy = IntegerType::get(ctx, 64);

    OpBuilder b(alloc);
    Location loc = alloc.getLoc();

    // Load the full-range lb/ub that the runtime will partition into tasks.
    Value lb = b.create<LLVM::LoadOp>(loc, iterTy, lbPtr);
    Value ub = b.create<LLVM::LoadOp>(loc, iterTy, ubPtr);

    // ── (E) Build omp.taskloop (task clauses only) ──────────────────────
    omp::TaskloopOperands tloOps;
    if (isUntied)    tloOps.untied    = b.getUnitAttr();
    if (isMergeable) tloOps.mergeable = b.getUnitAttr();
    if (isNogroup)   tloOps.nogroup   = b.getUnitAttr();
    if (grainOrNum) {
      if (schedMode == 1)      tloOps.grainsize = grainOrNum;
      else if (schedMode == 2) tloOps.numTasks  = grainOrNum;
    }

    auto taskloopOp = b.create<omp::TaskloopOp>(loc, tloOps);
    // TaskloopOp has SingleBlock+NoTerminator (same as WsloopOp):
    // region.front() is auto-created; never add omp.terminator.
    Block *tloBlk = &taskloopOp.getRegion().front();
    b.setInsertionPointToStart(tloBlk);

    // ── (F) Build nested omp.loop_nest (loop bounds go here, not above) ─
    omp::LoopNestOperands lnOps;
    lnOps.loopLowerBounds = {lb};
    lnOps.loopUpperBounds = {ub};
    lnOps.loopSteps       = {step};
    auto nestOp  = b.create<omp::LoopNestOp>(loc, lnOps);
    Block *nestBlk = b.createBlock(&nestOp.getRegion());
    BlockArgument iv = nestBlk->addArgument(iterTy, loc);
    b.setInsertionPointToStart(nestBlk);
    b.create<omp::YieldOp>(loc);   // placeholder; body inserted before this

    // ── (G) Inline per-iteration body into nestBlk ──────────────────────
    if (failed(inlineTaskloopBody(fn, nestBlk, b, loc, caps, lb, ub, iv))) {
      taskloopOp.erase();
      tlCall.emitWarning("LLVMToOmp: taskloop body inline failed — skipped");
      continue;
    }

    // ── (H) Erase in SSA-valid order ────────────────────────────────────
    // (1) tlCall directly uses alloc's result — erase first.
    tlCall.erase();
    // (2) Erase the shareds init chain (p2m → load → p2m → stores).
    eraseTaskSharedsInit(alloc);
    // (3) alloc result is now use-empty — safe to erase.
    alloc.erase();
    // (4) Remove the outlined entry function.
    fn.erase();

    LLVM_DEBUG({
      llvm::dbgs() << "[LLVMToOmp] __kmpc_taskloop → omp.taskloop";
      if (schedMode == 1 && grainOrNum) llvm::dbgs() << " grainsize";
      if (schedMode == 2 && grainOrNum) llvm::dbgs() << " num_tasks";
      if (isNogroup)                     llvm::dbgs() << " nogroup";
      llvm::dbgs() << "\n";
    });
  }
}

//===----------------------------------------------------------------------===//
// §15  Reductions — warn + leave (needs mem2reg)
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertReductions(ModuleOp mod) {
  mod.walk([&](LLVM::CallOp c){
    StringRef n = getCalleeName(c);
    if (n!="__kmpc_reduce" && n!="__kmpc_reduce_nowait") return;
    c.emitWarning("LLVMToOmp: '") << n
        << "' not converted — run mem2reg first so reduction GEPs can be "
           "traced.  Left as llvm.call.";
    StringRef en = (n=="__kmpc_reduce_nowait")
                       ? "__kmpc_end_reduce_nowait" : "__kmpc_end_reduce";
    if (LLVM::CallOp e = findNextCallTo(c, en))
      e.emitWarning("LLVMToOmp: ") << en << " also left as llvm.call";
  });
}

//===----------------------------------------------------------------------===//
// §16  Atomics
//
// IMPORTANT: AtomicReadOp, AtomicWriteOp, AtomicUpdateOp, AtomicCaptureOp
// do NOT have explicit *Operands builders in the td (no `let builders`
// listing).  The auto-generated positional builders are used instead.
//
// ── AtomicReadOp ──────────────────────────────────────────────────────────
//   td: arguments = !con((ins x:src_ptr, v:dst_ptr, element_type:TypeAttr),
//                        clausesArgs)    clausesArgs = [hint, memory_order]
//   Positional builder:
//     create<AtomicReadOp>(loc, x, v, element_type, hint, memory_order)
//   Semantics: *v = *x   (x = SOURCE, v = DESTINATION)
//
// ── AtomicWriteOp ─────────────────────────────────────────────────────────
//   td: arguments = !con((ins x:dst_ptr, expr:AnyType), clausesArgs)
//   Positional builder:
//     create<AtomicWriteOp>(loc, x, expr, hint, memory_order)
//   Semantics: *x = expr  (x = DESTINATION)
//
// ── AtomicUpdateOp ────────────────────────────────────────────────────────
//   td: arguments = !con((ins x:ptr, atomic_control:OptionalAttr), clausesArgs)
//       traits = [SingleBlockImplicitTerminator<"YieldOp">]
//   Positional builder:
//     create<AtomicUpdateOp>(loc, x, atomic_control, hint, memory_order)
//   After build(): region has one block (SingleBlock) + implicit empty YieldOp.
//   We add the block arg (current *x) and replace the YieldOp.
//
// ── AtomicCaptureOp ───────────────────────────────────────────────────────
//   td: traits = [SingleBlockImplicitTerminator<"TerminatorOp">]
//       clauses = [OpenMP_HintClause, OpenMP_MemoryOrderClause]
//   Positional builder:
//     create<AtomicCaptureOp>(loc, hint, memory_order)
//   After build(): ONE region, ONE block (SingleBlock), implicit TerminatorOp.
//   Valid contents: (update+read), (read+update), (read+write) + terminator.
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertAtomics(ModuleOp mod) {
  SmallVector<LLVM::CallOp> calls;
  mod.walk([&](LLVM::CallOp c){
    if (calleeStartsWith(c,"__kmpc_atomic_")) calls.push_back(c);
  });
  MLIRContext *ctx = mod.getContext();

  // Helper: build ClauseMemoryOrderKindAttr for seq_cst (conservative).
  auto seqCstAttr = [&]() {
    return omp::ClauseMemoryOrderKindAttr::get(
        ctx, omp::ClauseMemoryOrderKind::Seq_cst);
  };
  // hint arg: absent (OptionalAttr<IntegerAttr>).
  auto noHint = IntegerAttr{};

  auto safeArg = [](LLVM::CallOp op, unsigned i) -> Value {
    return i < op.getNumOperands() ? op.getOperand(i) : Value{};
  };

  for (LLVM::CallOp call : calls) {
    StringRef name = getCalleeName(call);
    Location  loc  = call.getLoc();
    OpBuilder b(call);

    AtomicDesc desc = parseAtomicName(name, ctx);
    if (!desc.elemType || desc.op == AtomicDesc::Op::Unknown) {
      call.emitWarning("LLVMToOmp: unrecognized atomic '")
          << name << "' — left as llvm.call";
      continue;
    }

    // ─── READ ────────────────────────────────────────────────────────────
    // __kmpc_atomic_TYPE_rd(ident, gtid, src_ptr) → elem_type
    // x = SOURCE ptr, v = DESTINATION ptr  (confirmed from td assembly:
    //   "$v `=` $x ... : type($v) , type($x)")
    if (desc.op == AtomicDesc::Op::Read) {
      Value src = safeArg(call,2);
      if (!src) { call.emitWarning("LLVMToOmp: atomic.rd missing src"); continue; }
      // Allocate a slot to serve as the destination (v).
      Value arraySize = b.create<arith::ConstantIndexOp>(loc, 1);
      Value slot = b.create<LLVM::AllocaOp>(
          loc,
          LLVM::LLVMPointerType::get(ctx),
          desc.elemType,
          arraySize,
          16u // alignment
      );
      // Positional: (loc, x=src, v=slot, element_type, hint, memory_order)
      b.create<omp::AtomicReadOp>(loc, src, slot,
                                   TypeAttr::get(desc.elemType),
                                   noHint, seqCstAttr());
      if (!call.getResults().empty()) {
        Value loaded = b.create<LLVM::LoadOp>(loc, desc.elemType, slot);
        call.getResult().replaceAllUsesWith(loaded);
      }
      call.erase();

    // ─── WRITE ───────────────────────────────────────────────────────────
    // __kmpc_atomic_TYPE_wr(ident, gtid, dst_ptr, val)
    // x = DESTINATION ptr, expr = value  (td assembly: "$x `=` $expr")
    } else if (desc.op == AtomicDesc::Op::Write) {
      Value dst = safeArg(call,2), val = safeArg(call,3);
      if (!dst||!val) {
        call.emitWarning("LLVMToOmp: atomic.wr missing args"); continue;
      }
      // Positional: (loc, x=dst, expr=val, hint, memory_order)
      b.create<omp::AtomicWriteOp>(loc, dst, val, noHint, seqCstAttr());
      call.erase();

    // ─── UPDATE ──────────────────────────────────────────────────────────
    // __kmpc_atomic_TYPE_OP(ident, gtid, x_ptr, rhs)
    } else if (desc.op == AtomicDesc::Op::Update) {
      Value xPtr = safeArg(call,2), rhs = safeArg(call,3);
      if (!xPtr||!rhs) {
        call.emitWarning("LLVMToOmp: atomic.update missing args"); continue;
      }
      // Positional: (loc, x, atomic_control=absent, hint, memory_order)
      // AtomicControlAttr{} = null optional attr.
      auto updOp = b.create<omp::AtomicUpdateOp>(
          loc, xPtr, omp::AtomicControlAttr{}, noHint, seqCstAttr());
      // §7 helper: add block arg, build binop, fix YieldOp.
      if (failed(buildAtomicUpdateRegion(b, loc, updOp,
                                          desc.elemType, desc.bin, rhs))) {
        call.emitWarning("LLVMToOmp: update binop not emittable for '")
            << name << "' — left as llvm.call";
        updOp.erase(); continue;
      }
      b.setInsertionPointAfter(updOp);
      call.erase();

    // ─── CAPTURE ─────────────────────────────────────────────────────────
    // __kmpc_atomic_TYPE_capture_OP(ident, gtid, res_ptr, x_ptr, val)
    // Emits update-then-read form: { *x op= val; v = *x }
    // td: ONE region, ONE block, implicit TerminatorOp.
    //   Valid forms in the block: (update+read), (read+update), (read+write)
    } else if (desc.op == AtomicDesc::Op::Capture) {
      Value res  = safeArg(call,2);
      Value xPtr = safeArg(call,3);
      Value val  = safeArg(call,4);
      if (!res||!xPtr||!val) {
        call.emitWarning("LLVMToOmp: atomic.capture missing args — "
                         "left as llvm.call"); continue;
      }
      // Determine update binop from suffix after "capture_".
      AtomicDesc::Bin capBin = AtomicDesc::Bin::Add;
      bool isWriteCapture = false;
      StringRef suf = name;
      if (auto pos = suf.find("capture_"); pos != StringRef::npos) {
        suf = suf.drop_front(pos + strlen("capture_"));
        if      (SW(suf,"add")) capBin = AtomicDesc::Bin::Add;
        else if (SW(suf,"sub")) capBin = AtomicDesc::Bin::Sub;
        else if (SW(suf,"mul")) capBin = AtomicDesc::Bin::Mul;
        else if (SW(suf,"wr")) isWriteCapture = true;
      }

      // Positional: (loc, hint, memory_order)
      // SingleBlockImplicitTerminator<"TerminatorOp"> → block auto-created.
      auto capOp = b.create<omp::AtomicCaptureOp>(loc, noHint, seqCstAttr());
      Block *capBlk = &capOp.getRegion().front(); // SingleBlock

      b.setInsertionPoint(capBlk->getTerminator());

      if (isWriteCapture) {
        // read-then-write form: { v = *x; *x = val }
        // Positional AtomicReadOp: (x=source, v=dest, elem_type, hint, mo)
        b.create<omp::AtomicReadOp>(loc, xPtr, res,
                                     TypeAttr::get(desc.elemType),
                                     noHint, seqCstAttr());
        b.create<omp::AtomicWriteOp>(loc, xPtr, val, noHint, seqCstAttr());
      } else {
        // update-then-read form: { *x op= val; v = *x }
        // (1) atomic.update
        auto innerUpd = b.create<omp::AtomicUpdateOp>(
            loc, xPtr, omp::AtomicControlAttr{}, noHint, seqCstAttr());
        if (failed(buildAtomicUpdateRegion(b, loc, innerUpd,
                                            desc.elemType, capBin, val))) {
          call.emitWarning("LLVMToOmp: capture update binop not emittable — "
                           "left as llvm.call");
          capOp.erase(); continue;
        }
        // (2) atomic.read  — insert before terminator
        b.setInsertionPoint(capBlk->getTerminator());
        // x=source, v=destination
        b.create<omp::AtomicReadOp>(loc, xPtr, res,
                                     TypeAttr::get(desc.elemType),
                                     noHint, seqCstAttr());
      }
      b.setInsertionPointAfter(capOp);
      call.erase();
      LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] atomic.capture converted\n");
    }
  }
}

//===----------------------------------------------------------------------===//
// §17  Leaf patterns (Phase 2 — greedy rewrite)
//===----------------------------------------------------------------------===//

/// __kmpc_barrier / __kmpc_cancel_barrier → omp.barrier
/// td: BarrierOp — assemblyFormat = "attr-dict", no arguments.
struct BarrierPat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    if (!isCallTo(op,"__kmpc_barrier") &&
        !isCallTo(op,"__kmpc_cancel_barrier")) return failure();
    rw.create<omp::BarrierOp>(op.getLoc());
    rw.eraseOp(op); return success();
  }
};

/// __kmpc_flush → omp.flush
/// td: FlushOp — arguments = !con((ins Variadic<AnyType>:$varList), clausesArgs)
///     clauses = [] (empty, TODO comment in td).
///     __kmpc_flush(ident_t*) has no user-facing varList → pass empty.
struct FlushPat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    if (!isCallTo(op,"__kmpc_flush")) return failure();
    rw.create<omp::FlushOp>(op.getLoc(), ValueRange{});
    rw.eraseOp(op); return success();
  }
};

/// __kmpc_omp_taskwait → omp.taskwait
/// td: TaskwaitOp — clauses=[OpenMP_DependClause, OpenMP_NowaitClause]
///     Builder: OpBuilder<(ins CArg<"const TaskwaitOperands &">:$clauses)>
///     __kmpc_omp_taskwait always blocks; nowait not set.
struct TaskwaitPat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    if (!isCallTo(op,"__kmpc_omp_taskwait")) return failure();
    omp::TaskwaitOperands twOps; // nowait not set — call always blocks
    rw.create<omp::TaskwaitOp>(op.getLoc(), twOps);
    rw.eraseOp(op); return success();
  }
};

/// __kmpc_omp_taskyield → omp.taskyield
/// td: TaskyieldOp — assemblyFormat = "attr-dict", no args, no explicit builder.
struct TaskyieldPat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    if (!isCallTo(op,"__kmpc_omp_taskyield")) return failure();
    rw.create<omp::TaskyieldOp>(op.getLoc());
    rw.eraseOp(op); return success();
  }
};

/// __kmpc_cancel → omp.cancel
/// td: CancelOp — clauses=[OpenMP_CancelDirectiveNameClause, OpenMP_IfClause]
///     Builder: OpBuilder<(ins CArg<"const CancelOperands &">:$clauses)>
///     CancelOperands.cancelDirective = ClauseCancellationConstructType
///     (ident, gtid, cancel_kind): 1=parallel,2=sections,4=loop,8=taskgroup
struct CancelPat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    if (!isCallTo(op,"__kmpc_cancel")) return failure();
    omp::ClauseCancellationConstructType kind =
        omp::ClauseCancellationConstructType::Parallel;
    if (auto v = getConstInt(op.getNumOperands()>2 ? op.getOperand(2) : Value{}))
      switch (*v) {
      case 1: kind = omp::ClauseCancellationConstructType::Parallel;  break;
      case 2: kind = omp::ClauseCancellationConstructType::Sections;  break;
      case 4: kind = omp::ClauseCancellationConstructType::Loop;      break;
      case 8: kind = omp::ClauseCancellationConstructType::Taskgroup; break;
      }
    omp::CancelOperands cOps;
    cOps.cancelDirective =
        omp::ClauseCancellationConstructTypeAttr::get(op.getContext(), kind); // OpenMP_CancelDirectiveNameClause
    rw.create<omp::CancelOp>(op.getLoc(), cOps);
    rw.eraseOp(op); return success();
  }
};

/// __kmpc_cancellationpoint → omp.cancellation_point
/// td: CancellationPointOp — clauses=[OpenMP_CancelDirectiveNameClause]
///     Builder: OpBuilder<(ins CArg<"const CancellationPointOperands &">)>
struct CancelPointPat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    if (!isCallTo(op,"__kmpc_cancellationpoint")) return failure();
    omp::ClauseCancellationConstructType kind =
        omp::ClauseCancellationConstructType::Parallel;
    if (auto v = getConstInt(op.getNumOperands()>2 ? op.getOperand(2) : Value{}))
      switch (*v) {
      case 1: kind = omp::ClauseCancellationConstructType::Parallel;  break;
      case 2: kind = omp::ClauseCancellationConstructType::Sections;  break;
      case 4: kind = omp::ClauseCancellationConstructType::Loop;      break;
      case 8: kind = omp::ClauseCancellationConstructType::Taskgroup; break;
      }
    omp::CancellationPointOperands cpOps;
    cpOps.cancelDirective =
        omp::ClauseCancellationConstructTypeAttr::get(op.getContext(), kind);
    rw.create<omp::CancellationPointOp>(op.getLoc(), cpOps);
    rw.eraseOp(op); return success();
  }
};

/// __kmpc_global_thread_num → constant 0
/// The gtid is internal bookkeeping; all outlined fn uses are already dropped.
struct GlobalTidPat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    if (!isCallTo(op,"__kmpc_global_thread_num")) return failure();
    if (!op.getResults().empty()) {
      Value z = rw.create<LLVM::ConstantOp>(
          op.getLoc(), rw.getI32Type(), rw.getI32IntegerAttr(0));
      op.getResult().replaceAllUsesWith(z);
    }
    rw.eraseOp(op); return success();
  }
};

/// GPU target init/deinit → hard error (out of scope for this CPU pass).
struct GpuTargetPat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &) const override {
    StringRef n = getCalleeName(op);
    if (!SW(n,"__kmpc_target_init")   && !SW(n,"__kmpc_target_deinit") &&
        !SW(n,"__kmpc_kernel_init")   && !SW(n,"__kmpc_spmd_kernel_init"))
      return failure();
    op.emitError("LLVMToOmp: GPU/SPMD call '") << n
        << "' is not supported by this CPU pass — "
           "use the device-side GPU lowering pipeline.";
    return failure(); // leave in IR so diagnostic is visible
  }
};

/// __kmpc_push_num_threads / _proc_bind / _num_teams
/// These are consumed by collectPendingClauses and have no OMP dialect
/// equivalent. Erase silently after Phase 1 has already extracted their
/// operand values.
struct PushClausePat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    StringRef n = getCalleeName(op);
    if (n != "__kmpc_push_num_threads" &&
        n != "__kmpc_push_proc_bind"   &&
        n != "__kmpc_push_num_teams")
      return failure();
    rw.eraseOp(op);
    return success();
  }
};

struct TaskCompleteIf0Pat : OpRewritePattern<LLVM::CallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &rw) const override {
    if (!isCallTo(op, "__kmpc_omp_task_begin_if0") &&
        !isCallTo(op, "__kmpc_omp_task_complete_if0"))
      return failure();
    rw.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// §17b  sections dispatch pattern
//
// Clang lowers #pragma omp sections to a static wsloop with index-based
// if-chain dispatch. After convertStaticWs inlines the scf.for body into the
// loop_nest, the canonical form is:
//
//   omp.wsloop [nowait] schedule(static=1) {
//     omp.loop_nest (%iv):i32 = (0) to (N-1) inclusive step (1) {
//       %idx = index_castui %iv
//       %eq0 = cmpi eq, %idx, 0
//       scf.if %eq0 { body0 } else {
//         %eq1 = cmpi eq, %idx, 1
//         scf.if %eq1 { body1 } else { … }
//       }
//       omp.yield
//     }
//   }
//
// Produced:
//   omp.sections [nowait] {
//     omp.section { body0; omp.terminator }
//     omp.section { body1; omp.terminator }
//     omp.terminator
//   }
//===----------------------------------------------------------------------===//

/// Walk the index-dispatch if-chain in `nestBody` and collect per-section
/// op lists (excluding yield terminators).  Returns false if the pattern
/// does not match.
static bool extractSectionBodies(
    Block *nestBody, Value iv,
    SmallVector<SmallVector<Operation *>> &out) {

  SmallVector<Operation *> ops;
  for (Operation &op : nestBody->without_terminator())
    ops.push_back(&op);
  if (ops.size() < 3) return false;

  // [0]: index_castui (or index_cast) of the loop IV
  Value idxVal;
  if (auto c = dyn_cast<arith::IndexCastUIOp>(ops[0])) {
    if (c.getIn() != iv) return false;
    idxVal = c.getResult();
  } else if (auto c = dyn_cast<arith::IndexCastOp>(ops[0])) {
    if (c.getIn() != iv) return false;
    idxVal = c.getResult();
  } else return false;

  // [1]: cmpi eq, %idx, 0
  auto firstCmp = dyn_cast<arith::CmpIOp>(ops[1]);
  if (!firstCmp ||
      firstCmp.getPredicate() != arith::CmpIPredicate::eq ||
      firstCmp.getLhs() != idxVal) return false;
  auto cst = getConstIndex(firstCmp.getRhs());
  if (!cst || *cst != 0) return false;

  // [2]: scf.if %cmp0 { body0 } else { chain... }
  auto outerIf = dyn_cast<scf::IfOp>(ops[2]);
  if (!outerIf || outerIf.getCondition() != firstCmp.getResult()) return false;

  // Section 0 — then-block of the outer if
  SmallVector<Operation *> sec0;
  for (Operation &op : outerIf.getThenRegion().front().without_terminator())
    sec0.push_back(&op);
  out.push_back(sec0);

  // Sections 1..N-1 — nested else-if chain
  Region *elseReg = &outerIf.getElseRegion();
  while (!elseReg->empty()) {
    Block *elseBlk = &elseReg->front();
    SmallVector<Operation *> elseOps;
    for (Operation &op : elseBlk->without_terminator())
      elseOps.push_back(&op);
    if (elseOps.empty()) break;
    if (elseOps.size() < 2) return false;

    // cmpi eq, %idx, k  (k == current section count)
    auto nextCmp = dyn_cast<arith::CmpIOp>(elseOps[0]);
    if (!nextCmp ||
        nextCmp.getPredicate() != arith::CmpIPredicate::eq ||
        nextCmp.getLhs() != idxVal) return false;
    auto nextCst = getConstIndex(nextCmp.getRhs());
    if (!nextCst || *nextCst != (int64_t)out.size()) return false;

    auto nextIf = dyn_cast<scf::IfOp>(elseOps[1]);
    if (!nextIf || nextIf.getCondition() != nextCmp.getResult()) return false;

    SmallVector<Operation *> secN;
    for (Operation &op : nextIf.getThenRegion().front().without_terminator())
      secN.push_back(&op);
    out.push_back(secN);

    // Recurse into the next else (empty for the last section)
    elseReg = &nextIf.getElseRegion();
  }
  return out.size() >= 2;
}

struct SectionsDispatchPat : OpRewritePattern<omp::WsloopOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::WsloopOp wsOp,
                                PatternRewriter &rw) const override {
    // ── (1) Body must be exactly one omp.loop_nest ──────────────────────
    // WsloopOp has NoTerminator so the region body has no terminator op.
    Block &wsBlk = wsOp.getRegion().front();
    if (std::next(wsBlk.begin()) != wsBlk.end()) return failure();
    auto nestOp = dyn_cast<omp::LoopNestOp>(&wsBlk.front());
    if (!nestOp) return failure();

    // ── (2) Single-dimension, lb=0, step=1, inclusive ───────────────────
    if (nestOp.getLoopLowerBounds().size() != 1 ||
        nestOp.getLoopUpperBounds().size() != 1 ||
        nestOp.getLoopSteps().size() != 1)   return failure();
    if (!nestOp.getLoopInclusiveAttr())       return failure();

    auto lb = getConstIndex(nestOp.getLoopLowerBounds()[0]);
    auto ub = getConstIndex(nestOp.getLoopUpperBounds()[0]);
    auto st = getConstIndex(nestOp.getLoopSteps()[0]);
    if (!lb || *lb != 0) return failure();
    if (!st || *st != 1) return failure();
    if (!ub || *ub < 1)  return failure(); // at least 2 sections

    // ── (3) Body must be the sections dispatch if-chain ─────────────────
    Block *nestBody = &nestOp.getRegion().front();
    Value iv = nestOp.getIVs()[0];
    SmallVector<SmallVector<Operation *>> sectionBodies;
    if (!extractSectionBodies(nestBody, iv, sectionBodies)) return failure();
    // Section count must exactly match the loop iteration count.
    if ((int64_t)sectionBodies.size() != *ub + 1)           return failure();

    Location loc = wsOp.getLoc();
    rw.setInsertionPoint(wsOp);

    // ── (4) Build omp.sections ───────────────────────────────────────────
    omp::SectionsOperands sOps;
    if (wsOp.getNowaitAttr()) sOps.nowait = rw.getUnitAttr();
    auto sectionsOp = rw.create<omp::SectionsOp>(loc, sOps);
    Block *secBody = rw.createBlock(&sectionsOp.getRegion());

    // ── (5) Build one omp.section per extracted body ─────────────────────
    for (auto &bodyOps : sectionBodies) {
      rw.setInsertionPointToEnd(secBody);
      auto sectionOp = rw.create<omp::SectionOp>(loc);
      Block *sBlk = rw.createBlock(&sectionOp.getRegion());
      rw.setInsertionPointToStart(sBlk);
      // Section bodies only reference values from outer scopes — no IV
      // remapping needed.
      IRMapping map;
      for (Operation *op : bodyOps)
        rw.clone(*op, map);
      rw.create<omp::TerminatorOp>(loc);
    }

    // ── (6) Terminate the sections region ───────────────────────────────
    rw.setInsertionPointToEnd(secBody);
    rw.create<omp::TerminatorOp>(loc);

    // ── (7) Erase the wsloop (and all nested ops) ────────────────────────
    rw.eraseOp(wsOp);
    LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] sections dispatch → omp.sections ("
                             << sectionBodies.size() << " sections)\n");
    return success();
  }
};

/// Catch-all: warn on any remaining __kmpc_* / omp_* calls (priority 0).
struct UnhandledPat : OpRewritePattern<LLVM::CallOp> {
  UnhandledPat(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(LLVM::CallOp op,
                                PatternRewriter &) const override {
    StringRef n = getCalleeName(op);
    if (!SW(n,"__kmpc_") && !SW(n,"omp_")) return failure();
    op.emitWarning("LLVMToOmp: '") << n
        << "' has no OMP dialect equivalent — left as llvm.call";
    return failure(); // leave in place intentionally
  }
};

void LLVMToOMPPass::applyLeafPatterns(ModuleOp mod) {
  RewritePatternSet pats(mod.getContext());
  pats.add<BarrierPat, FlushPat, TaskwaitPat, TaskyieldPat,
           CancelPat, CancelPointPat, GlobalTidPat,
           GpuTargetPat, PushClausePat, TaskCompleteIf0Pat, 
           SectionsDispatchPat, 
           UnhandledPat>(mod.getContext());
  (void)applyPatternsGreedily(mod, std::move(pats));
}

//===----------------------------------------------------------------------===//
// §19  Pass entry point
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::runOnOperation() {
  ModuleOp mod = getOperation();

  // Hard-reject GPU device modules (assumption F).
  if (auto t = mod->getAttrOfType<StringAttr>("llvm.target_triple")) {
    StringRef tv = t.getValue();
    if (tv.contains("nvptx")||tv.contains("amdgcn")||tv.contains("amdgpu")) {
      mod.emitError("LLVMToOmp: GPU device module (triple='")
          << tv << "') not supported.";
      signalPassFailure(); return;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] pass started\n");

  // Phase 1 — structural (order matters: teams before parallel so any
  // fork_call inside a teams outlined fn is reachable when convertParallel
  // runs).
  convertTeams(mod);
  convertParallel(mod);
  convertPaired(mod);     // single / master / critical / ordered / taskgroup
  convertStaticWs(mod);   // for_static_init/fini  → wsloop static
  // mod.print(llvm::outs(),
  //            OpPrintingFlags().useLocalScope());
  convertDynWs(mod);      // dispatch_init/next/fini→ wsloop dynamic
  convertTasks(mod);      // task_alloc + omp_task  → omp.task
  convertTaskloop(mod);
  convertAtomics(mod);    // __kmpc_atomic_*        → omp.atomic.*
  convertReductions(mod); // warn + leave

  
  // Phase 2 — leaf rewrites.
  applyLeafPatterns(mod);

  LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] pass complete\n");
}

} // namespace

//===----------------------------------------------------------------------===//
// §20  Close namespaces
//===----------------------------------------------------------------------===//

} // namespace (anonymous)
} // namespace mlir::enzyme