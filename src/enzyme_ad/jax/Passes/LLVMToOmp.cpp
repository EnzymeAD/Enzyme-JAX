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
    call.erase();
    fn.erase();
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
      LLVM::CallOp end = findNextCallTo(start,"__kmpc_end_critical");
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
      LLVM::CallOp end = findNextCallTo(start,"__kmpc_end_ordered");
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
      LLVM::CallOp end = findNextCallTo(start,"__kmpc_end_taskgroup");
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

void LLVMToOMPPass::convertStaticWs(ModuleOp mod) {
  SmallVector<LLVM::CallOp> inits;
  mod.walk([&](LLVM::CallOp c){
    if (calleeStartsWith(c,"__kmpc_for_static_init_")) inits.push_back(c);
  });
  for (LLVM::CallOp init : inits) {
    LLVM::CallOp fini = findNextCallTo(init,"__kmpc_for_static_fini");
    if (!fini) {
      init.emitWarning("LLVMToOmp: for_static_init without fini — skipped");
      continue;
    }

    // ── NEW: bail if init and fini are in different blocks.
    // The optimized CFG places the loop body in separate blocks with
    // block arguments (loop-carried IV). Moving those ops across block
    // boundaries breaks SSA — the block arg has no defining op to move.
    if (init->getBlock() != fini->getBlock()) {
      init.emitWarning("LLVMToOmp: for_static_init and fini in different "
                       "blocks (optimized CFG) — wsloop conversion skipped. "
                       "Compile with -O0 or implement CFG-aware conversion.");
      continue;
    }


    bool nw = detectNowait(fini, /*eraseBarrier=*/false);

    // Iteration type from suffix (_4 → i32, _8/_4u/_8u → i64 for 8).
    StringRef sfx = getCalleeName(init).drop_front(
        strlen("__kmpc_for_static_init_"));
    MLIRContext *ctx = mod.getContext();
    Type iterTy = SW(sfx,"8") ? (Type)IntegerType::get(ctx,64)
                               : (Type)IntegerType::get(ctx,32);

    // Args: ident(0) gtid(1) sched(2) p_last(3) p_lb(4) p_ub(5)
    //       p_stride(6) incr(7) chunk(8)
    omp::ClauseScheduleKind sched = omp::ClauseScheduleKind::Static;
    if (auto sv = getConstInt(init.getNumOperands()>2
                                  ? init.getOperand(2) : Value{}))
      sched = kmpSched(*sv);
    Value chunk = init.getNumOperands()>8 ? init.getOperand(8) : Value{};
    Value lbPtr = init.getNumOperands()>4 ? init.getOperand(4) : Value{};
    Value ubPtr = init.getNumOperands()>5 ? init.getOperand(5) : Value{};
    Value stPtr = init.getNumOperands()>6 ? init.getOperand(6) : Value{};
    if (!lbPtr||!ubPtr||!stPtr) {
      init.emitWarning("LLVMToOmp: for_static_init missing ptr args — skipped");
      continue;
    }

    OpBuilder b(init); Location loc = init.getLoc();
    Value lb = b.create<LLVM::LoadOp>(loc, iterTy, lbPtr);
    Value ub = b.create<LLVM::LoadOp>(loc, iterTy, ubPtr);
    Value st = b.create<LLVM::LoadOp>(loc, iterTy, stPtr);

    // WsloopOperands fields (OpenMP_ScheduleClause, OpenMP_NowaitClause):
    //   schedule        = ClauseScheduleKindAttr
    //   scheduleChunkVar = Value (optional)
    //   nowait          = UnitAttr (optional)
    omp::WsloopOperands wOps;
    wOps.scheduleKind = omp::ClauseScheduleKindAttr::get(ctx, sched);
    if (chunk) wOps.scheduleChunk = chunk;
    if (nw)    wOps.nowait           = b.getUnitAttr();

    auto wsOp = b.create<omp::WsloopOp>(loc, wOps);
    // SingleBlock trait: region.front() auto-created by build().
    // NoTerminator trait: do NOT add omp.terminator.
    Block *wsBlk = &wsOp.getRegion().front();
    b.setInsertionPointToStart(wsBlk);

    // LoopNestOp — no SingleBlock → create block manually.
    // LoopNestOperands fields (OpenMP_LoopRelatedClause):
    //   loopLowerBounds, loopUpperBounds, loopSteps (ValueRange)
    omp::LoopNestOperands lnOps;
    lnOps.loopLowerBounds = {lb};
    lnOps.loopUpperBounds = {ub};
    lnOps.loopSteps       = {st};
    auto nestOp = b.create<omp::LoopNestOp>(loc, lnOps);
    Block *nestBlk = b.createBlock(&nestOp.getRegion());
    nestBlk->addArgument(iterTy, loc); // IV = getIVs()[0]
    b.setInsertionPointToStart(nestBlk);
    // LoopNestOp terminated by omp.yield (YieldOp parentOneOf includes LoopNestOp)
    b.create<omp::YieldOp>(loc); // empty yield (no returned SSA values)

    auto bodyOps = opsBetween(init, fini);
    moveBeforeTerminator(bodyOps, nestBlk); // insert before omp.yield

    init.erase(); fini.erase();
    LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] static wsloop"
                             << (nw?" nowait":"") << "\n");
  }
}

//===----------------------------------------------------------------------===//
// §13  Dynamic work-sharing loop  (__kmpc_dispatch_init/next/fini)
//
// Same WsloopOp/LoopNestOp API as §12.
// Body ops are between dispatch_next and dispatch_fini.
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertDynWs(ModuleOp mod) {
  SmallVector<LLVM::CallOp> inits;
  mod.walk([&](LLVM::CallOp c){
    if (calleeStartsWith(c,"__kmpc_dispatch_init_")) inits.push_back(c);
  });
  for (LLVM::CallOp init : inits) {
    StringRef base = getCalleeName(init);
    StringRef tsuf = base.drop_front(strlen("__kmpc_dispatch_init_"));
    auto nextN = ("__kmpc_dispatch_next_" + tsuf).str();
    auto finiN = ("__kmpc_dispatch_fini_" + tsuf).str();
    LLVM::CallOp next = findNextCallTo(init, nextN);
    LLVM::CallOp fini = next ? findNextCallTo(next, finiN) : LLVM::CallOp{};
    if (!next||!fini) {
      init.emitWarning("LLVMToOmp: incomplete dispatch triple — skipped");
      continue;
    }
    // New: bail if init and fini are in different blocks.
    // The optimized CFG places the loop body in separate blocks with
    // block arguments (loop-carried IV). Moving those ops across block
    // boundaries breaks SSA — the block arg has no defining op to move.
    if (init->getBlock() != fini->getBlock()) {
      init.emitWarning("LLVMToOmp: for_static_init and fini in different "
                       "blocks (optimized CFG) — wsloop conversion skipped. "
                       "Compile with -O0 or implement CFG-aware conversion.");
      continue;
    }
    bool nw = detectNowait(fini, false);

    MLIRContext *ctx = mod.getContext();
    Type iterTy = SW(tsuf,"8") ? (Type)IntegerType::get(ctx,64)
                                : (Type)IntegerType::get(ctx,32);
    omp::ClauseScheduleKind sched = omp::ClauseScheduleKind::Dynamic;
    if (auto sv = getConstInt(init.getNumOperands()>2
                                  ? init.getOperand(2) : Value{}))
      sched = kmpSched(*sv);
    Value lb    = init.getNumOperands()>3 ? init.getOperand(3) : Value{};
    Value ub    = init.getNumOperands()>4 ? init.getOperand(4) : Value{};
    Value st    = init.getNumOperands()>5 ? init.getOperand(5) : Value{};
    Value chunk = init.getNumOperands()>6 ? init.getOperand(6) : Value{};
    if (!lb||!ub||!st) {
      init.emitWarning("LLVMToOmp: dispatch_init missing bounds — skipped");
      continue;
    }
    // Use per-chunk bounds from dispatch_next out-params if available.
    // next args: ident(0) gtid(1) &p_last(2) &p_lb(3) &p_ub(4) &p_stride(5)
    OpBuilder b(init); Location loc = init.getLoc();
    Value loopLb = lb, loopUb = ub, loopSt = st;
    if (next.getNumOperands() > 5) {
      loopLb = b.create<LLVM::LoadOp>(loc, iterTy, next.getOperand(3));
      loopUb = b.create<LLVM::LoadOp>(loc, iterTy, next.getOperand(4));
      loopSt = b.create<LLVM::LoadOp>(loc, iterTy, next.getOperand(5));
    }

    omp::WsloopOperands wOps;
    wOps.scheduleKind = omp::ClauseScheduleKindAttr::get(ctx, sched);;
    if (chunk) wOps.scheduleChunk = chunk;
    if (nw)    wOps.nowait           = b.getUnitAttr();

    auto wsOp = b.create<omp::WsloopOp>(loc, wOps);
    Block *wsBlk = &wsOp.getRegion().front(); // SingleBlock auto-created
    b.setInsertionPointToStart(wsBlk);

    omp::LoopNestOperands lnOps;
    lnOps.loopLowerBounds = {loopLb};
    lnOps.loopUpperBounds = {loopUb};
    lnOps.loopSteps       = {loopSt};
    auto nestOp = b.create<omp::LoopNestOp>(loc, lnOps);
    Block *nestBlk = b.createBlock(&nestOp.getRegion());
    nestBlk->addArgument(iterTy, loc);
    b.setInsertionPointToStart(nestBlk);
    b.create<omp::YieldOp>(loc);

    auto bodyOps = opsBetween(next, fini); // body between next and fini
    moveBeforeTerminator(bodyOps, nestBlk);

    init.erase(); next.erase(); fini.erase();
    LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] dynamic wsloop"
                             << (nw?" nowait":"") << "\n");
  }
}

//===----------------------------------------------------------------------===//
// §14  Tasks  (__kmpc_task_alloc + __kmpc_omp_task → omp.task)
//
// td: TaskOp — explicit builder OpBuilder<(ins CArg<"const TaskOperands &">)>
//     No SingleBlock → create block explicitly.
//     TaskOperands.untied = UnitAttr (OpenMP_UntiedClause)
//     TaskOperands.mergeable = UnitAttr (OpenMP_MergeableClause)
//===----------------------------------------------------------------------===//

void LLVMToOMPPass::convertTasks(ModuleOp mod) {
  SmallVector<LLVM::CallOp> allocs;
  mod.walk([&](LLVM::CallOp c){
    if (isCallTo(c,"__kmpc_task_alloc")) allocs.push_back(c);
  });
  for (LLVM::CallOp alloc : allocs) {
    Value td = alloc.getResult();
    LLVM::CallOp taskCall;

    for (Operation *u : llvm::make_early_inc_range(td.getUsers())) {
      if (auto uc = dyn_cast<LLVM::CallOp>(u)) {
        StringRef n = getCalleeName(uc);
        if (n=="__kmpc_omp_task"||n=="__kmpc_omp_task_with_deps")
          { taskCall = uc; break; }
        if (n=="__kmpc_omp_task_begin_if0")
          { taskCall = uc; break; }
      }
    }
    if (!taskCall) {
      alloc.emitWarning("LLVMToOmp: __kmpc_task_alloc without consumer — "
                        "left as llvm.call"); continue;
    }
    if (isCallTo(taskCall,"__kmpc_omp_task_with_deps"))
      taskCall.emitWarning("LLVMToOmp: task depend clauses dropped — "
                           "kmp_depend_info_t layout analysis needed");

    // alloc arg 2 = flags: bit0=untied, bit1=final, bit2=mergeable
    bool isUntied = false, isMergeable = false;
    if (auto fv = getConstInt(alloc.getNumOperands()>2
                                  ? alloc.getOperand(2) : Value{})) {
      isUntied    = *fv & 1;
      isMergeable = (*fv >> 2) & 1;
    }
    // alloc arg 5 = task entry outlined function pointer
    LLVM::LLVMFuncOp fn;
    if (alloc.getNumOperands() > 5)
      if (auto a = alloc.getOperand(5).getDefiningOp<LLVM::AddressOfOp>())
        fn = mod.lookupSymbol<LLVM::LLVMFuncOp>(a.getGlobalName());

    OpBuilder b(alloc); Location loc = alloc.getLoc();
    omp::TaskOperands tOps;
    if (isUntied)    tOps.untied    = b.getUnitAttr();
    if (isMergeable) tOps.mergeable = b.getUnitAttr();

    auto taskOp = b.create<omp::TaskOp>(loc, tOps);
    b.createBlock(&taskOp.getRegion()); // no SingleBlock
    b.setInsertionPointToStart(&taskOp.getRegion().front());

    if (fn) {
      SmallVector<Value> noCaps;
      if (failed(inlineOutlinedBody(fn, taskOp.getRegion(), b, noCaps))) {
        taskOp.erase();
        alloc.emitWarning("LLVMToOmp: failed to inline task body — skipped");
        continue;
      }
    } else {
      b.create<omp::TerminatorOp>(loc);
      alloc.emitWarning("LLVMToOmp: task entry fn unresolvable — "
                        "emitting empty omp.task");
    }

    alloc.erase(); 
    taskCall.erase();
    if (fn) fn.erase(); 
    LLVM_DEBUG(llvm::dbgs() << "[LLVMToOmp] task_alloc → omp.task\n");
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
           GpuTargetPat, PushClausePat, TaskCompleteIf0Pat, UnhandledPat>(mod.getContext());
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
  convertDynWs(mod);      // dispatch_init/next/fini→ wsloop dynamic
  convertTasks(mod);      // task_alloc + omp_task  → omp.task
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