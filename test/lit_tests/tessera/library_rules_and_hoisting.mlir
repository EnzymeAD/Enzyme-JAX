// RUN: enzymexlamlir-opt %s -parse-optimization-rules -tessera-propagate-properties -tessera-apply-pdl -tessera-hoist-invariant -split-input-file | FileCheck %s

// Rules, facts and hoisting together, as on a library of thin vector wrappers
// (IntVec over std::vector<int>), with what must stop each.

// A rule splits a lookup into a check and a lookup that uses its answer; the
// check leaves a loop that doesn't change the vector.
module {
  tessera.define private @vec.contains(!llvm.ptr {tessera.unchanged}, i32) -> i1 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @vec.is_sorted(!llvm.ptr {llvm.nocapture}) -> i32 attributes {argModes = [unit], pure = true}
  tessera.define private @vec.contains_if(!llvm.ptr {tessera.unchanged}, i32, i32) -> i1 attributes {argModes = [unit, unit, unit], pure = false}
  tessera.define private @vec.push_back(!llvm.ptr, i32) attributes {argModes = [unit, unit], pure = false}
  llvm.func @use(i1)

  tessera.optimizations {
    tessera.optimization "vec.contains(v, x) -> s = vec.is_sorted(v); r = vec.contains_if(v, s, x); r"
  }

  // CHECK-LABEL: llvm.func @queries
  // CHECK: %[[S:.*]] = tessera.call @vec.is_sorted
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: tessera.call @vec.contains_if(%{{.*}}, %[[S]], %{{.*}})
  llvm.func @queries(%v: !llvm.ptr {llvm.noalias}, %x: i32, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %r = tessera.call @vec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
      llvm.call @use(%r) : (i1) -> ()
    }
    llvm.return
  }

  // The loop changes the vector, so the check stays in it.
  // CHECK-LABEL: llvm.func @queries_changing
  // CHECK: scf.for
  // CHECK-NEXT: tessera.call @vec.push_back
  // CHECK-NEXT: tessera.call @vec.is_sorted
  // CHECK-NEXT: tessera.call @vec.contains_if
  llvm.func @queries_changing(%v: !llvm.ptr {llvm.noalias}, %x: i32, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      tessera.call @vec.push_back(%v, %x) : (!llvm.ptr, i32) -> ()
      %r = tessera.call @vec.contains(%v, %x) : (!llvm.ptr, i32) -> i1
      llvm.call @use(%r) : (i1) -> ()
    }
    llvm.return
  }
}

// -----

// An index a program builds, uses and frees each query is built once, and
// freed after the loop.
module {
  tessera.define private @vec.build_index(!llvm.ptr {llvm.nocapture}) -> !llvm.ptr attributes {argModes = [unit], pure = true}
  tessera.define private @idx.lookup(!llvm.ptr {tessera.unchanged}, i32) -> i1 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @vec.push_back(!llvm.ptr, i32) attributes {argModes = [unit, unit], pure = false}
  tessera.define private @c.free(!llvm.ptr) attributes {argModes = [unit], pure = false, tessera.original_name = "free"}
  llvm.func @use(i1)

  // CHECK-LABEL: llvm.func @lookups
  // CHECK: %[[I:.*]] = tessera.call @vec.build_index
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: tessera.call @idx.lookup(%[[I]]
  // CHECK: }
  // CHECK-NEXT: tessera.call @c.free(%[[I]])
  llvm.func @lookups(%v: !llvm.ptr {llvm.noalias}, %x: i32, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      %idx = tessera.call @vec.build_index(%v) : (!llvm.ptr) -> !llvm.ptr
      %h = tessera.call @idx.lookup(%idx, %x) : (!llvm.ptr, i32) -> i1
      llvm.call @use(%h) : (i1) -> ()
      tessera.call @c.free(%idx) : (!llvm.ptr) -> ()
    }
    llvm.return
  }

  // The loop changes the vector, so the index is rebuilt each query.
  // CHECK-LABEL: llvm.func @lookups_changing
  // CHECK: scf.for
  // CHECK-NEXT: tessera.call @vec.push_back
  // CHECK-NEXT: tessera.call @vec.build_index
  llvm.func @lookups_changing(%v: !llvm.ptr {llvm.noalias}, %x: i32, %n: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %ni = arith.index_cast %n : i64 to index
    scf.for %i = %c0 to %ni step %c1 {
      tessera.call @vec.push_back(%v, %x) : (!llvm.ptr, i32) -> ()
      %idx = tessera.call @vec.build_index(%v) : (!llvm.ptr) -> !llvm.ptr
      %h = tessera.call @idx.lookup(%idx, %x) : (!llvm.ptr, i32) -> i1
      llvm.call @use(%h) : (i1) -> ()
      tessera.call @c.free(%idx) : (!llvm.ptr) -> ()
    }
    llvm.return
  }
}

// -----

// The intersection of two sorted vectors is sorted, by a conditional
// guarantee, so a lookup in it can use binary search.
module {
  tessera.define private @vec.sort(!llvm.ptr {tessera.establishes = ["sorted"]}) attributes {argModes = [unit], pure = false}
  tessera.define private @set.intersect(!llvm.ptr {tessera.unchanged}, !llvm.ptr {tessera.unchanged}, !llvm.ptr) attributes {argModes = [unit, unit, unit], pure = false, tessera.conditional_establishes = [{arg = 2 : i64, condition = "sorted(a) && sorted(b)", constants = {}, names = {a = 0 : i64, b = 1 : i64}, property = "sorted"}]}
  tessera.define private @vec.contains(!llvm.ptr {tessera.unchanged}, i32) -> i1 attributes {argModes = [unit, unit], pure = false}
  tessera.define private @vec.binary_search(!llvm.ptr {tessera.assumes = ["sorted"], tessera.unchanged}, i32) -> i1 attributes {argModes = [unit, unit], pure = false}

  tessera.optimizations {
    tessera.optimization "if sorted(v), vec.contains(v, x) -> vec.binary_search(v, x)"
  }

  // CHECK-LABEL: llvm.func @both_sorted
  // CHECK: tessera.call @set.intersect
  // CHECK-NOT: tessera.call @vec.contains
  // CHECK: tessera.call @vec.binary_search
  llvm.func @both_sorted(%a: !llvm.ptr {llvm.noalias}, %b: !llvm.ptr {llvm.noalias}, %out: !llvm.ptr {llvm.noalias}, %x: i32) -> i1 {
    tessera.call @vec.sort(%a) : (!llvm.ptr) -> ()
    tessera.call @vec.sort(%b) : (!llvm.ptr) -> ()
    tessera.call @set.intersect(%a, %b, %out) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %r = tessera.call @vec.contains(%out, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }

  // One input isn't known sorted, so neither is the result.
  // CHECK-LABEL: llvm.func @one_sorted
  // CHECK: tessera.call @set.intersect
  // CHECK-NOT: tessera.call @vec.binary_search
  // CHECK: tessera.call @vec.contains
  llvm.func @one_sorted(%a: !llvm.ptr {llvm.noalias}, %b: !llvm.ptr {llvm.noalias}, %out: !llvm.ptr {llvm.noalias}, %x: i32) -> i1 {
    tessera.call @vec.sort(%a) : (!llvm.ptr) -> ()
    tessera.call @set.intersect(%a, %b, %out) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    %r = tessera.call @vec.contains(%out, %x) : (!llvm.ptr, i32) -> i1
    llvm.return %r : i1
  }
}

// -----

// A heap's maximum is its front while it is a heap: push_back breaks that,
// and make or push restore it.
module {
  tessera.define private @heap.make(!llvm.ptr {tessera.establishes = ["heap"]}) attributes {argModes = [unit], pure = false}
  tessera.define private @heap.push_back(!llvm.ptr, i32) attributes {argModes = [unit, unit], pure = false}
  tessera.define private @heap.push(!llvm.ptr {tessera.assumes = ["heap"], tessera.establishes = ["heap"]}, i32) attributes {argModes = [unit, unit], pure = false}
  tessera.define private @heap.max(!llvm.ptr {tessera.unchanged}) -> i32 attributes {argModes = [unit], pure = false}
  tessera.define private @heap.front(!llvm.ptr {tessera.assumes = ["heap"], tessera.unchanged}) -> i32 attributes {argModes = [unit], pure = false}
  llvm.func @use(i32)

  tessera.optimizations {
    tessera.optimization "if heap(h), heap.max(h) -> heap.front(h)"
  }

  // CHECK-LABEL: llvm.func @heap_cycle
  // CHECK: tessera.call @heap.make
  // CHECK-NOT: tessera.call @heap.max
  // CHECK: tessera.call @heap.front
  // CHECK: tessera.call @heap.push_back
  // CHECK-NOT: tessera.call @heap.front
  // CHECK: tessera.call @heap.max
  // CHECK: tessera.call @heap.make
  // CHECK: tessera.call @heap.push(
  // CHECK-NOT: tessera.call @heap.max
  // CHECK: tessera.call @heap.front
  // CHECK: llvm.return
  llvm.func @heap_cycle(%h: !llvm.ptr {llvm.noalias}, %x: i32) {
    tessera.call @heap.make(%h) : (!llvm.ptr) -> ()
    %m0 = tessera.call @heap.max(%h) : (!llvm.ptr) -> i32
    llvm.call @use(%m0) : (i32) -> ()
    tessera.call @heap.push_back(%h, %x) : (!llvm.ptr, i32) -> ()
    %m1 = tessera.call @heap.max(%h) : (!llvm.ptr) -> i32
    llvm.call @use(%m1) : (i32) -> ()
    tessera.call @heap.make(%h) : (!llvm.ptr) -> ()
    tessera.call @heap.push(%h, %x) : (!llvm.ptr, i32) -> ()
    %m2 = tessera.call @heap.max(%h) : (!llvm.ptr) -> i32
    llvm.call @use(%m2) : (i32) -> ()
    llvm.return
  }
}
