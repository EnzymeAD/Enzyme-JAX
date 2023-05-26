import pyllvm

f = """
#include <stdio.h>
int main() {
    printf("Hello World\\n");
}
double square(double x) { return x*x; }
double __enzyme_autodiff(void*, ...);
double dsquare(double x) {
    return __enzyme_autodiff((void*)square, x);
}
"""
import os
dn = os.path.dirname(pyllvm.__file__)
dn = os.path.join(dn, "external", "llvm-project", "clang", "staging")
res = pyllvm.compile(f, ["-v", "-resource-dir", dn, "-O2"])

print(res)


