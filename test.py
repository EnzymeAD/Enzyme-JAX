import pyllvm

f = """
#include <stdio.h>
int main() {
    printf("Hello World\\n");
}
"""
import os
dn = os.path.dirname(pyllvm.__file__)
print(dn)
dn = os.path.join(dn, "external", "llvm-project", "clang", "staging")
print(dn)
res = pyllvm.compile(f, ["-resource-dir", dn])

print(res)


