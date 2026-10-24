#!/usr/bin/env bash
# 一時的な実験（issue #264）。消す。
set -uo pipefail
C="${NABLA_IREE_HOME}/bin/iree-compile"
R="${NABLA_IREE_HOME}/bin/iree-run-module"
W="$(mktemp -d)"
gen() { # shape n
  python3 - "$1" "$2" <<'PY'
import sys
shape, n = sys.argv[1], int(sys.argv[2])
t = "tensor<3xf32>"
args = ", ".join(f"%a{i}: {t}" for i in range(n))
body = []
if shape == "chain":
    body.append(f"%r1 = stablehlo.add %a0, %a1 : {t}")
    for i in range(2, n):
        body.append(f"%r{i} = stablehlo.add %r{i-1}, %a{i} : {t}")
    ret, rt = f"%r{n-1}", t
elif shape == "tree":
    vals = [f"%a{i}" for i in range(n)]; k = 0
    while len(vals) > 1:
        nxt = []
        for j in range(0, len(vals) - 1, 2):
            body.append(f"%t{k} = stablehlo.add {vals[j]}, {vals[j+1]} : {t}"); nxt.append(f"%t{k}"); k += 1
        if len(vals) % 2: nxt.append(vals[-1])
        vals = nxt
    ret, rt = vals[0], t
elif shape == "sum1":
    # 全部を concat して reduce（1 出力、1 dispatch に多くの入力）
    body.append("%c = stablehlo.concatenate " + ", ".join(f"%a{i}" for i in range(n)) + f", dim = 0 : ({', '.join([t]*n)}) -> tensor<{3*n}xf32>")
    ret, rt = "%c", f"tensor<{3*n}xf32>"
print("module {\nfunc.func @main(" + args + f") -> {rt} {{")
for b in body: print("  " + b)
print(f"  return {ret} : {rt}")
print("}\n}")
PY
}
try() { # shape n flags...
  local shape=$1 n=$2; shift 2
  gen "$shape" "$n" > "$W/m.mlir"
  "$C" --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu "$@" "$W/m.mlir" -o "$W/m.vmfb" 2>/dev/null || { echo "RESULT $shape n=$n compile-fail"; return; }
  local inputs=(); for ((j = 0; j < n; j++)); do inputs+=("--input=3xf32=1"); done
  local out=""
  for dev in local-task local-sync; do
    local o code=0
    o=$("$R" --device=$dev --module="$W/m.vmfb" --function=main "${inputs[@]}" 2>&1) || code=$?
    local msg; msg=$(echo "$o" | grep -o 'binding count [0-9]* > [0-9]*' | head -1)
    out="$out $dev=$code${msg:+($msg)}"
  done
  echo "RESULT $shape n=$n flags=[$*]$out"
}
NOFUSE=--iree-dispatch-creation-fuse-multi-use=false
for n in 100 120 128 129 140 150 155 158 159 160 161 162; do try chain $n $NOFUSE; done
for n in 128 129 150 160 161 162; do try tree $n; try sum1 $n; done
for n in 160 161; do for k in 1 2 3; do try chain $n $NOFUSE; done; done
gen chain 161 > "$W/c.mlir"
"$C" --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu $NOFUSE --compile-to=flow "$W/c.mlir" -o "$W/f.mlir"
echo "chain161 nofuse dispatches=$(grep -c 'flow.dispatch @' "$W/f.mlir")"
gen chain 160 > "$W/c.mlir"
"$C" --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu $NOFUSE --compile-to=flow "$W/c.mlir" -o "$W/f.mlir"
echo "chain160 nofuse dispatches=$(grep -c 'flow.dispatch @' "$W/f.mlir")"
rm -rf "$W"
