#!/usr/bin/env bash
# 一時的な実験（issue #264）。CI で binding count の境界を測る。測り終えたら消す。
set -uo pipefail
NABLA_IREE_HOME="${NABLA_IREE_HOME:-${HOME}/.local/share/nabla/iree-3.11.0}"
C="${NABLA_IREE_HOME}/bin/iree-compile"
R="${NABLA_IREE_HOME}/bin/iree-run-module"
W="$(mktemp -d)"

gen() { # shape n width
  python3 - "$1" "$2" "$3" <<'PY'
import sys
shape, n, w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = f"tensor<{w}xf32>"
args = ", ".join(f"%a{i}: {t}" for i in range(n))
body = []
if shape == "chain":
    body.append(f"%r1 = stablehlo.add %a0, %a1 : {t}")
    for i in range(2, n):
        body.append(f"%r{i} = stablehlo.add %r{i-1}, %a{i} : {t}")
    ret, rt = f"%r{n-1}", t
elif shape == "chain-mul":
    body.append(f"%r1 = stablehlo.multiply %a0, %a1 : {t}")
    for i in range(2, n):
        body.append(f"%r{i} = stablehlo.multiply %r{i-1}, %a{i} : {t}")
    ret, rt = f"%r{n-1}", t
elif shape == "tree":
    vals = [f"%a{i}" for i in range(n)]
    k = 0
    while len(vals) > 1:
        nxt = []
        for j in range(0, len(vals) - 1, 2):
            body.append(f"%t{k} = stablehlo.add {vals[j]}, {vals[j+1]} : {t}")
            nxt.append(f"%t{k}"); k += 1
        if len(vals) % 2: nxt.append(vals[-1])
        vals = nxt
    ret, rt = vals[0], t
elif shape in ("loss", "reduce-chain"):
    s = "tensor<f32>"
    for i in range(n):
        x = f"%a{i}"
        if shape == "loss":
            body.append(f"%d{i} = stablehlo.add %a{i}, %a{i} : {t}")
            body.append(f"%h{i} = stablehlo.tanh %d{i} : {t}")
            x = f"%h{i}"
        body.append(f"%z{i} = stablehlo.constant dense<0.0> : {s}")
        body.append(f"%s{i} = stablehlo.reduce({x} init: %z{i}) applies stablehlo.add across dimensions = [0] : ({t}, {s}) -> {s}")
    body.append(f"%r1 = stablehlo.add %s0, %s1 : {s}")
    for i in range(2, n):
        body.append(f"%r{i} = stablehlo.add %r{i-1}, %s{i} : {s}")
    ret, rt = f"%r{n-1}", s
elif shape == "concat":
    rt = f"tensor<{w*n}xf32>"
    body.append(f"%c = stablehlo.concatenate " + ", ".join(f"%a{i}" for i in range(n)) + f", dim = 0 : ({', '.join([t]*n)}) -> {rt}")
    ret = "%c"
elif shape == "concat-tanh":
    rt = f"tensor<{w*n}xf32>"
    body.append(f"%c = stablehlo.concatenate " + ", ".join(f"%a{i}" for i in range(n)) + f", dim = 0 : ({', '.join([t]*n)}) -> {rt}")
    body.append(f"%h = stablehlo.tanh %c : {rt}")
    ret = "%h"
elif shape == "map":
    for i in range(n):
        body.append(f"%m{i} = stablehlo.tanh %a{i} : {t}")
    ret = ", ".join(f"%m{i}" for i in range(n)); rt = "(" + ", ".join([t]*n) + ")"
print("module {\nfunc.func @main(" + args + f") -> {rt} {{")
for b in body: print("  " + b)
print(f"  return {ret} : {rt.strip('()') if shape=='map' else rt}")
print("}\n}")
PY
}

try() { # shape n width flags...
  local shape=$1 n=$2 w=$3; shift 3
  gen "$shape" "$n" "$w" > "$W/m.mlir"
  local cerr
  if ! cerr=$("$C" --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu "$@" "$W/m.mlir" -o "$W/m.vmfb" 2>&1); then
    printf 'RESULT %-12s n=%-4s w=%-5s flags=[%s] compile=FAIL %s\n' "$shape" "$n" "$w" "$*" "$(echo "$cerr" | grep -m1 error | cut -c1-120)"
    return
  fi
  local inputs=(); for ((j = 0; j < n; j++)); do inputs+=("--input=${w}xf32=1"); done
  local out=""
  for dev in local-task local-sync; do
    local o code=0
    o=$("$R" --device=$dev --module="$W/m.vmfb" --function=main "${inputs[@]}" 2>&1) || code=$?
    local msg; msg=$(echo "$o" | grep -o 'binding count [0-9]* > [0-9]*' | head -1)
    out="$out $dev=$code${msg:+($msg)}"
  done
  printf 'RESULT %-12s n=%-4s w=%-5s flags=[%s]%s\n' "$shape" "$n" "$w" "$*" "$out"
}

"$C" --version | sed -n 2p
for shape in chain chain-mul tree loss reduce-chain concat concat-tanh map; do
  for n in 8 15 16 17 20 24 30 31 32 33 34 40 64 100 160; do
    try "$shape" "$n" 3
  done
done
for n in 16 17 31 32 33 40; do try chain "$n" 4096; try loss "$n" 4096; done
for flags in "--iree-opt-level=O0" "--iree-dispatch-creation-opt-level=O0" \
             "--iree-dispatch-creation-enable-aggressive-fusion=false" \
             "--iree-dispatch-creation-fuse-multi-use=false" "--iree-dispatch-creation-enable-fuse-horizontal-contractions=false"; do
  for shape in chain tree loss reduce-chain; do
    for n in 33 40 160; do try "$shape" "$n" 3 $flags; done
  done
done

# 失敗する形の dispatch の構造（flow まで下ろした IR の dispatch の行）
for shape in chain loss; do
  gen "$shape" 40 3 > "$W/m.mlir"
  echo "=== flow dispatches: $shape n=40"
  "$C" --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
       --compile-to=flow "$W/m.mlir" -o "$W/f.mlir" 2>&1 | head -5
  grep -c 'flow.dispatch @' "$W/f.mlir"
  grep 'flow.dispatch @' "$W/f.mlir" | sed -E 's/tensor<[^>]*>//g' | cut -c1-400 | head -10
done
gen chain 40 3 > "$W/m.mlir"
echo "=== passes that touch dispatch regions (chain n=40)"
"$C" --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
     --compile-to=flow --mlir-print-ir-after-all --mlir-print-ir-module-scope=false "$W/m.mlir" -o /dev/null 2>"$W/ir.txt"
grep -n 'IR Dump After' "$W/ir.txt" | awk -F'IR Dump After ' '{print $2}' | uniq | head -150
echo "--- first dump containing flow.dispatch.region"
awk '/IR Dump After/{hdr=$0} /flow.dispatch.region/{print hdr; exit}' "$W/ir.txt"
rm -rf "$W"
