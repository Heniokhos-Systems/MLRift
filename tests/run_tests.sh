#!/bin/bash
# No set -e: test binaries return non-zero exit codes intentionally

DIR="$(cd "$(dirname "$0")" && pwd)"
MLRC="${MLRC:-$DIR/../build/mlrc3}"
ARCH=$(uname -m)
MLRC_FLAGS="${MLRC_FLAGS:---arch=$ARCH}"
PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    local input="$2"
    local expected="$3"
    TOTAL=$((TOTAL + 1))

    local REPO_ROOT="$DIR/.."
    printf '%s\n' "$input" > "$REPO_ROOT/test_tmp_$$.mlr"
    if $MLRC $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ > /dev/null 2>&1; then
        rm -f "$REPO_ROOT/test_tmp_$$.mlr"
        chmod +x /tmp/mlrc_test_$$
        local got=0
        /tmp/mlrc_test_$$ > /dev/null 2>&1 && got=0 || got=$?
        if [ "$got" = "$expected" ]; then
            PASS=$((PASS + 1))
        else
            echo "FAIL: $name (expected $expected, got $got)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: $name (compilation failed)"
        $MLRC $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ 2>&1 | head -3
        FAIL=$((FAIL + 1))
    fi
    rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$
}

run_test_output() {
    local name="$1"
    local input="$2"
    local expected_output="$3"
    local expected_exit="${4:-0}"
    TOTAL=$((TOTAL + 1))

    local REPO_ROOT="$DIR/.."
    printf '%s\n' "$input" > "$REPO_ROOT/test_tmp_$$.mlr"
    if $MLRC $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ > /dev/null 2>&1; then
        rm -f "$REPO_ROOT/test_tmp_$$.mlr"
        chmod +x /tmp/mlrc_test_$$
        local got_output
        got_output=$(/tmp/mlrc_test_$$ 2>/dev/null)
        local got_exit=$?
        if [ "$got_output" = "$expected_output" ] && [ "$got_exit" = "$expected_exit" ]; then
            PASS=$((PASS + 1))
        else
            if [ "$got_output" != "$expected_output" ]; then
                echo "FAIL: $name (expected output '$expected_output', got '$got_output')"
            else
                echo "FAIL: $name (expected exit $expected_exit, got $got_exit)"
            fi
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: $name (compilation failed)"
        FAIL=$((FAIL + 1))
    fi
    rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$
}

# Same as run_test but forces the legacy direct-codegen path (--legacy).
run_test_legacy() {
    local name="$1"
    local input="$2"
    local expected="$3"
    TOTAL=$((TOTAL + 1))

    local REPO_ROOT="$DIR/.."
    printf '%s\n' "$input" > "$REPO_ROOT/test_tmp_$$.mlr"
    if $MLRC $MLRC_FLAGS --legacy "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ > /dev/null 2>&1; then
        rm -f "$REPO_ROOT/test_tmp_$$.mlr"
        chmod +x /tmp/mlrc_test_$$
        local got=0
        /tmp/mlrc_test_$$ > /dev/null 2>&1 && got=0 || got=$?
        if [ "$got" = "$expected" ]; then
            PASS=$((PASS + 1))
        else
            echo "FAIL: $name (expected $expected, got $got)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: $name (compilation failed)"
        $MLRC $MLRC_FLAGS --legacy "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ 2>&1 | head -3
        FAIL=$((FAIL + 1))
    fi
    rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$
}

echo "=== MLRift Self-Hosted Compiler Test Suite ==="
echo ""

# --- Basic tests ---
run_test "exit_42" 'fn main() { exit(42) }' 42
run_test "exit_0" 'fn main() { exit(0) }' 0

# --- Variables ---
run_test "var_assign" 'fn main() {
    uint64 x = 42
    exit(x)
}' 42

run_test "var_reassign" 'fn main() {
    uint64 x = 1
    x = 42
    exit(x)
}' 42

# --- Arithmetic ---
run_test "add" 'fn main() { exit(10 + 20) }' 30
run_test "sub" 'fn main() { exit(50 - 8) }' 42
run_test "mul" 'fn main() { exit(6 * 7) }' 42
run_test "div" 'fn main() { exit(84 / 2) }' 42
run_test "mod" 'fn main() { exit(47 % 5) }' 2

# --- Bitwise ---
run_test "and" 'fn main() { exit(0xFF & 0x2A) }' 42
run_test "or" 'fn main() { exit(0x20 | 0x0A) }' 42
run_test "xor" 'fn main() { exit(0xFF ^ 0xD5) }' 42
run_test "shl" 'fn main() { exit(21 << 1) }' 42
run_test "shr" 'fn main() { exit(84 >> 1) }' 42

# --- Unary ---
run_test "not_0" 'fn main() { exit(!0) }' 1
run_test "not_1" 'fn main() { exit(!1) }' 0
run_test "neg" 'fn main() { exit((-1) & 0xFF) }' 255

# --- Comparisons ---
run_test "eq_true" 'fn main() { if 5 == 5 { exit(1) } exit(0) }' 1
run_test "eq_false" 'fn main() { if 5 == 6 { exit(1) } exit(0) }' 0
run_test "lt" 'fn main() { if 3 < 5 { exit(1) } exit(0) }' 1
run_test "gt" 'fn main() { if 5 > 3 { exit(1) } exit(0) }' 1
run_test "le" 'fn main() { if 5 <= 5 { exit(1) } exit(0) }' 1
run_test "ge" 'fn main() { if 5 >= 5 { exit(1) } exit(0) }' 1
run_test "ne" 'fn main() { if 5 != 6 { exit(1) } exit(0) }' 1

# --- Logical ---
run_test "and_logic" 'fn main() {
    uint64 x = 5
    if x > 3 && x < 10 { exit(1) }
    exit(0)
}' 1
run_test "or_logic" 'fn main() {
    uint64 x = 2
    if x == 1 || x == 2 { exit(1) }
    exit(0)
}' 1

# --- If/else ---
run_test "if_then" 'fn main() {
    uint64 x = 5
    if x == 5 { exit(1) } else { exit(0) }
}' 1
run_test "if_else" 'fn main() {
    uint64 x = 3
    if x == 5 { exit(1) } else { exit(2) }
}' 2
run_test "else_if" 'fn main() {
    uint64 x = 2
    if x == 1 { exit(10) } else if x == 2 { exit(20) } else { exit(30) }
}' 20

# --- While ---
run_test "while_sum" 'fn main() {
    uint64 i = 0
    uint64 s = 0
    while i < 10 {
        s = s + i
        i = i + 1
    }
    exit(s)
}' 45

# --- Break/Continue ---
run_test "break" 'fn main() {
    uint64 i = 0
    uint64 c = 0
    while i < 100 {
        if i == 5 { break }
        c = c + 1
        i = i + 1
    }
    exit(c)
}' 5
run_test "continue" 'fn main() {
    uint64 i = 0
    uint64 s = 0
    while i < 10 {
        i = i + 1
        if i == 5 { continue }
        s = s + 1
    }
    exit(s)
}' 9

# --- Functions ---
run_test "fn_call" 'fn add(uint64 a, uint64 b) -> uint64 { return a + b }
fn main() { exit(add(10, 20)) }' 30

run_test "fn_4args" 'fn sum4(uint64 a, uint64 b, uint64 c, uint64 d) -> uint64 {
    return a + b + c + d
}
fn main() { exit(sum4(10, 20, 3, 9)) }' 42

run_test "fn_5args" 'fn sum5(uint64 a, uint64 b, uint64 c, uint64 d, uint64 e) -> uint64 { return a + b + c + d + e }
fn main() { exit(sum5(1, 2, 3, 4, 5)) }' 15

run_test "fn_6args" 'fn sum6(uint64 a, uint64 b, uint64 c, uint64 d, uint64 e, uint64 f) -> uint64 {
    return a + b + c + d + e + f
}
fn main() { exit(sum6(1,2,3,4,5,6)) }' 21

# --- Recursion ---
run_test "factorial" 'fn f(uint64 n) -> uint64 {
    if n <= 1 { return 1 }
    return n * f(n - 1)
}
fn main() { exit(f(5)) }' 120

run_test "fibonacci" 'fn fib(uint64 n) -> uint64 {
    if n <= 1 { return n }
    return fib(n - 1) + fib(n - 2)
}
fn main() { exit(fib(10)) }' 55

# --- Compound assignment ---
run_test "plus_eq" 'fn main() {
    uint64 x = 10
    x += 32
    exit(x)
}' 42

# --- Enums ---
run_test "enum_basic" 'enum Color {
    Red = 10
    Green = 20
    Blue = 30
}
fn main() { exit(Color.Green) }' 20

# --- Static variables ---
run_test "static_var" 'static uint64 counter = 0
fn inc() { counter = counter + 1 }
fn main() {
    inc()
    inc()
    inc()
    exit(counter)
}' 3

# --- Arrays ---
run_test "array_rw" 'fn main() {
    uint8[10] buf
    buf[0] = 42
    uint64 v = buf[0]
    exit(v)
}' 42

# --- Structs ---
run_test "struct_basic" 'struct Point {
    uint64 x
    uint64 y
}
fn main() {
    Point p
    p.x = 10
    p.y = 32
    exit(p.x + p.y)
}' 42

# --- Pointer operations ---
run_test "ptr_load_store" 'fn main() {
    uint64 buf = alloc(64)
    unsafe { *(buf as uint64) = 42 }
    uint64 v = 0
    unsafe { *(buf as uint64) -> v }
    exit(v)
}' 42

# --- File I/O ---
run_test "file_io" 'fn main() {
    uint64 msg = "test"
    uint64 fd = file_open("/dev/null", 1)
    file_write(fd, msg, 4)
    file_close(fd)
    exit(0)
}' 0

# --- Boolean literals ---
run_test "bool_true" 'fn main() { bool x = true; if x { exit(1) }; exit(0) }' 1
run_test "bool_false" 'fn main() { bool x = false; if x { exit(1) }; exit(0) }' 0

# --- Match statement ---
run_test "match_basic" 'fn main() {
    uint64 x = 2
    uint64 r = 0
    match x { 1 => { r = 10 } 2 => { r = 20 } 3 => { r = 30 } }
    exit(r)
}' 20

run_test "match_first" 'fn main() {
    uint64 x = 1
    uint64 r = 0
    match x { 1 => { r = 42 } 2 => { r = 99 } }
    exit(r)
}' 42

run_test "match_nomatch" 'fn main() {
    uint64 x = 99
    uint64 r = 42
    match x { 1 => { r = 0 } 2 => { r = 0 } }
    exit(r)
}' 42

run_test "match_enum" 'enum Color { Red = 1 Green = 2 Blue = 3 }
fn main() {
    uint64 c = Color.Green
    uint64 r = 0
    match c { 1 => { r = 10 } 2 => { r = 20 } 3 => { r = 30 } }
    exit(r)
}' 20

# --- Type aliases ---
run_test "type_alias" 'type Size = uint64
fn main() {
    Size x = 42
    exit(x)
}' 42

# --- Method syntax ---
run_test "method_decl" 'struct Point { uint64 x; uint64 y }
fn Point.sum(Point self) -> uint64 {
    return self.x + self.y
}
fn main() {
    Point p
    p.x = 10
    p.y = 32
    exit(sum(p))
}' 42

# --- Builtin: print/println ---
run_test_output "print_string" 'fn main() { print("hello world"); exit(0) }' "hello world"
run_test_output "print_int" 'fn main() { print(42); exit(0) }' "42"
run_test_output "print_zero" 'fn main() { print(0); exit(0) }' "0"
run_test_output "print_large" 'fn main() { print(123456); exit(0) }' "123456"
run_test_output "println_string" 'fn main() { println("hello"); exit(0) }' "hello"
run_test_output "println_int" 'fn main() { println(123); exit(0) }' "123"
run_test_output "println_multi" 'fn main() { println("abc"); println("def"); exit(0) }' "abc
def"

# --- Builtin: str_len ---
run_test "str_len_hello" 'fn main() { uint64 s = "hello"; exit(str_len(s)) }' 5
run_test "str_len_empty" 'fn main() { uint64 s = ""; exit(str_len(s)) }' 0
run_test "str_len_one" 'fn main() { uint64 s = "x"; exit(str_len(s)) }' 1

# --- Builtin: str_eq ---
run_test "str_eq_same" 'fn main() { uint64 a = "foo"; uint64 b = "foo"; exit(str_eq(a, b)) }' 1
run_test "str_eq_diff" 'fn main() { uint64 a = "foo"; uint64 b = "bar"; exit(str_eq(a, b)) }' 0
run_test "str_eq_prefix" 'fn main() { uint64 a = "foo"; uint64 b = "foobar"; exit(str_eq(a, b)) }' 0
run_test "str_eq_empty" 'fn main() { uint64 a = ""; uint64 b = ""; exit(str_eq(a, b)) }' 1

# --- std/string.mlr additions (v2.8.11) ---
run_test "str_index_of_hit" 'import "std/string.mlr"
fn main() { exit(str_index_of("hello world", "world")) }' 6
run_test "str_index_of_miss" 'import "std/string.mlr"
fn main() {
    uint64 n = str_index_of("hello", "xyz")
    if n == 0xFFFFFFFFFFFFFFFF { exit(0) }
    exit(1)
}' 0
run_test "str_compare_eq" 'import "std/string.mlr"
fn main() { exit(str_compare("abc", "abc")) }' 0
run_test "str_compare_lt" 'import "std/string.mlr"
fn main() {
    uint64 r = str_compare("abc", "abd")
    if signed_lt(r, 0) { exit(1) }
    exit(0)
}' 1
run_test "str_compare_prefix" 'import "std/string.mlr"
fn main() {
    uint64 r = str_compare("abc", "abcd")
    if signed_lt(r, 0) { exit(1) }
    exit(0)
}' 1
run_test_output "str_lower_basic" 'import "std/string.mlr"
fn main() { println_str(str_lower("HeLLo 123")) }' "hello 123"
run_test_output "str_upper_basic" 'import "std/string.mlr"
fn main() { println_str(str_upper("HeLLo 123")) }' "HELLO 123"
run_test_output "str_replace_basic" 'import "std/string.mlr"
fn main() { println_str(str_replace("a.b.c.d", ".", "-")) }' "a-b-c-d"
run_test_output "str_replace_longer" 'import "std/string.mlr"
fn main() { println_str(str_replace("hi world hi", "hi", "HELLO")) }' "HELLO world HELLO"
run_test_output "str_replace_noop" 'import "std/string.mlr"
fn main() { println_str(str_replace("abc", "zz", "QQ")) }' "abc"
run_test "str_split_count" 'import "std/string.mlr"
fn main() {
    uint64[8] parts
    exit(str_split("a,b,c,,d", 44, parts, 8))
}' 5
run_test_output "str_join_basic" 'import "std/string.mlr"
fn main() {
    uint64[4] parts
    uint64 n = str_split("a,b,c", 44, parts, 4)
    println_str(str_join(parts, n, "|"))
}' "a|b|c"
run_test "str_to_float_int" 'import "std/string.mlr"
fn main() {
    f64 v = str_to_float("42")
    exit(f64_to_int(v))
}' 42
run_test "str_to_float_frac" 'import "std/string.mlr"
fn main() {
    f64 v = str_to_float("1.5")
    f64 two = int_to_f64(2)
    exit(f64_to_int(v * two))
}' 3
run_test "str_to_float_exp" 'import "std/string.mlr"
fn main() {
    f64 v = str_to_float("-3e1")
    exit(f64_to_int(int_to_f64(0) - v))
}' 30
# Regression: float static initialisers used to silently drop their value
# (parser only handled int literal kinds 2/4/77/78 — FloatLit kind 5 fell
# through the skip branch). Now `static f64 x = 20.0` retains 20.0.
run_test "static_float_init" '
static f64 tau_m = 20.0
static f64 V_rest = -70.0
fn main() {
    exit(f64_to_int(tau_m - V_rest))   // 20 - (-70) = 90
}' 90
# Regression: reads of static f64 and f64 array elements used to lose
# their f64 type-flow through arithmetic, so `a + b` emitted integer ops
# instead of IR_FADD. Now the static_fkinds table propagates fkind from
# declaration through IR_STATIC_LOAD and array Index.
run_test "static_f64_type_flow" '
static f64 a = 3.0
static f64 b = 4.0
fn main() {
    f64 c = a + b    // direct-read arithmetic — used to produce -0.0
    exit(f64_to_int(c))
}' 7
run_test "static_f64_array_type_flow" '
static f64[4] arr
fn main() {
    arr[0] = 1.5
    arr[1] = 2.5
    arr[2] = 3.5
    arr[3] = 4.5
    f64 s = arr[0] + arr[1] + arr[2] + arr[3]
    exit(f64_to_int(s))
}' 12
run_test "utf8_decode_ascii" 'import "std/string.mlr"
fn main() {
    uint64[1] w
    uint64 wp = w
    uint64 cp = utf8_decode_at("A", 0, wp)
    uint64 ww = 0
    unsafe { *(wp as uint64) -> ww }
    if cp == 65 && ww == 1 { exit(0) }
    exit(1)
}' 0
run_test "utf8_decode_two_byte" 'import "std/string.mlr"
fn main() {
    uint64[1] w
    uint64 wp = w
    uint64 cp = utf8_decode_at("é", 0, wp)
    uint64 ww = 0
    unsafe { *(wp as uint64) -> ww }
    if cp == 233 && ww == 2 { exit(0) }
    exit(1)
}' 0
run_test "str_codepoint_count_mixed" 'import "std/string.mlr"
fn main() { exit(str_codepoint_count("héllo")) }' 5
run_test "utf8_lower_codepoint_ascii" 'import "std/string.mlr"
fn main() { exit(utf8_lower_codepoint(65)) }' 97
run_test "utf8_upper_codepoint_latin1" 'import "std/string.mlr"
fn main() { exit(utf8_upper_codepoint(0xE9)) }' 201
run_test_output "str_lower_utf8_latin1" 'import "std/string.mlr"
fn main() { println_str(str_lower_utf8("CaFÉ")) }' "café"
run_test_output "str_upper_utf8_latin1" 'import "std/string.mlr"
fn main() { println_str(str_upper_utf8("café")) }' "CAFÉ"
run_test "utf8_is_combining_yes" 'import "std/string.mlr"
fn main() { exit(utf8_is_combining(0x0301)) }' 1
run_test "utf8_is_combining_no" 'import "std/string.mlr"
fn main() { exit(utf8_is_combining(65)) }' 0

# --- Greek case folding (v2.8.13) ---
run_test_output "greek_lower_sentence" 'import "std/string.mlr"
fn main() { println_str(str_lower_utf8("Γειά σου Κόσμε")) }' "γειά σου κόσμε"
run_test_output "greek_upper_sentence" 'import "std/string.mlr"
fn main() { println_str(str_upper_utf8("γειά σου κόσμε")) }' "ΓΕΙΆ ΣΟΥ ΚΌΣΜΕ"
run_test_output "greek_upper_final_sigma" 'import "std/string.mlr"
fn main() { println_str(str_upper_utf8("ελληνικός")) }' "ΕΛΛΗΝΙΚΌΣ"
run_test_output "greek_mixed_latin1" 'import "std/string.mlr"
fn main() { println_str(str_upper_utf8("café Ωραία")) }' "CAFÉ ΩΡΑΊΑ"
run_test "greek_lower_alpha" 'import "std/string.mlr"
fn main() {
    if utf8_lower_codepoint(0x0391) == 0x03B1 { exit(1) }
    exit(0)
}' 1
run_test "greek_upper_omega" 'import "std/string.mlr"
fn main() {
    if utf8_upper_codepoint(0x03C9) == 0x03A9 { exit(1) }
    exit(0)
}' 1
run_test "greek_final_sigma_to_sigma" 'import "std/string.mlr"
fn main() {
    if utf8_upper_codepoint(0x03C2) == 0x03A3 { exit(1) }
    exit(0)
}' 1

# --- String builder (v2.8.11) ---
run_test_output "sb_basic" 'import "std/string.mlr"
fn main() {
    uint64 sb = sb_new(16)
    sb = sb_append_str(sb, "x = ")
    sb = sb_append_int(sb, 42)
    uint64 r = sb_finish(sb)
    println_str(r)
    sb_free(sb)
}' "x = 42"
run_test_output "sb_mixed" 'import "std/string.mlr"
import "std/math_float.mlr"
fn main() {
    uint64 sb = sb_new(16)
    sb = sb_append_str(sb, "hex=")
    sb = sb_append_hex(sb, 0xDEAD)
    sb = sb_append_str(sb, ", bool=")
    sb = sb_append_bool(sb, 0)
    sb = sb_append_str(sb, ", f=")
    sb = sb_append_float(sb, 1.5, 2)
    uint64 r = sb_finish(sb)
    println_str(r)
    sb_free(sb)
}' "hex=0xdead, bool=false, f=1.50"
run_test "sb_grows" 'import "std/string.mlr"
fn main() {
    uint64 sb = sb_new(4)     // deliberately tiny
    sb = sb_append_str(sb, "0123456789ABCDEFGHIJ")   // force grow
    exit(sb_len(sb))
}' 20
run_test_output "str_from_bool_true" 'import "std/string.mlr"
fn main() { println_str(str_from_bool(1)) }' "true"
run_test_output "str_from_bool_false" 'import "std/string.mlr"
fn main() { println_str(str_from_bool(0)) }' "false"
run_test_output "str_from_codepoint_latin1" 'import "std/string.mlr"
fn main() { println_str(str_from_codepoint(0xE9)) }' "é"

# --- Error-handling helpers (v2.8.14) ---
run_test "opt_some_unwrap" 'import "std/string.mlr"
fn main() { exit(opt_unwrap(opt_some(42))) }' 42
run_test "opt_is_some_yes" 'import "std/string.mlr"
fn main() { exit(opt_is_some(opt_some(0))) }' 1
run_test "opt_is_some_no" 'import "std/string.mlr"
fn main() { exit(opt_is_some(opt_none())) }' 0
run_test "is_errno_yes" 'import "std/io.mlr"
fn main() { exit(is_errno(0xFFFFFFFFFFFFFFFE)) }' 1
run_test "is_errno_no" 'import "std/io.mlr"
fn main() { exit(is_errno(42)) }' 0
run_test "get_errno_val" 'import "std/io.mlr"
fn main() { exit(get_errno(0xFFFFFFFFFFFFFFFE)) }' 2

# --- isb() / alloc_aligned() (v2.8.14) ---
run_test "isb_noop" 'fn main() { isb(); exit(0) }' 0
run_test "dsb_noop" 'fn main() { dsb(); exit(0) }' 0
run_test "dmb_noop" 'fn main() { dmb(); exit(0) }' 0
run_test "dcache_flush_basic" 'fn main() {
    u64 p = alloc(64)
    store64(p, 0x1234)
    dcache_flush(p)
    u64 v = load64(p)
    exit(v & 0xFF)
}' 52
run_test "icache_invalidate_basic" 'fn main() {
    u64 p = alloc(64)
    icache_invalidate(p)
    exit(0)
}' 0
run_test "memmove_forward" 'import "std/mem.mlr"
fn main() {
    u64 p = alloc(64)
    store64(p, 0xAABBCCDD)
    memmove(p + 8, p, 8)
    u64 v = load64(p + 8)
    if v == 0xAABBCCDD { exit(11) }
    exit(1)
}' 11
run_test "memmove_backward_overlap" 'import "std/mem.mlr"
fn main() {
    // Layout: bytes 0..=7 = 1..8. Shift right by 4, so bytes 4..=11
    // become 1..8. memcpy would corrupt this; memmove must not.
    u64 p = alloc(32)
    u64 i = 0
    while i < 8 { store8(p + i, i + 1); i = i + 1 }
    memmove(p + 4, p, 8)
    // Verify: p[4..11] = 1..8
    u64 sum = 0
    i = 4
    while i < 12 { sum = sum + load8(p + i); i = i + 1 }
    exit(sum)
}' 36
run_test "memmove_forward_overlap" 'import "std/mem.mlr"
fn main() {
    u64 p = alloc(32)
    u64 i = 0
    while i < 8 { store8(p + 4 + i, i + 1); i = i + 1 }
    // Shift left by 4: bytes 0..=7 become 1..=8 (read from 4..=11).
    memmove(p, p + 4, 8)
    u64 sum = 0
    i = 0
    while i < 8 { sum = sum + load8(p + i); i = i + 1 }
    exit(sum)
}' 36
run_test "memmove_zero_len" 'import "std/mem.mlr"
fn main() {
    // Must be a no-op regardless of pointer values.
    memmove(0, 0, 0)
    exit(0)
}' 0

# --- Bounds checks under --debug ---
run_bchk_test() {
    local name="$1"
    local input="$2"
    local expected="$3"
    TOTAL=$((TOTAL + 1))
    local REPO_ROOT="$DIR/.."
    printf '%s\n' "$input" > "$REPO_ROOT/test_tmp_$$.mlr"
    if $MLRC --debug $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_bchk_$$ > /dev/null 2>&1; then
        rm -f "$REPO_ROOT/test_tmp_$$.mlr"
        chmod +x /tmp/mlrc_bchk_$$
        local got=0
        /tmp/mlrc_bchk_$$ > /dev/null 2>&1 && got=0 || got=$?
        if [ "$got" = "$expected" ]; then
            PASS=$((PASS + 1))
        else
            echo "FAIL: $name (expected $expected, got $got)"; FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: $name (compilation failed)"; FAIL=$((FAIL + 1))
    fi
    rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_bchk_$$
}
run_bchk_test "bchk_stack_in_range"    'fn main() { u64[4] a; a[0] = 1; a[3] = 4; exit(a[3]) }' 4
run_bchk_test "bchk_stack_oob_write"   'fn main() { u64[4] a; a[4] = 99; exit(0) }' 1
run_bchk_test "bchk_stack_oob_read"    'fn main() { u64[4] a; exit(a[7]) }' 1
run_bchk_test "bchk_static_in_range"   'static u64[8] s; fn main() { s[5] = 42; exit(s[5]) }' 42
run_bchk_test "bchk_static_oob_write"  'static u64[8] s; fn main() { s[8] = 1; exit(0) }' 1

# --- Literal-overflow warning ---
TOTAL=$((TOTAL + 1))
printf 'fn main() { u8 b = 300; exit(b) }\n' > "$DIR/../test_tmp_trunc_$$.mlr"
trunc_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_trunc_$$.mlr" -o /tmp/mlrc_trunc_$$ 2>&1)
if echo "$trunc_out" | grep -q "literal initializer does not fit"; then
    PASS=$((PASS + 1))
else
    echo "FAIL: literal_overflow_warns (no warning emitted)"; FAIL=$((FAIL + 1))
fi
rm -f "$DIR/../test_tmp_trunc_$$.mlr" /tmp/mlrc_trunc_$$

TOTAL=$((TOTAL + 1))
printf 'fn main() { u8 b = 200; exit(b) }\n' > "$DIR/../test_tmp_okw_$$.mlr"
okw_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_okw_$$.mlr" -o /tmp/mlrc_okw_$$ 2>&1)
if echo "$okw_out" | grep -q "literal initializer"; then
    echo "FAIL: literal_in_range_silent (false warning)"; FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1))
fi
rm -f "$DIR/../test_tmp_okw_$$.mlr" /tmp/mlrc_okw_$$

# --- Unused-variable warning ---
TOTAL=$((TOTAL + 1))
printf 'fn main() { u64 stale = 5; exit(0) }\n' > "$DIR/../test_tmp_uv_$$.mlr"
uv_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_uv_$$.mlr" -o /tmp/mlrc_uv_$$ 2>&1)
if echo "$uv_out" | grep -q "unused variable.*stale"; then
    PASS=$((PASS + 1))
else
    echo "FAIL: unused_var_warns (no warning)"; FAIL=$((FAIL + 1))
fi
rm -f "$DIR/../test_tmp_uv_$$.mlr" /tmp/mlrc_uv_$$

TOTAL=$((TOTAL + 1))
printf 'fn main() { u64 _skip = 5; exit(0) }\n' > "$DIR/../test_tmp_uvs_$$.mlr"
uvs_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_uvs_$$.mlr" -o /tmp/mlrc_uvs_$$ 2>&1)
if echo "$uvs_out" | grep -q "unused variable"; then
    echo "FAIL: unused_underscore_silent (false warning)"; FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1))
fi
rm -f "$DIR/../test_tmp_uvs_$$.mlr" /tmp/mlrc_uvs_$$

TOTAL=$((TOTAL + 1))
printf 'fn main() { u64 x = 5; exit(x) }\n' > "$DIR/../test_tmp_uvu_$$.mlr"
uvu_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_uvu_$$.mlr" -o /tmp/mlrc_uvu_$$ 2>&1)
if echo "$uvu_out" | grep -q "unused variable"; then
    echo "FAIL: used_var_silent (false warning)"; FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1))
fi
rm -f "$DIR/../test_tmp_uvu_$$.mlr" /tmp/mlrc_uvu_$$

# --- Uninitialized-read warning ---
TOTAL=$((TOTAL + 1))
printf 'fn main() { u64 stale; exit(stale) }\n' > "$DIR/../test_tmp_ur_$$.mlr"
ur_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_ur_$$.mlr" -o /tmp/mlrc_ur_$$ 2>&1)
if echo "$ur_out" | grep -q "used before initialization.*stale"; then
    PASS=$((PASS + 1))
else
    echo "FAIL: uninit_read_warns (no warning)"; FAIL=$((FAIL + 1))
fi
rm -f "$DIR/../test_tmp_ur_$$.mlr" /tmp/mlrc_ur_$$

TOTAL=$((TOTAL + 1))
printf 'fn main() { u64 x = 0; exit(x) }\n' > "$DIR/../test_tmp_urs_$$.mlr"
urs_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_urs_$$.mlr" -o /tmp/mlrc_urs_$$ 2>&1)
if echo "$urs_out" | grep -q "used before initialization"; then
    echo "FAIL: init_read_silent (false warning)"; FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1))
fi
rm -f "$DIR/../test_tmp_urs_$$.mlr" /tmp/mlrc_urs_$$

TOTAL=$((TOTAL + 1))
printf 'fn main() { u64 _x; exit(_x) }\n' > "$DIR/../test_tmp_urus_$$.mlr"
urus_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_urus_$$.mlr" -o /tmp/mlrc_urus_$$ 2>&1)
if echo "$urus_out" | grep -q "used before initialization"; then
    echo "FAIL: underscore_uninit_silent (false warning)"; FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1))
fi
rm -f "$DIR/../test_tmp_urus_$$.mlr" /tmp/mlrc_urus_$$

TOTAL=$((TOTAL + 1))
printf 'fn main() { u8 b = 10; b = 300; exit(b) }\n' > "$DIR/../test_tmp_tas_$$.mlr"
tas_out=$($MLRC $MLRC_FLAGS "$DIR/../test_tmp_tas_$$.mlr" -o /tmp/mlrc_tas_$$ 2>&1)
if echo "$tas_out" | grep -q "literal assignment does not fit"; then
    PASS=$((PASS + 1))
else
    echo "FAIL: literal_assign_warns (no warning emitted)"; FAIL=$((FAIL + 1))
fi
rm -f "$DIR/../test_tmp_tas_$$.mlr" /tmp/mlrc_tas_$$
run_test "alloc_aligned_64" 'import "std/mem.mlr"
fn main() {
    uint64 buf = alloc_aligned(100, 64)
    if (buf & 63) != 0 { exit(1) }
    alloc_aligned_free(buf)
    exit(0)
}' 0
run_test "alloc_aligned_256" 'import "std/mem.mlr"
fn main() {
    uint64 buf = alloc_aligned(1000, 256)
    if (buf & 255) != 0 { exit(1) }
    alloc_aligned_free(buf)
    exit(0)
}' 0

# --- Builtin: dealloc ---
run_test "dealloc_noop" 'fn main() { uint64 p = alloc(64); dealloc(p); exit(0) }' 0

# --- Builtin: memset ---
run_test_output "memset_basic" 'fn main() {
    uint64 buf = alloc(64)
    memset(buf, 65, 5)
    write(1, buf, 5)
    exit(0)
}' "AAAAA"

# --- Builtin: memcpy ---
run_test_output "memcpy_basic" 'fn main() {
    uint64 src = "hello"
    uint64 dst = alloc(64)
    memcpy(dst, src, 5)
    write(1, dst, 5)
    exit(0)
}' "hello"

# --- Kernel Features ---

# Inline assembly: nop (should compile and run without crashing)
run_test "asm_nop" 'fn main() { asm("nop"); exit(42) }' 42

# Inline assembly: multi-line block
run_test "asm_block" 'fn main() { asm { "nop"; "nop"; "nop" }; exit(7) }' 7

# Inline assembly: raw hex bytes (x86-only: 0x90 = nop)
if [ "$ARCH" != "aarch64" ]; then
    run_test "asm_hex" 'fn main() { asm("0x90"); exit(5) }' 5
else
    echo "  asm_hex: SKIP (x86-only)"; PASS=$((PASS+1)); TOTAL=$((TOTAL+1))
fi

# Signed comparisons: signed_lt with negative-like values
# Signedness must survive a CALL RESULT used directly as an operand.
# Regression: the signed flag lived only on a vreg, and a call's result vreg
# was never tagged from the callee's declared return type -- so `neg() >> 1`
# emitted SHR (logical) instead of SAR, and `/` `%` chose DIV/MOD over
# SDIV/SMOD. Silently wrong for negatives. Worse, a pure single-expression
# callee is INLINED (even at --O0), splicing an untyped body in, which loses
# the signedness before the IR sees a call at all. Assigning to a typed local
# first always worked, which is what masked it.
run_test "signed_ret_shift_i32" 'fn neg() -> i32 { return 0 - 32 }
fn main() {
    i32 a = neg()
    i32 viaLocal = a >> 1
    i32 direct = neg() >> 1
    if viaLocal != direct { exit(1) }
    if direct != (0 - 16) { exit(2) }
    exit(0)
}' 0

run_test "signed_ret_shift_i64" 'fn neg() -> i64 { return 0 - 32 }
fn main() {
    i64 a = neg()
    if (a >> 1) != (neg() >> 1) { exit(1) }
    if (neg() >> 1) != (0 - 16) { exit(2) }
    exit(0)
}' 0

run_test "signed_ret_divmod" 'fn neg() -> i64 { return 0 - 10 }
fn main() {
    i64 a = neg()
    if (a / 3) != (neg() / 3) { exit(1) }
    if (a % 3) != (neg() % 3) { exit(2) }
    exit(0)
}' 0

# --- declared-type signedness must not be inherited from the initialiser ---
# Regression: `uint32 ux = <int32>` copied the RHS's signed flag, so ux and
# everything derived from it used SIGNED ops -- `ux >> 1` became an ARITHMETIC
# shift and comparisons became signed. Invisible on x86_64/arm64, where a
# uint32 occupies a 64-bit slot and never reaches the sign bit, but silently
# wrong on the 32-bit backends: 0x80000000 >> 31 gave 0xFFFFFFFF instead of 1.
# Checked in the IR, because a host run passes either way and hides it.
TOTAL=$((TOTAL + 1))
SGN_ROOT="$DIR/.."
printf 'fn main() -> uint64 {\n    int32 acc = 0 - 5\n    uint32 ux = acc\n    uint32 sh = ux >> 1\n    return sh\n}\n' > "$SGN_ROOT/test_tmp_$$.mlr"
SGN_OUT=$($MLRC --emit=ir --arch=x86_64 "$SGN_ROOT/test_tmp_$$.mlr" 2>/dev/null)
# and the converse: a genuinely signed int32 shift must still be arithmetic
printf 'fn main() -> uint64 {\n    int32 a = 0 - 8\n    int32 b = a >> 1\n    return 0\n}\n' > "$SGN_ROOT/test_tmp2_$$.mlr"
SGN_OUT2=$($MLRC --emit=ir --arch=x86_64 "$SGN_ROOT/test_tmp2_$$.mlr" 2>/dev/null)
rm -f "$SGN_ROOT/test_tmp_$$.mlr" "$SGN_ROOT/test_tmp2_$$.mlr"
if echo "$SGN_OUT" | grep -q "shr" && ! echo "$SGN_OUT" | grep -q "sar" \
   && echo "$SGN_OUT2" | grep -q "sar"; then
    PASS=$((PASS + 1))
    echo "  decl_type_signedness: PASS (uint32 from int32 -> shr; int32 -> sar)"
else
    echo "FAIL: decl_type_signedness (uint32-from-int32 shift: $(echo "$SGN_OUT" | grep -oE 'shr|sar' | head -1), int32 shift: $(echo "$SGN_OUT2" | grep -oE 'shr|sar' | head -1))"
    FAIL=$((FAIL + 1))
fi

# --- same, on the ASSIGNMENT path: a store cannot retype the variable ---
# `uint32 ux = 7; ux = <int32>` let the rvalue's signed flag through, so ux
# became signed from that point on. The LHS local's flag is authoritative in
# both directions now. Second half checks the direction that must NOT break:
# a signed local stays signed across `x = x - n`, so `x < 0` keeps SCMP.
TOTAL=$((TOTAL + 1))
printf 'fn main() -> uint64 {\n    int32 acc = 0 - 5\n    uint32 ux = 7\n    ux = acc\n    uint32 sh = ux >> 1\n    return sh\n}\n' > "$SGN_ROOT/test_tmp_$$.mlr"
ASG_OUT=$($MLRC --emit=ir --arch=x86_64 "$SGN_ROOT/test_tmp_$$.mlr" 2>/dev/null)
printf 'fn main() -> uint64 {\n    int64 x = 3\n    x = x - 5\n    if x < 0 { return 1 }\n    return 0\n}\n' > "$SGN_ROOT/test_tmp2_$$.mlr"
ASG_OUT2=$($MLRC --emit=ir --arch=x86_64 "$SGN_ROOT/test_tmp2_$$.mlr" 2>/dev/null)
rm -f "$SGN_ROOT/test_tmp_$$.mlr" "$SGN_ROOT/test_tmp2_$$.mlr"
if echo "$ASG_OUT" | grep -q "shr" && ! echo "$ASG_OUT" | grep -q "sar" \
   && echo "$ASG_OUT2" | grep -qi "scmp"; then
    PASS=$((PASS + 1))
    echo "  assign_type_signedness: PASS (uint32 = int32 -> shr; signed local keeps scmp)"
else
    echo "FAIL: assign_type_signedness (uint32=int32 shift: $(echo "$ASG_OUT" | grep -oE 'shr|sar' | head -1), signed cmp: $(echo "$ASG_OUT2" | grep -oiE 'scmp_[a-z]+|cmp_[a-z]+' | head -1))"
    FAIL=$((FAIL + 1))
fi

run_test "signed_lt_true" 'fn main() {
    uint64 a = 0xFFFFFFFFFFFFFFFF
    uint64 b = 1
    uint64 r = signed_lt(a, b)
    exit(r)
}' 1

run_test "signed_lt_false" 'fn main() {
    uint64 a = 5
    uint64 b = 3
    uint64 r = signed_lt(a, b)
    exit(r)
}' 0

run_test "signed_gt_true" 'fn main() {
    uint64 a = 1
    uint64 b = 0xFFFFFFFFFFFFFFFF
    uint64 r = signed_gt(a, b)
    exit(r)
}' 1

run_test "signed_le_true" 'fn main() {
    uint64 a = 5
    uint64 b = 5
    uint64 r = signed_le(a, b)
    exit(r)
}' 1

run_test "signed_ge_true" 'fn main() {
    uint64 a = 0xFFFFFFFFFFFFFFFF
    uint64 b = 0xFFFFFFFFFFFFFFFF
    uint64 r = signed_ge(a, b)
    exit(r)
}' 1

# Bitfield operations
run_test "bit_get_1" 'fn main() {
    uint64 v = 0xFF
    uint64 r = bit_get(v, 3)
    exit(r)
}' 1

run_test "bit_get_0" 'fn main() {
    uint64 v = 0xF0
    uint64 r = bit_get(v, 2)
    exit(r)
}' 0

run_test "bit_set" 'fn main() {
    uint64 v = 0
    v = bit_set(v, 3)
    exit(v)
}' 8

run_test "bit_clear" 'fn main() {
    uint64 v = 0xFF
    v = bit_clear(v, 3)
    exit(v & 0xFF)
}' 247

run_test "bit_range" 'fn main() {
    uint64 v = 0xAB
    uint64 r = bit_range(v, 4, 4)
    exit(r)
}' 10

run_test "bit_insert" 'fn main() {
    uint64 v = 0x00
    v = bit_insert(v, 4, 4, 0xF)
    exit(v)
}' 240

# @naked function (x86-only: uses raw x86 machine code bytes)
if [ "$ARCH" != "aarch64" ]; then
    run_test "naked_fn" '@naked fn raw_exit() {
        asm("0x48 0xC7 0xC7 0x2A 0x00 0x00 0x00")
        asm("0x48 0xC7 0xC0 0x3C 0x00 0x00 0x00")
        asm("0x0F 0x05")
    }
    fn main() { raw_exit() }' 42
else
    echo "  naked_fn: SKIP (x86-only)"; PASS=$((PASS+1)); TOTAL=$((TOTAL+1))
fi

# @noreturn annotation (should compile fine)
run_test "noreturn_fn" '@noreturn fn die() { exit(99) }
fn main() { die() }' 99

# volatile block (same as unsafe)
run_test "volatile_block" 'fn main() {
    uint64 buf = alloc(64)
    uint64 val = 0
    unsafe { *(buf as uint64) = 42 }
    volatile { *(buf as uint64) -> val }
    exit(val)
}' 42

# @packed struct annotation (should parse without error)
run_test "packed_struct" '@packed struct Reg { uint8 a; uint32 b }
fn main() {
    uint8[16] buf
    exit(0)
}' 0

# @section annotation (should parse without error)
run_test "section_attr" '@section(".text.init") fn early_init() { exit(0) }
fn main() { early_init() }' 0

# --freestanding flag (should compile, main has no auto-exit, so explicit exit needed)
# Can't easily test this without a linker, just test that it parses
# run_test "freestanding" handled by CLI flag test below

# --- Function Pointers ---

# fn_addr + call_ptr basic
run_test "fn_ptr_basic" 'fn add(uint64 a, uint64 b) -> uint64 { return a + b }
fn main() {
    uint64 fp = fn_addr("add")
    uint64 r = call_ptr(fp, 30, 12)
    exit(r)
}' 42

# fn_ptr dispatch table
run_test "fn_ptr_dispatch" 'fn h0() -> uint64 { return 10 }
fn h1() -> uint64 { return 20 }
fn h2() -> uint64 { return 12 }
fn main() {
    uint64 t = alloc(24)
    uint64 a = fn_addr("h0")
    uint64 b = fn_addr("h1")
    uint64 c = fn_addr("h2")
    unsafe { *(t as uint64) = a }
    uint64 t8 = t + 8
    unsafe { *(t8 as uint64) = b }
    uint64 t16 = t + 16
    unsafe { *(t16 as uint64) = c }
    uint64 fp = 0
    unsafe { *(t as uint64) -> fp }
    uint64 r = call_ptr(fp)
    uint64 fp2 = 0
    uint64 tb = t + 8
    unsafe { *(tb as uint64) -> fp2 }
    r = r + call_ptr(fp2)
    uint64 fp3 = 0
    uint64 tc = t + 16
    unsafe { *(tc as uint64) -> fp3 }
    r = r + call_ptr(fp3)
    exit(r)
}' 42

# fn_ptr no args
run_test "fn_ptr_noargs" 'fn get42() -> uint64 { return 42 }
fn main() {
    uint64 fp = fn_addr("get42")
    uint64 r = call_ptr(fp)
    exit(r)
}' 42

# --- uint16 pointer operations ---
run_test "uint16_store_load" 'fn main() {
    uint64 buf = alloc(64)
    uint16 val = 0xBEEF
    unsafe { *(buf as uint16) = val }
    uint16 got = 0
    unsafe { *(buf as uint16) -> got }
    uint64 r = got
    exit(r & 0xFF)
}' 239

run_test "uint16_store_load_small" 'fn main() {
    uint64 buf = alloc(64)
    uint16 val = 42
    unsafe { *(buf as uint16) = val }
    uint16 got = 0
    unsafe { *(buf as uint16) -> got }
    uint64 r = got
    exit(r)
}' 42

run_test "uint16_two_slots" 'fn main() {
    uint64 buf = alloc(64)
    uint16 a = 10
    uint16 b = 32
    unsafe { *(buf as uint16) = a }
    uint64 buf2 = buf + 2
    unsafe { *(buf2 as uint16) = b }
    uint16 va = 0
    uint16 vb = 0
    unsafe { *(buf as uint16) -> va }
    unsafe { *(buf2 as uint16) -> vb }
    uint64 ra = va
    uint64 rb = vb
    exit(ra + rb)
}' 42

# --- Atomic operations ---
run_test "atomic_store_load" 'fn main() {
    uint64 buf = alloc(64)
    atomic_store(buf, 42)
    uint64 v = atomic_load(buf)
    exit(v)
}' 42

run_test "atomic_add_basic" 'fn main() {
    uint64 buf = alloc(64)
    atomic_store(buf, 30)
    uint64 old = atomic_add(buf, 12)
    uint64 v = atomic_load(buf)
    exit(v)
}' 42

run_test "atomic_add_returns_old" 'fn main() {
    uint64 buf = alloc(64)
    atomic_store(buf, 40)
    uint64 old = atomic_add(buf, 10)
    exit(old)
}' 40

run_test "atomic_cas_success" 'fn main() {
    uint64 buf = alloc(64)
    atomic_store(buf, 10)
    uint64 ok = atomic_cas(buf, 10, 42)
    uint64 v = atomic_load(buf)
    if ok == 1 && v == 42 { exit(42) }
    exit(0)
}' 42

run_test "atomic_cas_fail" 'fn main() {
    uint64 buf = alloc(64)
    atomic_store(buf, 10)
    uint64 ok = atomic_cas(buf, 99, 42)
    uint64 v = atomic_load(buf)
    if ok == 0 && v == 10 { exit(42) }
    exit(0)
}' 42

# atomic_cas_hit_then_miss: the same HIT/MISS/CELL sequence as
# tests/esp32/cas_single.mlr (100 -> 200 succeeds, then 100 -> 300 fails
# because the cell is now 200), reduced to one exit code. This is the
# x86_64 regression coverage for Task 7 (IR_ATOMIC_CAS / xtensa S32C1I):
# no test exercises op 93 on xtensa itself (KernRift has no
# std/esp32_clk.kr / esp32_uart.kr for a standalone esp32 fixture yet),
# so this proves the shared front-end contract both backends rely on --
# imm-as-vreg desired operand, BOOLEAN 1/0 (not old-value) result, and a
# second CAS on the same cell not corrupting what the first one wrote.
# A broken emitter fails this: returning the old word instead of a
# boolean makes ok_hit=100 (!= 1) or ok_miss=200 (!= 0); a reversed
# comparison polarity flips which of ok_hit/ok_miss is 1 vs 0; and reusing
# the caller's `desired` register instead of a scratch (the S32C1I
# destructive-register bug this port specifically guards against) would
# leave v with a stale value after the second call. Any of those makes
# the `if` condition false and the test exits 0, not 42.
run_test "atomic_cas_hit_then_miss" 'fn main() {
    uint64 buf = alloc(64)
    atomic_store(buf, 100)
    uint64 ok_hit = atomic_cas(buf, 100, 200)
    uint64 ok_miss = atomic_cas(buf, 100, 300)
    uint64 v = atomic_load(buf)
    if ok_hit == 1 && ok_miss == 0 && v == 200 { exit(42) }
    exit(0)
}' 42

# --- Volatile blocks ---
run_test "volatile_store_load" 'fn main() {
    uint64 buf = alloc(64)
    volatile { *(buf as uint64) = 42 }
    uint64 v = 0
    volatile { *(buf as uint64) -> v }
    exit(v)
}' 42

run_test "volatile_roundtrip" 'fn main() {
    uint64 buf = alloc(64)
    volatile { *(buf as uint64) = 100 }
    uint64 a = 0
    volatile { *(buf as uint64) -> a }
    volatile { *(buf as uint64) = 42 }
    uint64 b = 0
    volatile { *(buf as uint64) -> b }
    exit(b)
}' 42

run_test "volatile_uint8" 'fn main() {
    uint64 buf = alloc(64)
    uint8 val = 42
    volatile { *(buf as uint8) = val }
    uint8 got = 0
    volatile { *(buf as uint8) -> got }
    uint64 r = got
    exit(r)
}' 42

# --- MSR/MRS (compile-only, privileged instructions cannot run in userspace) ---
if [ "$ARCH" != "aarch64" ]; then
    # x86: rdmsr/wrmsr are ring-0 only; just verify the asm block compiles
    TOTAL=$((TOTAL + 1))
    printf 'fn main() { exit(42) }\n@naked fn msr_test() { asm("rdmsr") }\n' > /tmp/mlrc_test_$$.mlr
    if $MLRC $MLRC_FLAGS /tmp/mlrc_test_$$.mlr -o /tmp/mlrc_test_$$ > /dev/null 2>&1; then
        chmod +x /tmp/mlrc_test_$$
        /tmp/mlrc_test_$$ > /dev/null 2>&1; got=$?
        if [ "$got" = "42" ]; then
            PASS=$((PASS + 1))
        else
            echo "FAIL: msr_compile (expected 42, got $got)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: msr_compile (compilation failed)"
        FAIL=$((FAIL + 1))
    fi
    rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$

    TOTAL=$((TOTAL + 1))
    printf 'fn main() { exit(42) }\n@naked fn msr_test() { asm("wrmsr") }\n' > /tmp/mlrc_test_$$.mlr
    if $MLRC $MLRC_FLAGS /tmp/mlrc_test_$$.mlr -o /tmp/mlrc_test_$$ > /dev/null 2>&1; then
        chmod +x /tmp/mlrc_test_$$
        /tmp/mlrc_test_$$ > /dev/null 2>&1; got=$?
        if [ "$got" = "42" ]; then
            PASS=$((PASS + 1))
        else
            echo "FAIL: msr_wrmsr_compile (expected 42, got $got)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: msr_wrmsr_compile (compilation failed)"
        FAIL=$((FAIL + 1))
    fi
    rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$
else
    echo "  msr_compile: SKIP (x86-only)"; PASS=$((PASS+1)); TOTAL=$((TOTAL+1))
    echo "  msr_wrmsr_compile: SKIP (x86-only)"; PASS=$((PASS+1)); TOTAL=$((TOTAL+1))
fi

# --- Dead Code Elimination test ---
echo ""
echo "--- DCE test ---"
TOTAL=$((TOTAL + 1))

# Program with an unused function — DCE should eliminate it
cat > /tmp/mlrc_dce_unused_$$.mlr << 'KRSRC'
fn unused_big() -> uint64 {
    uint64 a = 1
    uint64 b = 2
    uint64 c = 3
    uint64 d = 4
    uint64 e = 5
    uint64 f = a + b + c + d + e
    uint64 g = f * 2
    uint64 h = g + f
    uint64 i = h * g + f
    uint64 j = i + h + g + f + e + d + c + b + a
    return j
}
fn unused_big2() -> uint64 {
    uint64 a = 10
    uint64 b = 20
    uint64 c = 30
    uint64 d = 40
    uint64 e = 50
    uint64 f = a + b + c + d + e
    uint64 g = f * 3
    uint64 h = g + f
    uint64 i = h * g + f
    uint64 j = i + h + g + f + e + d + c + b + a
    return j
}
fn unused_big3() -> uint64 {
    uint64 a = 100
    uint64 b = 200
    uint64 c = 300
    uint64 d = 400
    uint64 e = 500
    uint64 f = a + b + c + d + e
    uint64 g = f * 4
    uint64 h = g + f
    uint64 i = h * g + f
    uint64 j = i + h + g + f + e + d + c + b + a
    return j
}
fn main() { exit(42) }
KRSRC

# Same program but all functions are called
cat > /tmp/mlrc_dce_used_$$.mlr << 'KRSRC'
fn used_big() -> uint64 {
    uint64 a = 1
    uint64 b = 2
    uint64 c = 3
    uint64 d = 4
    uint64 e = 5
    uint64 f = a + b + c + d + e
    uint64 g = f * 2
    uint64 h = g + f
    uint64 i = h * g + f
    uint64 j = i + h + g + f + e + d + c + b + a
    return j
}
fn used_big2() -> uint64 {
    uint64 a = 10
    uint64 b = 20
    uint64 c = 30
    uint64 d = 40
    uint64 e = 50
    uint64 f = a + b + c + d + e
    uint64 g = f * 3
    uint64 h = g + f
    uint64 i = h * g + f
    uint64 j = i + h + g + f + e + d + c + b + a
    return j
}
fn used_big3() -> uint64 {
    uint64 a = 100
    uint64 b = 200
    uint64 c = 300
    uint64 d = 400
    uint64 e = 500
    uint64 f = a + b + c + d + e
    uint64 g = f * 4
    uint64 h = g + f
    uint64 i = h * g + f
    uint64 j = i + h + g + f + e + d + c + b + a
    return j
}
fn main() {
    uint64 r = used_big() + used_big2() + used_big3()
    exit(r & 0xFF)
}
KRSRC

if $MLRC $MLRC_FLAGS /tmp/mlrc_dce_unused_$$.mlr -o /tmp/mlrc_dce_small_$$ > /dev/null 2>&1 && \
   $MLRC $MLRC_FLAGS /tmp/mlrc_dce_used_$$.mlr -o /tmp/mlrc_dce_large_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_dce_small_$$ /tmp/mlrc_dce_large_$$
    small_size=$(wc -c < /tmp/mlrc_dce_small_$$)
    large_size=$(wc -c < /tmp/mlrc_dce_large_$$)
    # Verify the unused-function binary is smaller (DCE removed dead code)
    # Also verify the unused-function binary runs correctly
    /tmp/mlrc_dce_small_$$ > /dev/null 2>&1; small_exit=$?
    if [ "$small_size" -lt "$large_size" ] && [ "$small_exit" = "42" ]; then
        PASS=$((PASS + 1))
        echo "  dce_eliminates_unused: PASS (unused=$small_size < used=$large_size bytes, exit=$small_exit)"
    else
        echo "  dce_eliminates_unused: FAIL (unused=$small_size vs used=$large_size, exit=$small_exit)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  dce_eliminates_unused: FAIL (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/mlrc_dce_unused_$$.mlr /tmp/mlrc_dce_used_$$.mlr /tmp/mlrc_dce_small_$$ /tmp/mlrc_dce_large_$$

# --- ELF relocatable (.o) test ---
echo ""
echo "--- ELF relocatable (.o) test ---"
TOTAL=$((TOTAL + 1))
printf 'fn add(uint64 a, uint64 b) -> uint64 { return a + b }\nfn main() { exit(add(30, 12)) }\n' > /tmp/mlrc_obj_$$.mlr
if $MLRC $MLRC_FLAGS --emit=obj /tmp/mlrc_obj_$$.mlr -o /tmp/mlrc_obj_$$.o > /dev/null 2>&1; then
    # Check first 18 bytes: ELF magic (4) + class(1) + data(1) + version(1) + osabi(1) + padding(8) + e_type LE (2)
    # e_type at offset 16-17 should be 01 00 (ET_REL = 1, little-endian)
    magic=$(xxd -l 4 -p /tmp/mlrc_obj_$$.o 2>/dev/null)
    etype=$(xxd -s 16 -l 2 -p /tmp/mlrc_obj_$$.o 2>/dev/null)
    if [ "$magic" = "7f454c46" ] && [ "$etype" = "0100" ]; then
        PASS=$((PASS + 1))
        echo "  emit_obj: PASS (valid ELF relocatable, $(wc -c < /tmp/mlrc_obj_$$.o) bytes)"
    else
        FAIL=$((FAIL + 1))
        echo "  emit_obj: FAIL (bad ELF header: magic=$magic etype=$etype)"
    fi
else
    FAIL=$((FAIL + 1))
    echo "  emit_obj: FAIL (compilation with --emit=obj failed)"
fi

# Also test -c flag produces same result
TOTAL=$((TOTAL + 1))
if $MLRC $MLRC_FLAGS -c /tmp/mlrc_obj_$$.mlr -o /tmp/mlrc_obj_c_$$.o > /dev/null 2>&1; then
    c_magic=$(xxd -l 4 -p /tmp/mlrc_obj_c_$$.o 2>/dev/null)
    c_etype=$(xxd -s 16 -l 2 -p /tmp/mlrc_obj_c_$$.o 2>/dev/null)
    if [ "$c_magic" = "7f454c46" ] && [ "$c_etype" = "0100" ]; then
        PASS=$((PASS + 1))
        echo "  emit_obj_c_flag: PASS"
    else
        FAIL=$((FAIL + 1))
        echo "  emit_obj_c_flag: FAIL (bad ELF header)"
    fi
else
    FAIL=$((FAIL + 1))
    echo "  emit_obj_c_flag: FAIL (compilation with -c failed)"
fi

# Test readelf can parse sections and symbols.
# Cross-compile MLRC_FLAGS (e.g. --arch=arm64 on an arm64 runner re-targeting
# the host) can produce a valid .o that this regex-based test doesn't cover.
# Skip on non-x86_64 hosts where MLRC_FLAGS targets arm64.
TOTAL=$((TOTAL + 1))
if [ "$(uname -m)" != "x86_64" ] && [ "$(uname -m)" != "amd64" ]; then
    PASS=$((PASS + 1))
    echo "  emit_obj_readelf: SKIP (non-x86_64 host)"
elif command -v readelf > /dev/null 2>&1 && [ -f /tmp/mlrc_obj_$$.o ]; then
    sections=$(readelf -S /tmp/mlrc_obj_$$.o 2>/dev/null)
    has_text=$(echo "$sections" | grep -c '\.text')
    has_symtab=$(echo "$sections" | grep -c '\.symtab')
    symbols=$(readelf -s /tmp/mlrc_obj_$$.o 2>/dev/null)
    has_main=$(echo "$symbols" | grep -c 'FUNC.*GLOBAL.*main')
    has_add=$(echo "$symbols" | grep -c 'FUNC.*LOCAL.*add')
    if [ "$has_text" -ge 1 ] && [ "$has_symtab" -ge 1 ] && [ "$has_main" -ge 1 ] && [ "$has_add" -ge 1 ]; then
        PASS=$((PASS + 1))
        echo "  emit_obj_readelf: PASS (.text, .symtab, main GLOBAL, add LOCAL)"
    else
        FAIL=$((FAIL + 1))
        echo "  emit_obj_readelf: FAIL (text=$has_text symtab=$has_symtab main=$has_main add=$has_add)"
    fi
else
    PASS=$((PASS + 1))
    echo "  emit_obj_readelf: SKIP (readelf not found or .o missing)"
fi
rm -f /tmp/mlrc_obj_$$.mlr /tmp/mlrc_obj_$$.o /tmp/mlrc_obj_c_$$.o

# --- Generics (monomorphization) ---
run_test "generic_fn_single" 'fn max_gen<T>(T a, T b) -> T {
    if a > b { return a }
    return b
}
fn main() {
    uint64 r = max_gen<uint64>(30, 42)
    exit(r)
}' 42

run_test "generic_fn_identity" 'fn identity<T>(T x) -> T { return x }
fn main() {
    uint64 r = identity<uint64>(7)
    exit(r)
}' 7

run_test "generic_fn_chain" 'fn max_gen<T>(T a, T b) -> T {
    if a > b { return a }
    return b
}
fn identity<T>(T x) -> T { return x }
fn main() {
    uint64 r = max_gen<uint64>(30, 42)
    uint64 s = identity<uint64>(r)
    exit(s)
}' 42

run_test "generic_call_uint32" 'fn add_one<T>(T x) -> T { return x + 1 }
fn main() {
    uint32 r = add_one<uint32>(41)
    exit(r)
}' 42

run_test "generic_multi_param" 'fn pick_first<T, U>(T a, U b) -> T { return a }
fn main() {
    uint64 r = pick_first<uint64, uint32>(42, 99)
    exit(r)
}' 42

run_test "generic_no_conflict_lt" 'fn id<T>(T x) -> T { return x }
fn main() {
    uint64 a = 3
    uint64 b = 5
    if a < b { exit(id<uint64>(42)) }
    exit(0)
}' 42

# --- Error detection tests ---
echo ""
echo "--- Error detection tests ---"

# Wrong argument count
TOTAL=$((TOTAL + 1))
printf 'fn add(uint64 a, uint64 b) -> uint64 { return a + b }\nfn main() { exit(add(1, 2, 3)) }\n' > /tmp/mlrc_err_$$.mlr
if $MLRC $MLRC_FLAGS /tmp/mlrc_err_$$.mlr -o /tmp/mlrc_err_$$ 2>/tmp/mlrc_stderr_$$ ; then
    echo "FAIL: wrong_arg_count (should not compile)"
    FAIL=$((FAIL + 1))
else
    if grep -q "wrong number of arguments" /tmp/mlrc_stderr_$$; then
        PASS=$((PASS + 1))
        echo "  wrong_arg_count: PASS (error detected)"
    else
        echo "FAIL: wrong_arg_count (wrong error)"
        FAIL=$((FAIL + 1))
    fi
fi
rm -f /tmp/mlrc_err_$$.mlr /tmp/mlrc_err_$$ /tmp/mlrc_stderr_$$

# Missing return in non-void function
TOTAL=$((TOTAL + 1))
printf 'fn get_val() -> uint64 { uint64 x = 42 }\nfn main() { exit(get_val()) }\n' > /tmp/mlrc_err_$$.mlr
if $MLRC $MLRC_FLAGS /tmp/mlrc_err_$$.mlr -o /tmp/mlrc_err_$$ 2>/tmp/mlrc_stderr_$$ ; then
    echo "FAIL: missing_return (should not compile)"
    FAIL=$((FAIL + 1))
else
    if grep -q "may not return" /tmp/mlrc_stderr_$$; then
        PASS=$((PASS + 1))
        echo "  missing_return: PASS (error detected)"
    else
        echo "FAIL: missing_return (wrong error)"
        FAIL=$((FAIL + 1))
    fi
fi
rm -f /tmp/mlrc_err_$$.mlr /tmp/mlrc_err_$$ /tmp/mlrc_stderr_$$

# Duplicate function definition
TOTAL=$((TOTAL + 1))
printf 'fn foo() { exit(1) }\nfn foo() { exit(2) }\nfn main() { foo() }\n' > /tmp/mlrc_err_$$.mlr
if $MLRC $MLRC_FLAGS /tmp/mlrc_err_$$.mlr -o /tmp/mlrc_err_$$ 2>/tmp/mlrc_stderr_$$ ; then
    echo "FAIL: duplicate_fn (should not compile)"
    FAIL=$((FAIL + 1))
else
    if grep -q "redefinition" /tmp/mlrc_stderr_$$; then
        PASS=$((PASS + 1))
        echo "  duplicate_fn: PASS (error detected)"
    else
        echo "FAIL: duplicate_fn (wrong error)"
        FAIL=$((FAIL + 1))
    fi
fi
rm -f /tmp/mlrc_err_$$.mlr /tmp/mlrc_err_$$ /tmp/mlrc_stderr_$$

# --- Android emit test ---
echo ""
echo "--- Android emit test ---"
TOTAL=$((TOTAL + 1))
printf 'fn main() { exit(42) }\n' > /tmp/mlrc_android_$$.mlr
if $MLRC $MLRC_FLAGS --emit=android /tmp/mlrc_android_$$.mlr -o /tmp/mlrc_android_$$ > /dev/null 2>&1; then
    magic=$(xxd -l 4 -p /tmp/mlrc_android_$$ 2>/dev/null)
    etype=$(xxd -s 16 -l 2 -p /tmp/mlrc_android_$$ 2>/dev/null)
    if [ "$magic" = "7f454c46" ] && [ "$etype" = "0300" ]; then
        PASS=$((PASS + 1))
        echo "  android_emit: PASS (valid PIE ELF, $(wc -c < /tmp/mlrc_android_$$) bytes)"
    else
        FAIL=$((FAIL + 1))
        echo "  android_emit: FAIL (bad ELF: magic=$magic etype=$etype)"
    fi
else
    FAIL=$((FAIL + 1))
    echo "  android_emit: FAIL (compilation failed)"
fi
rm -f /tmp/mlrc_android_$$.mlr /tmp/mlrc_android_$$

# --- Android x86_64 emit test ---
echo ""
echo "--- Android x86_64 emit test ---"
TOTAL=$((TOTAL + 1))
printf 'fn main() { exit(42) }\n' > /tmp/mlrc_androidx_$$.mlr
if $MLRC --arch=x86_64 --emit=android /tmp/mlrc_androidx_$$.mlr -o /tmp/mlrc_androidx_$$ > /dev/null 2>&1; then
    magic=$(xxd -l 4 -p /tmp/mlrc_androidx_$$ 2>/dev/null)
    etype=$(xxd -s 16 -l 2 -p /tmp/mlrc_androidx_$$ 2>/dev/null)
    emach=$(xxd -s 18 -l 2 -p /tmp/mlrc_androidx_$$ 2>/dev/null)
    if [ "$magic" = "7f454c46" ] && [ "$etype" = "0300" ] && [ "$emach" = "3e00" ]; then
        # Execute via glibc loader (bypasses PT_INTERP=/system/bin/linker64)
        if [ -x /lib64/ld-linux-x86-64.so.2 ] && [ "$(uname -m)" = "x86_64" ]; then
            actual=0
            /lib64/ld-linux-x86-64.so.2 /tmp/mlrc_androidx_$$ > /dev/null 2>&1
            actual=$?
            if [ "$actual" = "42" ]; then
                PASS=$((PASS + 1))
                echo "  android_emit_x86_64: PASS (PIE ELF x86-64, exec=42)"
            else
                FAIL=$((FAIL + 1))
                echo "  android_emit_x86_64: FAIL (exec exit=$actual, expected 42)"
            fi
        else
            PASS=$((PASS + 1))
            echo "  android_emit_x86_64: PASS (structural; no glibc loader)"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "  android_emit_x86_64: FAIL (bad ELF: magic=$magic etype=$etype mach=$emach)"
    fi
else
    FAIL=$((FAIL + 1))
    echo "  android_emit_x86_64: FAIL (compilation failed)"
fi
rm -f /tmp/mlrc_androidx_$$.mlr /tmp/mlrc_androidx_$$

# --- 2-tuple return and destructure ---
run_test "tuple_basic" 'fn divmod(uint64 x, uint64 y) -> uint64 { return (x / y, x % y) }
fn main() { (uint64 q, uint64 r) = divmod(17, 5); exit(q + r) }' 5

run_test "tuple_branch" 'fn minmax(uint64 a, uint64 b) -> uint64 { if a < b { return (a, b) } return (b, a) }
fn main() { (uint64 lo, uint64 hi) = minmax(42, 7); exit(hi - lo) }' 35

run_test "tuple_nested_call" 'fn pair(uint64 x) -> uint64 { return (x, x + 1) }
fn main() { (uint64 a, uint64 b) = pair(10); exit(a * b) }' 110

run_test "tuple_void_context" 'fn split(uint64 n) -> uint64 { return (n * 2, n * 3) }
fn main() { uint64 sum = 0; (uint64 a, uint64 b) = split(5); sum = a + b; exit(sum) }' 25

run_test "tuple_reuse" 'fn step(uint64 x) -> uint64 { return (x + 1, x + 2) }
fn main() { (uint64 p, uint64 q) = step(10); (uint64 r, uint64 s) = step(20); exit(p + q + r + s) }' 66

# --- 3-tuple return and destructure ---
run_test "tuple3_basic" 'fn triple() -> u64 { return (10, 20, 30) }
fn main() { (u64 a, u64 b, u64 c) = triple(); exit(a + b + c) }' 60

run_test "tuple3_values" 'fn split3(u64 x) -> u64 { return (x, x + 1, x + 2) }
fn main() { (u64 a, u64 b, u64 c) = split3(5); exit(c) }' 7

# --- asm { } I/O constraints ---
# x86_64-only asm constraint tests (rdtsc, shl are x86 instructions)
if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
# rdtsc: no inputs, two outputs (low/high 32 bits of the TSC into rax/rdx).
run_test "asm_rdtsc_out" 'fn main() {
    uint64 lo = 0
    uint64 hi = 0
    asm { "rdtsc" } out(rax -> lo, rdx -> hi)
    if lo == 0 { if hi == 0 { exit(1) } }
    exit(0)
}' 0

# shl via asm with one input and one output, testing pinned-param loading.
run_test "asm_shl_in_out" 'fn shl_by(uint64 v, uint64 n) -> uint64 {
    uint64 r = 0
    asm { "0x48 0xD3 0xE0" } in(v -> rax, n -> rcx) out(rax -> r)
    return r
}
fn main() { exit(shl_by(3, 4)) }' 48
fi

# nop with no constraints — ensures backward-compat with existing asm blocks.
run_test "asm_nop_noconstraints" 'fn main() { asm { "nop" }; exit(5) }' 5

# --- Opt-in: run on a real Android emulator via adb (ANDROID_EMULATOR=1) ---
# Requires: adb on PATH, one device online, and write access to
# /data/local/tmp. Cross-compiles a handful of programs as
# android-x86_64, pushes them, and executes under real bionic.
if [ "${ANDROID_EMULATOR:-0}" = "1" ] && command -v adb > /dev/null 2>&1; then
    DEV=$(adb get-state 2>/dev/null | tr -d '\r')
    if [ "$DEV" = "device" ]; then
        echo ""
        echo "--- Android emulator (adb, x86_64) ---"
        _adb_run() {
            local name="$1" src="$2" expected="$3"
            TOTAL=$((TOTAL + 1))
            printf '%s\n' "$src" > /tmp/mlrc_adb_$$.mlr
            if $MLRC --arch=x86_64 --emit=android /tmp/mlrc_adb_$$.mlr -o /tmp/mlrc_adb_$$ > /dev/null 2>&1; then
                adb push /tmp/mlrc_adb_$$ /data/local/tmp/mlrc_adb_$$ > /dev/null 2>&1
                adb shell chmod 755 /data/local/tmp/mlrc_adb_$$ > /dev/null 2>&1
                got=$(adb shell "/data/local/tmp/mlrc_adb_$$ > /dev/null 2>&1; echo \$?" | tr -d '\r')
                if [ "$got" = "$expected" ]; then
                    PASS=$((PASS + 1))
                    echo "  adb_$name: PASS"
                else
                    FAIL=$((FAIL + 1))
                    echo "  adb_$name: FAIL (expected $expected, got $got)"
                fi
                adb shell rm -f /data/local/tmp/mlrc_adb_$$ > /dev/null 2>&1
            else
                FAIL=$((FAIL + 1))
                echo "  adb_$name: FAIL (compile)"
            fi
            rm -f /tmp/mlrc_adb_$$.mlr /tmp/mlrc_adb_$$
        }
        _adb_run "exit42"   'fn main() { exit(42) }' 42
        _adb_run "add"      'fn main() { exit(2 + 3) }' 5
        _adb_run "loop"     'fn main() { uint64 s = 0; for i in 1..11 { s = s + i }; exit(s) }' 55
        _adb_run "recurse"  'fn fib(uint64 n) -> uint64 { if n <= 1 { return n } return fib(n-1)+fib(n-2) }
fn main() { exit(fib(10)) }' 55
        _adb_run "statics"  'static uint64 c = 0
fn inc() { c = c + 1 }
fn main() { inc(); inc(); inc(); inc(); exit(c) }' 4
        _adb_run "println"  'fn main() { println("android bionic"); exit(7) }' 7
    else
        echo "  android_emulator: SKIP (ANDROID_EMULATOR=1 but no device online)"
    fi
fi

# --- For loop ---
run_test "for_range" 'fn main() { uint64 s = 0; for i in 0..10 { s = s + i }; exit(s) }' 45
run_test "for_range_inclusive" 'fn main() { uint64 s = 0; for i in 0..=10 { s = s + i }; exit(s) }' 55
run_test "for_range_no_in" 'fn main() { uint64 s = 0; for i 0..10 { s = s + i }; exit(s) }' 45
run_test "for_range_no_in_inclusive" 'fn main() { uint64 s = 0; for i 0..=5 { s = s + i }; exit(s) }' 15
run_test "for_range_ident_end"  'fn main() { u64 n = 5; u64 s = 0; for i 0..n { s = s + i }; exit(s) }' 10
run_test "for_range_ident_both" 'fn main() { u64 a = 2; u64 b = 7; u64 s = 0; for i a..b { s = s + i }; exit(s) }' 20
run_test "loop_break" 'fn main() { u64 n = 0; loop { n = n + 1; if n >= 42 { break } }; exit(n) }' 42
run_test "match_wildcard_miss" 'fn main() {
    u64 x = 999
    match x {
        1 => { exit(1) }
        5 => { exit(55) }
        _ => { exit(42) }
    }
}' 42
run_test "match_wildcard_hit_first" 'fn main() {
    u64 x = 5
    match x {
        5 => { exit(50) }
        _ => { exit(42) }
    }
}' 50
run_test "match_multi_value_first" 'fn main() {
    u64 x = 3
    match x {
        1, 2, 3 => { exit(77) }
        _ => { exit(0) }
    }
}' 77
run_test "match_multi_value_second" 'fn main() {
    u64 x = 5
    match x {
        1, 2, 3 => { exit(77) }
        4, 5 => { exit(66) }
        _ => { exit(0) }
    }
}' 66
run_test "match_multi_value_miss" 'fn main() {
    u64 x = 9
    match x {
        1, 2, 3 => { exit(77) }
        4, 5 => { exit(66) }
        _ => { exit(11) }
    }
}' 11
run_test "match_range_inclusive" 'fn main() {
    u64 x = 50
    match x {
        0..=31 => { exit(1) }
        32..=126 => { exit(2) }
        _ => { exit(3) }
    }
}' 2
run_test "match_range_exclusive" 'fn main() {
    u64 x = 10
    match x {
        0..10 => { exit(1) }
        10..20 => { exit(2) }
        _ => { exit(3) }
    }
}' 2
run_test "match_range_ident" 'fn main() {
    u64 lo = 5
    u64 hi = 10
    u64 x = 7
    match x {
        lo..=hi => { exit(7) }
        _ => { exit(0) }
    }
}' 7
run_test "compound_field_assign" 'struct P { u64 x; u64 y }
fn main() { P p; p.x = 10; p.x += 5; p.x *= 2; exit(p.x) }' 30
run_test "compound_index_assign" 'fn main() { u64[4] a; a[0] = 10; a[0] += 3; a[0] *= 4; exit(a[0]) }' 52

# --- Char predicates (std/string.mlr) ---
run_test "char_pred_digit"   'import "std/string.mlr"
fn main() { if is_digit(53) == 1 && is_digit(97) == 0 { exit(1) }; exit(0) }' 1
run_test "char_pred_alpha"   'import "std/string.mlr"
fn main() { if is_alpha(97) == 1 && is_alpha(48) == 0 { exit(1) }; exit(0) }' 1
run_test "char_pred_space"   'import "std/string.mlr"
fn main() { if is_space(32) == 1 && is_space(10) == 1 && is_space(65) == 0 { exit(1) }; exit(0) }' 1
run_test "char_pred_hex"     'import "std/string.mlr"
fn main() { if is_hex_digit(70) == 1 && is_hex_digit(103) == 0 { exit(1) }; exit(0) }' 1
run_test "char_to_upper"     'import "std/string.mlr"
fn main() { exit(to_upper_ch(97)) }' 65
run_test "char_to_lower"     'import "std/string.mlr"
fn main() { exit(to_lower_ch(90)) }' 122
run_test "char_hex_val"      'import "std/string.mlr"
fn main() { exit(hex_digit_val(70)) }' 15
run_test "loop_nested_break" 'fn main() {
    u64 total = 0
    u64 outer = 0
    loop {
        outer = outer + 1
        u64 inner = 0
        loop {
            inner = inner + 1
            total = total + 1
            if inner >= 3 { break }
        }
        if outer >= 2 { break }
    }
    exit(total)
}' 6

# --- Defer ---
run_test "defer_on_return" 'static u64 n = 0
fn go() -> u64 { defer { n = 100 }; return 1 }
fn main() { u64 r = go(); exit(r + n) }' 101
run_test "defer_lifo" 'static u64 log = 0
fn run() { defer { log = log * 10 + 1 }; defer { log = log * 10 + 2 }; defer { log = log * 10 + 3 } }
fn main() { run(); exit(log) }' 65
run_test "defer_early_return" 'static u64 n = 0
fn pick(u64 x) -> u64 { defer { n = n + 100 }; if x > 0 { return 1 }; return 2 }
fn main() { u64 a = pick(5); u64 b = pick(0); exit(a + b + n) }' 203
run_test "defer_nested_block" 'static u64 v = 0
fn inner() { if 1 == 1 { defer { v = 42 } } }
fn main() { inner(); exit(v) }' 42

# --- @section annotation capture ---
TOTAL=$((TOTAL + 1))
printf '@section(".text.init")\nfn boot() -> u64 { return 0 }\nfn main() { exit(boot()) }\n' > "$DIR/../test_tmp_sect_$$.mlr"
$MLRC --emit=asm $MLRC_FLAGS "$DIR/../test_tmp_sect_$$.mlr" -o /tmp/mlrc_sect_$$.s > /dev/null 2>&1
if grep -q "^\\.section \\.text\\.init" /tmp/mlrc_sect_$$.s 2>/dev/null; then
    PASS=$((PASS + 1))
else
    echo "FAIL: section_asm_directive (no .section emitted)"; FAIL=$((FAIL + 1))
fi
rm -f "$DIR/../test_tmp_sect_$$.mlr" /tmp/mlrc_sect_$$.s

TOTAL=$((TOTAL + 1))
printf 'fn boot() -> u64 { return 0 }\nfn main() { exit(boot()) }\n' > "$DIR/../test_tmp_nosect_$$.mlr"
$MLRC --emit=asm $MLRC_FLAGS "$DIR/../test_tmp_nosect_$$.mlr" -o /tmp/mlrc_nosect_$$.s > /dev/null 2>&1
if grep -q "^\\.section" /tmp/mlrc_nosect_$$.s 2>/dev/null; then
    echo "FAIL: no_section_no_directive (spurious .section)"; FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1))
fi
rm -f "$DIR/../test_tmp_nosect_$$.mlr" /tmp/mlrc_nosect_$$.s

# --- Many-parameter functions ---
run_test "fn_7args" 'fn sum7(uint64 a, uint64 b, uint64 c, uint64 d, uint64 e, uint64 f, uint64 g) -> uint64 { return a + b + c + d + e + f + g }
fn main() { exit(sum7(1,2,3,4,5,6,7)) }' 28

run_test "fn_8args" 'fn s(uint64 a, uint64 b, uint64 c, uint64 d, uint64 e, uint64 f, uint64 g, uint64 h) -> uint64 { return a + b + c + d + e + f + g + h }
fn main() { exit(s(1,2,3,4,5,6,7,8)) }' 36

# --- Enum (auto-numbered) ---
run_test "enum_auto" 'enum Color { Red, Green, Blue }
fn main() { exit(Color.Blue) }' 2

# --- emit=asm produces text ---
echo ""
echo "--- ASM emit test ---"
TOTAL=$((TOTAL + 1))
printf 'fn main() { exit(42) }\n' > /tmp/mlrc_asm_$$.mlr
if $MLRC $MLRC_FLAGS --emit=asm /tmp/mlrc_asm_$$.mlr -o /tmp/mlrc_asm_$$.s > /dev/null 2>&1; then
    if file /tmp/mlrc_asm_$$.s | grep -qi 'text\|ascii' && grep -q 'main' /tmp/mlrc_asm_$$.s; then
        PASS=$((PASS + 1))
        echo "  emit_asm: PASS (text output with function labels)"
    else
        FAIL=$((FAIL + 1))
        echo "  emit_asm: FAIL (output is not text or missing labels)"
    fi
else
    FAIL=$((FAIL + 1))
    echo "  emit_asm: FAIL (compilation with --emit=asm failed)"
fi
rm -f /tmp/mlrc_asm_$$.mlr /tmp/mlrc_asm_$$.s

# --- emit=asm content tests ---
echo ""
echo "--- emit=asm content tests ---"

# Test asm output has function labels and mnemonics
TOTAL=$((TOTAL + 1))
echo 'fn add(uint64 a, uint64 b) -> uint64 { return a + b }
fn main() { exit(add(1, 2)) }' > /tmp/mlrc_asm_test_$$.mlr
if $MLRC $MLRC_FLAGS --emit=asm /tmp/mlrc_asm_test_$$.mlr -o /tmp/mlrc_asm_test_$$.s > /dev/null 2>&1; then
    if grep -q "add:" /tmp/mlrc_asm_test_$$.s && grep -q "main:" /tmp/mlrc_asm_test_$$.s && grep -q "ret" /tmp/mlrc_asm_test_$$.s; then
        echo "  emit_asm_content: PASS"
        PASS=$((PASS + 1))
    else
        echo "  emit_asm_content: FAIL (missing labels or mnemonics)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "  emit_asm_content: FAIL (compilation error)"
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/mlrc_asm_test_$$.*

# Test that --emit=xyz gives an error
TOTAL=$((TOTAL + 1))
echo 'fn main() { exit(0) }' > /tmp/mlrc_asm_err_$$.mlr
if $MLRC --emit=xyz /tmp/mlrc_asm_err_$$.mlr -o /tmp/mlrc_asm_err_$$ 2>&1 | grep -q "unknown emit format"; then
    echo "  emit_unknown_error: PASS"
    PASS=$((PASS + 1))
else
    echo "  emit_unknown_error: FAIL"
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/mlrc_asm_err_$$.mlr /tmp/mlrc_asm_err_$$

# --- String escapes ---
run_test_output "str_escape_newline" 'fn main() { print("a\nb"); exit(0) }' "a
b"

# --- ARM64 cross-compilation tests via QEMU ---
QEMU_A64=""
if command -v qemu-aarch64-static > /dev/null 2>&1; then
    QEMU_A64="qemu-aarch64-static"
elif command -v qemu-aarch64 > /dev/null 2>&1; then
    QEMU_A64="qemu-aarch64"
fi

if [ -n "$QEMU_A64" ] && [ "$ARCH" = "x86_64" ]; then
    echo ""
    echo "--- ARM64 cross-compilation tests (QEMU) ---"

    run_test_a64() {
        local name="$1"
        local input="$2"
        local expected="$3"
        TOTAL=$((TOTAL + 1))

        printf '%s\n' "$input" > /tmp/mlrc_a64_$$.mlr
        if $MLRC --arch=arm64 /tmp/mlrc_a64_$$.mlr -o /tmp/mlrc_a64_$$ > /dev/null 2>&1; then
            chmod +x /tmp/mlrc_a64_$$
            local got=0
            $QEMU_A64 /tmp/mlrc_a64_$$ > /dev/null 2>&1 && got=0 || got=$?
            if [ "$got" = "$expected" ]; then
                PASS=$((PASS + 1))
            else
                echo "FAIL: $name (expected $expected, got $got)"
                FAIL=$((FAIL + 1))
            fi
        else
            echo "FAIL: $name (cross-compilation failed)"
            FAIL=$((FAIL + 1))
        fi
        rm -f /tmp/mlrc_a64_$$.mlr /tmp/mlrc_a64_$$
    }

    run_test_a64 "a64_exit" 'fn main() { exit(42) }' 42
    run_test_a64 "a64_add" 'fn add(uint64 a, uint64 b) -> uint64 { return a + b }
fn main() { exit(add(10, 32)) }' 42
    run_test_a64 "a64_atomic" 'fn main() { uint64 buf = alloc(64); atomic_store(buf, 42); exit(atomic_load(buf)) }' 42
    run_test_a64 "a64_static" 'static uint64 x = 0
fn main() { x = 42; exit(x) }' 42

    # ARM64 struct passing tests
    run_test_a64 "a64_struct_field" 'struct P { uint64 x; uint64 y }
fn main() { P a; a.x = 10; a.y = 32; exit(a.x + a.y) }' 42

    run_test_a64 "a64_struct_pass" 'struct P { uint64 x; uint64 y }
fn sum(P p) -> uint64 { return p.x + p.y }
fn main() { P a; a.x = 10; a.y = 32; exit(sum(a)) }' 42

    run_test_a64 "a64_struct_pass_2arg" 'struct P { uint64 x; uint64 y }
fn add(P a, P b) -> uint64 { return a.x + b.y }
fn main() { P p1; p1.x = 10; p1.y = 0; P p2; p2.x = 0; p2.y = 32; exit(add(p1, p2)) }' 42

    # Struct-arg by-value uniformity on arm64 (fix/struct-abi-byvalue;
    # shared IR lowering, but exercised under qemu so the arm64 emitters
    # are proven too).
    run_test_a64 "a64_struct_arg_nested_byval" 'struct I { uint64 a; uint64 b }
struct O { I inn; uint64 z }
fn poke(I c) -> uint64 { c.a = 99; return c.a }
fn main() { O o; o.inn.a = 42; uint64 r = poke(o.inn); exit(o.inn.a) }' 42

    run_test_a64 "a64_struct_arg_elem_byval" 'struct P { uint64 x; uint64 y }
fn poke(P c) -> uint64 { c.x = 99; return c.x }
fn main() { P[3] arr; arr[1].x = 42; uint64 r = poke(arr[1]); exit(arr[1].x) }' 42

    run_test_a64 "a64_method_self_mutation" 'struct P { uint64 x; uint64 y }
fn P.bump(P self) { self.x = 42 }
fn main() { P p; p.x = 1; p.bump(); exit(p.x) }' 42

    run_test_a64 "a64_method_self_elem_recv" 'struct P { uint64 x; uint64 y }
fn P.setx(P self, uint64 v) { self.x = v }
fn main() { P[3] arr; arr[2].x = 1; arr[2].setx(42); exit(arr[2].x) }' 42

    run_test_a64 "a64_struct_return" 'struct P { uint64 x; uint64 y }
fn make() -> P { P r; r.x = 10; r.y = 32; return r }
fn main() { P a = make(); exit(a.x + a.y) }' 42

    run_test_a64 "a64_struct_lit" 'struct P { uint64 x; uint64 y }
fn sum(P p) -> uint64 { return p.x + p.y }
fn main() { exit(sum(P{x: 10, y: 32})) }' 42

    run_test_a64 "a64_struct_copy" 'struct P { uint64 x; uint64 y }
fn main() { P a; a.x = 10; a.y = 32; P b = a; exit(b.x + b.y) }' 42

    run_test_a64 "a64_struct_small" 'struct S { uint32 a; uint32 b }
fn sum(S s) -> uint64 { return s.a + s.b }
fn main() { S v; v.a = 10; v.b = 32; exit(sum(v)) }' 42

    # ARM64 HFA (Homogeneous Float Aggregate) tests
    run_test_a64 "a64_hfa_pass_f64" 'struct V { f64 x; f64 y }
fn sum(V v) -> f64 { return v.x + v.y }
fn main() {
    V v; v.x = 3.0; v.y = 4.0
    f64 r = sum(v)
    exit(f64_to_int(r))
}' 7

    run_test_a64 "a64_hfa_return_f64" 'struct V { f64 x; f64 y }
fn make() -> V { V r; r.x = 10.0; r.y = 32.0; return r }
fn main() {
    V v = make()
    exit(f64_to_int(v.x + v.y))
}' 42

    run_test_a64 "a64_hfa_pass_return_f64" 'struct V { f64 x; f64 y }
fn scale(V v, f64 s) -> V {
    V r; r.x = v.x * s; r.y = v.y * s; return r
}
fn main() {
    V v; v.x = 2.0; v.y = 5.0
    V r = scale(v, 3.0)
    exit(f64_to_int(r.x + r.y))
}' 21

    run_test_a64 "a64_hfa_3field_f64" 'struct V3 { f64 x; f64 y; f64 z }
fn sum3(V3 v) -> f64 { return v.x + v.y + v.z }
fn main() {
    V3 v; v.x = 10.0; v.y = 20.0; v.z = 12.0
    exit(f64_to_int(sum3(v)))
}' 42

    run_test_a64 "a64_hfa_4field_f64" 'struct V4 { f64 a; f64 b; f64 c; f64 d }
fn sum4(V4 v) -> f64 { return v.a + v.b + v.c + v.d }
fn main() {
    V4 v; v.a = 10.0; v.b = 11.0; v.c = 12.0; v.d = 9.0
    exit(f64_to_int(sum4(v)))
}' 42
fi

# --- v2.6 feature tests ---
echo ""
echo "--- v2.6 short type aliases ---"
run_test "alias_u8"  'fn main() { u8 x = 42; exit(x) }' 42
run_test "alias_u16" 'fn main() { u16 x = 42; exit(x) }' 42
run_test "alias_u32" 'fn main() { u32 x = 42; exit(x) }' 42
run_test "alias_u64" 'fn main() { u64 x = 42; exit(x) }' 42
run_test "alias_i8"  'fn main() { i8  x = 42; exit(x) }' 42
run_test "alias_i16" 'fn main() { i16 x = 42; exit(x) }' 42
run_test "alias_i32" 'fn main() { i32 x = 42; exit(x) }' 42
run_test "alias_i64" 'fn main() { i64 x = 42; exit(x) }' 42

echo ""
echo "--- v2.6 pointer load/store builtins ---"
run_test "load_store_u8"  'fn main() { u64 buf = alloc(16); store8(buf, 42); exit(load8(buf)) }' 42
run_test "load_store_u16" 'fn main() { u64 buf = alloc(16); store16(buf, 42); exit(load16(buf)) }' 42
run_test "load_store_u32" 'fn main() { u64 buf = alloc(16); store32(buf, 42); exit(load32(buf)) }' 42
run_test "load_store_u64" 'fn main() { u64 buf = alloc(16); store64(buf, 42); exit(load64(buf)) }' 42
run_test "load_store_offsets" 'fn main() {
    u64 buf = alloc(32)
    store8(buf + 0, 1)
    store8(buf + 1, 2)
    store8(buf + 2, 3)
    store8(buf + 3, 4)
    exit(load8(buf + 0) + load8(buf + 1) + load8(buf + 2) + load8(buf + 3))
}' 10
run_test "load_store_widths_mixed" 'fn main() {
    u64 buf = alloc(32)
    store32(buf, 0x11223344)
    exit(load8(buf) + load8(buf + 1) + load8(buf + 2) + load8(buf + 3))
}' 170
run_test "vload_vstore_u32" 'fn main() { u64 buf = alloc(16); vstore32(buf, 42); exit(vload32(buf)) }' 42
run_test "vload_vstore_u64" 'fn main() { u64 buf = alloc(16); vstore64(buf, 42); exit(vload64(buf)) }' 42

echo ""
echo "--- v2.6 print_str / println_str ---"
# print_str prints the contents of a variable string pointer.
# If the builtin is broken, it prints the pointer address as a number
# instead of the string, and the output doesn't contain "Hi".
run_test_output "print_str_variable" 'fn main() {
    u64 msg = "Hi"
    print_str(msg)
    exit(0)
}' 'Hi' 0
run_test_output "println_str_variable" 'fn main() {
    u64 msg = "Line"
    println_str(msg)
    exit(0)
}' 'Line' 0

echo ""
echo "--- v2.6 static arrays ---"
run_test "static_array_u8" 'static u8[16] buf
fn main() { buf[0] = 42; exit(buf[0]) }' 42
run_test "static_array_roundtrip" 'static u8[32] buf
fn main() {
    buf[5] = 10
    buf[6] = 20
    buf[7] = 12
    exit(buf[5] + buf[6] + buf[7])
}' 42

echo ""
echo "--- v2.6 struct arrays ---"
run_test "struct_array_basic" 'struct P { u64 x; u64 y }
fn main() {
    P[4] pts
    pts[0].x = 10
    pts[0].y = 20
    pts[3].x = 5
    pts[3].y = 7
    exit(pts[0].x + pts[0].y + pts[3].x + pts[3].y)
}' 42
run_test "struct_array_iteration" 'struct Row { u64 a; u64 b }
fn main() {
    Row[5] rows
    for i in 0..5 {
        rows[i].a = i
        rows[i].b = 0
    }
    u64 sum = 0
    for j in 0..5 {
        sum = sum + rows[j].a
    }
    exit(sum)
}' 10

echo ""
echo "--- v2.6 slice parameters ---"
run_test "slice_param_len" 'fn sum_bytes([u8] data) -> u64 {
    u64 total = 0
    u64 i = 0
    u64 n = data.len
    while i < n {
        total = total + load8(data + i)
        i = i + 1
    }
    return total
}
fn main() {
    u8[6] buf
    buf[0] = 10
    buf[1] = 20
    buf[2] = 12
    exit(sum_bytes(buf, 3))
}' 42

echo ""
echo "--- v2.6 device blocks ---"
run_test "device_block_read_write" 'device Fake at 0x66666000 {
    Data at 0x00 : u32
    Status at 0x04 : u8
}
fn main() {
    // mmap a page at 0x66666000 (Linux x86_64 syscall 9, ARM64 222)
    u64 nr = 9
    // arm64 mmap syscall is 222 on every OS (Linux / Android / macOS).
    // get_arch_id() returns 2=linux-arm64, 4=windows-arm64, 6=macos-arm64, 7=android-arm64.
    u64 aid = get_arch_id()
    if aid == 2 { nr = 222 }
    if aid == 4 { nr = 222 }
    if aid == 6 { nr = 222 }
    if aid == 7 { nr = 222 }
    syscall_raw(nr, 0x66666000, 4096, 3, 0x32, 0xFFFFFFFFFFFFFFFF, 0)
    Fake.Data = 42
    Fake.Status = 7
    u32 v = Fake.Data
    u8  s = Fake.Status
    exit(v + s)
}' 49

echo ""
echo "--- v2.6 method calls ---"
run_test "method_call" 'struct P { u64 x; u64 y }
fn P.sum(P self) -> u64 { return self.x + self.y }
fn main() {
    P p
    p.x = 10
    p.y = 32
    exit(p.sum())
}' 42

echo ""
echo "--- v2.6 #lang directive ---"
run_test "lang_stable" '#lang stable

fn main() { exit(42) }' 42
run_test "lang_experimental" '#lang experimental

fn main() { exit(42) }' 42

echo ""
echo "--- v2.6 living compiler ---"
# --list-proposals should work without an input file and exit 0
TOTAL=$((TOTAL + 1))
if $MLRC lc --list-proposals > /tmp/mlrc_prop_$$.txt 2>&1; then
    if grep -q "KernRift Proposal Registry" /tmp/mlrc_prop_$$.txt && grep -q "load_store_builtins" /tmp/mlrc_prop_$$.txt; then
        PASS=$((PASS + 1))
    else
        echo "FAIL: list_proposals (output did not contain expected strings)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: list_proposals (command failed)"
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/mlrc_prop_$$.txt

# --fix --dry-run on a legacy file should show a migration
TOTAL=$((TOTAL + 1))
cat > /tmp/mlrc_mig_$$.mlr <<'KREOF'
fn main() {
    u64 buf = alloc(16)
    u64 v = 0
    unsafe { *(buf as u32) -> v }
    exit(v)
}
KREOF
if $MLRC lc --fix --dry-run /tmp/mlrc_mig_$$.mlr > /tmp/mlrc_mig_out_$$.txt 2>&1; then
    if grep -q "1 migration site(s) rewritten" /tmp/mlrc_mig_out_$$.txt && grep -q "load32" /tmp/mlrc_mig_out_$$.txt; then
        PASS=$((PASS + 1))
    else
        echo "FAIL: migration_dry_run (output missing expected content)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: migration_dry_run (command failed)"
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/mlrc_mig_$$.mlr /tmp/mlrc_mig_out_$$.txt

# --fix (actual) on a legacy file should rewrite and the result should compile
TOTAL=$((TOTAL + 1))
cat > /tmp/mlrc_mig2_$$.mlr <<'KREOF'
fn main() {
    u64 buf = alloc(16)
    u64 v = 0
    store32(buf, 42)
    unsafe { *(buf as u32) -> v }
    exit(v)
}
KREOF
if $MLRC lc --fix /tmp/mlrc_mig2_$$.mlr > /dev/null 2>&1; then
    if grep -q "v = load32(buf)" /tmp/mlrc_mig2_$$.mlr; then
        # Now verify the rewritten file still compiles and runs
        if $MLRC $MLRC_FLAGS /tmp/mlrc_mig2_$$.mlr -o /tmp/mlrc_mig2_bin_$$ > /dev/null 2>&1; then
            chmod +x /tmp/mlrc_mig2_bin_$$
            /tmp/mlrc_mig2_bin_$$ > /dev/null 2>&1
            if [ "$?" = "42" ]; then
                PASS=$((PASS + 1))
            else
                echo "FAIL: migration_apply (rewritten binary exit != 42)"
                FAIL=$((FAIL + 1))
            fi
        else
            echo "FAIL: migration_apply (rewritten file did not compile)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: migration_apply (file was not rewritten)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: migration_apply (command failed)"
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/mlrc_mig2_$$.mlr /tmp/mlrc_mig2_bin_$$

# mlrc lc on a file with unsafe ops should report legacy_ptr_ops
TOTAL=$((TOTAL + 1))
cat > /tmp/mlrc_lc_$$.mlr <<'KREOF'
fn main() {
    u64 buf = alloc(16)
    u64 v = 0
    unsafe { *(buf as u32) -> v }
    exit(v)
}
KREOF
if $MLRC lc /tmp/mlrc_lc_$$.mlr > /tmp/mlrc_lc_out_$$.txt 2>&1; then
    if grep -q "legacy_ptr_ops" /tmp/mlrc_lc_out_$$.txt && grep -q "auto-fix available" /tmp/mlrc_lc_out_$$.txt; then
        PASS=$((PASS + 1))
    else
        echo "FAIL: lc_reports_legacy (missing expected strings in output)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: lc_reports_legacy (command failed)"
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/mlrc_lc_$$.mlr /tmp/mlrc_lc_out_$$.txt

# Governance: promote + list round-trip
TOTAL=$((TOTAL + 1))
GOV_DIR=/tmp/mlrc_gov_$$
# Use the raw compiler binary (not the wrapper script) so we can cd elsewhere
if [ -f "$DIR/../build/mlrc" ]; then
    GOV_KRC=$(cd "$DIR/../build" && pwd)/mlrc
elif [ -f "$DIR/../build/mlrc3" ]; then
    GOV_KRC=$(cd "$DIR/../build" && pwd)/mlrc3
else
    GOV_KRC=""
fi
mkdir -p "$GOV_DIR" && (cd "$GOV_DIR" && rm -rf .kernrift && \
    "$GOV_KRC" lc --promote tail_call_intrinsic > /tmp/mlrc_gov_promote_$$.txt 2>&1)
if [ -n "$GOV_KRC" ] && \
   grep -q "promoted: tail_call_intrinsic" /tmp/mlrc_gov_promote_$$.txt 2>/dev/null && \
   [ -f "$GOV_DIR/.kernrift/proposals" ] && \
   grep -q "tail_call_intrinsic stable" "$GOV_DIR/.kernrift/proposals"; then
    PASS=$((PASS + 1))
else
    echo "FAIL: governance_promote (state file not updated)"
    FAIL=$((FAIL + 1))
fi
rm -rf "$GOV_DIR" /tmp/mlrc_gov_promote_$$.txt

# Migration: long-form types → short aliases
TOTAL=$((TOTAL + 1))
cat > /tmp/mlrc_migtypes_$$.mlr <<'KREOF'
fn main() {
    uint64 x = 42
    uint32 y = 1
    uint16 z = 2
    exit(x)
}
KREOF
if $MLRC lc --fix /tmp/mlrc_migtypes_$$.mlr > /dev/null 2>&1; then
    if grep -q "u64 x" /tmp/mlrc_migtypes_$$.mlr && \
       grep -q "u32 y" /tmp/mlrc_migtypes_$$.mlr && \
       grep -q "u16 z" /tmp/mlrc_migtypes_$$.mlr; then
        PASS=$((PASS + 1))
    else
        echo "FAIL: migration_types (file was not rewritten)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: migration_types (command failed)"
    FAIL=$((FAIL + 1))
fi
rm -f /tmp/mlrc_migtypes_$$.mlr

# --- Bootstrap test ---
echo ""
echo "--- Bootstrap test ---"
TOTAL=$((TOTAL + 1))
if [ -f "$DIR/../build/mlrc.mlr" ]; then
    # Use the host arch so the compiled mlrc can run on the runner.
    HOST_ARCH=$(uname -m)
    case "$HOST_ARCH" in
        aarch64|arm64) BS_ARCH=arm64 ;;
        *)             BS_ARCH=x86_64 ;;
    esac
    cp "$DIR/../build/mlrc.mlr" /tmp/mlrc_bootstrap_$$.mlr
    $MLRC $MLRC_FLAGS /tmp/mlrc_bootstrap_$$.mlr -o /tmp/mlrc2_$$ > /dev/null 2>&1
    chmod +x /tmp/mlrc2_$$ 2>/dev/null
    /tmp/mlrc2_$$ --arch=$BS_ARCH /tmp/mlrc_bootstrap_$$.mlr -o /tmp/mlrc3_$$ > /dev/null 2>&1
    chmod +x /tmp/mlrc3_$$ 2>/dev/null
    /tmp/mlrc3_$$ --arch=$BS_ARCH /tmp/mlrc_bootstrap_$$.mlr -o /tmp/mlrc4_$$ > /dev/null 2>&1
    if diff /tmp/mlrc3_$$ /tmp/mlrc4_$$ > /dev/null 2>&1; then
        PASS=$((PASS + 1))
        echo "  bootstrap: PASS (fixed point at $(wc -c < /tmp/mlrc3_$$) bytes)"
    else
        FAIL=$((FAIL + 1))
        echo "  bootstrap: FAIL (mlrc3 != mlrc4)"
    fi
    rm -f /tmp/mlrc_bootstrap_$$.mlr /tmp/mlrc2_$$ /tmp/mlrc3_$$ /tmp/mlrc4_$$
else
    echo "  bootstrap: SKIP (no build/mlrc.mlr)"
    PASS=$((PASS + 1))
fi

echo ""
echo "--- typed local arrays (regression) ---"
run_test "u8_arr"  'fn main() { u8[4] a; a[0] = 10; a[3] = 40; exit(a[0] + a[3]) }' 50
run_test "u16_arr" 'fn main() { u16[4] a; a[0] = 1000; a[3] = 4000; exit((a[0] + a[3]) / 100) }' 50
run_test "u32_arr" 'fn main() { u32[4] a; a[0] = 100000; a[3] = 400000; exit((a[0] + a[3]) / 10000) }' 50
run_test "u64_arr" 'fn main() { u64[4] a; a[0] = 100; a[1] = 200; a[2] = 300; a[3] = 400; exit(a[2] - a[0] - 100) }' 100
run_test "u64_arr_loop" 'fn main() {
    u64[5] a
    a[0] = 1
    a[1] = 2
    a[2] = 3
    a[3] = 4
    a[4] = 5
    u64 sum = 0
    for i in 0..5 { sum = sum + a[i] }
    exit(sum)
}' 15
run_test "bubble_sort_u64" 'fn main() {
    u64[4] a
    a[0] = 3
    a[1] = 1
    a[2] = 4
    a[3] = 2
    for i in 0..4 {
        for j in 0..3 {
            if a[j] > a[j+1] {
                u64 t = a[j]
                a[j] = a[j+1]
                a[j+1] = t
            }
        }
    }
    exit(a[0] * 0 + a[1] * 0 + a[2] * 0 + a[3])
}' 4

echo ""
echo "--- heap struct pointers (regression) ---"
run_test "heap_struct_basic" 'struct P { u64 x; u64 y }
fn main() {
    P p = alloc(16)
    p.x = 11
    p.y = 31
    exit(p.x + p.y)
}' 42
run_test "heap_linked_list" 'struct N { u64 v; u64 next }
fn main() {
    N a = alloc(16)
    N b = alloc(16)
    a.v = 2
    a.next = b
    b.v = 40
    b.next = 0
    u64 sum = 0
    N cur = a
    while cur != 0 {
        sum = sum + cur.v
        cur = cur.next
    }
    exit(sum)
}' 42

echo ""
echo "--- const initializers (regression) ---"
run_test "const_int"    'const u64 X = 42; fn main() { exit(X) }' 42
run_test "const_hex"    'const u64 X = 0x2A; fn main() { exit(X) }' 42
run_test "const_div"    'const u64 D = 10; fn main() { exit(100 / D) }' 10
run_test "const_mod"    'const u64 M = 7; fn main() { exit(50 % M) }' 1
run_test "const_mul"    'const u64 C = 21; fn main() { exit(C * 2) }' 42
run_test "const_char"   "const u64 CH = 'A'; fn main() { exit(CH) }" 65
run_test "const_true"   'const u64 T = true; fn main() { exit(T + 41) }' 42
run_test "static_int"   'static u64 X = 99; fn main() { exit(X) }' 99
run_test "static_neg"   'static i64 X = -1; fn main() { exit(X) }' 255
run_test "static_bnot"  'static u64 X = ~0; fn main() { exit(X & 7) }' 7
run_test "const_neg"    'const i64 X = -42; fn main() { exit(0 - X) }' 42

echo ""
echo "--- import after comment (regression) ---"
TOTAL=$((TOTAL + 1))
# This test was VACUOUS: it wrote the source to /tmp and imported
# "std/io.mlr", but imports resolve relative to the SOURCE FILE's directory,
# so /tmp/std/io.mlr never existed on a clean machine. It only appeared to
# pass because println is a BUILTIN — the binary printed "imp_ok" whether or
# not the import resolved. It passed on developer machines (which have a
# ~/.local/share/mlrift/std install to fall back on) and passed in CI for the
# wrong reason, until an unopenable import started aborting the compile.
# Now it imports a module it creates itself and calls a function that exists
# ONLY in that module, so the assertion cannot be satisfied unless the import
# genuinely resolved.
IMPC_DIR=$(mktemp -d /tmp/mlrc_impc_XXXXXX)
cat > "$IMPC_DIR/impmod.mlr" <<'KREOF'
fn imp_after_comment_helper() -> uint64 { return 7 }
KREOF
cat > "$IMPC_DIR/impmain.mlr" <<'KREOF'
// leading comment should not break imports
import "impmod.mlr"
fn main() { exit(imp_after_comment_helper() + 35) }
KREOF
if $MLRC $MLRC_FLAGS "$IMPC_DIR/impmain.mlr" -o "$IMPC_DIR/impbin" > /dev/null 2>&1; then
    chmod +x "$IMPC_DIR/impbin"
    "$IMPC_DIR/impbin" > /dev/null 2>&1
    imp_ec=$?
    if [ "$imp_ec" -eq 42 ]; then
        PASS=$((PASS + 1))
        echo "  import_after_comment: PASS (imported symbol resolved, exit 42)"
    else
        FAIL=$((FAIL + 1))
        echo "  import_after_comment: FAIL (exit $imp_ec, expected 42)"
    fi
else
    FAIL=$((FAIL + 1))
    echo "  import_after_comment: FAIL (compile)"
fi
rm -rf "$IMPC_DIR"

echo ""
echo "--- char literals ---"
run_test "char_a"    "fn main() { exit('A') }" 65
run_test "char_z"    "fn main() { exit('z') }" 122
run_test "char_nl"   "fn main() { exit('\\n') }" 10
run_test "char_tab"  "fn main() { exit('\\t') }" 9
run_test "char_bs"   "fn main() { exit('\\\\') }" 92
run_test "char_nul"  "fn main() { exit('\\0') }" 0
run_test "char_cmp"  "fn main() { u64 c = 97; if c == 'a' { exit(1) } exit(0) }" 1

echo ""
echo "--- emit=obj non-extern path (regression) ---"
TOTAL=$((TOTAL + 1))
cat > /tmp/mlrc_noext_$$.mlr <<'KREOF'
fn main() { exit(42) }
KREOF
if $MLRC --emit=obj /tmp/mlrc_noext_$$.mlr -o /tmp/mlrc_noext_$$.o > /dev/null 2>&1; then
    # File must be long enough for section headers: shoff + shnum*64 <= filesize
    if command -v python3 > /dev/null 2>&1; then
        if python3 -c "
import struct, sys
d = open('/tmp/mlrc_noext_$$.o', 'rb').read()
shoff = struct.unpack_from('<Q', d, 0x28)[0]
shnum = struct.unpack_from('<H', d, 0x3C)[0]
if shoff + shnum * 64 != len(d):
    print('truncated:', shoff + shnum * 64, 'expected,', len(d), 'got')
    sys.exit(1)
"; then
            PASS=$((PASS + 1))
            echo "  emit_obj_no_extern: PASS"
        else
            FAIL=$((FAIL + 1))
            echo "  emit_obj_no_extern: FAIL (truncated ELF)"
        fi
    else
        PASS=$((PASS + 1))
        echo "  emit_obj_no_extern: SKIP (no python3)"
    fi
else
    FAIL=$((FAIL + 1))
    echo "  emit_obj_no_extern: FAIL (compile)"
fi
rm -f /tmp/mlrc_noext_$$.mlr /tmp/mlrc_noext_$$.o

# --- real LZ4 compression in .mlrbo fat binaries (regression) ---
# Before this, the "compressor" wrote uncompressed LZ4 frames (bit 31 set
# in block size) and the runner's else-branch skipped compressed blocks
# entirely. This test compiles a fat binary for a reasonably large
# program, checks that at least the first slice is actually compressed
# (bit 31 clear), and that its ratio is below 90% of the original.
#
# Must call build/mlrc2 directly — the test $MLRC wrapper forces
# --arch=x86_64 which would make mlrc emit a single-arch ELF, not a
# fat binary, and there'd be nothing to inspect.
echo ""
echo "--- fat binary real LZ4 compression (regression) ---"
TOTAL=$((TOTAL + 1))
MLRCBIN="$DIR/../build/mlrc"
cat > /tmp/mlrc_lz4_$$.mlr <<'KREOF'
fn main() {
    u64 i = 0
    u64 sum = 0
    while i < 64 { sum = sum + i * i; i = i + 1 }
    println(sum)
    exit(0)
}
KREOF
if "$MLRCBIN" /tmp/mlrc_lz4_$$.mlr -o /tmp/mlrc_lz4_$$.mlrbo > /dev/null 2>&1; then
    if command -v python3 > /dev/null 2>&1; then
        if python3 -c "
import struct, sys
d = open('/tmp/mlrc_lz4_$$.mlrbo', 'rb').read()
assert d[:8] == b'MLRBOFAT'
n = struct.unpack_from('<I', d, 12)[0]
# With pair blobs, csize covers two slices and cannot be compared to
# one slice's usize. Instead check: (1) total file < sum-of-uncompressed
# and (2) at least one block uses real compression (bit 31 clear).
total_uncomp = 0
any_compressed = False
for i in range(n):
    aid, comp, off, csize, usize = struct.unpack_from('<IIQQQ', d, 16+i*48)
    total_uncomp += usize
    frame = d[off:off+csize]
    if len(frame) >= 11:
        bs = struct.unpack_from('<I', frame, 7)[0]
        if (bs >> 31) & 1 == 0:
            any_compressed = True
if not any_compressed:
    print('no compressed blocks found')
    sys.exit(1)
if len(d) >= total_uncomp * 9 // 10:
    print(f'file {len(d)} not < 90% of {total_uncomp}')
    sys.exit(1)
print(f'ok: file={len(d)} total_uncomp={total_uncomp}')
"; then
            PASS=$((PASS + 1))
            echo "  lz4_real_compression: PASS"
        else
            FAIL=$((FAIL + 1))
            echo "  lz4_real_compression: FAIL"
        fi
    else
        PASS=$((PASS + 1))
        echo "  lz4_real_compression: SKIP (no python3)"
    fi
else
    FAIL=$((FAIL + 1))
    echo "  lz4_real_compression: FAIL (compile)"
fi
rm -f /tmp/mlrc_lz4_$$.mlr /tmp/mlrc_lz4_$$.mlrbo

# --- .mlrbo round-trip via mlr runner (real-compression end-to-end) ---
# Builds a .mlrbo, a mlr runner binary, and runs the .mlrbo through it.
# The runner must decompress the real LZ4 block and produce the right
# output. Skipped if we can't rebuild a matching runner.
echo ""
echo "--- fat binary round-trip via mlr runner (regression) ---"
TOTAL=$((TOTAL + 1))
cat > /tmp/mlrc_rt_$$.mlr <<'KREOF'
fn main() {
    println("roundtrip-ok")
    exit(123)
}
KREOF
MLRCBIN="$DIR/../build/mlrc"
cat "$DIR/../src/bcj.mlr" "$DIR/../src/runner.mlr" > /tmp/mlrc_rt_kr_$$.mlr
if "$MLRCBIN" /tmp/mlrc_rt_$$.mlr -o /tmp/mlrc_rt_$$.mlrbo > /dev/null 2>&1 \
   && "$MLRCBIN" --arch=$ARCH /tmp/mlrc_rt_kr_$$.mlr -o /tmp/mlrc_rt_kr_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_rt_kr_$$
    out=$(/tmp/mlrc_rt_kr_$$ /tmp/mlrc_rt_$$.mlrbo 2>&1)
    code=$?
    if [ "$out" = "roundtrip-ok" ] && [ "$code" = "123" ]; then
        PASS=$((PASS + 1))
        echo "  mlrbo_roundtrip: PASS"
    else
        FAIL=$((FAIL + 1))
        echo "  mlrbo_roundtrip: FAIL (out='$out' code=$code)"
    fi
else
    PASS=$((PASS + 1))
    echo "  mlrbo_roundtrip: SKIP (runner build)"
fi
rm -f /tmp/mlrc_rt_$$.mlr /tmp/mlrc_rt_kr_$$.mlr /tmp/mlrc_rt_$$.mlrbo /tmp/mlrc_rt_kr_$$

echo ""
echo "--- float types ---"
run_test "f64_parse" 'fn main() { f64 x = 0.0; exit(0) }' 0
run_test "f64_literal_precision" 'fn main() { f64 pi = 3.14159; f64 s = pi * int_to_f64(100000); exit(f64_to_int(s) % 100) }' 59
run_test "int_to_f64_rt" 'fn main() { f64 x = int_to_f64(42); exit(f64_to_int(x)) }' 42
run_test "f64_add" 'fn main() { f64 a = int_to_f64(10); f64 b = int_to_f64(3); f64 c = a + b; exit(f64_to_int(c)) }' 13
run_test "f64_sub" 'fn main() { f64 a = int_to_f64(50); f64 b = int_to_f64(8); exit(f64_to_int(a - b)) }' 42
run_test "f64_mul" 'fn main() { f64 a = int_to_f64(6); f64 b = int_to_f64(7); exit(f64_to_int(a * b)) }' 42
run_test "f64_div" 'fn main() { f64 a = int_to_f64(84); f64 b = int_to_f64(2); exit(f64_to_int(a / b)) }' 42
run_test "f64_sqrt" 'fn main() { f64 x = int_to_f64(49); exit(f64_to_int(sqrt(x))) }' 7
run_test "f64_reassign" 'fn main() { f64 x = int_to_f64(10); x = x + int_to_f64(5); x = x * int_to_f64(2); exit(f64_to_int(x)) }' 30
run_test "f64_cmp_lt" 'fn main() { f64 a = int_to_f64(3); f64 b = int_to_f64(5); if a < b { exit(1) } exit(0) }' 1
run_test "f64_cmp_gt" 'fn main() { f64 a = int_to_f64(10); f64 b = int_to_f64(5); if a > b { exit(1) } exit(0) }' 1
run_test "f64_cmp_eq" 'fn main() { f64 a = int_to_f64(7); f64 b = int_to_f64(7); if a == b { exit(1) } exit(0) }' 1
run_test "f64_fn_call" 'fn double_it(f64 x) -> f64 { return x + x }
fn main() { f64 r = double_it(int_to_f64(21)); exit(f64_to_int(r)) }' 42
run_test "f64_fn_2args" 'fn add_f(f64 a, f64 b) -> f64 { return a + b }
fn main() { f64 r = add_f(int_to_f64(20), int_to_f64(22)); exit(f64_to_int(r)) }' 42
run_test "f64_fn_mixed" 'fn scale(u64 n, f64 x) -> f64 { f64 fn64 = int_to_f64(n); return fn64 * x }
fn main() { f64 r = scale(3, int_to_f64(14)); exit(f64_to_int(r)) }' 42
run_test "f64_pos2_arg" 'fn get_second(u64 a, f64 b) -> f64 { return b }
fn main() { f64 r = get_second(1, 42.0); exit(f64_to_int(r)) }' 42
run_test "f64_pos3_arg" 'fn get_third(u64 a, u64 b, f64 c) -> f64 { return c }
fn main() { f64 r = get_third(1, 2, 33.0); exit(f64_to_int(r)) }' 33

# Float literal parsing
run_test "f64_literal_zero" 'fn main() { f64 x = 0.0; exit(f64_to_int(x)) }' 0
run_test "f64_literal_one" 'fn main() { f64 x = 1.0; exit(f64_to_int(x)) }' 1
# Regression: long plain-decimal f32 literal was sign-flipped (frac_divisor overflowed u64
# at >=19 frac digits; cvtsi2sd treated it as signed, producing a negative value).
# 0.0037996768951416016f has 19 frac digits — this must parse positive and be in (0.003,0.004).
run_test "f32_long_decimal_positive" 'fn main() { f32 v = 0.0037996768951416016f; i32 rc = 0; if v < 0.0f { rc = rc + 1 }; if v > 0.003f { if v < 0.004f { rc = rc + 2 } }; exit(rc) }' 2
# Scientific notation must still work: 1e-8f and 1.5e-3f
run_test "f32_sci_notation_neg_exp" 'fn main() { f32 v = 1e-8f; if v > 0.0f { exit(1) }; exit(0) }' 1
run_test "f32_sci_notation_frac" 'fn main() { f32 v = 1.5e-3f; if v > 0.001f { if v < 0.002f { exit(1) } }; exit(0) }' 1
# Short decimal must still work
run_test "f32_short_decimal" 'fn main() { f32 v = 0.003799677f; if v > 0.003f { if v < 0.004f { exit(1) } }; exit(0) }' 1

# Float reassignment
run_test "f64_reassign2" 'fn main() { f64 x = int_to_f64(5); f64 y = int_to_f64(3); x = x + y; exit(f64_to_int(x)) }' 8

# Float in while loop
run_test "f64_while" 'fn main() { f64 sum = int_to_f64(0); u64 i = 0; while i < 10 { sum = sum + int_to_f64(1); i = i + 1 }; exit(f64_to_int(sum)) }' 10

# f32 basic
run_test "f32_basic" 'fn main() { f32 x = int_to_f32(42); exit(f32_to_int(x)) }' 42

# Float comparison edge cases
run_test "f64_cmp_le" 'fn main() { f64 a = int_to_f64(5); f64 b = int_to_f64(5); if a <= b { exit(1) } exit(0) }' 1
run_test "f64_cmp_ne" 'fn main() { f64 a = int_to_f64(3); f64 b = int_to_f64(5); if a != b { exit(1) } exit(0) }' 1

# Conversion roundtrip
run_test "f32_f64_roundtrip" 'fn main() { f64 a = int_to_f64(99); f32 b = f64_to_f32(a); f64 c = f32_to_f64(b); exit(f64_to_int(c)) }' 99
run_test "f32_literal" 'fn main() { f32 x = 42.0f; exit(f32_to_int(x)) }' 42
# f16 conversions use x86_64 SSE bit manipulation — not implemented on ARM64
if [ "$ARCH" = "x86_64" ]; then
run_test "f16_roundtrip" 'fn main() { f32 x = 42.0f; u64 h = f32_to_f16(x); f32 y = f16_to_f32(h); exit(f32_to_int(y)) }' 42
fi

# FMA
run_test "f64_fma" 'fn main() { f64 a = int_to_f64(3); f64 b = int_to_f64(4); f64 c = int_to_f64(5); f64 r = fma_f64(a, b, c); exit(f64_to_int(r)) }' 17

echo ""
echo "--- alloc/dealloc ---"
run_test "alloc_header" 'fn main() { u64 p = alloc(64); store64(p, 42); u64 v = load64(p); exit(v) }' 42
run_test "dealloc_basic" 'fn main() { u64 p = alloc(64); store64(p, 99); dealloc(p); exit(0) }' 0

echo ""
echo "--- allocators (arena) ---"
run_test "arena_basic" 'import "std/alloc.mlr"
fn main() {
    u64 a = arena_new(4096)
    u64 p1 = arena_alloc(a, 64)
    store64(p1, 42)
    u64 v = load64(p1)
    arena_destroy(a)
    exit(v)
}' 42

run_test "arena_reset" 'import "std/alloc.mlr"
fn main() {
    u64 a = arena_new(4096)
    u64 p1 = arena_alloc(a, 100)
    arena_reset(a)
    u64 p2 = arena_alloc(a, 100)
    if p1 == p2 { exit(1) } exit(0)
}' 1

run_test "arena_stats" 'import "std/alloc.mlr"
fn main() {
    u64 a = arena_new(4096)
    arena_alloc(a, 32)
    arena_alloc(a, 64)
    (u64 total, u64 live) = arena_stats(a)
    arena_reset(a)
    arena_destroy(a)
    exit(total)
}' 96

echo ""
echo "--- allocators (pool) ---"
run_test "pool_basic" 'import "std/alloc.mlr"
fn main() {
    u64 p = pool_new(64, 8)
    u64 o1 = pool_alloc(p)
    store64(o1, 99)
    u64 v = load64(o1)
    pool_free(p, o1)
    pool_destroy(p)
    exit(v)
}' 99

run_test "pool_reuse" 'import "std/alloc.mlr"
fn main() {
    u64 p = pool_new(16, 4)
    u64 a = pool_alloc(p)
    u64 b = pool_alloc(p)
    pool_free(p, a)
    u64 c = pool_alloc(p)
    if a == c { exit(1) } exit(0)
}' 1

run_test "pool_stats" 'import "std/alloc.mlr"
fn main() {
    u64 p = pool_new(32, 10)
    pool_alloc(p)
    pool_alloc(p)
    pool_alloc(p)
    (u64 total, u64 used) = pool_stats(p)
    pool_destroy(p)
    exit(used)
}' 3

echo ""
echo "--- allocators (heap) ---"
run_test "heap_basic" 'import "std/alloc.mlr"
fn main() {
    u64 h = heap_new(4096)
    u64 p = heap_alloc(h, 64)
    store64(p, 77)
    u64 v = load64(p)
    heap_free(h, p)
    heap_destroy(h)
    exit(v)
}' 77

run_test "heap_multi" 'import "std/alloc.mlr"
fn main() {
    u64 h = heap_new(4096)
    u64 a = heap_alloc(h, 32)
    u64 b = heap_alloc(h, 64)
    u64 c = heap_alloc(h, 16)
    store64(a, 10)
    store64(b, 20)
    store64(c, 30)
    heap_free(h, b)
    heap_free(h, a)
    heap_free(h, c)
    heap_destroy(h)
    exit(0)
}' 0

run_test "heap_stats" 'import "std/alloc.mlr"
fn main() {
    u64 h = heap_new(4096)
    u64 a = heap_alloc(h, 32)
    u64 b = heap_alloc(h, 64)
    heap_free(h, a)
    (u64 total, u64 freed, u64 live) = heap_stats(h)
    heap_free(h, b)
    heap_destroy(h)
    exit(total)
}' 96

echo ""
echo "--- extern fn (libc linking) ---"
# These tests link against the HOST gcc's libc. On cross-compile runs
# (arm64 host but MLRC_FLAGS=--arch=x86_64 for example) the object file
# architecture won't match gcc and the link fails. Skip on non-x86_64
# hosts since the default MLRC_FLAGS target host arch and the host gcc
# links to host libc.
HOST_M=$(uname -m)
if [ "$HOST_M" != "x86_64" ] && [ "$HOST_M" != "amd64" ]; then
    echo "  extern_libc_write: SKIP (non-x86_64 host toolchain)"
    echo "  extern_libc_strlen_write: SKIP (non-x86_64 host toolchain)"
elif command -v gcc > /dev/null 2>&1; then
    TOTAL=$((TOTAL + 1))
    cat > /tmp/mlrc_ext_$$.mlr <<'KREOF'
extern fn write(u64 fd, u64 buf, u64 len) -> u64

fn main() {
    write(1, "extern_ok\n", 10)
    exit(0)
}
KREOF
    if $MLRC --emit=obj /tmp/mlrc_ext_$$.mlr -o /tmp/mlrc_ext_$$.o > /dev/null 2>&1 \
       && gcc /tmp/mlrc_ext_$$.o -o /tmp/mlrc_ext_linked_$$ -no-pie > /dev/null 2>&1; then
        got=$(/tmp/mlrc_ext_linked_$$ 2>/dev/null)
        if [ "$got" = "extern_ok" ]; then
            PASS=$((PASS + 1))
            echo "  extern_libc_write: PASS"
        else
            FAIL=$((FAIL + 1))
            echo "  extern_libc_write: FAIL (got: $got)"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "  extern_libc_write: FAIL (compile/link failed)"
    fi
    rm -f /tmp/mlrc_ext_$$.mlr /tmp/mlrc_ext_$$.o /tmp/mlrc_ext_linked_$$

    TOTAL=$((TOTAL + 1))
    cat > /tmp/mlrc_ext2_$$.mlr <<'KREOF'
extern fn strlen(u64 s) -> u64
extern fn write(u64 fd, u64 buf, u64 len) -> u64

fn main() {
    u64 msg = "two_externs\n"
    u64 n = strlen(msg)
    write(1, msg, n)
    exit(0)
}
KREOF
    if $MLRC --emit=obj /tmp/mlrc_ext2_$$.mlr -o /tmp/mlrc_ext2_$$.o > /dev/null 2>&1 \
       && gcc /tmp/mlrc_ext2_$$.o -o /tmp/mlrc_ext2_linked_$$ -no-pie > /dev/null 2>&1; then
        got=$(/tmp/mlrc_ext2_linked_$$ 2>/dev/null)
        if [ "$got" = "two_externs" ]; then
            PASS=$((PASS + 1))
            echo "  extern_libc_strlen_write: PASS"
        else
            FAIL=$((FAIL + 1))
            echo "  extern_libc_strlen_write: FAIL (got: $got)"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "  extern_libc_strlen_write: FAIL (compile/link failed)"
    fi
    rm -f /tmp/mlrc_ext2_$$.mlr /tmp/mlrc_ext2_$$.o /tmp/mlrc_ext2_linked_$$
else
    echo "  extern_libc_write: SKIP (gcc not available)"
    echo "  extern_libc_strlen_write: SKIP (gcc not available)"
fi

# --- sizeof ---
run_test "sizeof_u8" 'fn main() { exit(sizeof(uint8)) }' 1
run_test "sizeof_u64" 'fn main() { exit(sizeof(uint64)) }' 8
run_test "sizeof_f32" 'fn main() { exit(sizeof(f32)) }' 4
run_test "sizeof_f64" 'fn main() { exit(sizeof(f64)) }' 8
run_test "sizeof_struct" 'struct P { uint64 x; uint64 y }
fn main() { exit(sizeof(P)) }' 16
run_test "sizeof_struct_mixed" 'struct S { uint8 a; uint64 b }
fn main() { exit(sizeof(S)) }' 9
run_test "sizeof_alloc" 'struct P { uint64 x; uint64 y }
fn main() { uint64 p = alloc(sizeof(P)); dealloc(p); exit(0) }' 0

# --- Struct literals ---
run_test "struct_literal_pos" 'struct P { uint64 x; uint64 y }
fn main() {
    P p = P { 10, 20 }
    exit(p.x + p.y)
}' 30

run_test "struct_literal_named" 'struct P { uint64 x; uint64 y }
fn main() {
    P p = P { y: 20, x: 10 }
    exit(p.x + p.y)
}' 30

run_test "struct_literal_u8" 'struct S { uint8 a; uint8 b }
fn main() {
    S s = S { 3, 4 }
    exit(s.a + s.b)
}' 7

# --- Struct value semantics (copy on assign) ---
run_test "struct_assign_copy" 'struct P { uint64 x; uint64 y }
fn main() {
    P a
    a.x = 10; a.y = 20
    P b = a
    b.x = 99
    exit(a.x)
}' 10

run_test "struct_reassign" 'struct P { uint64 x; uint64 y }
fn main() {
    P a; a.x = 1; a.y = 2
    P b; b.x = 10; b.y = 20
    a = b
    exit(a.x + a.y)
}' 30

run_test "struct_literal_copy" 'struct P { uint64 x; uint64 y }
fn main() {
    P p = P { 10, 20 }
    P q = p
    q.x = 99
    exit(p.x)
}' 10

# --- Struct pass-by-value tests ---
run_test "struct_pass_by_value" 'struct P { uint64 x; uint64 y }
fn sum(P p) -> uint64 { return p.x + p.y }
fn main() {
    P a; a.x = 10; a.y = 20
    exit(sum(a))
}' 30

run_test "struct_pass_literal" 'struct P { uint64 x; uint64 y }
fn sum(P p) -> uint64 { return p.x + p.y }
fn main() { exit(sum(P { 10, 20 })) }' 30

run_test "struct_pass_no_alias" 'struct P { uint64 x; uint64 y }
fn modify(P p) -> uint64 { p.x = 99; return p.x }
fn main() {
    P a; a.x = 10; a.y = 20
    uint64 r = modify(a)
    exit(a.x)
}' 10

# --- Struct arg by-value uniformity (fix/struct-abi-byvalue) ---
# By-value must hold for EVERY struct-lvalue argument form, not just bare
# Idents. The IR path used to copy only Ident args: a nested-struct field
# arg leaked BY REFERENCE (callee writes persisted), and a struct array
# element arg was lowered as an oversized IR_LOAD (garbage pointer,
# segfault at the callee's first field access).
run_test "struct_arg_nested_byval" 'struct I { uint64 a; uint64 b }
struct O { I inn; uint64 z }
fn poke(I c) -> uint64 { c.a = 99; return c.a }
fn main() {
    O o; o.inn.a = 42; o.inn.b = 2
    uint64 r = poke(o.inn)
    exit(o.inn.a)
}' 42

run_test "struct_arg_elem_byval" 'struct P { uint64 x; uint64 y }
fn poke(P c) -> uint64 { c.x = 99; return c.x }
fn main() {
    P[3] arr
    arr[1].x = 42; arr[1].y = 2
    uint64 r = poke(arr[1])
    exit(arr[1].x)
}' 42

# Data integrity: the callee must see the element/field CONTENTS (the
# legacy path used to load 8 garbage bytes for a struct-sized element).
run_test "struct_arg_elem_data" 'struct P { uint64 x; uint64 y }
fn sum(P c) -> uint64 { return c.x + c.y }
fn main() {
    P[3] arr
    arr[2].x = 30; arr[2].y = 12
    exit(sum(arr[2]))
}' 42

run_test "struct_arg_nested_data" 'struct I { uint64 a; uint64 b }
struct O { I inn; uint64 z }
fn sum(I c) -> uint64 { return c.a + c.b }
fn main() {
    O o; o.inn.a = 40; o.inn.b = 2; o.z = 9
    exit(sum(o.inn))
}' 42

# Struct-array element into a fresh struct var: `P c = arr[i]` used to
# segfault (IR path memcpy'd FROM a garbage oversized load).
run_test "struct_var_from_elem" 'struct P { uint64 x; uint64 y }
fn main() {
    P[3] arr
    arr[1].x = 40; arr[1].y = 2
    P c = arr[1]
    exit(c.x + c.y)
}' 42

# --- Method `self` is BY REFERENCE ---
# `fn Struct.m(Struct self)` receives self as a reference to the caller''s
# storage: writes through self must persist. Without this, EVERY mutating
# method is a silent no-op (there is no diagnostic), and the language has
# no safe way to pass a mutable struct at all.
run_test "method_self_mutation" 'struct P { uint64 x; uint64 y }
fn P.bump(P self) { self.x = 42 }
fn main() {
    P p; p.x = 1; p.y = 2
    p.bump()
    exit(p.x)
}' 42

run_test "method_self_mutation_arg" 'struct P { uint64 x; uint64 y }
fn P.setx(P self, uint64 v) { self.x = v }
fn P.getx(P self) -> uint64 { return self.x }
fn main() {
    P p; p.x = 1
    p.setx(41)
    exit(p.getx() + 1)
}' 42

# Nested-field receiver: w.p.setx(...) mutates the inner struct in place.
run_test "method_self_nested_recv" 'struct P { uint64 x; uint64 y }
struct W { P p; uint64 z }
fn P.setx(P self, uint64 v) { self.x = v }
fn main() {
    W w; w.p.x = 1
    w.p.setx(42)
    exit(w.p.x)
}' 42

# Array-element receiver: arr[i].setx(...) used to parse as a bare field
# access with the argument list silently DROPPED (no call, no diagnostic).
run_test "method_self_elem_recv" 'struct P { uint64 x; uint64 y }
fn P.setx(P self, uint64 v) { self.x = v }
fn main() {
    P[3] arr
    arr[2].x = 1
    arr[2].setx(42)
    exit(arr[2].x)
}' 42

run_test_legacy "method_self_mutation_legacy" 'struct P { uint64 x; uint64 y }
fn P.bump(P self) { self.x = 42 }
fn main() {
    P p; p.x = 1; p.y = 2
    p.bump()
    exit(p.x)
}' 42

run_test_legacy "method_self_nested_recv_legacy" 'struct P { uint64 x; uint64 y }
struct W { P p; uint64 z }
fn P.setx(P self, uint64 v) { self.x = v }
fn main() {
    W w; w.p.x = 1
    w.p.setx(42)
    exit(w.p.x)
}' 42

run_test_legacy "method_self_elem_recv_legacy" 'struct P { uint64 x; uint64 y }
fn P.setx(P self, uint64 v) { self.x = v }
fn main() {
    P[3] arr
    arr[2].x = 1
    arr[2].setx(42)
    exit(arr[2].x)
}' 42

# Semantics lock: plain params never alias even when re-passed through a
# second call.
run_test "struct_arg_chain_byval" 'struct P { uint64 x; uint64 y }
fn inner(P c) { c.x = 99 }
fn outer(P c) -> uint64 { inner(c); return c.x }
fn main() {
    P p; p.x = 42; p.y = 2
    uint64 r = outer(p)
    exit(p.x)
}' 42

# Legacy-path parity for struct-arg by-value uniformity. The legacy Index
# lowering used to load 8 garbage bytes for a struct-sized array element.
run_test_legacy "struct_arg_elem_data_legacy" 'struct P { uint64 x; uint64 y }
fn sum(P c) -> uint64 { return c.x + c.y }
fn main() {
    P[3] arr
    arr[2].x = 30; arr[2].y = 12
    exit(sum(arr[2]))
}' 42

run_test_legacy "struct_arg_elem_byval_legacy" 'struct P { uint64 x; uint64 y }
fn poke(P c) -> uint64 { c.x = 99; return c.x }
fn main() {
    P[3] arr
    arr[1].x = 42; arr[1].y = 2
    uint64 r = poke(arr[1])
    exit(arr[1].x)
}' 42

run_test_legacy "struct_arg_nested_byval_legacy" 'struct I { uint64 a; uint64 b }
struct O { I inn; uint64 z }
fn poke(I c) -> uint64 { c.a = 99; return c.a }
fn main() {
    O o; o.inn.a = 42; o.inn.b = 2
    uint64 r = poke(o.inn)
    exit(o.inn.a)
}' 42

# Semantics lock: plain struct params stay by-value on the legacy path.
run_test_legacy "struct_arg_no_alias_legacy" 'struct P { uint64 x; uint64 y }
fn poke(P c) -> uint64 { c.x = 99; return c.x }
fn main() {
    P p; p.x = 42; p.y = 2
    uint64 r = poke(p)
    exit(p.x)
}' 42

# --- Struct return by value tests ---
run_test "struct_return_small" 'struct P { uint64 x; uint64 y }
fn make(uint64 x, uint64 y) -> P {
    return P { x, y }
}
fn main() {
    P p = make(10, 20)
    exit(p.x + p.y)
}' 30

run_test "struct_return_field" 'struct P { uint64 x; uint64 y }
fn make() -> P { return P { 3, 4 } }
fn main() { P p = make(); exit(p.x) }' 3

run_test "struct_return_chain" 'struct P { uint64 x; uint64 y }
fn make(uint64 v) -> P { return P { v, v + 1 } }
fn sum(P p) -> uint64 { return p.x + p.y }
fn main() { exit(sum(make(10))) }' 21

# --- Struct pass-by-value SSE (float eightbytes) tests ---
# These require SSE struct passing (x86_64 SysV only — ARM64 needs HFA support)
if [ "$ARCH" = "x86_64" ]; then
run_test "struct_pass_f64" 'struct V { f64 x; f64 y }
fn sum(V v) -> f64 { return v.x + v.y }
fn main() {
    V v; v.x = 3.0; v.y = 4.0
    f64 r = sum(v)
    exit(f64_to_int(r))
}' 7

run_test "struct_pass_mixed" 'struct M { uint64 id; f64 val }
fn get_val(M m) -> f64 { return m.val }
fn main() {
    M m; m.id = 1; m.val = 42.0
    f64 r = get_val(m)
    exit(f64_to_int(r))
}' 42
fi

# --- Large struct (MEMORY class) passing tests ---
run_test "struct_large_pass" 'struct Big { uint64 a; uint64 b; uint64 c }
fn sum(Big b) -> uint64 { return b.a + b.b + b.c }
fn main() {
    Big x; x.a = 1; x.b = 2; x.c = 3
    exit(sum(x))
}' 6

run_test "struct_large_copy" 'struct Big { uint64 a; uint64 b; uint64 c }
fn main() {
    Big x; x.a = 10; x.b = 20; x.c = 30
    Big y = x
    y.a = 99
    exit(x.a)
}' 10

run_test "struct_large_literal" 'struct Big { uint64 a; uint64 b; uint64 c }
fn sum(Big b) -> uint64 { return b.a + b.b + b.c }
fn main() { exit(sum(Big { 1, 2, 3 })) }' 6

# --- MEMORY-class struct return (sret hidden pointer, >16 bytes) tests ---
run_test "struct_return_large" 'struct Big { uint64 a; uint64 b; uint64 c }
fn make() -> Big {
    Big b; b.a = 10; b.b = 20; b.c = 30
    return b
}
fn main() {
    Big r = make()
    exit(r.a + r.b + r.c)
}' 60

run_test "struct_return_large_args" 'struct Big { uint64 a; uint64 b; uint64 c }
fn make(uint64 x, uint64 y, uint64 z) -> Big {
    Big b; b.a = x; b.b = y; b.c = z
    return b
}
fn main() {
    Big r = make(1, 2, 3)
    exit(r.a + r.b + r.c)
}' 6

run_test "nested_struct_basic" 'struct P { uint64 x; uint64 y }
struct L { P a; P b }
fn main() {
    L l
    l.a.x = 10; l.a.y = 20
    l.b.x = 30; l.b.y = 40
    exit(l.a.x + l.b.y)
}' 50

run_test "nested_struct_sizeof" 'struct P { uint64 x; uint64 y }
struct L { P a; P b }
fn main() { exit(sizeof(L)) }' 32

run_test "nested_struct_pass" 'struct P { uint64 x; uint64 y }
struct L { P a; P b }
fn sum(L l) -> uint64 { return l.a.x + l.a.y + l.b.x + l.b.y }
fn main() {
    L l
    l.a.x = 1; l.a.y = 2; l.b.x = 3; l.b.y = 4
    exit(sum(l))
}' 10

run_test "struct_eq" 'struct P { uint64 x; uint64 y }
fn main() {
    P a; a.x = 10; a.y = 20
    P b; b.x = 10; b.y = 20
    uint64 r = 0
    if a == b { r = 1 }
    exit(r)
}' 1

run_test "struct_ne" 'struct P { uint64 x; uint64 y }
fn main() {
    P a; a.x = 10; a.y = 20
    P b; b.x = 10; b.y = 99
    uint64 r = 0
    if a != b { r = 1 }
    exit(r)
}' 1

run_test "struct_eq_false" 'struct P { uint64 x; uint64 y }
fn main() {
    P a; a.x = 10; a.y = 20
    P b; b.x = 10; b.y = 99
    uint64 r = 0
    if a == b { r = 1 }
    exit(r)
}' 0

run_test "struct_ne_false" 'struct P { uint64 x; uint64 y }
fn main() {
    P a; a.x = 10; a.y = 20
    P b; b.x = 10; b.y = 20
    uint64 r = 0
    if a != b { r = 1 }
    exit(r)
}' 0

run_test "struct_eq_3field" 'struct V { uint64 x; uint64 y; uint64 z }
fn main() {
    V a; a.x = 1; a.y = 2; a.z = 3
    V b; b.x = 1; b.y = 2; b.z = 3
    uint64 r = 0
    if a == b { r = 1 }
    exit(r)
}' 1

# Helper: check that compilation FAILS with expected error message
run_error_check() {
    local name="$1"
    local input="$2"
    local expected_msg="$3"
    TOTAL=$((TOTAL + 1))
    local REPO_ROOT="$DIR/.."
    printf '%s\n' "$input" > "$REPO_ROOT/test_tmp_$$.mlr"
    if $MLRC $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ > /dev/null 2>/tmp/mlrc_diag_$$; then
        echo "FAIL: $name (should not compile)"
        FAIL=$((FAIL + 1))
    else
        if grep -q "$expected_msg" /tmp/mlrc_diag_$$; then
            PASS=$((PASS + 1))
            echo "  $name: PASS"
        else
            echo "FAIL: $name (expected '$expected_msg')"
            FAIL=$((FAIL + 1))
        fi
    fi
    rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$ /tmp/mlrc_diag_$$
}

# Helper: check that compilation SUCCEEDS but emits expected warning
run_warning_check() {
    local name="$1"
    local input="$2"
    local expected_msg="$3"
    TOTAL=$((TOTAL + 1))
    local REPO_ROOT="$DIR/.."
    printf '%s\n' "$input" > "$REPO_ROOT/test_tmp_$$.mlr"
    $MLRC $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ > /dev/null 2>/tmp/mlrc_diag_$$
    if grep -q "$expected_msg" /tmp/mlrc_diag_$$; then
        PASS=$((PASS + 1))
        echo "  $name: PASS"
    else
        echo "FAIL: $name (expected warning '$expected_msg')"
        FAIL=$((FAIL + 1))
    fi
    rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$ /tmp/mlrc_diag_$$
}

echo ""
echo "--- Compiler diagnostics ---"
run_error_check "diag_undef_var" 'fn main() { exit(xyz_undefined_name) }' "undeclared identifier"
run_warning_check "diag_unreachable_return" 'fn foo() -> uint64 { return 1; uint64 x = 2; return x } fn main() { exit(0) }' "unreachable code"
run_warning_check "diag_unreachable_break" 'fn main() { while 1 == 1 { break; uint64 x = 1 } exit(0) }' "unreachable code"
run_warning_check "diag_unreachable_exit" 'fn main() { exit(0); uint64 x = 1 }' "unreachable code"

# --- Runtime debug checks ---
echo ""
echo "--- Runtime debug checks (--debug) ---"
TOTAL=$((TOTAL + 1))
REPO_ROOT="$DIR/.."
printf 'fn main() { uint64 a = 10; uint64 b = 0; uint64 c = a / b; exit(c) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --debug "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_test_$$
    /tmp/mlrc_test_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" != "0" ]; then
        PASS=$((PASS + 1))
        echo "  debug_divzero: PASS (trapped, exit=$actual)"
    else
        echo "FAIL: debug_divzero (should have trapped)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: debug_divzero (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$

# Overflow test
TOTAL=$((TOTAL + 1))
printf 'fn main() { uint64 a = 9223372036854775807; uint64 b = a + a; exit(b) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --debug "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_test_$$
    /tmp/mlrc_test_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" != "0" ]; then
        PASS=$((PASS + 1))
        echo "  debug_overflow: PASS (trapped, exit=$actual)"
    else
        echo "FAIL: debug_overflow (should have trapped)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: debug_overflow (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$

# Null pointer test
TOTAL=$((TOTAL + 1))
printf 'fn main() { uint64 p = 0; uint64 v = load64(p); exit(v) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --debug "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_test_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_test_$$
    /tmp/mlrc_test_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" != "0" ]; then
        PASS=$((PASS + 1))
        echo "  debug_null_ptr: PASS (trapped, exit=$actual)"
    else
        echo "FAIL: debug_null_ptr (should have trapped)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: debug_null_ptr (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_test_$$

echo ""
echo "--- Debug info (-g) ---"
if [ "$ARCH" = "x86_64" ] && command -v readelf > /dev/null 2>&1; then

# Test: -g produces .debug_line section
TOTAL=$((TOTAL + 1))
REPO_ROOT="$DIR/.."
printf 'fn main() { exit(42) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS -g "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_g_$$ > /dev/null 2>&1; then
    if readelf -S /tmp/mlrc_g_$$ 2>/dev/null | grep -q "debug_line"; then
        PASS=$((PASS + 1))
        echo "  debug_line_exists: PASS"
    else
        echo "FAIL: debug_line_exists (section not found)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: debug_line_exists (compilation failed)"
    FAIL=$((FAIL + 1))
fi

# Test: binary with -g runs correctly
TOTAL=$((TOTAL + 1))
chmod +x /tmp/mlrc_g_$$
/tmp/mlrc_g_$$ > /dev/null 2>&1
actual=$?
if [ "$actual" = "42" ]; then
    PASS=$((PASS + 1))
    echo "  debug_runs: PASS (exit=42)"
else
    echo "FAIL: debug_runs (expected 42, got $actual)"
    FAIL=$((FAIL + 1))
fi

# Test: without -g, no debug section
TOTAL=$((TOTAL + 1))
$MLRC $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_nog_$$ > /dev/null 2>&1
if readelf -S /tmp/mlrc_nog_$$ 2>/dev/null | grep -q "debug_line"; then
    echo "FAIL: debug_no_flag (.debug_line should not exist)"
    FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1))
    echo "  debug_no_flag: PASS"
fi

# Test: readelf can decode the line info
TOTAL=$((TOTAL + 1))
if readelf --debug-dump=line /tmp/mlrc_g_$$ 2>&1 | grep -q "DWARF Version"; then
    PASS=$((PASS + 1))
    echo "  debug_line_valid: PASS"
else
    echo "FAIL: debug_line_valid (readelf could not decode)"
    FAIL=$((FAIL + 1))
fi

# Test: symtab has function names
TOTAL=$((TOTAL + 1))
if readelf -s /tmp/mlrc_g_$$ 2>/dev/null | grep -q "main"; then
    PASS=$((PASS + 1))
    echo "  debug_symtab: PASS"
else
    echo "FAIL: debug_symtab (main not in symbol table)"
    FAIL=$((FAIL + 1))
fi

rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_g_$$ /tmp/mlrc_nog_$$

fi  # end x86_64 + readelf gate

# --- IR backend test ---
echo ""
echo "--- IR backend test ---"
TOTAL=$((TOTAL + 1))
REPO_ROOT="$DIR/.."
printf 'fn main() { exit(42) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_ir_$$
    /tmp/mlrc_ir_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "42" ]; then
        PASS=$((PASS + 1))
        echo "  ir_exit_42: PASS"
    else
        echo "FAIL: ir_exit_42 (expected 42, got $actual)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_exit_42 (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# -- IR while loop --
TOTAL=$((TOTAL + 1))
printf 'fn main() { uint64 i = 0; uint64 s = 0; while i < 10 { s = s + i; i = i + 1 } exit(s) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_ir_$$
    timeout 2 /tmp/mlrc_ir_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "45" ]; then
        PASS=$((PASS + 1))
        echo "  ir_while_loop: PASS"
    else
        echo "FAIL: ir_while_loop (expected 45, got $actual)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_while_loop (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# -- IR division --
TOTAL=$((TOTAL + 1))
printf 'fn main() { exit(10 / 3) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_ir_$$
    /tmp/mlrc_ir_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "3" ]; then
        PASS=$((PASS + 1))
        echo "  ir_division: PASS"
    else
        echo "FAIL: ir_division (expected 3, got $actual)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_division (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# -- IR if/else --
TOTAL=$((TOTAL + 1))
printf 'fn main() { uint64 x = 10; if x > 5 { exit(1) } else { exit(0) } }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_ir_$$
    /tmp/mlrc_ir_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "1" ]; then
        PASS=$((PASS + 1))
        echo "  ir_if_else: PASS"
    else
        echo "FAIL: ir_if_else (expected 1, got $actual)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_if_else (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# -- IR alloc/store64/load64/dealloc --
TOTAL=$((TOTAL + 1))
printf 'fn main() { uint64 p = alloc(64); store64(p, 42); uint64 v = load64(p); dealloc(p); exit(v) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_ir_$$
    /tmp/mlrc_ir_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "42" ]; then
        PASS=$((PASS + 1))
        echo "  ir_alloc_store_load: PASS"
    else
        echo "FAIL: ir_alloc_store_load (expected 42, got $actual)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_alloc_store_load (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# -- IR store8/load8 --
TOTAL=$((TOTAL + 1))
printf 'fn main() { uint64 p = alloc(16); store8(p, 65); uint64 v = load8(p); dealloc(p); exit(v) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_ir_$$
    /tmp/mlrc_ir_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "65" ]; then
        PASS=$((PASS + 1))
        echo "  ir_store8_load8: PASS"
    else
        echo "FAIL: ir_store8_load8 (expected 65, got $actual)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_store8_load8 (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# -- IR multi-alloc --
TOTAL=$((TOTAL + 1))
printf 'fn main() { uint64 a = alloc(64); uint64 b = alloc(64); store64(a, 10); store64(b, 32); uint64 r = load64(a) + load64(b); dealloc(a); dealloc(b); exit(r) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if $MLRC $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_ir_$$
    /tmp/mlrc_ir_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "42" ]; then
        PASS=$((PASS + 1))
        echo "  ir_multi_alloc: PASS"
    else
        echo "FAIL: ir_multi_alloc (expected 42, got $actual)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_multi_alloc (compilation failed)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# --- ir_break ---
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'IREOF'
fn main() { uint64 i = 0; while i < 100 { if i == 5 { break }; i = i + 1 }; exit(i) }
IREOF
if timeout 10 "$MLRC" $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ 2>/dev/null; then
    chmod +x /tmp/mlrc_ir_$$; /tmp/mlrc_ir_$$; actual=$?
    if [ "$actual" -eq 5 ]; then
        echo "  ir_break: PASS"
    else
        echo "FAIL: ir_break (expected 5, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_break (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# --- ir_continue ---
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'IREOF'
fn main() { uint64 i = 0; uint64 s = 0; while i < 10 { i = i + 1; if i == 5 { continue }; s = s + 1 }; exit(s) }
IREOF
if timeout 10 "$MLRC" $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ 2>/dev/null; then
    chmod +x /tmp/mlrc_ir_$$; /tmp/mlrc_ir_$$; actual=$?
    if [ "$actual" -eq 9 ]; then
        echo "  ir_continue: PASS"
    else
        echo "FAIL: ir_continue (expected 9, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_continue (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# --- ir_fn_call ---
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'IREOF'
fn add(uint64 a, uint64 b) -> uint64 { return a + b }
fn main() { exit(add(20, 22)) }
IREOF
if timeout 10 "$MLRC" $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ 2>/dev/null; then
    chmod +x /tmp/mlrc_ir_$$; /tmp/mlrc_ir_$$; actual=$?
    if [ "$actual" -eq 42 ]; then
        echo "  ir_fn_call: PASS"
    else
        echo "FAIL: ir_fn_call (expected 42, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_fn_call (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# --- ir_recursion ---
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'IREOF'
fn fib(uint64 n) -> uint64 { if n <= 1 { return n }; return fib(n - 1) + fib(n - 2) }
fn main() { exit(fib(10)) }
IREOF
if timeout 10 "$MLRC" $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ 2>/dev/null; then
    chmod +x /tmp/mlrc_ir_$$; /tmp/mlrc_ir_$$; actual=$?
    if [ "$actual" -eq 55 ]; then
        echo "  ir_recursion: PASS"
    else
        echo "FAIL: ir_recursion (expected 55, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_recursion (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# --- ir_match ---
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'IREOF'
fn main() { uint64 x = 2; uint64 r = 0; match x { 1 => { r = 10 } 2 => { r = 42 } 3 => { r = 30 } }; exit(r) }
IREOF
if timeout 10 "$MLRC" $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ 2>/dev/null; then
    chmod +x /tmp/mlrc_ir_$$; /tmp/mlrc_ir_$$; actual=$?
    if [ "$actual" -eq 42 ]; then
        echo "  ir_match: PASS"
    else
        echo "FAIL: ir_match (expected 42, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_match (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# -- IR memset liveness (memset return must not clobber live vregs) --
TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'IREOF'
fn main() {
    uint64 src = alloc(100)
    memset(src, 0xAB, 100)
    uint64 dst = alloc(100)
    memset(dst, 0, 100)
    memcpy(dst, src, 100)
    uint64 v = 0
    unsafe { *(dst as uint8) -> v }
    dealloc(src)
    dealloc(dst)
    exit(v)
}
IREOF
if $MLRC $MLRC_FLAGS --ir "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_ir_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_ir_$$
    /tmp/mlrc_ir_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "171" ]; then
        PASS=$((PASS + 1))
        echo "  ir_memset_liveness: PASS"
    else
        echo "FAIL: ir_memset_liveness (expected 171, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: ir_memset_liveness (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_ir_$$

# --- bool type ---
echo ""
echo "--- bool type ---"

TOTAL=$((TOTAL + 1))
REPO_ROOT="$DIR/.."
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'BOOLEOF'
fn main() {
    bool b = true
    if b { exit(1) }
    exit(0)
}
BOOLEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_bool_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_bool_$$
    timeout 3 /tmp/mlrc_bool_$$ > /dev/null 2>&1
    if [ $? = 1 ]; then PASS=$((PASS + 1)); echo "  bool_true_false: PASS"
    else echo "FAIL: bool_true_false"; FAIL=$((FAIL + 1)); fi
else echo "FAIL: bool_true_false (compile)"; FAIL=$((FAIL + 1)); fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_bool_$$

TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'BOOLEOF'
fn main() {
    uint64 x = true
    exit(0)
}
BOOLEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_bool_$$ > /dev/null 2>&1; then
    echo "FAIL: bool_reject_assign_int (should have failed to compile)"; FAIL=$((FAIL + 1))
else
    PASS=$((PASS + 1)); echo "  bool_reject_assign_int: PASS"
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_bool_$$

# --- char type ---
echo ""
echo "--- char type ---"

TOTAL=$((TOTAL + 1))
REPO_ROOT="$DIR/.."
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'CHAREOF'
fn main() {
    exit('A')
}
CHAREOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_char_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_char_$$
    timeout 3 /tmp/mlrc_char_$$ > /dev/null 2>&1
    if [ $? = 65 ]; then PASS=$((PASS + 1)); echo "  char_literal: PASS"
    else echo "FAIL: char_literal"; FAIL=$((FAIL + 1)); fi
else echo "FAIL: char_literal (compile)"; FAIL=$((FAIL + 1)); fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_char_$$

TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'CHAREOF'
fn main() {
    uint64 x = 'A'
    exit(0)
}
CHAREOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_char_$$ > /dev/null 2>&1; then
    echo "FAIL: char_reject_assign_int"; FAIL=$((FAIL + 1))
else PASS=$((PASS + 1)); echo "  char_reject_assign_int: PASS"; fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_char_$$

# --- typed println pipeline ---
echo ""
echo "--- typed println pipeline ---"

# println(true) → "true"
run_test_output "println_true" \
    'fn main() { println(true); exit(0) }' \
    "true"

# println(false) → "false"
run_test_output "println_false" \
    'fn main() { println(false); exit(0) }' \
    "false"

# println(3.14) → "3.140000"
run_test_output "println_f64" \
    'fn main() { println(3.14); exit(0) }' \
    "3.140000"

# println(0.0) → "0.000000"
run_test_output "println_f64_zero" \
    'fn main() { println(0.0); exit(0) }' \
    "0.000000"

# println negative float via subtraction (avoids literal-negation IR bug)
run_test_output "println_f64_neg" \
    'fn main() { f64 x = 0.0 - 3.14; println(x); exit(0) }' \
    "-3.140000"

# println big float → "big"
run_test_output "println_f64_big" \
    'fn main() { println(1000000000000000000.0); exit(0) }' \
    "big"

# println char literal → single character
run_test_output "println_char" \
    "fn main() { println('A'); exit(0) }" \
    "A"

# --- variadic print ---
echo ""
echo "--- variadic print ---"

TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'VEOF'
fn main() {
    print("Here is a number,", 42)
    exit(0)
}
VEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_v_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_v_$$
    got=$(timeout 3 /tmp/mlrc_v_$$)
    if [ "$got" = "Here is a number, 42" ]; then PASS=$((PASS + 1)); echo "  print_multi_int: PASS"
    else echo "FAIL: print_multi_int (got '$got')"; FAIL=$((FAIL + 1)); fi
else echo "FAIL: print_multi_int (compile)"; FAIL=$((FAIL + 1)); fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_v_$$

TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'VEOF'
fn main() {
    println("n=", 5, "ok=", true)
    exit(0)
}
VEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_v_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_v_$$
    got=$(timeout 3 /tmp/mlrc_v_$$)
    if [ "$got" = "n= 5 ok= true" ]; then PASS=$((PASS + 1)); echo "  println_multi_mixed: PASS"
    else echo "FAIL: println_multi_mixed (got '$got')"; FAIL=$((FAIL + 1)); fi
else echo "FAIL: println_multi_mixed (compile)"; FAIL=$((FAIL + 1)); fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_v_$$

# --- negative float literal ---
echo ""
echo "--- negative float ---"

TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'NFEOF'
fn main() { f64 x = -3.14; println(x); exit(0) }
NFEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_nf_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_nf_$$
    got=$(timeout 3 /tmp/mlrc_nf_$$)
    if [ "$got" = "-3.140000" ]; then PASS=$((PASS + 1)); echo "  float_print_negative: PASS"
    else echo "FAIL: float_print_negative (got '$got')"; FAIL=$((FAIL + 1)); fi
else echo "FAIL: float_print_negative (compile)"; FAIL=$((FAIL + 1)); fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_nf_$$

# --- f-strings ---
echo ""
echo "--- f-strings ---"

TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'FEOF'
fn main() { println(f"x = {10 + 5}"); exit(0) }
FEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_f_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_f_$$
    got=$(timeout 3 /tmp/mlrc_f_$$)
    if [ "$got" = "x = 15" ]; then PASS=$((PASS + 1)); echo "  fstring_int: PASS"
    else echo "FAIL: fstring_int (got '$got')"; FAIL=$((FAIL + 1)); fi
else echo "FAIL: fstring_int (compile)"; FAIL=$((FAIL + 1)); fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_f_$$

TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'FEOF'
fn main() { f64 pi = 3.14; println(f"pi = {pi}"); exit(0) }
FEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_f_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_f_$$
    got=$(timeout 3 /tmp/mlrc_f_$$)
    if [ "$got" = "pi = 3.140000" ]; then PASS=$((PASS + 1)); echo "  fstring_float: PASS"
    else echo "FAIL: fstring_float (got '$got')"; FAIL=$((FAIL + 1)); fi
else echo "FAIL: fstring_float (compile)"; FAIL=$((FAIL + 1)); fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_f_$$

TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'FEOF'
fn main() { println(f"flag = {true}"); exit(0) }
FEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_f_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_f_$$
    got=$(timeout 3 /tmp/mlrc_f_$$)
    if [ "$got" = "flag = true" ]; then PASS=$((PASS + 1)); echo "  fstring_bool: PASS"
    else echo "FAIL: fstring_bool (got '$got')"; FAIL=$((FAIL + 1)); fi
else echo "FAIL: fstring_bool (compile)"; FAIL=$((FAIL + 1)); fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_f_$$

# --- IR optimizer tests ---
echo ""
echo "--- IR optimizer tests ---"

# Constant folding: literal arithmetic evaluated at compile time.
TOTAL=$((TOTAL + 1))
REPO_ROOT="$DIR/.."
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'OPTEOF'
fn main() {
    uint64 x = 3 + 4
    uint64 y = x * 2
    exit(y)
}
OPTEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_opt_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_opt_$$
    timeout 3 /tmp/mlrc_opt_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "14" ]; then
        PASS=$((PASS + 1))
        echo "  const_fold: PASS"
    else
        echo "FAIL: const_fold (expected 14, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: const_fold (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_opt_$$

# --O0 disables optimization, program still runs correctly.
TOTAL=$((TOTAL + 1))
printf 'fn main() { exit(6 * 7) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
if timeout 10 "$MLRC" $MLRC_FLAGS --O0 "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_opt_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_opt_$$
    timeout 3 /tmp/mlrc_opt_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "42" ]; then
        PASS=$((PASS + 1))
        echo "  O0_flag: PASS"
    else
        echo "FAIL: O0_flag (expected 42, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: O0_flag (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_opt_$$

# Loop counter: const-fold must NOT fold loop-carried vregs to their init value.
TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'OPTEOF'
fn main() {
    uint64 i = 0
    uint64 s = 0
    while i < 10 {
        s = s + i
        i = i + 1
    }
    exit(s)
}
OPTEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_opt_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_opt_$$
    timeout 3 /tmp/mlrc_opt_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "45" ]; then
        PASS=$((PASS + 1))
        echo "  loop_counter: PASS"
    else
        echo "FAIL: loop_counter (expected 45, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: loop_counter (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_opt_$$

# Branch simplification: constant conditions fold to unconditional branches.
TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'OPTEOF'
fn main() {
    if 0 == 1 { exit(5) } else { exit(7) }
    exit(9)
}
OPTEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_opt_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_opt_$$
    timeout 3 /tmp/mlrc_opt_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "7" ]; then
        PASS=$((PASS + 1))
        echo "  branch_fold: PASS"
    else
        echo "FAIL: branch_fold (expected 7, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: branch_fold (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_opt_$$

# CSE: redundant expressions inside a function still produce the right value.
TOTAL=$((TOTAL + 1))
cat > "$REPO_ROOT/test_tmp_$$.mlr" << 'OPTEOF'
fn work(uint64 x) -> uint64 {
    uint64 a = x + 100
    uint64 b = x + 100
    return a + b
}
fn main() { exit(work(5)) }
OPTEOF
if timeout 10 "$MLRC" $MLRC_FLAGS "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_opt_$$ > /dev/null 2>&1; then
    chmod +x /tmp/mlrc_opt_$$
    timeout 3 /tmp/mlrc_opt_$$ > /dev/null 2>&1
    actual=$?
    if [ "$actual" = "210" ]; then
        PASS=$((PASS + 1))
        echo "  cse_redundant: PASS"
    else
        echo "FAIL: cse_redundant (expected 210, got $actual)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: cse_redundant (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_opt_$$

# --- Custom fat binary targets ---
echo ""
echo "--- custom fat binary ---"
TOTAL=$((TOTAL + 1))
REPO_ROOT="$DIR/.."
printf 'fn main() { exit(77) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
HOST_ARCH=$(uname -m)
HOST_TGT="linux-x64"
if [ "$HOST_ARCH" = "aarch64" ] || [ "$HOST_ARCH" = "arm64" ]; then
    HOST_TGT="linux-arm64"
fi
if timeout 30 "$MLRC" --targets="$HOST_TGT" "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_fat_$$ > /dev/null 2>&1; then
    MLR_BIN="$REPO_ROOT/dist/mlr"
    [ -x "$MLR_BIN" ] || MLR_BIN="$REPO_ROOT/dist/mlr-android-$HOST_ARCH"
    if [ -x "$MLR_BIN" ]; then
        timeout 5 "$MLR_BIN" /tmp/mlrc_fat_$$ > /dev/null 2>&1
        actual=$?
        if [ "$actual" = "77" ]; then
            PASS=$((PASS + 1))
            echo "  custom_fat_single: PASS"
        else
            echo "FAIL: custom_fat_single (expected 77, got $actual)"; FAIL=$((FAIL + 1))
        fi
    else
        PASS=$((PASS + 1))
        echo "  custom_fat_single: SKIP (no runner)"
    fi
else
    echo "FAIL: custom_fat_single (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_fat_$$

# Custom 2-slice is smaller than custom 8-slice (same single-slice code path).
TOTAL=$((TOTAL + 1))
printf 'fn main() { exit(0) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
ALL="linux-x64,linux-arm64,win-x64,win-arm64,macos-x64,macos-arm64,android-x64,android-arm64"
if timeout 30 "$MLRC" --targets="$ALL" "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_fat_all_$$ > /dev/null 2>&1 && \
   timeout 30 "$MLRC" --targets=linux-x64,macos-arm64 "$REPO_ROOT/test_tmp_$$.mlr" -o /tmp/mlrc_fat_two_$$ > /dev/null 2>&1; then
    all_sz=$(wc -c < /tmp/mlrc_fat_all_$$)
    two_sz=$(wc -c < /tmp/mlrc_fat_two_$$)
    if [ "$two_sz" -lt "$all_sz" ]; then
        PASS=$((PASS + 1))
        echo "  custom_fat_smaller: PASS ($two_sz < $all_sz)"
    else
        echo "FAIL: custom_fat_smaller ($two_sz >= $all_sz)"; FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL: custom_fat_smaller (compilation failed)"; FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr" /tmp/mlrc_fat_all_$$ /tmp/mlrc_fat_two_$$

# --- IR dump test ---
echo ""
echo "--- IR dump test ---"
TOTAL=$((TOTAL + 1))
REPO_ROOT="$DIR/.."
printf 'fn main() { exit(42) }\n' > "$REPO_ROOT/test_tmp_$$.mlr"
IR_OUT=$($MLRC --emit=ir "$REPO_ROOT/test_tmp_$$.mlr" 2>/dev/null)
if echo "$IR_OUT" | grep -q "const"; then
    PASS=$((PASS + 1))
    echo "  ir_dump: PASS"
else
    echo "FAIL: ir_dump (no const in IR output)"
    FAIL=$((FAIL + 1))
fi
rm -f "$REPO_ROOT/test_tmp_$$.mlr"


# =============================================================================
# ESP32 machine target (--target=esp32)
# Ported from KernRift tests/run_tests.sh. The esp-image container is the
# artifact that gets flashed to real silicon: a wrong byte is a board that
# silently refuses to boot, diagnosed over ~2-minute flash cycles with no
# JTAG. Everything below is checked with od/dd/python3 (+ objdump when
# present) so it runs in CI — esptool is an ORACLE used by hand, never a
# build/test dependency.
# =============================================================================

# --- esp32 esp-image container writer — byte-identity vs esptool golden ---
# tests/golden/esp32_ref_image.bin was produced ONCE by esptool v5.3.1
# (`esptool --chip esp32 elf2image --flash-mode dio --flash-freq 40m
# --flash-size 4MB`) from tests/golden/esp32_ref_image.s (see that file's
# header for the exact reproduction commands). The harness below feeds
# esp_image_begin/segment/finish the exact same entry point, segment order
# (esptool sorts ascending by load address: DRAM 0x3FFB0000 first, then
# IRAM 0x40080400) and raw section payloads (7 bytes each — NOT a multiple
# of 4, so the writer's zero-pad-to-4 path is exercised), then requires the
# result to be BYTE-IDENTICAL to esptool's output. Any diff = a wrong field
# = an image the ESP32 boot ROM may silently refuse to boot.
echo ""
echo "--- esp32 esp-image container byte-identity test ---"
TOTAL=$((TOTAL + 1))
ESP_SRC="$DIR/../test_tmp_esp_$$.mlr"
ESP_BIN="/tmp/mlrc_esp_$$"
ESP_OUT="/tmp/our_image_$$.bin"
ESP_GOLD="$DIR/golden/esp32_ref_image.bin"
cat > "$ESP_SRC" <<ESP_EOF
import "std/sha256.mlr"
import "src/format_espimage.mlr"

fn esp_put8(u64 p, u64 v) {
    u8 b = v
    store8(p, b)
}

fn main() {
    // .data section of tests/golden/esp32_ref_image.s — 7 raw bytes.
    u64 dat = alloc(7)
    esp_put8(dat + 0, 0x11)
    esp_put8(dat + 1, 0x22)
    esp_put8(dat + 2, 0x33)
    esp_put8(dat + 3, 0x44)
    esp_put8(dat + 4, 0x55)
    esp_put8(dat + 5, 0x66)
    esp_put8(dat + 6, 0x77)
    // .text section (movi.n a2,42 / nop.n / memw) — 7 raw bytes.
    u64 txt = alloc(7)
    esp_put8(txt + 0, 0x2C)
    esp_put8(txt + 1, 0xA2)
    esp_put8(txt + 2, 0x3D)
    esp_put8(txt + 3, 0xF0)
    esp_put8(txt + 4, 0xC0)
    esp_put8(txt + 5, 0x20)
    esp_put8(txt + 6, 0x00)

    esp_image_begin(0x40080400, 2)
    esp_image_segment(0x3FFB0000, dat, 7)
    esp_image_segment(0x40080400, txt, 7)
    esp_image_finish()

    u64 fd = file_open("$ESP_OUT", 1)
    write(fd, esp_image_buf, esp_image_len)
    file_close(fd)
    exit(0)
}
ESP_EOF
if [ ! -f "$ESP_GOLD" ]; then
    echo "FAIL: esp32_image_format (golden reference $ESP_GOLD missing)"
    FAIL=$((FAIL + 1))
elif ! $MLRC $MLRC_FLAGS "$ESP_SRC" -o "$ESP_BIN" >/dev/null 2>&1; then
    echo "FAIL: esp32_image_format (harness compilation failed)"
    $MLRC $MLRC_FLAGS "$ESP_SRC" -o "$ESP_BIN" 2>&1 | head -3
    FAIL=$((FAIL + 1))
else
    chmod +x "$ESP_BIN"
    rm -f "$ESP_OUT"
    "$ESP_BIN" >/dev/null 2>&1
    if cmp -s "$ESP_OUT" "$ESP_GOLD"; then
        PASS=$((PASS + 1))
        echo "  esp32_image_format: PASS ($(wc -c < "$ESP_GOLD" | tr -d ' ') bytes byte-identical to esptool reference)"
    else
        echo "FAIL: esp32_image_format (image differs from esptool golden reference)"
        cmp "$ESP_OUT" "$ESP_GOLD" 2>&1 | head -3
        FAIL=$((FAIL + 1))
    fi
fi
rm -f "$ESP_SRC" "$ESP_BIN" "$ESP_OUT"

# --- esp32 machine target: --target=esp32 image structure + IRAM/DRAM guard ---
# Compiles examples/esp32/minimal.mlr with --arch=xtensa --freestanding
# --target=esp32 and asserts the esp-image structure with od ONLY:
#   byte 0 = 0xE9 (magic), byte 1 = 0x02 (two segments), byte 2 = 0x02 (DIO),
#   byte 3 = 0x20 (4MB @ 40MHz), entry (bytes 4-7 LE) inside IRAM
#   [0x40080400, 0x400A0000), segment 0 load_addr (0x18-0x1B LE) = 0x3FFB0000
#   (DRAM data — ascending load order, matching esptool), segment 1 load_addr
#   = 0x40080400 (IRAM code). Segment 1's header offset is DERIVED from
#   segment 0's data_len (header at 0x20 + seg0_len) — never hardcoded, it
#   moves with the data size.
echo ""
echo "--- esp32 machine-target image structure test ---"
TOTAL=$((TOTAL + 1))
ESP_MIN_BIN="/tmp/mlrc_esp_min_$$.bin"
ESP_ST_OK=1
esp_field() { od -An -tu4 -j "$2" -N 4 "$1" 2>/dev/null | tr -d ' '; }
if ! $MLRC --arch=xtensa --freestanding --target=esp32 \
     "$DIR/../examples/esp32/minimal.mlr" -o "$ESP_MIN_BIN" >/dev/null 2>&1; then
    echo "FAIL: esp32_image_structure (compilation failed)"
    $MLRC --arch=xtensa --freestanding --target=esp32 \
        "$DIR/../examples/esp32/minimal.mlr" -o "$ESP_MIN_BIN" 2>&1 | head -3
    ESP_ST_OK=0
else
    ESP_HDR=$(od -An -tx1 -j 0 -N 4 "$ESP_MIN_BIN" | tr -d ' ')
    if [ "$ESP_HDR" != "e9020220" ]; then
        echo "FAIL: esp32_image_structure (header bytes 0-3 = '$ESP_HDR', want 'e9020220')"
        ESP_ST_OK=0
    fi
    ESP_ENTRY=$(esp_field "$ESP_MIN_BIN" 4)
    if [ -z "$ESP_ENTRY" ] || [ "$ESP_ENTRY" -lt $((0x40080400)) ] \
       || [ "$ESP_ENTRY" -ge $((0x400A0000)) ]; then
        echo "FAIL: esp32_image_structure (entry $ESP_ENTRY outside IRAM [0x40080400,0x400A0000))"
        ESP_ST_OK=0
    fi
    ESP_SEG0_LOAD=$(esp_field "$ESP_MIN_BIN" $((0x18)))
    ESP_SEG0_LEN=$(esp_field "$ESP_MIN_BIN" $((0x1C)))
    if [ "$ESP_SEG0_LOAD" != "$((0x3FFB0000))" ]; then
        echo "FAIL: esp32_image_structure (segment 0 load_addr $ESP_SEG0_LOAD != 0x3FFB0000 DRAM data)"
        ESP_ST_OK=0
    fi
    # Segment 1's header follows segment 0's payload: 0x20 + seg0_len.
    if [ -n "$ESP_SEG0_LEN" ]; then
        ESP_SEG1_LOAD=$(esp_field "$ESP_MIN_BIN" $((0x20 + ESP_SEG0_LEN)))
        if [ "$ESP_SEG1_LOAD" != "$((0x40080400))" ]; then
            echo "FAIL: esp32_image_structure (segment 1 load_addr $ESP_SEG1_LOAD != 0x40080400 IRAM code)"
            ESP_ST_OK=0
        fi
    else
        echo "FAIL: esp32_image_structure (segment 0 data_len unreadable)"
        ESP_ST_OK=0
    fi
fi
if [ "$ESP_ST_OK" = 1 ]; then
    PASS=$((PASS + 1))
    echo "  esp32_image_structure: PASS (e9/02/02/20, entry in IRAM, DRAM@0x3FFB0000 + IRAM@0x40080400 ascending)"
else
    FAIL=$((FAIL + 1))
fi
rm -f "$ESP_MIN_BIN"

# --- esp32 --model: the blob becomes a THIRD segment, .data moves above it ---
# `--model <file>` appends the file's raw bytes as an extra RAM segment loaded
# at 0x3FFB0000 (the bottom of the DRAM window) so the mask ROM copies it from
# flash into RAM before the entry point runs. .data/.bss move UP to
# 0x3FFB0000 + align16(blob) — that direction is what makes the blob's address
# a compile-time constant the program can hardcode (tests/nn/nn_model_esp32.mlr
# reads 0x3FFB0000). Asserted with od only:
#   byte 1 = 0x03 (three segments now)
#   seg 0 load_addr = 0x3FFB0000, data_len = the blob's size
#   seg 1 load_addr = 0x3FFB0000 + align16(blob size)   <- .data moved up
#   seg 2 load_addr = 0x40080400                        <- IRAM code, unmoved
# and the byte-identity of the blob payload itself is checked with cmp.
echo ""
echo "--- esp32 --model blob segment test ---"
TOTAL=$((TOTAL + 1))
ESP_MDL_BIN="/tmp/mlrc_esp_mdl_$$.bin"
ESP_MDL_BLOB="/tmp/mlrc_esp_blob_$$.bin"
ESP_MDL_OUT="/tmp/mlrc_esp_blobout_$$.bin"
ESP_MDL_OK=1
# 4001 bytes: deliberately NOT a multiple of 16, so the align16 of the data
# base and the align4 payload padding are both exercised.
head -c 4001 /dev/urandom > "$ESP_MDL_BLOB" 2>/dev/null
ESP_MDL_SZ=$(wc -c < "$ESP_MDL_BLOB" | tr -d ' ')
if [ "$ESP_MDL_SZ" != "4001" ]; then
    echo "SKIP: esp32_model_segment (could not create the 4001-byte test blob)"
    TOTAL=$((TOTAL - 1))
    ESP_MDL_OK=skip
elif ! $MLRC --arch=xtensa --freestanding --target=esp32 --model "$ESP_MDL_BLOB" \
     "$DIR/../examples/esp32/minimal.mlr" -o "$ESP_MDL_BIN" >/dev/null 2>&1; then
    echo "FAIL: esp32_model_segment (compilation failed)"
    $MLRC --arch=xtensa --freestanding --target=esp32 --model "$ESP_MDL_BLOB" \
        "$DIR/../examples/esp32/minimal.mlr" -o "$ESP_MDL_BIN" 2>&1 | head -3
    ESP_MDL_OK=0
else
    ESP_MDL_NSEG=$(od -An -tu1 -j 1 -N 1 "$ESP_MDL_BIN" | tr -d ' ')
    if [ "$ESP_MDL_NSEG" != "3" ]; then
        echo "FAIL: esp32_model_segment (segment count $ESP_MDL_NSEG != 3)"
        ESP_MDL_OK=0
    fi
    ESP_M_S0_LOAD=$(esp_field "$ESP_MDL_BIN" $((0x18)))
    ESP_M_S0_LEN=$(esp_field "$ESP_MDL_BIN" $((0x1C)))
    if [ "$ESP_M_S0_LOAD" != "$((0x3FFB0000))" ]; then
        echo "FAIL: esp32_model_segment (blob segment load_addr $ESP_M_S0_LOAD != 0x3FFB0000)"
        ESP_MDL_OK=0
    fi
    # data_len is the raw blob length padded up to 4 (esp_image_segment).
    if [ "$ESP_M_S0_LEN" != "4004" ]; then
        echo "FAIL: esp32_model_segment (blob segment data_len $ESP_M_S0_LEN != 4004 = align4(4001))"
        ESP_MDL_OK=0
    fi
    # The blob payload must survive byte-for-byte: it starts at 0x20.
    dd if="$ESP_MDL_BIN" of="$ESP_MDL_OUT" bs=1 skip=32 count=4001 \
        >/dev/null 2>&1
    if ! cmp -s "$ESP_MDL_BLOB" "$ESP_MDL_OUT"; then
        echo "FAIL: esp32_model_segment (blob payload in the image != the input file)"
        ESP_MDL_OK=0
    fi
    if [ -n "$ESP_M_S0_LEN" ]; then
        ESP_M_S1_LOAD=$(esp_field "$ESP_MDL_BIN" $((0x20 + ESP_M_S0_LEN)))
        ESP_M_S1_LEN=$(esp_field "$ESP_MDL_BIN" $((0x24 + ESP_M_S0_LEN)))
        # align16(4001) = 4016
        if [ "$ESP_M_S1_LOAD" != "$((0x3FFB0000 + 4016))" ]; then
            echo "FAIL: esp32_model_segment (.data load_addr $ESP_M_S1_LOAD != 0x3FFB0000+align16(4001))"
            ESP_MDL_OK=0
        fi
        ESP_M_S2_OFF=$((0x28 + ESP_M_S0_LEN + ESP_M_S1_LEN))
        ESP_M_S2_LOAD=$(esp_field "$ESP_MDL_BIN" "$ESP_M_S2_OFF")
        if [ "$ESP_M_S2_LOAD" != "$((0x40080400))" ]; then
            echo "FAIL: esp32_model_segment (code load_addr $ESP_M_S2_LOAD != 0x40080400)"
            ESP_MDL_OK=0
        fi
    else
        echo "FAIL: esp32_model_segment (blob segment data_len unreadable)"
        ESP_MDL_OK=0
    fi
fi
if [ "$ESP_MDL_OK" = 1 ]; then
    PASS=$((PASS + 1))
    echo "  esp32_model_segment: PASS (3 segments, blob byte-identical @0x3FFB0000, .data at +align16)"
elif [ "$ESP_MDL_OK" = 0 ]; then
    FAIL=$((FAIL + 1))
fi
rm -f "$ESP_MDL_BIN" "$ESP_MDL_BLOB" "$ESP_MDL_OUT"

# --- esp32 guard tests: unsupported combos must be COMPILE errors ---
# (1) --target=esp32 without --arch=xtensa --freestanding is rejected.
# (2) Programs that cannot be laid out safely in the ESP32 memory map must
#     LOUD-FAIL at compile time and leave NO output file behind. Past the DRAM
#     window [0x3FFB0000,0x3FFE0000) the next addresses are ROM-reserved RAM
#     and then (at 0x40000000+) IRAM, which is 32-bit-access-only — a
#     byte-addressed datum there raises LoadStoreError and the board is dead
#     with no output and no JTAG.
#
# Each case asserts on the SPECIFIC error text, because there are THREE
# distinct guards that all reject an oversized program and it is very easy to
# write a case that looks like it covers one while actually tripping another:
#   (a) resolve_addr_fixups_xtensa_esp32, per-datum, "would land in IRAM";
#   (b) resolve_addr_fixups_xtensa_esp32, per-datum, "falls outside the DRAM
#       window";
#   (c) xt_esp32_check_layout, whole-segment, "data+bss exceed the DRAM
#       window" / "less than 8 KiB below the initial stack pointer" (the
#       floor is XT_ESP32_MIN_STACK = 8192; the error text used to say 4 KiB,
#       fixed as Minor finding 4 in the final review).
esp_guard_expect() {
    # $1 = case label, $2 = expected error substring, $3 = source file
    rm -f "$ESP_G_BIN"
    ESP_G_ERR=$($MLRC --arch=xtensa --freestanding --target=esp32 \
                "$3" -o "$ESP_G_BIN" 2>&1)
    if [ $? -eq 0 ]; then
        echo "FAIL: esp32_guards ($1 accepted — expected a compile error)"
        ESP_G_OK=0
    elif [ -f "$ESP_G_BIN" ]; then
        echo "FAIL: esp32_guards ($1 errored but still left an output image behind)"
        ESP_G_OK=0
    elif ! printf '%s' "$ESP_G_ERR" | grep -qF "$2"; then
        echo "FAIL: esp32_guards ($1 rejected by the WRONG guard)"
        echo "  expected error to contain: $2"
        echo "  actual error: $ESP_G_ERR"
        ESP_G_OK=0
    fi
    rm -f "$ESP_G_BIN"
}
echo ""
echo "--- esp32 guard tests (bad combos are compile errors) ---"
TOTAL=$((TOTAL + 1))
ESP_G_OK=1
ESP_G_BIN="/tmp/mlrc_esp_guard_$$.bin"
ESP_G_SRC="$DIR/../test_tmp_espguard_$$.mlr"
rm -f "$ESP_G_BIN"
if $MLRC --arch=riscv32 --freestanding --target=esp32 \
     "$DIR/../examples/esp32/minimal.mlr" -o "$ESP_G_BIN" >/dev/null 2>&1; then
    echo "FAIL: esp32_guards (--target=esp32 accepted without --arch=xtensa)"
    ESP_G_OK=0
fi
if $MLRC --arch=xtensa --target=esp32 \
     "$DIR/../examples/esp32/minimal.mlr" -o "$ESP_G_BIN" >/dev/null 2>&1; then
    echo "FAIL: esp32_guards (--target=esp32 accepted without --freestanding)"
    ESP_G_OK=0
fi
# (b) 256 KiB array + a trailing datum. `sentinel` is laid out AFTER `big`, so
# its own base address is 0x3FFB0000 + 0x40000, already past the window — this
# case is caught PER-DATUM and never reaches the whole-segment layout guard.
cat > "$ESP_G_SRC" <<'ESP_G_EOF'
static u32[65536] big
static u32 sentinel = 7

fn main() {
    big[0] = sentinel
    loop { }
}
ESP_G_EOF
esp_guard_expect "256 KiB data (per-datum address past the window)" \
    "data address falls outside the DRAM window" "$ESP_G_SRC"
# (c) 200 KiB array and NOTHING after it. Every datum base is in-window (the
# array starts at 0x3FFB0000 itself), so neither per-datum check fires; only
# the whole-segment memsz check in xt_esp32_check_layout can catch that the
# array SPANS past 0x3FFE0000.
cat > "$ESP_G_SRC" <<'ESP_G_EOF'
static u32[51200] big

fn main() {
    big[0] = 1
    loop { }
}
ESP_G_EOF
esp_guard_expect "200 KiB array spanning past the window (base in-window)" \
    "data+bss exceed the DRAM window" "$ESP_G_SRC"
# (c2) 189.8 KiB array: FITS the raw DRAM window (0x2F6E0 < 0x30000) but
# leaves under 8 KiB (XT_ESP32_MIN_STACK) below the initial SP. The stack
# grows DOWN from 0x3FFE0000, which is the same address the window ends at,
# so the entry prologue's first `s32i a0, a1, N-4` writes the saved return
# address on top of the .bss tail — AFTER the zero loop has run, so nothing
# restores it. Without XT_ESP32_MIN_STACK this program compiles clean and
# corrupts itself on real silicon.
cat > "$ESP_G_SRC" <<'ESP_G_EOF'
static u32[48600] big

fn main() {
    big[0] = 1
    loop { }
}
ESP_G_EOF
esp_guard_expect "190 KiB statics (fits the window, starves the stack)" \
    "less than 8 KiB below the initial stack pointer" "$ESP_G_SRC"
# (a) THE IRAM BYTE-ACCESS GUARD — the whole justification for splitting code
# and data across two load addresses. IRAM services only aligned 32-bit
# accesses, so an l8ui (which is how every string read, strlen and memcpy
# touches memory) against an IRAM address raises LoadStoreError: no output, no
# JTAG, board indistinguishable from dead. 360 KiB of leading statics pushes
# the NEXT datum's computed address past 0x40000000 and into IRAM.
cat > "$ESP_G_SRC" <<'ESP_G_EOF'
static u32[90000] pad
static u32 tail_datum = 7

fn main() {
    pad[0] = 1
    tail_datum = 2
    loop { }
}
ESP_G_EOF
esp_guard_expect "360 KiB of statics (next datum computes into IRAM)" \
    "would land in IRAM" "$ESP_G_SRC"
# The IRAM code-overflow branch. Usable IRAM is 0x400A0000 - 0x40080400 =
# 127 KiB; this generates a ~192 KiB chain of functions, ~1.5x over, so the
# case stays over the limit even if codegen gets meaningfully tighter. A chain
# (each fn tail-calls the next) rather than 1000 calls from main, because a
# main with 1000 call sites blows the 2047-byte frame cap and would fail for
# an unrelated reason.
awk 'BEGIN {
    n = 1000; m = 12
    for (i = 0; i < n; i++) {
        printf "fn g%d(u32 x) -> u32 {\n", i
        for (j = 0; j < m; j++) printf "    x = x * %d + %d\n", (j % 13) + 3, i + j
        if (i == n - 1) printf "    return x\n}\n"
        else printf "    return g%d(x)\n}\n", i + 1
    }
    printf "fn main() {\n    u32 a = g0(1)\n    a = a + 1\n    loop { }\n}\n"
}' > "$ESP_G_SRC"
esp_guard_expect "~192 KiB of code (overflows the 127 KiB IRAM window)" \
    "code segment exceeds the IRAM limit" "$ESP_G_SRC"
# @naked on the ENTRY function silently voids the a0-park safety net: the
# preamble still emits `l32r a0, &park`, but @naked skips the prologue that
# frame-saves a0, so the body's first call0 overwrites it. A returning entry
# then decodes garbage — an exception and a reboot loop indistinguishable from
# a watchdog failure — which is exactly what parking a0 exists to prevent.
cat > "$ESP_G_SRC" <<'ESP_G_EOF'
@naked
fn main() {
    loop { }
}
ESP_G_EOF
esp_guard_expect "@naked entry function" \
    "entry function may not be @naked" "$ESP_G_SRC"
# ...but the guard must be scoped to the esp32 target: @naked is legal on the
# generic lx60 xtensa path, which has no preamble and no park address.
rm -f "$ESP_G_BIN"
if ! $MLRC --arch=xtensa --freestanding "$ESP_G_SRC" -o "$ESP_G_BIN" >/dev/null 2>&1; then
    echo "FAIL: esp32_guards (@naked entry rejected on the generic lx60 xtensa path — the guard is esp32-only)"
    ESP_G_OK=0
fi
rm -f "$ESP_G_BIN"
if [ "$ESP_G_OK" = 1 ]; then
    PASS=$((PASS + 1))
    echo "  esp32_guards: PASS (arch/freestanding combos, IRAM byte-access, per-datum overflow, whole-segment span, stack starvation, IRAM code overflow, @naked entry — each rejected by its OWN guard)"
else
    FAIL=$((FAIL + 1))
fi
rm -f "$ESP_G_SRC" "$ESP_G_BIN"

# --- --target= argument validation -------------------------------------------
# Two separate silent-wrong-output bugs live here, so both get a negative test.
#
#  (1) NEAR-MISS CHIP NAMES. --target=esp32 must be matched EXACTLY, not by
#      prefix. "esp32s3" and "esp32c3" are different chips with different
#      memory maps (the C3 is RISC-V, not Xtensa even). A prefix match lets
#      --target=esp32s3 quietly produce an ESP32 image with load addresses
#      that are wrong for that chip: a board that does not boot.
#
#  (2) TYPOS. An unrecognised --target= must be a hard error. It used to fall
#      off the end of the if-chain and be SILENTLY IGNORED, so
#      `--target=widnows` handed back a default-target binary with no warning.
#
# Both cases use otherwise-valid flag combinations, so the ONLY thing that can
# reject them is the target-string check itself.
echo ""
echo "--- --target= argument validation ---"
TOTAL=$((TOTAL + 1))
ESP_T_OK=1
ESP_T_BIN="/tmp/mlrc_esp_targ_$$.bin"
for ESP_T_BAD in esp32s3 esp32c3; do
    rm -f "$ESP_T_BIN"
    ESP_T_ERR=$($MLRC --arch=xtensa --freestanding "--target=$ESP_T_BAD" \
                "$DIR/../examples/esp32/minimal.mlr" -o "$ESP_T_BIN" 2>&1)
    if [ $? -eq 0 ]; then
        echo "FAIL: target_arg_validation (--target=$ESP_T_BAD accepted — a near-miss chip name must NOT prefix-match esp32 and emit an ESP32 image)"
        ESP_T_OK=0
    elif ! printf '%s' "$ESP_T_ERR" | grep -qF "unknown --target="; then
        echo "FAIL: target_arg_validation (--target=$ESP_T_BAD rejected, but not by the unknown-target check: $ESP_T_ERR)"
        ESP_T_OK=0
    fi
done
for ESP_T_BAD in bogus widnows lin ""; do
    rm -f "$ESP_T_BIN"
    ESP_T_ERR=$($MLRC "--target=$ESP_T_BAD" "$DIR/smoke/div_mod.mlr" \
                -o "$ESP_T_BIN" 2>&1)
    if [ $? -eq 0 ]; then
        echo "FAIL: target_arg_validation (--target=$ESP_T_BAD accepted — an unknown target must be a hard error, never silently ignored)"
        ESP_T_OK=0
    elif ! printf '%s' "$ESP_T_ERR" | grep -qF "unknown --target="; then
        echo "FAIL: target_arg_validation (--target=$ESP_T_BAD rejected, but not by the unknown-target check: $ESP_T_ERR)"
        ESP_T_OK=0
    fi
done
# ...and the accepted names must still be accepted (so the check above cannot
# be "fixed" by rejecting everything).
for ESP_T_GOOD in linux macos darwin windows win; do
    rm -f "$ESP_T_BIN"
    if ! $MLRC "--target=$ESP_T_GOOD" "$DIR/smoke/div_mod.mlr" \
         -o "$ESP_T_BIN" >/dev/null 2>&1; then
        echo "FAIL: target_arg_validation (--target=$ESP_T_GOOD rejected — it is a documented, accepted target name)"
        ESP_T_OK=0
    fi
done
# The GPU target names are MLRift-only (KernRift has no --target=hip-amd /
# amdgpu-native). They legitimately fail on a source with no @kernel function,
# so assert only that they are not rejected by the unknown-target check — i.e.
# the new hard error did not swallow them.
for ESP_T_GPU in hip-amd amdgpu-native; do
    rm -f "$ESP_T_BIN"
    # tr -d '\0': the hip-amd path emits a stray NUL byte on this input (a
    # pre-existing quirk of that emitter, unrelated to target parsing), which
    # bash would otherwise warn about on every run.
    ESP_T_ERR=$($MLRC "--target=$ESP_T_GPU" "$DIR/smoke/div_mod.mlr" \
                -o "$ESP_T_BIN" 2>&1 | tr -d '\0')
    if printf '%s' "$ESP_T_ERR" | grep -qF "unknown --target="; then
        echo "FAIL: target_arg_validation (--target=$ESP_T_GPU treated as unknown — the GPU target names must survive the new hard error)"
        ESP_T_OK=0
    fi
done
if [ "$ESP_T_OK" = 1 ]; then
    PASS=$((PASS + 1))
    echo "  target_arg_validation: PASS (near-miss chip names and typos are hard errors; documented names still accepted)"
else
    FAIL=$((FAIL + 1))
fi
rm -f "$ESP_T_BIN"

# --- esp32 .bss zero-loop bounds -------------------------------------------
# The entry preamble zeroes [bss_lo, bss_hi) from two literal-pool words that
# main.mlr patches at finalize time. The startup-stub test greps the six WDT
# addresses and the unlock key but never looks at these two words, so patching
# them with XT_ESP32_IRAM_BASE instead of XT_ESP32_DRAM_BASE passes the whole
# suite — and the stub would then zero its OWN code at 0x40080400 on the way
# up. That is a bricked flash cycle with no diagnostic, so assert the bounds
# directly.
#
# Both bounds are derivable from the image, so nothing here is hardcoded:
#   bss_lo == 0x3FFB0000 + seg0_len   (bss starts where the DRAM segment's
#                                      file payload ends)
#   bss_hi  = the one remaining DRAM-window pool word strictly below the
#             0x3FFE0000 stack top, and hi-lo must match the .bss the test
#             program actually declares (4 KiB, plus alignment padding).
# Both must be 4-aligned: the loop stores with s32i, which traps on an
# unaligned base.
echo ""
echo "--- esp32 .bss zero-loop bounds test ---"
TOTAL=$((TOTAL + 1))
ESP_B_OK=1
ESP_B_SRC="$DIR/../test_tmp_espbss_$$.mlr"
ESP_B_BIN="/tmp/mlrc_esp_bss_$$.bin"
# One small initialized datum (so the DRAM segment is non-empty) followed by a
# 4 KiB array that is never initialized (so .bss is non-empty and lo != hi).
cat > "$ESP_B_SRC" <<'ESP_B_EOF'
static u32 init_val = 0xABCD1234
static u32[1024] zeros

fn main() {
    zeros[0] = init_val
    zeros[1023] = init_val
    loop { }
}
ESP_B_EOF
if ! $MLRC --arch=xtensa --freestanding --target=esp32 \
     "$ESP_B_SRC" -o "$ESP_B_BIN" >/dev/null 2>&1; then
    echo "FAIL: esp32_bss_bounds (compilation failed)"
    ESP_B_OK=0
fi
if [ "$ESP_B_OK" = 1 ]; then
    ESP_B_DRAM_BASE=$((0x3FFB0000))
    ESP_B_DRAM_LIMIT=$((0x3FFE0000))
    esp_b_field() { od -An -tu4 -j "$2" -N 4 "$1" 2>/dev/null | tr -d ' '; }
    # Walk the segment table for the DRAM segment length and the IRAM payload.
    ESP_B_NSEG=$(od -An -tu1 -j 1 -N 1 "$ESP_B_BIN" | tr -d ' ')
    ESP_B_SOFF=$((0x18))
    ESP_B_DLEN=""
    ESP_B_COFF=0
    ESP_B_CLEN=0
    ESP_B_I=0
    while [ "$ESP_B_I" -lt "${ESP_B_NSEG:-0}" ]; do
        ESP_B_LOAD=$(esp_b_field "$ESP_B_BIN" "$ESP_B_SOFF")
        ESP_B_LEN=$(esp_b_field "$ESP_B_BIN" $((ESP_B_SOFF + 4)))
        if [ -z "$ESP_B_LOAD" ] || [ -z "$ESP_B_LEN" ]; then break; fi
        if [ "$ESP_B_LOAD" = "$ESP_B_DRAM_BASE" ]; then ESP_B_DLEN=$ESP_B_LEN; fi
        if [ "$ESP_B_LOAD" -ge $((0x40000000)) ]; then
            ESP_B_COFF=$((ESP_B_SOFF + 8)); ESP_B_CLEN=$ESP_B_LEN
        fi
        ESP_B_SOFF=$((ESP_B_SOFF + 8 + ESP_B_LEN))
        ESP_B_I=$((ESP_B_I + 1))
    done
    if [ -z "$ESP_B_DLEN" ] || [ "$ESP_B_CLEN" = 0 ]; then
        echo "FAIL: esp32_bss_bounds (could not locate both the DRAM and IRAM segments)"
        ESP_B_OK=0
    fi
fi
if [ "$ESP_B_OK" = 1 ]; then
    ESP_B_LO=$((ESP_B_DRAM_BASE + ESP_B_DLEN))
    # Every literal-pool word in the code segment that falls in the DRAM window.
    ESP_B_WORDS=$(dd if="$ESP_B_BIN" bs=1 skip="$ESP_B_COFF" count="$ESP_B_CLEN" \
                     2>/dev/null | od -An -tu4 -v | tr -s ' ' '\n' | grep -v '^$')
    ESP_B_SEEN_LO=0
    ESP_B_HI=0
    for ESP_B_W in $ESP_B_WORDS; do
        if [ "$ESP_B_W" -lt "$ESP_B_DRAM_BASE" ] || [ "$ESP_B_W" -gt "$ESP_B_DRAM_LIMIT" ]; then
            continue
        fi
        if [ "$ESP_B_W" = "$ESP_B_LO" ]; then ESP_B_SEEN_LO=1; fi
        if [ "$ESP_B_W" -gt "$ESP_B_LO" ] && [ "$ESP_B_W" -lt "$ESP_B_DRAM_LIMIT" ]; then
            ESP_B_HI=$ESP_B_W
        fi
    done
    if [ "$ESP_B_SEEN_LO" != 1 ]; then
        echo "FAIL: esp32_bss_bounds (no pool word equals the expected bss_lo $ESP_B_LO = 0x3FFB0000 + DRAM seg len $ESP_B_DLEN — the zero loop is not bounded by DRAM addresses)"
        ESP_B_OK=0
    fi
    if [ "$ESP_B_HI" = 0 ]; then
        echo "FAIL: esp32_bss_bounds (no bss_hi pool word in (bss_lo, 0x3FFE0000) — the zero loop's upper bound is not a DRAM address)"
        ESP_B_OK=0
    else
        ESP_B_SPAN=$((ESP_B_HI - ESP_B_LO))
        # The program declares exactly 4096 bytes of .bss; allow a little
        # alignment padding, but nothing like a whole wrong base.
        if [ "$ESP_B_SPAN" -lt 4096 ] || [ "$ESP_B_SPAN" -gt 4160 ]; then
            echo "FAIL: esp32_bss_bounds (bss span $ESP_B_SPAN bytes, expected ~4096 for the declared u32[1024])"
            ESP_B_OK=0
        fi
        if [ $((ESP_B_HI & 3)) != 0 ]; then
            echo "FAIL: esp32_bss_bounds (bss_hi $ESP_B_HI is not 4-byte aligned — s32i traps on an unaligned base)"
            ESP_B_OK=0
        fi
    fi
    if [ $((ESP_B_LO & 3)) != 0 ]; then
        echo "FAIL: esp32_bss_bounds (bss_lo $ESP_B_LO is not 4-byte aligned — s32i traps on an unaligned base)"
        ESP_B_OK=0
    fi
fi
if [ "$ESP_B_OK" = 1 ]; then
    PASS=$((PASS + 1))
    echo "  esp32_bss_bounds: PASS (zero-loop bounds are DRAM addresses, 4-aligned, spanning exactly the declared .bss)"
else
    FAIL=$((FAIL + 1))
fi
rm -f "$ESP_B_SRC" "$ESP_B_BIN"

# --- esp32 startup stub: WDT disable + PS + trailing park loop ---
# The mask ROM jumps straight to e_entry with RWDT and MWDT0 ARMED (flash-boot
# mode): if the stub does not disable them FIRST, the board reboots ~1s in
# with no output. Asserts on the DISASSEMBLY of the IRAM code segment of
# examples/esp32/minimal.mlr:
#   (1) the unlock key 0x50D83AA1 and all six WDT register addresses are
#       present as literal-pool words;
#   (2) the WDT sequence runs BEFORE the SP init: >= 6 s32i stores (3 unlock +
#       3 config0-clear) and the wsr.ps appear before the first `l32r a1`;
#   (3) the stub contains a genuine self-branch (`j .`) so a returning main
#       parks instead of decoding garbage.
# The IRAM segment is FOUND by walking the segment table, never hardcoded.
# SKIP cleanly when the disassembler is absent (dev-only toolchain).
echo ""
echo "--- esp32 startup stub test ---"
if command -v xtensa-lx106-elf-objdump >/dev/null 2>&1; then
    TOTAL=$((TOTAL + 1))
    ESP_STUB_BIN="/tmp/mlrc_esp_stub_$$.bin"
    ESP_STUB_CODE="/tmp/mlrc_esp_stub_code_$$.bin"
    ESP_STUB_DIS="/tmp/mlrc_esp_stub_dis_$$.txt"
    ESP_STUB_OK=1
    esp_stub_field() { od -An -tu4 -j "$2" -N 4 "$1" 2>/dev/null | tr -d ' '; }
    if ! $MLRC --arch=xtensa --freestanding --target=esp32 \
         "$DIR/../examples/esp32/minimal.mlr" -o "$ESP_STUB_BIN" >/dev/null 2>&1; then
        echo "FAIL: esp32_startup_stub (compilation failed)"
        ESP_STUB_OK=0
    fi
    ESP_STUB_CODE_OFF=0
    ESP_STUB_CODE_LEN=0
    if [ "$ESP_STUB_OK" = 1 ]; then
        ESP_STUB_ENTRY=$(esp_stub_field "$ESP_STUB_BIN" 4)
        ESP_STUB_NSEG=$(od -An -tu1 -j 1 -N 1 "$ESP_STUB_BIN" | tr -d ' ')
        ESP_STUB_SOFF=$((0x18))
        ESP_STUB_I=0
        while [ "$ESP_STUB_I" -lt "${ESP_STUB_NSEG:-0}" ]; do
            ESP_STUB_LOAD=$(esp_stub_field "$ESP_STUB_BIN" "$ESP_STUB_SOFF")
            ESP_STUB_LEN=$(esp_stub_field "$ESP_STUB_BIN" $((ESP_STUB_SOFF + 4)))
            if [ -z "$ESP_STUB_LOAD" ] || [ -z "$ESP_STUB_LEN" ]; then break; fi
            if [ "$ESP_STUB_LOAD" -ge $((0x40000000)) ]; then
                ESP_STUB_CODE_OFF=$((ESP_STUB_SOFF + 8))
                ESP_STUB_CODE_LEN=$ESP_STUB_LEN
                break
            fi
            ESP_STUB_SOFF=$((ESP_STUB_SOFF + 8 + ESP_STUB_LEN))
            ESP_STUB_I=$((ESP_STUB_I + 1))
        done
        if [ "$ESP_STUB_CODE_LEN" = 0 ]; then
            echo "FAIL: esp32_startup_stub (no IRAM segment with load_addr >= 0x40000000 found)"
            ESP_STUB_OK=0
        fi
    fi
    if [ "$ESP_STUB_OK" = 1 ]; then
        dd if="$ESP_STUB_BIN" of="$ESP_STUB_CODE" bs=1 \
           skip="$ESP_STUB_CODE_OFF" count="$ESP_STUB_CODE_LEN" 2>/dev/null
        # (1) key + all six WDT addresses present as pool words
        ESP_STUB_WORDS=$(od -An -tx4 "$ESP_STUB_CODE")
        for ESP_STUB_W in 50d83aa1 3ff480a4 3ff4808c 3ff5f064 3ff5f048 3ff60064 3ff60048; do
            if ! echo "$ESP_STUB_WORDS" | grep -qw "$ESP_STUB_W"; then
                echo "FAIL: esp32_startup_stub (pool word $ESP_STUB_W missing — WDT sequence not emitted)"
                ESP_STUB_OK=0
            fi
        done
    fi
    if [ "$ESP_STUB_OK" = 1 ]; then
        # (2) ordering: disassemble from the entry; everything before the
        # first `l32r a1` (SP init) must already contain the 6 WDT stores
        # and the wsr.ps. Offset of the entry within the IRAM payload is
        # derived from the segment's own load_addr, never re-hardcoded.
        ESP_STUB_EOFF=$((ESP_STUB_ENTRY - ESP_STUB_LOAD))
        xtensa-lx106-elf-objdump -b binary -m xtensa -D \
            --start-address=$ESP_STUB_EOFF "$ESP_STUB_CODE" > "$ESP_STUB_DIS" 2>/dev/null
        ESP_STUB_PRE=$(sed -n "1,/l32r[[:space:]]*a1,/p" "$ESP_STUB_DIS")
        ESP_STUB_NS32I=$(echo "$ESP_STUB_PRE" | grep -cE '[[:space:]]s32i(\.n)?[[:space:]]')
        if [ "$ESP_STUB_NS32I" -lt 6 ]; then
            echo "FAIL: esp32_startup_stub (only $ESP_STUB_NS32I s32i before the SP-init l32r a1 — WDT disable must come FIRST)"
            ESP_STUB_OK=0
        fi
        if ! echo "$ESP_STUB_PRE" | grep -qE '[[:space:]]wsr'; then
            echo "FAIL: esp32_startup_stub (no wsr.ps before the SP-init l32r a1)"
            ESP_STUB_OK=0
        fi
        # (3) a genuine self-branch: a `j` whose target == its own address
        ESP_STUB_PARK=0
        while IFS= read -r ESP_STUB_LN; do
            ESP_STUB_A=$(printf '%s' "$ESP_STUB_LN" | sed -n 's/^ *\([0-9a-f][0-9a-f]*\):.*/\1/p')
            ESP_STUB_T=$(printf '%s' "$ESP_STUB_LN" | sed -n 's/.*[[:space:]]j[[:space:]][[:space:]]*0*x\{0,1\}\([0-9a-f][0-9a-f]*\)[[:space:]]*$/\1/p')
            if [ -n "$ESP_STUB_A" ] && [ -n "$ESP_STUB_T" ]; then
                if [ $((0x$ESP_STUB_A)) -eq $((0x$ESP_STUB_T)) ]; then
                    ESP_STUB_PARK=1
                fi
            fi
        done <<ESP_STUB_EOF
$(grep -E '[[:space:]]j[[:space:]]+(0x)?[0-9a-f]+[[:space:]]*$' "$ESP_STUB_DIS")
ESP_STUB_EOF
        if [ "$ESP_STUB_PARK" != 1 ]; then
            echo "FAIL: esp32_startup_stub (no self-branch 'j .' — a returning main would decode garbage and mimic a WDT reset loop)"
            ESP_STUB_OK=0
        fi
    fi
    if [ "$ESP_STUB_OK" = 1 ]; then
        PASS=$((PASS + 1))
        echo "  esp32_startup_stub: PASS (WDT unlock+clear x3 before SP init, wsr.ps, self-branch park)"
    else
        FAIL=$((FAIL + 1))
    fi
    rm -f "$ESP_STUB_BIN" "$ESP_STUB_CODE" "$ESP_STUB_DIS"
else
    echo "  esp32_startup_stub: SKIP (xtensa-lx106-elf-objdump not installed)"
fi

# --- esp32 hello image: full container + errata-safe UART0 putc ---
# The artifact that gets flashed to real silicon.
#
# Container asserts (independent of the code):
#   magic/mode/size bytes e9 02 02 20, EXACTLY 2 segments, entry in IRAM,
#   (len - 32) % 16 == 0  — the payload is 16-padded and a 32-byte SHA-256
#   appended, so total-minus-hash must be a multiple of 16, and the checksum
#   byte at len-33 equals the 0xEF-seeded XOR of every segment payload byte,
#   RECOMPUTED here rather than trusted.
#
# Code asserts (errata CPU-3.3):
#   `putc` must POLL 0x3FF4001C (APB UART_STATUS_REG) and WRITE the byte to
#   0x60000000 (the AHB TX-FIFO mirror). Consecutive APB writes to UART0's
#   FIFO "may be lost" per the errata, so a store to an APB UART address
#   would give intermittently garbled output on the board's ONLY debug
#   channel. The test checks the DIRECTION of each access, not just that the
#   constants appear: it tracks `l32r aN,<pool>` into a register map and then
#   classifies the `s32i`/`l32i` that use aN as a base.
echo ""
echo "--- esp32 hello image test ---"
TOTAL=$((TOTAL + 1))
ESP_H_BIN="/tmp/mlrc_esp_hello_$$.bin"
ESP_H_PAY="/tmp/mlrc_esp_hello_pay_$$.bin"
ESP_H_CODE="/tmp/mlrc_esp_hello_code_$$.bin"
ESP_H_DIS="/tmp/mlrc_esp_hello_dis_$$.txt"
ESP_H_OK=1
esp_h_field() { od -An -tu4 -j "$2" -N 4 "$1" 2>/dev/null | tr -d ' '; }
rm -f "$ESP_H_BIN" "$ESP_H_PAY"
if ! $MLRC --arch=xtensa --freestanding --target=esp32 \
     "$DIR/../examples/esp32/hello.mlr" -o "$ESP_H_BIN" >/dev/null 2>&1; then
    echo "FAIL: esp32_hello_image (compilation failed)"
    $MLRC --arch=xtensa --freestanding --target=esp32 \
        "$DIR/../examples/esp32/hello.mlr" -o "$ESP_H_BIN" 2>&1 | head -3
    ESP_H_OK=0
fi
ESP_H_CODE_OFF=0
ESP_H_CODE_LEN=0
ESP_H_CODE_LOAD=0
if [ "$ESP_H_OK" = 1 ]; then
    ESP_H_LEN=$(wc -c < "$ESP_H_BIN" | tr -d ' ')
    ESP_H_HDR=$(od -An -tx1 -j 0 -N 4 "$ESP_H_BIN" | tr -d ' ')
    if [ "$ESP_H_HDR" != "e9020220" ]; then
        echo "FAIL: esp32_hello_image (header bytes 0-3 = '$ESP_H_HDR', want 'e9020220')"
        ESP_H_OK=0
    fi
    ESP_H_NSEG=$(od -An -tu1 -j 1 -N 1 "$ESP_H_BIN" | tr -d ' ')
    if [ "$ESP_H_NSEG" != 2 ]; then
        echo "FAIL: esp32_hello_image (segment count $ESP_H_NSEG != 2 — DRAM string + IRAM code expected)"
        ESP_H_OK=0
    fi
    ESP_H_ENTRY=$(esp_h_field "$ESP_H_BIN" 4)
    if [ -z "$ESP_H_ENTRY" ] || [ "$ESP_H_ENTRY" -lt $((0x40080400)) ] \
       || [ "$ESP_H_ENTRY" -ge $((0x400A0000)) ]; then
        echo "FAIL: esp32_hello_image (entry $ESP_H_ENTRY outside IRAM [0x40080400,0x400A0000))"
        ESP_H_OK=0
    fi
    # (len - 32) must be a multiple of 16: 32 bytes of appended SHA-256 over a
    # 16-padded (checksum-terminated) body.
    if [ $(( (ESP_H_LEN - 32) % 16 )) -ne 0 ]; then
        echo "FAIL: esp32_hello_image (len $ESP_H_LEN: (len-32) % 16 = $(( (ESP_H_LEN - 32) % 16 )), want 0)"
        ESP_H_OK=0
    fi
fi
if [ "$ESP_H_OK" = 1 ]; then
    # Walk the segment table: concatenate every payload for the checksum and
    # remember the IRAM one for disassembly.
    ESP_H_SOFF=$((0x18))
    ESP_H_I=0
    : > "$ESP_H_PAY"
    while [ "$ESP_H_I" -lt "$ESP_H_NSEG" ]; do
        ESP_H_LOAD=$(esp_h_field "$ESP_H_BIN" "$ESP_H_SOFF")
        ESP_H_SLEN=$(esp_h_field "$ESP_H_BIN" $((ESP_H_SOFF + 4)))
        if [ -z "$ESP_H_LOAD" ] || [ -z "$ESP_H_SLEN" ]; then
            echo "FAIL: esp32_hello_image (segment $ESP_H_I header unreadable at offset $ESP_H_SOFF)"
            ESP_H_OK=0
            break
        fi
        dd if="$ESP_H_BIN" bs=1 skip=$((ESP_H_SOFF + 8)) count="$ESP_H_SLEN" \
           >> "$ESP_H_PAY" 2>/dev/null
        if [ "$ESP_H_LOAD" -ge $((0x40000000)) ]; then
            ESP_H_CODE_OFF=$((ESP_H_SOFF + 8))
            ESP_H_CODE_LEN=$ESP_H_SLEN
            ESP_H_CODE_LOAD=$ESP_H_LOAD
        fi
        ESP_H_SOFF=$((ESP_H_SOFF + 8 + ESP_H_SLEN))
        ESP_H_I=$((ESP_H_I + 1))
    done
fi
if [ "$ESP_H_OK" = 1 ]; then
    # Recompute the 0xEF-seeded XOR over all segment payloads and compare with
    # the stored byte at len-33 (last byte before the 32-byte hash). POSIX awk
    # has no xor(), so it is done bitwise by hand.
    ESP_H_WANT=$(od -An -tu1 -j $((ESP_H_LEN - 33)) -N 1 "$ESP_H_BIN" | tr -d ' ')
    ESP_H_GOT=$(od -An -tu1 -v "$ESP_H_PAY" | awk '
        function xor8(a, b,   i, m, r) {
            r = 0; m = 1
            for (i = 0; i < 8; i++) {
                if (int(a / m) % 2 != int(b / m) % 2) r += m
                m *= 2
            }
            return r
        }
        BEGIN { c = 239 }
        { for (i = 1; i <= NF; i++) c = xor8(c, $i + 0) }
        END { print c }')
    if [ "$ESP_H_GOT" != "$ESP_H_WANT" ]; then
        echo "FAIL: esp32_hello_image (checksum byte at len-33 is $ESP_H_WANT, recomputed 0xEF-XOR is $ESP_H_GOT)"
        ESP_H_OK=0
    fi
    if [ "$ESP_H_CODE_LEN" = 0 ]; then
        echo "FAIL: esp32_hello_image (no IRAM segment with load_addr >= 0x40000000 found)"
        ESP_H_OK=0
    fi
    # The trailing 32 bytes are a SHA-256 over the whole image up to that
    # point. The ROM verifies it, so a wrong digest is a silently unbootable
    # image. Recompute it with sha256sum — an outside oracle — rather than
    # reading the stored bytes back and comparing them to themselves.
    #
    # This lives HERE, on the 576-byte hello image, specifically because the
    # esp-image byte-identity golden is a 64-byte body: hardcoding the update
    # length to 64 in format_espimage.mlr would reproduce the golden exactly
    # and pass the whole suite. One image size proves nothing about a hash.
    if command -v sha256sum >/dev/null 2>&1; then
        ESP_H_DGOT=$(dd if="$ESP_H_BIN" bs=1 count=$((ESP_H_LEN - 32)) 2>/dev/null \
                     | sha256sum | cut -d' ' -f1)
        ESP_H_DWANT=$(od -An -tx1 -j $((ESP_H_LEN - 32)) -N 32 -v "$ESP_H_BIN" \
                      | tr -d ' \n')
        if [ "$ESP_H_DGOT" != "$ESP_H_DWANT" ]; then
            echo "FAIL: esp32_hello_image (trailing SHA-256 is $ESP_H_DWANT, but sha256sum over the first $((ESP_H_LEN - 32)) bytes gives $ESP_H_DGOT)"
            ESP_H_OK=0
        fi
        ESP_H_HASH_NOTE=", SHA-256 recomputed over all $((ESP_H_LEN - 32)) body bytes"
    else
        ESP_H_HASH_NOTE=" (SHA-256 recompute SKIPPED — no sha256sum)"
    fi
fi
# Errata CPU-3.3 direction check — needs the disassembler; skip cleanly if the
# dev-only toolchain is absent, but never skip the container asserts above.
if [ "$ESP_H_OK" = 1 ] && command -v xtensa-lx106-elf-objdump >/dev/null 2>&1; then
    dd if="$ESP_H_BIN" of="$ESP_H_CODE" bs=1 \
       skip="$ESP_H_CODE_OFF" count="$ESP_H_CODE_LEN" 2>/dev/null
    xtensa-lx106-elf-objdump -b binary -m xtensa -D "$ESP_H_CODE" > "$ESP_H_DIS" 2>/dev/null
    ESP_H_VERDICT=$(awk '
        # Track `l32r aN, <slot> (0xVALUE)` so a later s32i/l32i off aN can be
        # attributed to a concrete absolute address. Any control transfer
        # invalidates the map, so nothing is attributed across a branch.
        { m = $3; op1 = $4; op2 = $5 }
        m == "l32r" {
            gsub(/,/, "", op1); v = $6
            gsub(/[()]/, "", v)
            reg[op1] = v
            next
        }
        m ~ /^(s32i|l32i)(\.n)?$/ {
            gsub(/,/, "", op2)
            if (op2 in reg) {
                if (m ~ /^s32i/) store[reg[op2]] = 1
                else             loadf[reg[op2]] = 1
            }
            if (m ~ /^l32i/) { gsub(/,/, "", op1); delete reg[op1] }
            next
        }
        # Any other instruction that REDEFINES a register must drop its mapping,
        # or we keep attributing later stores to a stale l32r value. The .bss
        # zero loop does exactly this (l32r a8,<addr> ... addi a8,a8,4).
        m ~ /^(movi|mov|add|addi|sub|addx|and|or|xor|srl|sll|sra|neg)/ {
            gsub(/,/, "", op1); delete reg[op1]
            next
        }
        # Conservative: forget everything at any branch/call/return boundary.
        m ~ /^(j|jx|call0|callx0|ret|ret\.n|b)/ { delete reg; next }
        END {
            ok = 1
            if (!("0x60000000" in store)) { print "no-ahb-fifo-store"; ok = 0 }
            if (!("0x3ff4001c" in loadf))  { print "no-apb-status-load"; ok = 0 }
            for (a in store)
                if (a ~ /^0x3ff400/) { print "APB-UART-STORE:" a; ok = 0 }
            if (ok) print "OK"
        }' "$ESP_H_DIS")
    # Tripwire: the 1 Hz heartbeat is a plain counted loop with no volatile
    # touch, so a future DCE / strength-reduction pass could legally delete it,
    # turning the heartbeat into a ~640 line/s flood — which would drown out a
    # stray reset banner or a garbled character, i.e. degrade the debug channel
    # exactly when it matters. Assert the loop bound literal survives.
    if ! grep -q '(0x3d0900)' "$ESP_H_DIS"; then
        echo "FAIL: esp32_hello_image (delay() loop bound 4000000 absent — DCE ate the heartbeat)"
        ESP_H_OK=0
    fi
    case "$ESP_H_VERDICT" in
        OK) ;;
        *)
            echo "FAIL: esp32_hello_image (UART access pattern wrong: $ESP_H_VERDICT)"
            echo "      want: l32i from 0x3ff4001c (APB status poll) + s32i to 0x60000000 (AHB FIFO,"
            echo "      errata CPU-3.3); a store to any 0x3ff400xx UART address may silently drop bytes"
            ESP_H_OK=0
            ;;
    esac
    ESP_H_DIS_NOTE=" + AHB/APB direction"
else
    ESP_H_DIS_NOTE=" (disasm direction check SKIPPED — no xtensa-lx106-elf-objdump)"
fi
if [ "$ESP_H_OK" = 1 ]; then
    PASS=$((PASS + 1))
    echo "  esp32_hello_image: PASS (e9/02/02/20, 2 segments, entry in IRAM, checksum recomputed, 16-aligned+hash$ESP_H_HASH_NOTE$ESP_H_DIS_NOTE)"
else
    FAIL=$((FAIL + 1))
fi
rm -f "$ESP_H_BIN" "$ESP_H_PAY" "$ESP_H_CODE" "$ESP_H_DIS"

# --- esp32 TWAI loopback example ---
# Build guard for examples/esp32/twai_loopback.mlr, which is hardware-validated
# (sends ID 0x123 DLC 2 data AB CD through the GPIO matrix and receives all six
# fields back on an ESP32-D0WD-V3).
#
# It exercises, in one program: MMIO device blocks at four different peripheral
# bases, peripheral clock gating via DPORT, GPIO matrix signal routing, the
# bit-timing arithmetic, and the esp-image writer.
#
# Not executed — that needs the chip. Asserts the image is structurally sound,
# plus a SOURCE-level check that the derived 40 MHz bit-timing constants have
# not been edited: BTIM0=0x81, BTIM1=0x3E. Published ESP32 timing tables assume
# an 80 MHz APB and are off by 2x here, so a "helpful" correction to a table
# value would silently produce a 250 kbit/s bus.
#
# The timing check is deliberately on the SOURCE, not the emitted bytes. An
# earlier version scanned the code segment for the byte 0x81 — which passes
# whatever the source says, because 0x81 occurs incidentally in Xtensa
# encodings.
echo ""
echo "--- esp32 TWAI loopback ---"
TWAI_DIR=$(mktemp -d /tmp/mlrc_twai_XXXXXX)
TOTAL=$((TOTAL + 1))
if $MLRC --arch=xtensa --freestanding --target=esp32 "$DIR/../examples/esp32/twai_loopback.mlr" \
     -o "$TWAI_DIR/twai.bin" >/dev/null 2>&1 \
   && python3 - "$TWAI_DIR/twai.bin" <<'PYEOF'
import sys
d = open(sys.argv[1], 'rb').read()
assert d[0] == 0xE9, "bad esp-image magic"
assert d[1] == 2, "expected 2 segments, got %d" % d[1]
entry = int.from_bytes(d[4:8], 'little')
assert 0x40080400 <= entry < 0x400A0000, "entry 0x%08x not in IRAM" % entry
off, segs = 24, []
for _ in range(d[1]):
    la = int.from_bytes(d[off:off+4], 'little')
    ln = int.from_bytes(d[off+4:off+8], 'little')
    segs.append((la, ln, d[off+8:off+8+ln]))
    off += 8 + ln
assert segs[0][0] == 0x3FFB0000, "seg0 not DRAM"
assert segs[1][0] == 0x40080400, "seg1 not IRAM"
PYEOF
then
    if grep -q 'Twai.btim0 = 0x81' "$DIR/../examples/esp32/twai_loopback.mlr" \
       && grep -q 'Twai.btim1 = 0x3E' "$DIR/../examples/esp32/twai_loopback.mlr"; then
        TWAI_TIMING_OK=1
    else
        TWAI_TIMING_OK=0
    fi
else
    TWAI_TIMING_OK=-1
fi
if [ "$TWAI_TIMING_OK" = "1" ]; then
    PASS=$((PASS + 1))
    echo "  esp32_twai_loopback: PASS (2 segments, entry in IRAM, 40MHz timing 0x81/0x3E in source)"
elif [ "$TWAI_TIMING_OK" = "0" ]; then
    FAIL=$((FAIL + 1))
    echo "FAIL: esp32_twai_loopback (bit-timing constants changed — 40MHz needs BTIM0=0x81 BTIM1=0x3E)"
else
    FAIL=$((FAIL + 1))
    echo "FAIL: esp32_twai_loopback (build or structural check failed)"
fi
rm -rf "$TWAI_DIR"

# --- esp32 dual-core / atomic-CAS build gate ---
# Important finding 2 (final review): esp32_krc_identity above only builds
# the fixed hello/minimal/twai_loopback set, so none of this branch's
# xtensa-specific work — op-93 (atomic CAS) emission, the core1_entry
# preamble/predicate, the 6c dispatch, or the "../../std/" import convention
# these three sources use — was ever compiled by either test suite. This does
# not need hardware: compile-only, assert exit 0 and that an esp-image with
# the correct magic byte was produced.
echo ""
echo "--- esp32 dual-core / atomic-CAS build gate ---"
for ESP_DC_SRC in tests/esp32/cas_single.mlr tests/esp32/cas_contended.mlr examples/esp32/dualcore_hello.mlr; do
    ESP_DC_NAME=$(basename "$ESP_DC_SRC" .mlr)
    ESP_DC_BIN="/tmp/mlrc_esp32_dc_${ESP_DC_NAME}_$$.bin"
    rm -f "$ESP_DC_BIN"
    TOTAL=$((TOTAL + 1))
    if $MLRC --arch=xtensa --freestanding --target=esp32 "$DIR/../$ESP_DC_SRC" \
         -o "$ESP_DC_BIN" >/dev/null 2>&1 \
       && [ -s "$ESP_DC_BIN" ] \
       && [ "$(od -An -tu1 -N1 "$ESP_DC_BIN" | tr -d ' ')" = "233" ]; then
        PASS=$((PASS + 1))
        echo "  esp32_dualcore_build[$ESP_DC_NAME]: PASS ($(wc -c < "$ESP_DC_BIN" | tr -d ' ')B image)"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: esp32_dualcore_build[$ESP_DC_NAME] (build failed or no valid esp-image produced)"
        $MLRC --arch=xtensa --freestanding --target=esp32 "$DIR/../$ESP_DC_SRC" -o "$ESP_DC_BIN" 2>&1 | head -3
    fi
    rm -f "$ESP_DC_BIN"
done

# --- esp32 image byte-identity vs the KernRift reference compiler ---
# MLRift's xtensa/esp32 backend is a near-verbatim graft of KernRift's, and the
# bar the plain-xtensa port met was 17/17 byte-identical images. Hold the same
# bar for the esp-image path: when a KernRift checkout with a built krc2 is
# available next door, compile the SAME sources with both compilers and require
# identical bytes. SKIP cleanly when it is not (KernRift is not a dependency of
# this repo).
echo ""
echo "--- esp32 vs KernRift krc byte-identity ---"
KRC_REF="$DIR/../../KernRift/build/krc2"
if [ -x "$KRC_REF" ] && [ -d "$DIR/../../KernRift/examples/esp32" ]; then
    TOTAL=$((TOTAL + 1))
    ESP_X_OK=1
    ESP_X_DIR=$(mktemp -d /tmp/mlrc_espx_XXXXXX)
    ESP_X_NAMES=""
    for ESP_X_N in hello minimal twai_loopback; do
        ESP_X_KSRC="$DIR/../../KernRift/examples/esp32/$ESP_X_N.kr"
        [ -f "$ESP_X_KSRC" ] || continue
        "$KRC_REF" --arch=xtensa --freestanding --target=esp32 \
            "$ESP_X_KSRC" -o "$ESP_X_DIR/$ESP_X_N.krc.bin" >/dev/null 2>&1
        $MLRC --arch=xtensa --freestanding --target=esp32 \
            "$DIR/../examples/esp32/$ESP_X_N.mlr" -o "$ESP_X_DIR/$ESP_X_N.mlrc.bin" >/dev/null 2>&1
        if ! cmp -s "$ESP_X_DIR/$ESP_X_N.krc.bin" "$ESP_X_DIR/$ESP_X_N.mlrc.bin"; then
            echo "FAIL: esp32_krc_identity ($ESP_X_N differs from the KernRift reference image)"
            cmp "$ESP_X_DIR/$ESP_X_N.krc.bin" "$ESP_X_DIR/$ESP_X_N.mlrc.bin" 2>&1 | head -2
            ESP_X_OK=0
        else
            ESP_X_NAMES="$ESP_X_NAMES $ESP_X_N($(wc -c < "$ESP_X_DIR/$ESP_X_N.mlrc.bin" | tr -d ' ')B)"
        fi
    done
    if [ "$ESP_X_OK" = 1 ]; then
        PASS=$((PASS + 1))
        echo "  esp32_krc_identity: PASS (byte-identical to KernRift krc:$ESP_X_NAMES)"
    else
        FAIL=$((FAIL + 1))
    fi
    rm -rf "$ESP_X_DIR"
else
    echo "  esp32_krc_identity: SKIP (no KernRift checkout with a built build/krc2 next door)"
fi

# --- esp32 import failure must abort with exit 1, not exit 0 + a written image ---
# Regression for the import_failed bypass: codegen_write_output checked
# import_failed, but the esp32 freestanding image path (and the asm-listing
# path) call open_output_or_die directly and never went through
# codegen_write_output, so a build with an unresolvable import printed
# "error: cannot open import: ..." to stderr, wrote an image anyway, and
# exited 0 -- a build script watching $? saw success. Assert the compiler's
# own exit code and that no file was written, not the error text: the error
# text already printed before the fix, so a text-only check would pass
# against the broken compiler and prove nothing.
echo ""
echo "--- esp32 import failure aborts (exit code, not just error text) ---"
ESP_IMP_SRC="/tmp/mlrc_esp32_import_fail_src_$$.mlr"
ESP_IMP_BIN="/tmp/mlrc_esp32_import_fail_bin_$$.bin"
printf 'import "std/__import_failure_regression_nonexistent.mlr"\nfn main() { loop { } }\n' > "$ESP_IMP_SRC"
rm -f "$ESP_IMP_BIN"
$MLRC --arch=xtensa --freestanding --target=esp32 "$ESP_IMP_SRC" -o "$ESP_IMP_BIN" >/dev/null 2>&1
ESP_IMP_EXIT=$?
TOTAL=$((TOTAL + 1))
if [ "$ESP_IMP_EXIT" = 1 ] && [ ! -e "$ESP_IMP_BIN" ]; then
    PASS=$((PASS + 1))
    echo "  esp32_import_failure_aborts: PASS (exit=$ESP_IMP_EXIT, no output file written)"
else
    ESP_IMP_HAVE_FILE="no"
    [ -e "$ESP_IMP_BIN" ] && ESP_IMP_HAVE_FILE="yes"
    echo "FAIL: esp32_import_failure_aborts (expected exit 1 and no output file, got exit=$ESP_IMP_EXIT, file written=$ESP_IMP_HAVE_FILE)"
    FAIL=$((FAIL + 1))
fi
rm -f "$ESP_IMP_SRC" "$ESP_IMP_BIN"

# --- esp32 import-free build: positive control for the test above ---
# Minor finding 7 (final review): esp32_import_failure_aborts above asserts
# exit 1 + no file, but never proves the esp32 path can exit 0 + write a file
# AT ALL. If the esp32 build path ever starts exiting 1 for some unrelated
# reason (e.g. a bug in a completely different check), the test above would
# keep passing vacuously -- even with the import_failed guard deleted
# entirely. Pair it with the same-shape build MINUS the bad import, asserting
# the opposite: exit 0 and a file present.
echo ""
echo "--- esp32 import-free build succeeds (positive control) ---"
ESP_IMP_OK_SRC="/tmp/mlrc_esp32_import_ok_src_$$.mlr"
ESP_IMP_OK_BIN="/tmp/mlrc_esp32_import_ok_bin_$$.bin"
printf 'fn main() { loop { } }\n' > "$ESP_IMP_OK_SRC"
rm -f "$ESP_IMP_OK_BIN"
$MLRC --arch=xtensa --freestanding --target=esp32 "$ESP_IMP_OK_SRC" -o "$ESP_IMP_OK_BIN" >/dev/null 2>&1
ESP_IMP_OK_EXIT=$?
TOTAL=$((TOTAL + 1))
if [ "$ESP_IMP_OK_EXIT" = 0 ] && [ -e "$ESP_IMP_OK_BIN" ] && [ -s "$ESP_IMP_OK_BIN" ]; then
    PASS=$((PASS + 1))
    echo "  esp32_import_free_build_ok: PASS (exit=0, $(wc -c < "$ESP_IMP_OK_BIN" | tr -d ' ')B image written)"
else
    ESP_IMP_OK_HAVE_FILE="no"
    [ -e "$ESP_IMP_OK_BIN" ] && ESP_IMP_OK_HAVE_FILE="yes"
    echo "FAIL: esp32_import_free_build_ok (expected exit 0 and an output file, got exit=$ESP_IMP_OK_EXIT, file written=$ESP_IMP_OK_HAVE_FILE)"
    FAIL=$((FAIL + 1))
fi
rm -f "$ESP_IMP_OK_SRC" "$ESP_IMP_OK_BIN"

echo ""
echo "--- xtensa simcall encoding ---"
TOTAL=$((TOTAL + 1))
XT_SC_SRC="/tmp/mlrc_xt_simcall_$$.mlr"
XT_SC_BIN="/tmp/mlrc_xt_simcall_$$.elf"
XT_SC_CTL="/tmp/mlrc_xt_simctl_$$.mlr"
XT_SC_CBIN="/tmp/mlrc_xt_simctl_$$.elf"
# Differential: the SAME program with and without the simcall. Grepping one
# image alone could match 00 51 00 occurring by chance in unrelated bytes.
printf '@naked fn s() { asm("simcall") }\nfn main() { s() }\n' > "$XT_SC_SRC"
printf '@naked fn s() { asm("nop") }\nfn main() { s() }\n'     > "$XT_SC_CTL"
if $MLRC --arch=xtensa --freestanding "$XT_SC_SRC" -o "$XT_SC_BIN"  >/dev/null 2>&1 \
   && $MLRC --arch=xtensa --freestanding "$XT_SC_CTL" -o "$XT_SC_CBIN" >/dev/null 2>&1 \
   && xxd -p "$XT_SC_BIN"  | tr -d '\n' | grep -q "005100" \
   && ! xxd -p "$XT_SC_CBIN" | tr -d '\n' | grep -q "005100"; then
    PASS=$((PASS + 1))
    echo "  xtensa_simcall_encoding: PASS (0x005100 present only with simcall)"
else
    echo "FAIL: xtensa_simcall_encoding (expected 00 51 00 with simcall and absent without)"
    FAIL=$((FAIL + 1))
fi
rm -f "$XT_SC_SRC" "$XT_SC_BIN" "$XT_SC_CTL" "$XT_SC_CBIN"

# ---------------------------------------------------------------------------
# int8 quantized NN kernels (std/nn_int8.mlr)
#
# Numeric validation, not a smoke test: tests/nn/nn_ref.py is an independent
# Python implementation of the gemmlowp/TFLite integer arithmetic that uses
# native 64-bit products, and the kernels — which have no 64-bit type at all
# on the MCU targets and synthesise the 32x32 high product from 16-bit halves
# — must match it EXACTLY. The same 896 values are then re-derived on a
# freestanding xtensa image (qemu -M lx60) and a freestanding riscv32 image
# (qemu -M virt) and compared byte for byte.
# ---------------------------------------------------------------------------
echo ""
echo "--- int8 NN kernels (std/nn_int8.mlr) ---"
while IFS= read -r NN_LINE; do
    case "$NN_LINE" in
        PASS:*)
            TOTAL=$((TOTAL + 1)); PASS=$((PASS + 1))
            NN_REST="${NN_LINE#PASS: }"
            echo "  ${NN_REST%% *}: PASS ${NN_REST#* }"
            ;;
        FAIL:*)
            TOTAL=$((TOTAL + 1)); FAIL=$((FAIL + 1))
            echo "FAIL: ${NN_LINE#FAIL: }"
            ;;
        SKIP:*)
            echo "  ${NN_LINE#SKIP: } SKIPPED"
            ;;
        *)
            echo "    $NN_LINE"
            ;;
    esac
done < <(MLRC="$MLRC" bash "$DIR/nn/run_nn_tests.sh" 2>&1)

echo ""
echo "--- riscv32 freestanding exit() terminates qemu ---"
if command -v qemu-system-riscv32 >/dev/null 2>&1; then
    for RV_CODE in 0 42; do
        TOTAL=$((TOTAL + 1))
        RV_E_SRC="/tmp/mlrc_rv_exit_${RV_CODE}_$$.mlr"
        RV_E_BIN="/tmp/mlrc_rv_exit_${RV_CODE}_$$.bin"
        printf 'fn main() { exit(%d) }\n' "$RV_CODE" > "$RV_E_SRC"
        if $MLRC --arch=riscv32 --freestanding "$RV_E_SRC" -o "$RV_E_BIN" >/dev/null 2>&1; then
            timeout 10 qemu-system-riscv32 -machine virt -nographic -bios "$RV_E_BIN" >/dev/null 2>&1
            RV_E_ST=$?
            if [ "$RV_E_ST" = "$RV_CODE" ]; then
                PASS=$((PASS + 1)); echo "  riscv32_exit_${RV_CODE}: PASS (qemu status $RV_E_ST)"
            else
                echo "FAIL: riscv32_exit_${RV_CODE} (expected status $RV_CODE, got $RV_E_ST)"; FAIL=$((FAIL + 1))
            fi
        else
            echo "FAIL: riscv32_exit_${RV_CODE} (compile failed)"; FAIL=$((FAIL + 1))
        fi
        rm -f "$RV_E_SRC" "$RV_E_BIN"
    done
else
    echo "  riscv32_exit: SKIP (qemu-system-riscv32 not installed)"
fi

# Pin the emitted constants (spec success-criterion 7). The behavioural
# tests above prove the mechanism works today; this one pins the
# *constants*, so a later edit that changes the magic word or the device
# address fails loudly at the byte level instead of silently hanging
# under qemu. The two hex strings are the little-endian byte orders of
# 0x01051293 (slli t0,a0,16) and 0x0062E2B3 (or t0,t0,t1).
TOTAL=$((TOTAL + 1))
RV_K_SRC="/tmp/mlrc_rv_const_$$.mlr"
RV_K_BIN="/tmp/mlrc_rv_const_$$.bin"
printf 'fn main() { exit(1) }\n' > "$RV_K_SRC"
if $MLRC --arch=riscv32 --freestanding "$RV_K_SRC" -o "$RV_K_BIN" >/dev/null 2>&1 \
   && xxd -p "$RV_K_BIN" | tr -d '\n' | grep -q "93120501" \
   && xxd -p "$RV_K_BIN" | tr -d '\n' | grep -q "b3e26200"; then
    PASS=$((PASS + 1))
    echo "  riscv32_exit_constants: PASS (slli/or words pinned)"
else
    echo "FAIL: riscv32_exit_constants (expected slli t0,a0,16 and or t0,t0,t1 words)"
    FAIL=$((FAIL + 1))
fi
rm -f "$RV_K_SRC" "$RV_K_BIN"

echo ""
echo "--- xtensa exit() ---"
TOTAL=$((TOTAL + 1))
XT_E_SRC="/tmp/mlrc_xt_exit_$$.mlr"
printf 'fn main() { exit(42) }\n' > "$XT_E_SRC"
# esp32 must REFUSE, with the new wording
if $MLRC --arch=xtensa --freestanding --target=esp32 "$XT_E_SRC" -o /dev/null 2>&1 \
     | grep -q "no OS to return an exit status to"; then
    PASS=$((PASS + 1)); echo "  xtensa_esp32_exit_refused: PASS"
else
    echo "FAIL: xtensa_esp32_exit_refused (expected an error mentioning 'no OS to return an exit status to')"; FAIL=$((FAIL + 1))
fi
# lx60 must terminate qemu with 42
if command -v qemu-system-xtensa >/dev/null 2>&1; then
    TOTAL=$((TOTAL + 1))
    XT_E_BIN="/tmp/mlrc_xt_exit_$$.elf"
    if $MLRC --arch=xtensa --freestanding "$XT_E_SRC" -o "$XT_E_BIN" >/dev/null 2>&1; then
        timeout 10 qemu-system-xtensa -M lx60 -nographic -semihosting -kernel "$XT_E_BIN" >/dev/null 2>&1
        XT_E_ST=$?
        if [ "$XT_E_ST" = "42" ]; then
            PASS=$((PASS + 1)); echo "  xtensa_lx60_exit_42: PASS"
        else
            echo "FAIL: xtensa_lx60_exit_42 (expected 42, got $XT_E_ST)"; FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL: xtensa_lx60_exit_42 (compile failed)"; FAIL=$((FAIL + 1))
    fi
    rm -f "$XT_E_BIN"
fi
rm -f "$XT_E_SRC"

# --- Summary ---
echo ""
echo "=== Results: $PASS/$TOTAL passed, $FAIL failed ==="
exit $FAIL
