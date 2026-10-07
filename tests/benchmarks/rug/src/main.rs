//! The Rug worker of the consolidated report (tests/benchmarks/report.py): GMP,
//! MPFR and MPC. Development only; apn_mojo never calls them. The protocol is
//! apn_worker.mojo's: the catalog file as the argument, then `check i`,
//! `time i target repeats`, `version` and `quit` on standard input, one JSON
//! line each. Each operation resolves to a function once per case, so a timed
//! call is one indirect call; `black_box` keeps it in the loop.
use rug::{integer::IsPrime, ops::Pow, Complex, Float, Integer, Rational};
use serde_json::{json, Value};
use std::{ffi::CStr, hint::black_box, io::{self, BufRead, Write}, time::Instant};

type IntFn = fn(&Integer, &Integer, &Integer, u32) -> Integer;
type RatFn = fn(&Rational, &Rational, i32) -> Rational;
type RealFn = fn(&Float, &Float, &Float, i32, u32) -> Float;
type CplxFn = fn(&Complex, &Complex, i32, u32) -> Complex;

fn int_op(name: &str) -> IntFn {
    match name {
        "add" => |x, y, _, _| Integer::from(x + y),
        "subtract" => |x, y, _, _| Integer::from(x - y),
        "multiply" => |x, y, _, _| Integer::from(x * y),
        "square" => |x, _, _, _| Integer::from(x.square_ref()),
        // The operands are positive, so truncation is floor division.
        "floordiv" => |x, y, _, _| Integer::from(x / y),
        "mod" => |x, y, _, _| Integer::from(x % y),
        "and" => |x, y, _, _| Integer::from(x & y),
        "or" => |x, y, _, _| Integer::from(x | y),
        "xor" => |x, y, _, _| Integer::from(x ^ y),
        "shift_left" => |x, _, _, n| Integer::from(x << n),
        "shift_right" => |x, _, _, n| Integer::from(x >> n),
        "gcd" => |x, y, _, _| Integer::from(x.gcd_ref(y)),
        "lcm" => |x, y, _, _| Integer::from(x.lcm_ref(y)),
        "isqrt" => |x, _, _, _| Integer::from(x.sqrt_ref()),
        "iroot" => |x, _, _, n| Integer::from(x.root_ref(n)),
        "div_exact" => |x, y, _, _| Integer::from(x.div_exact_ref(y)),
        "inverse_mod" => |x, y, _, _| Integer::from(x.invert_ref(y).unwrap()),
        "pow_mod" => |x, y, z, _| Integer::from(x.pow_mod_ref(y, z).unwrap()),
        // GMP's test with 24 or fewer repetitions is Baillie-PSW.
        "is_prime" => |x, _, _, _| Integer::from((x.is_probably_prime(24) != IsPrime::No) as u32),
        "factorial" => |_, _, _, n| Integer::from(Integer::factorial(n)),
        "binomial" => |x, _, _, n| Integer::from(x.binomial_ref(n)),
        _ => panic!("unsupported integer operation: {name}"),
    }
}

fn rat_op(name: &str) -> RatFn {
    match name {
        "add" => |x, y, _| Rational::from(x + y),
        "subtract" => |x, y, _| Rational::from(x - y),
        "multiply" => |x, y, _| Rational::from(x * y),
        "divide" => |x, y, _| Rational::from(x / y),
        "square" => |x, _, _| Rational::from(x.square_ref()),
        "pow_int" => |x, _, n| Rational::from(x.pow(n)),
        _ => panic!("unsupported rational operation: {name}"),
    }
}

fn real_op(name: &str) -> RealFn {
    match name {
        "add" => |x, y, _, _, p| Float::with_val(p, x + y),
        "subtract" => |x, y, _, _, p| Float::with_val(p, x - y),
        "multiply" => |x, y, _, _, p| Float::with_val(p, x * y),
        "divide" => |x, y, _, _, p| Float::with_val(p, x / y),
        "square" => |x, _, _, _, p| Float::with_val(p, x.square_ref()),
        "sqrt" => |x, _, _, _, p| Float::with_val(p, x.sqrt_ref()),
        "fma" => |x, y, z, _, p| Float::with_val(p, x.mul_add_ref(y, z)),
        "pow_int" => |x, _, _, n, p| Float::with_val(p, x.pow(n)),
        "exp" => |x, _, _, _, p| Float::with_val(p, x.exp_ref()),
        "expm1" => |x, _, _, _, p| Float::with_val(p, x.exp_m1_ref()),
        "exp2" => |x, _, _, _, p| Float::with_val(p, x.exp2_ref()),
        "log" => |x, _, _, _, p| Float::with_val(p, x.ln_ref()),
        "log1p" => |x, _, _, _, p| Float::with_val(p, x.ln_1p_ref()),
        "log2" => |x, _, _, _, p| Float::with_val(p, x.log2_ref()),
        "log10" => |x, _, _, _, p| Float::with_val(p, x.log10_ref()),
        "sin" => |x, _, _, _, p| Float::with_val(p, x.sin_ref()),
        "cos" => |x, _, _, _, p| Float::with_val(p, x.cos_ref()),
        "tan" => |x, _, _, _, p| Float::with_val(p, x.tan_ref()),
        "atan" => |x, _, _, _, p| Float::with_val(p, x.atan_ref()),
        "asin" => |x, _, _, _, p| Float::with_val(p, x.asin_ref()),
        "acos" => |x, _, _, _, p| Float::with_val(p, x.acos_ref()),
        "sinh" => |x, _, _, _, p| Float::with_val(p, x.sinh_ref()),
        "cosh" => |x, _, _, _, p| Float::with_val(p, x.cosh_ref()),
        "tanh" => |x, _, _, _, p| Float::with_val(p, x.tanh_ref()),
        "asinh" => |x, _, _, _, p| Float::with_val(p, x.asinh_ref()),
        "acosh" => |x, _, _, _, p| Float::with_val(p, x.acosh_ref()),
        "atanh" => |x, _, _, _, p| Float::with_val(p, x.atanh_ref()),
        "gamma" => |x, _, _, _, p| Float::with_val(p, x.gamma_ref()),
        "gammaln" => |x, _, _, _, p| Float::with_val(p, x.ln_gamma_ref()),
        "digamma" => |x, _, _, _, p| Float::with_val(p, x.digamma_ref()),
        "erf" => |x, _, _, _, p| Float::with_val(p, x.erf_ref()),
        "erfc" => |x, _, _, _, p| Float::with_val(p, x.erfc_ref()),
        "expi" => |x, _, _, _, p| Float::with_val(p, x.eint_ref()),
        "atan2" => |x, y, _, _, p| Float::with_val(p, x.atan2_ref(y)),
        "pow" => |x, y, _, _, p| Float::with_val(p, x.pow(y)),
        _ => panic!("unsupported float operation: {name}"),
    }
}

fn cplx_op(name: &str) -> CplxFn {
    match name {
        "add" => |x, y, _, p| Complex::with_val(p, x + y),
        "subtract" => |x, y, _, p| Complex::with_val(p, x - y),
        "multiply" => |x, y, _, p| Complex::with_val(p, x * y),
        "divide" => |x, y, _, p| Complex::with_val(p, x / y),
        "square" => |x, _, _, p| Complex::with_val(p, x.square_ref()),
        "sqrt" => |x, _, _, p| Complex::with_val(p, x.sqrt_ref()),
        "pow_int" => |x, _, n, p| Complex::with_val(p, x.pow(n)),
        "exp" => |x, _, _, p| Complex::with_val(p, x.exp_ref()),
        "log" => |x, _, _, p| Complex::with_val(p, x.ln_ref()),
        "sin" => |x, _, _, p| Complex::with_val(p, x.sin_ref()),
        "cos" => |x, _, _, p| Complex::with_val(p, x.cos_ref()),
        "tan" => |x, _, _, p| Complex::with_val(p, x.tan_ref()),
        "sinh" => |x, _, _, p| Complex::with_val(p, x.sinh_ref()),
        "cosh" => |x, _, _, p| Complex::with_val(p, x.cosh_ref()),
        "tanh" => |x, _, _, p| Complex::with_val(p, x.tanh_ref()),
        "asin" => |x, _, _, p| Complex::with_val(p, x.asin_ref()),
        "acos" => |x, _, _, p| Complex::with_val(p, x.acos_ref()),
        "atan" => |x, _, _, p| Complex::with_val(p, x.atan_ref()),
        "asinh" => |x, _, _, p| Complex::with_val(p, x.asinh_ref()),
        "acosh" => |x, _, _, p| Complex::with_val(p, x.acosh_ref()),
        "atanh" => |x, _, _, p| Complex::with_val(p, x.atanh_ref()),
        "pow" => |x, y, _, p| Complex::with_val(p, x.pow(y)),
        _ => panic!("unsupported complex operation: {name}"),
    }
}

enum Out { Int(Integer), Rat(Rational), Real(Float), Cplx(Complex), Text(String), Reals(Vec<Float>), Ints(Vec<Integer>) }

enum Case {
    Int { f: IntFn, x: Integer, y: Integer, z: Integer, n: u32 },
    Rat { f: RatFn, x: Rational, y: Rational, n: i32 },
    Real { f: RealFn, x: Float, y: Float, z: Float, n: i32, p: u32 },
    Cplx { f: CplxFn, x: Complex, y: Complex, n: i32, p: u32 },
    IntText { parse: bool, text: String, x: Integer },
    RealText { parse: bool, text: String, x: Float, digits: usize, p: u32 },
    RealBatch { op: String, xs: Vec<Float>, ys: Vec<Float>, p: u32 },
    IntBatch { op: String, xs: Vec<Integer>, ys: Vec<Integer> },
}

fn integer(text: &str) -> Integer { Integer::from_str_radix(text, 10).unwrap() }
fn rational(text: &str) -> Rational { Rational::from_str_radix(text, 10).unwrap() }
fn real(text: &str, p: u32) -> Float { Float::with_val(p, rational(text)) }
fn cplx(text: &str, p: u32) -> Complex {
    let (re, im) = text.split_once(';').unwrap();
    Complex::with_val(p, (real(re, p), real(im, p)))
}

fn prepare(row: &str) -> Case {
    let f: Vec<&str> = row.split('\t').collect();
    let (family, op) = (f[1], f[2]);
    let p: u32 = f[3].parse().unwrap();
    let param: i64 = f[4].parse().unwrap();
    let a = &f[5..];
    let arg = |i: usize| a.get(i).copied();
    match family {
        "integer" => Case::Int {
            f: int_op(op),
            x: arg(0).map_or(Integer::new(), integer),
            y: arg(1).map_or(Integer::new(), integer),
            z: arg(2).map_or(Integer::new(), integer),
            n: param as u32,
        },
        "rational" => Case::Rat {
            f: rat_op(op), x: rational(a[0]), y: arg(1).map_or(Rational::new(), rational), n: param as i32,
        },
        "float" => Case::Real {
            f: real_op(op),
            x: real(a[0], p),
            y: arg(1).map_or(Float::new(p), |t| real(t, p)),
            z: arg(2).map_or(Float::new(p), |t| real(t, p)),
            n: param as i32,
            p,
        },
        "complex" => Case::Cplx {
            f: cplx_op(op), x: cplx(a[0], p), y: arg(1).map_or(Complex::new(p), |t| cplx(t, p)), n: param as i32, p,
        },
        "integer_text" => Case::IntText { parse: op == "parse", text: a[0].to_string(), x: integer(a[0]) },
        "float_text" => Case::RealText {
            parse: op == "parse",
            text: a[0].to_string(),
            x: if op == "parse" { Float::new(p) } else { real(a[0], p) },
            digits: param as usize,
            p,
        },
        "float_batch" => Case::RealBatch {
            op: op.to_string(),
            xs: a[0].split(',').map(|t| real(t, p)).collect(),
            ys: a[1].split(',').map(|t| real(t, p)).collect(),
            p,
        },
        "integer_batch" => Case::IntBatch {
            op: op.to_string(),
            xs: a[0].split(',').map(integer).collect(),
            ys: a[1].split(',').map(integer).collect(),
        },
        _ => panic!("unsupported family: {family}"),
    }
}

fn evaluate(case: &Case) -> Out {
    match case {
        Case::Int { f, x, y, z, n } => Out::Int(f(x, y, z, *n)),
        Case::Rat { f, x, y, n } => Out::Rat(f(x, y, *n)),
        Case::Real { f, x, y, z, n, p } => Out::Real(f(x, y, z, *n, *p)),
        Case::Cplx { f, x, y, n, p } => Out::Cplx(f(x, y, *n, *p)),
        Case::IntText { parse, text, x } => {
            if *parse { Out::Int(Integer::from(Integer::parse(text).unwrap())) } else { Out::Text(x.to_string()) }
        }
        Case::RealText { parse, text, x, digits, p } => {
            if *parse {
                Out::Real(Float::with_val(*p, Float::parse(text).unwrap()))
            } else {
                Out::Text(x.to_string_radix(10, Some(*digits)))
            }
        }
        Case::RealBatch { op, xs, ys, p } => match op.as_str() {
            // A `_loop` case times apn_mojo's plain loop; Rug's loop is the same either way.
            "add" | "add_loop" => Out::Reals(xs.iter().zip(ys).map(|(a, b)| Float::with_val(*p, a + b)).collect()),
            "multiply" => Out::Reals(xs.iter().zip(ys).map(|(a, b)| Float::with_val(*p, a * b)).collect()),
            "exp" | "exp_loop" => Out::Reals(xs.iter().map(|a| Float::with_val(*p, a.exp_ref())).collect()),
            "sum" => Out::Real(Float::with_val(*p, Float::sum(xs.iter()))),
            "dot" => Out::Real(Float::with_val(*p, Float::dot(xs.iter().zip(ys)))),
            _ => panic!("unsupported float batch operation: {op}"),
        },
        Case::IntBatch { op, xs, ys } => match op.as_str() {
            "add" | "add_loop" => Out::Ints(xs.iter().zip(ys).map(|(a, b)| Integer::from(a + b)).collect()),
            "multiply" => Out::Ints(xs.iter().zip(ys).map(|(a, b)| Integer::from(a * b)).collect()),
            "sum" => Out::Int(Integer::from(Integer::sum(xs.iter()))),
            _ => panic!("unsupported integer batch operation: {op}"),
        },
    }
}

/// One timed call: never inlined, its result reduced to whether it is nonzero.
#[inline(never)]
fn call(case: &Case) -> bool {
    match evaluate(case) {
        Out::Int(z) => z != 0,
        Out::Rat(z) => z != 0,
        Out::Real(z) => !z.is_zero(),
        Out::Cplx(z) => !(z.real().is_zero() && z.imag().is_zero()),
        Out::Text(s) => !s.is_empty(),
        Out::Reals(v) => !v.is_empty(),
        Out::Ints(v) => !v.is_empty(),
    }
}

fn real_json(x: &Float) -> Value {
    if x.is_nan() { return json!({"class": "nan"}); }
    if x.is_infinite() { return json!({"class": "inf", "negative": x.is_sign_negative()}); }
    if x.is_zero() { return json!({"class": "zero", "negative": x.is_sign_negative()}); }
    let (m, e) = x.to_integer_exp().unwrap();
    json!({"class": "finite", "significand": m.to_string(), "exponent": e})
}

fn encode(out: &Out) -> Value {
    match out {
        Out::Int(z) => json!({"integer": z.to_string()}),
        Out::Rat(z) => json!({"numerator": z.numer().to_string(), "denominator": z.denom().to_string()}),
        Out::Real(z) => real_json(z),
        Out::Cplx(z) => json!({"real": real_json(z.real()), "imag": real_json(z.imag())}),
        Out::Text(s) => json!({"text": s}),
        Out::Reals(v) => json!({"list": v.iter().map(real_json).collect::<Vec<_>>()}),
        Out::Ints(v) => json!({"list": v.iter().map(|z| json!({"integer": z.to_string()})).collect::<Vec<_>>()}),
    }
}

fn elapsed(case: &Case, number: u64) -> u64 {
    let mut sink = 0u64;
    let start = Instant::now();
    for _ in 0..number {
        sink += call(black_box(case)) as u64;
    }
    let t = start.elapsed().as_nanos() as u64;
    black_box(sink);
    t
}

/// The timeit protocol: 1, 2, 5, 10, 20, ... calls until one sample takes
/// `target` ns, then `repeats` samples of that many calls.
fn time(case: &Case, target: u64, repeats: usize) -> Value {
    let (mut number, mut decade) = (1u64, 1u64);
    while elapsed(case, number) < target {
        if number == 5 * decade {
            decade *= 10;
            number = decade;
        } else {
            number = if number == decade { 2 * decade } else { 5 * decade };
        }
    }
    let samples: Vec<u64> = (0..repeats).map(|_| elapsed(case, number)).collect();
    json!({"number": number, "samples": samples})
}

fn c_text(text: *const std::ffi::c_char) -> String {
    unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
}

fn main() {
    let path = std::env::args().nth(1).expect("catalog file");
    let rows: Vec<String> = std::fs::read_to_string(path).unwrap().lines().map(str::to_string).collect();
    let mut stdout = io::stdout().lock();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let words: Vec<&str> = line.split(' ').collect();
        let answer = match words[0] {
            "quit" => break,
            "version" => json!({
                "gmp": c_text(unsafe { gmp_mpfr_sys::gmp::version }),
                "mpfr": c_text(unsafe { gmp_mpfr_sys::mpfr::get_version() }),
                "mpc": c_text(unsafe { gmp_mpfr_sys::mpc::get_version() }),
                "rug": "1.30.0",
            }),
            command => {
                let case = prepare(&rows[words[1].parse::<usize>().unwrap()]);
                if command == "check" {
                    json!({"result": encode(&evaluate(&case))})
                } else {
                    time(&case, words[2].parse().unwrap(), words[3].parse().unwrap())
                }
            }
        };
        writeln!(stdout, "{answer}").unwrap();
        stdout.flush().unwrap();
    }
}
