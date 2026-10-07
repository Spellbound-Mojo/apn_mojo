# Integer

`Integer` keeps whole-number arithmetic exact as values grow from a native
word to many limbs. Compact handles and shared kernels support both scalar
and batch work. The [reference](../reference/integer.md) describes public calls.

## Representation

On the qualified target, an Integer is a 16-byte handle with one of four
representations:

| Variant | Purpose and Invariants |
|---|---|
| **Inline signed 64-bit** | Stores values within $[-2^{63}, 2^{63}-1]$, including zero. |
| **Inline unsigned 64-bit** | Stores wide unsigned values within $[2^{63}, 2^{64}-1]$ without heap allocation. |
| **Static literal words** | References compile-time arbitrary-precision integer literals. |
| **Shared sign-magnitude** | Heap-allocated 32-bit limbs for dynamic runtime values exceeding 64 bits. |

Equality, comparisons, hashing, text, and JSON all depend on the mathematical
value, regardless of how it is stored. Zero has one canonical form and no
negative sign. When a value shrinks to 64 bits, it returns to inline storage.

Heap magnitudes use one `malloc` block: an atomic reference-count header
followed by 32-bit limbs. Copies share the block until an update needs its
own storage, so changing an `Integer` leaves existing copies and dictionary
keys intact. To build a large integer, the caller allocates the required
space, fills its words, and exposes the handle only when it is complete.
Kernels write every word they cover; only skipped words need clearing.

System allocations use `std.ffi.external_call`. Adopted Mojo `List` buffers
must instead be freed by the native runtime allocator, so the two sources
are tracked separately. Every allocation has a checked size, though physical
out-of-memory remains fatal.

Keeping the handle at 16 bytes lets scalar operations pass a compact value.
View metadata belongs to the batch container, where it does not enlarge
every scalar argument and result.

Heap magnitudes keep 32-bit words, but the kernels that stream through them do
64-bit work. Addition, subtraction, shifts, comparison, and the conversion to and
from the 64-bit limbs of division and square roots load or store two adjacent
words at once (alignment 4, so any word offset is valid), carrying through 64-bit
arithmetic; multiplication multiplies word pairs as 64-bit limbs. The in-place
updates (`+=`, `<<=`) share these kernels. Pairing the words uses the target's
64-bit arithmetic while preserving the existing storage representation.

## Arithmetic

Addition, subtraction, and comparison use native paths for small values and
word kernels for larger magnitudes. Comparisons can stop once a differing
word or magnitude length determines the order.
In-place addition (`+=`) mutates the underlying buffer directly when the value
uniquely owns its limb allocation and signs are compatible; otherwise, it falls
back to out-of-place addition. In place, the shared addition kernel writes the
sum over the left operand's own words, which is safe because it reads each pair
of words before writing it; the carry word goes into spare capacity, and the
words move to a larger allocation only when there is none.

- **Bitwise operations (`&`, `|`, `^`, `~`)**: Follow Python's semantics on
  infinite two's complement. One kernel reads both magnitudes two words at a
  time, converts a negative operand on the fly ($-m$ is $\lnot m + 1$, extended
  with ones past its words), combines the pairs, and converts a negative result
  back the same way. The result has one word more than the longer operand, which
  holds the sign extension. Batch updates in place use the same kernel.

- **Multiplication**: Inspects operand magnitudes and dispatches to specialized
  kernels based on word width: native limb multiplication for small values, a
  dedicated squaring kernel for identical operands, Karatsuba multiplication for
  medium-sized integers, and blocked multiplication for operands of mismatched
  lengths. Multiplication kernels never allocate dynamically; callers prepare
  result and scratch buffers before entering limb loops.
- **Division (`//`, `%`, `div_rem_*`)**: Implements Knuth's Algorithm D over
  64-bit limbs. A single scratch buffer (allocated on the stack for inputs up to
  80 limbs) holds normalized operands with the divisor's leading bit set. Each
  quotient digit is approximated from the two leading limbs using a precomputed
  reciprocal (requiring two multiplications instead of hardware division, based
  on Möller and Granlund, *Improved division by invariant integers*, 2011), and
  corrected by the third limb to ensure error never exceeds one unit. The quotient
  and remainder are written directly into target allocations. The floating-point
  single-rounding core uses the same division engine. A divisor of at
  most 64 bits needs no scratch block: one pass from the dividend's top pair
  shifts each pair by the divisor's normalization as it reads it, divides it
  with the same reciprocal, and writes the quotient straight into its
  allocation; the remainder is a native value. `%`, and so `gcd`, skips the
  quotient. A quotient that floor division (operands of opposite signs) or
  Euclidean division (a negative dividend) must round away from zero gains
  its one in its words before it becomes an Integer, and the remainder
  becomes the divisor less the truncated one.
- **Greatest Common Divisor (`gcd`)**: Magnitudes of up to two 64-bit limbs
  take Stein's binary gcd, after one division when the larger has at least 16
  more bits: Stein's steps, a shift and a subtraction each, cost less than
  Euclid's divisions, but take about one step per bit by which the larger
  operand exceeds the smaller. Each step takes its shift from the wrapped
  difference, whose trailing zeros are its magnitude's, so counting them
  overlaps the rest of the step. While either operand has two limbs, a step
  forms both differences and picks one by the operands' order without a
  branch (Stein's step; Knuth, *The Art of Computer Programming*, vol. 2,
  §4.5.2, Algorithm B), and counts the low limb's trailing zeros alone; then one-limb
  steps finish. Longer operands take Lehmer's algorithm on 64-bit limbs in one
  scratch block. Each step reduces the leading 128 bits of
  both operands, taken at the same bit position, by Euclid's algorithm into a
  $2 \times 2$ matrix of determinant 1 with entries below $2^{64}$: the
  double-digit step of Möller (*On Schönhage's algorithm and subquadratic
  integer gcd computation*, Mathematics of Computation 77, 2008). It stops while the
  remainders' high limbs are still at least 2, so its quotients are those of the
  full operands, and applying the matrix's inverse, in one pass over the limbs,
  leaves both nonnegative and about 62 bits shorter. A step that cannot proceed
  (a quotient past 64 bits, or nearly equal operands) takes one division, as do
  operands more than a limb apart; once both fit two limbs, the binary gcd
  finishes. Rational normalization relies directly on this kernel.
- **Factorials (`factorial`, `factorial2`)**: $n!$ for $n \le 20$ and
  $n!!$ for $n \le 33$, the values that fit an `Int64`, come from tables.
  Beyond, $n!$ is its odd part times $2^{n - \operatorname{popcount}(n)}$
  (Legendre's formula for the power of two), and the odd part
  is the product over $a \ge 0$ of the odd numbers up to
  $\lfloor n / 2^a \rfloor$, each odd $j$ coming in once for every $a$ with
  $j 2^a \le n$. An odd double factorial is one such level, and
  $(2m)!! = 2^m m!$. The odd numbers pack into 64-bit words in fixed groups:
  in a level whose terms have at most $b$ bits, any $\lfloor 64 / b \rfloor$
  of them fit a word, so no product is checked. The words multiply on limbs
  in one scratch block, into one accumulator a limb at a time up to 96 words
  and in halves beyond, so that products past the Karatsuba threshold meet
  at the top. The power of two is a shift as the result's words are stored,
  in one allocation.
- **Square roots (`isqrt`)**: Zimmermann's Karatsuba square root (Brent and
  Zimmermann, *Modern Computer Arithmetic*, Algorithm 1.12) returns the root and
  its remainder together. Write $a = (a_3 b + a_2) b^2 + a_1 b + a_0$ with
  $a_3 \ge b / 4$. With $(s', r')$ the root and remainder of $a_3 b + a_2$, and
  $(q, u)$ the quotient and remainder of $(r' b + a_1) / (2 s')$, the root
  $s = s' b + q$ has the remainder $u b + a_0 - q^2$. Since $s' \ge b / 2$,
  $q \le b$ and $q^2 \le 2s - 1$, so a negative remainder needs one correction:
  $s - 1$, with $r + 2s - 1$.
  - Up to 256 bits the step runs in native integers, with $b = 2^k$ and
    $k = \lfloor(\text{bits} + 1) / 4\rfloor$ between 32 and 64, over a 128-bit
    root from a `Float64` estimate and one Newton step. The top part has at most
    128 bits, so $s' < 2^{64}$ and $r' b + a_1 < 2^{129}$; its quotient by $2s'$
    is that of $(r' b + a_1) \gg 1$ by $s'$, a 128-by-64-bit division, and the
    dropped bit returns in the remainder.
  - Beyond, the step runs on 64-bit limbs (Zimmermann, *Karatsuba Square
    Root*, INRIA Research Report 3805, 1999), in one
    scratch block of $3.5n + 1$ limbs for an $n$-limb root (on the stack up to
    $n = 45$). The radicand is shifted left by $2k$ bits so that its top limb is at
    least $2^{62}$. The top half's root is then $s' \ge B^h / 2$, already a
    normalized divisor for the in-place limb division, which divides by $s'$ and
    halves the quotient: an odd quotient adds $s'$ to the remainder. $q^2$ goes to
    limbs the division has freed. When $s'$ is all ones and $q = B^l$, the
    uncorrected root $B^n$ overflows its limbs; the correction then adds $2B^n$ to
    the remainder and brings the root back within them. With $4^k a = S^2 + R$
    and $s_0 = S \bmod 2^k$, the root is $S \gg k$ and the remainder
    $(R + 2 S s_0 - s_0^2) / 4^k$.
- **Roots (`iroot`)**: Degree 2 takes the square root above. Higher degrees
  take Zimmermann's root (Brent and Zimmermann, *Modern Computer Arithmetic*,
  Cambridge University Press, 2010, chapter 1), which finds the root's
  bits from the top. With $S$ the floor $k$-th root of a prefix $P$ of the
  radicand, and $R$ the remainder $P - S^k$ followed by the radicand's next
  $b$ bits, one Newton step gives the root of the longer prefix as
  $S 2^b + q$, for $q = \lfloor R / (k S^{k-1}) \rfloor$ clamped below $2^b$.
  Each level adds at most $(x + \log_2 k) / 2$ bits to reach an $x$-bit root,
  about doubling it, and the result is then at most one too large.
  - Every level's root is certified exactly before the next step: from above
    while $S^k > P$, and from below through $(S + 1)^k - S^k > k S^{k-1}$, which
    the remainder settles unless it exceeds $k S^{k-1}$. For small roots of
    high degrees, where that bound is weak, a `Float64` comparison of
    $P / S^k$ with $(1 + 1/S)^k$ settles it, with a margin of
    $(k + 64) 2^{-50}$ above the worst rounding of both sides (correctly
    rounded operations err by at most $2^{-53}$ each). Only when neither does,
    and not after a step down, is $(S + 1)^k$ computed. Exactness depends
    neither on the book's bound nor on the seed.
  - The seed is a `Float64` estimate: `exp2` and `log2` of the leading 64 bits,
    with the exponent split so that the root taken stays below $2^{1 + 64/k}$,
    then one Newton step in `Float64` for roots past 28 bits. Mojo's `exp2` and
    `log2` are good to about $2^{-31}$ relatively (measured), the refined
    estimate to about $2^{-50}$.
  - Radicands of up to 256 bits never reach the levels. Native Newton steps on
    the exact residual, $\lfloor (P - S^k) / (k S^{k-1}) \rfloor$, take the
    estimate to the root's unit in a step or two. Products are checked for
    overflow by bit lengths, splitting a factor at half the width when the
    lengths sum to one past it. A power past $2^{256}$ steps down by
    $S (1 - P / S^k) / k$ from `Float64` values, or by at least $S \gg 52$:
    just below the root of $2^{256} - 1$ the `Float64` step is noise. Up to 64
    bits every power fits 128 bits, unchecked, and roots of at most 11 bits
    take their bits from the top instead, one power each, chosen by selects
    rather than branches: a chain of `Float64` operations takes longer.
  - Longer radicands start from a native seed: up to $\lfloor 256/k \rfloor$
    bits for degrees up to 8, the largest whose prefix fits 256 bits, and 30
    bits, within a unit of the estimate, for higher degrees, certified on limbs.
    The levels then run on 64-bit limbs in one scratch block (on the stack for
    cube roots of up to 2,880 bits), with the word kernels' products and the
    limb division.
- **Exact scaled addition (`axpy`)**: Computes $a \cdot x_i + y_i$ exactly across
  all elements of vector batches, pairing elements by logical index across any
  stride or slice selection. The scalar coefficient broadcasts across elements,
  and execution routes through the unified batch driver.

## Number theory

Number-theory functions support exact simplification, such as removing square
factors from a radical, and primality testing. The calculations are exact;
`is_prime` above $2^{64}$ gives a probable-prime answer, as described below.

- **Factor removal (`remove_factor`)**: Divides by $p, p^2, p^4, \ldots$ while
  each power divides, then back down through the same powers, so a multiplicity
  $k$ costs about $2 \log_2 k$ divisions instead of $k$. Factors of two come
  from the count of trailing zero bits.
- **Trial division (`trial_division`)**: Tries 2, 3, 5 and then the numbers
  coprime to 30, eight in every thirty, instead of keeping a table of primes. A
  composite candidate never divides, because its prime factors were divided out
  before it, so the result equals division by primes alone. Remainders by a
  one-word candidate run limb by limb without allocating, and once the rest
  fits in 64 bits the search continues on native integers. It stops when a
  candidate's square exceeds the rest, which is then 1 or a prime; that prime
  is reported as a factor when it is below the bound.
- **Exact roots (`iroot_exact`)**: Cheap tests reject most inputs before a root
  is taken. The trailing zero count must be a multiple of the degree. A square
  must be a quadratic residue modulo 64, 63, 65 and 11, which together pass
  fewer than 1% of non-squares. For another degree $d$, the two smallest primes
  $\ell = kd + 1$ serve as witnesses: an $x$ coprime to $\ell$ is a $d$-th power
  modulo $\ell$ only if $x^{(\ell - 1)/d} \equiv 1 \pmod{\ell}$. A survivor
  takes the floor root with its remainder (Zimmermann's square root for
  squares, the root kernel above otherwise): a zero remainder is the exact
  root.
- **Perfect powers (`perfect_power`)**: Tries each prime exponent up to the bit
  length, odd ones only for a negative input. After a root is found, the search
  continues on the root with the same prime, so composite exponents build up as
  products.
- **Primality (`is_prime`)**: Below $2^{64}$, trial division by the primes up to
  37 and then Miller–Rabin with the first $k$ prime bases, stopping once $n$ is
  below $\psi_k$, the smallest composite that passes those bases (Jaeschke 1993;
  Sorenson and Webster 2017). Twelve bases cover every 64-bit $n$, since
  $\psi_{12} \approx 3.2 \times 10^{23}$, so these answers are proven. Above
  $2^{64}$ the test is Baillie–PSW: trial division by the primes below 1000, a
  strong probable-prime test to base 2, a perfect-square check (without which
  the parameter search below would not end), and a strong Lucas test with
  Selfridge's parameters $P = 1$, $Q = (1 - D)/4$ for the first $D$ in
  $5, -7, 9, -11, \ldots$ with Jacobi symbol $-1$. No composite is known to pass
  Baillie–PSW, but that is not a proof.
- **Jacobi symbol (`jacobi`)**: The binary algorithm with quadratic
  reciprocity, on native words once both operands fit in 64 bits.
- **Rational gcd, lcm and roots**: `rational.gcd` is the gcd of the numerators
  over the lcm of the denominators. This is already in lowest terms, because a
  prime that divides both numerators divides neither denominator; `lcm` is the
  dual. A fraction in lowest terms is a perfect power exactly when its
  numerator and denominator are, which is how `root_exact` decides.

Through `vmap`, `iroot_exact` and `root_exact` give two results: the roots,
with 0 where there is none, and a `Mask` of where there is one (see
[Batches and execution](batches.md)). `trial_division` returns a
`TrialDivision`, which a batch cannot hold, so it applies to one value at a time.
The package-level `gcd` and `lcm` remain the Integer versions, so `vmap[gcd]`
keeps working; the Rational versions are `apn_mojo.rational.gcd` and
`apn_mojo.rational.lcm`.

The suite `tests/test_number_theory.mojo` checks these functions against
`tests/fixtures/number_theory.txt`, which `tests/fixtures/number_theory.py`
writes with gmpy2. The fixture holds the 162 strong pseudoprimes to base 2
below $10^7$, the 43 Carmichael numbers below $10^6$, the 58 strong Lucas
pseudoprimes below $10^6$, the Mersenne prime exponents below 1300, 27 random
primes of 64 to 4096 bits with their odd neighbours, and Jacobi symbols of
multiword values. The suite also compares `is_prime` with a sieve for every
$n < 10^6$, and Baillie–PSW alone for every odd $n < 10^5$, since `is_prime`
never reaches it below $2^{64}$.

## Hashing and dictionary keys

`Integer` implements Mojo's `Hashable` trait for use in `Dict` and `Set`.
Hashing reads the canonical sign, length, and limb words without allocating.
Equal values have equal hashes, regardless of representation. These hashes
are process-local and are unsuitable for persistent storage or cryptography.
For a hash that is the same in every process and release, use `stable_hash`
(see [Float](float.md#representation-identity-and-hashing)).

## Measured performance

The [benchmark results](benchmark-results.md) summarize Integer arithmetic
and number theory, with a full report for individual operations and sizes.
Compare like workloads: division by a small word and GCD of two wide
operands exercise different paths.
