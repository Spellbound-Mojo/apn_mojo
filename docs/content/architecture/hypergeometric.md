# Hypergeometric functions

The hypergeometric implementation combines a shared series engine with
transformations that make each problem easier to evaluate. `hyp1f1` and
`hyp2f1` supply the confluent and Gauss hypergeometric functions;
`gammainc`, `gammaincc`, and `betainc` reuse the same machinery for the
regularized incomplete gamma and beta functions. All five support real
Float and Ball arguments.

Every numerical path produces an enclosure. A Ball result retains that
bound; a Float result uses it to certify rounding, increasing the working
precision when necessary. Choosing a promising formula and proving its
answer are separate steps: estimates guide the computation, while explicit
remainder bounds and outward rounding establish correctness.

## Implementation structure

The source files below are relative to `src/apn_mojo/`.

| File | Responsibility |
|---|---|
| `ball/_hypergeometric.mojo` | Series planning, tail bounds, Kummer's U expansion, and the transformations for `hyp1f1` and `hyp2f1` |
| `ball/_fixed_series.mojo` | Fixed-point summation with bounds for truncation and input uncertainty |
| `ball/_incomplete.mojo` | Incomplete gamma and beta formulas, complements, and monotone interval evaluation |
| `float/_special.mojo` | Exact cases, domain and range handling, and adapters to the rounding loop |
| `ball/_certified.mojo` | Shared enclosure-to-Float certification and precision retries |

Gamma, reciprocal gamma, digamma, and elementary functions use the existing
scalar kernels. Batches reach these functions through the shared
vectorization machinery. Production evaluation is pure Mojo.

### Precision and exact values

Ball entry points add eight working bits once, through `_BALL_GUARD`, then
round the final enclosure to the requested precision. Internal kernels take
the precision supplied by their caller. Extra bits are reserved where a
formula needs them, particularly for cancellation within a series or when
forming a small complement.

Below, $w$ denotes the precision requested of a numerical kernel, before
any additional guard chosen for its formula.

Float entry points first handle domain boundaries and recognized exact
values. Otherwise, they pass an enclosure evaluator to the Ziv loop in
[Correct rounding](certified.md). That loop returns only when the enclosure
certifies the requested value and rounding status. If it cannot do so
within the context's precision budget, it raises.

Exact cases matter for termination. An enclosure that straddles an exactly
representable answer can keep straddling a rounding boundary at every
precision. Terminating polynomials and other recognized rational values
therefore use Integer and Rational arithmetic before the rounding loop.
This recognition is bounded: the shared polynomial evaluator allows degree
at most $2^{16}$ and an estimated exact size of at most $2^{20}$ bits.
Function-specific checks may impose smaller limits.

## The series engine

Write $(a)_k=a(a+1)\cdots(a+k-1)$, with $(a)_0=1$. The shared recurrence is

$$
T_0=1,\qquad
T_{k+1}=T_k\,z\,
\frac{\prod_{i=1}^{r}(u_i+k)}
     {(k+1)\prod_{j=1}^{s}(l_j+k)},\qquad
S=\sum_{k=0}^{\infty}T_k.
$$

A summation kernel returns an enclosure of the partial sum
$S_N=\sum_{k=0}^{N-1}T_k$ and a bound on the first omitted term $T_N$.
The caller supplies the remainder theorem: a geometric bound for a
convergent series, or an asymptotic bound for a finite expansion.

An exact upper parameter $-n$ terminates the series after $T_n$. The finite
summation path stops there without forming the next ratio. This also
handles accepted polynomial cases in which that next ratio would contain
an undefined $0/0$ at a lower-parameter pole.

### Bounding a convergent tail

For $r\le s+1$, include the factorial as one more lower parameter:
$(v_1,\ldots,v_{s+1})=(l_1,\ldots,l_s,1)$. Pair each upper parameter with a
lower parameter, and choose $N$ so every $v_j+N>0$. Then, for $k\ge N$,

$$
\frac{|u_i+k|}{v_i+k}
\le 1+\frac{|u_i-v_i|}{v_i+N}.
$$

Consequently, an upper bound for every subsequent term ratio is

$$
D_N=|z|
\prod_{i=1}^{r}\left(1+\frac{|u_i-v_i|}{v_i+N}\right)
\prod_{j=r+1}^{s+1}\frac{1}{v_j+N}.
$$

If $D_N<1$, summing the geometric majorant gives

$$
\left|S-S_N\right|\le\frac{|T_N|}{1-D_N}.
$$

This is the real-parameter specialization of the bound in
[Johansson, *Computing hypergeometric functions rigorously*, Theorem 1](https://arxiv.org/abs/1606.06977).
For interval inputs, `_ratio_bound` uses upper bounds on the numerator
magnitudes and strictly positive lower bounds on the denominators. The
resulting tail bound is added to the partial sum's radius.

If the planned $N$ does not establish $D_N<1$, the engine doubles $N$ and
recomputes the sum, up to the $2^{22}$-term limit. For fixed parameters,
$D_N\to0$ when $r\le s$, and $D_N\to|z|$ when $r=s+1$. These limits explain
convergence, but do not guarantee that a usable bound will be reached within
the implementation's budget.

### Planning the work

`_plan` walks the recurrence in native double precision, tracking
$\log_2|T_k|$ and a signed partial sum scaled by the largest term. This
avoids overflowing the estimate when intermediate terms greatly exceed the
answer. The estimated loss

$$
\log_2\frac{\max_k|T_k|}{|S|}
$$

sets the cancellation guard, together with a term-count allowance and a
small fixed reserve. For a convergent series, the planner looks for an
estimated tail at least $w+8$ bits below the sum. It also keeps the estimated
ratio away from one: the current test requires $\log_2 D_N<-0.0015$.

Near an integer parameter, conversion to a double can erase the distance
to that integer. The planner therefore keeps the sign and logarithm of the
exact midpoint defect $p-\operatorname{round}(p)$. When $p+k$ reaches that
defect, it uses this record instead of subtracting rounded doubles. A nearby
pole or terminating parameter is then distinguishable from an exact one.

These estimates do not certify the tail or the final rounding. The
summation and remainder calculations produce their own bounds; a poor plan
can leave a wide enclosure or fail to find a usable path.

### Summation kernels

The engine first tries fixed-point summation. If its workspace or range
checks fail, it tries exact dyadic term ratios, then ordinary Ball
operations. All three implement the same recurrence and return the same
partial-sum and omitted-term contract.

For dyadic parameters $p=N_p/2^{d_p}$, the ratio can be written

$$
\frac{T_{k+1}}{T_k}=2^h\frac{P(k)}{Q(k)},\qquad
h=\sum_j d_{l_j}-\sum_i d_{u_i}-d_z,
$$

$$
P(k)=N_z\prod_i(N_{u_i}+k2^{d_{u_i}}),\qquad
Q(k)=(k+1)\prod_j(N_{l_j}+k2^{d_{l_j}}).
$$

The integer-ratio path replaces several Ball operations with one Integer
multiplication and division. It limits the estimated denominator size to
192 bits and numerator size to $w+128+\min(w,512)$ bits. The argument $z$
is folded into $P$ when it fits; otherwise it remains a separate Ball
multiplication. Small binary shifts go into $P$ or $Q$, and larger shifts
scale the term directly. In the general Ball path, shifted parameters are
incremented in place rather than reconstructed for each term.

### Fixed-point summation and its error budget

`_fixed_terms` stores a signed integer $V_k$ representing $T_k2^{-E}$.
The scale $E$ comes from the estimated sum size and working precision;
the largest estimated term determines the required limb capacity. Terms,
stepped dyadic factors, division scratch space, and two accumulators share
one stack block of 4096 64-bit limbs. Positive and negative terms are added
separately and subtracted at the end, so accumulation itself is exact.

Each recurrence step applies factors separately: multiply by a dyadic
numerator and shift, or shift and divide by a denominator. One-limb factors
use the corresponding limb operations. This keeps the working term near
its required width and avoids allocating an Integer or Ball for each
factor operation.

There are two distinct errors to carry. First, if $e_k$ bounds the term's
error in fixed-point units, an operation with nonnegative magnitude factor
$g$ updates that bound as

$$
e\gets ge+1
$$

when it can discard a fractional unit, or $e\gets ge$ when it is exact.
Starting from $e_0=0$, the partial sum's truncation error is bounded by
$2^E\sum_{k<N}e_k$.

Second, uncertain inputs perturb the midpoint recurrence. Writing bars for
midpoints and $r$ for radii, a bound on one step's relative perturbation is

$$
1+\delta_k=
\left(1+\frac{r_z}{|\bar z|}\right)
\prod_i\left(1+\frac{r_{u_i}}{|\bar u_i+k|}\right)
\prod_j\frac{1}{1-r_{l_j}/|\bar l_j+k|},
$$

where the denominators must be nonzero and each lower-parameter relative
radius must be less than one. With $R_0=0$, successive steps give

$$
R_{k+1}=R_k+\delta_k+R_k\delta_k.
$$

The expanded form preserves a small $\delta_k$ that would disappear in a
floating-point evaluation of $1+\delta_k$. Input uncertainty contributes
at most

$$
2^E\sum_{k<N}R_k\bigl(|V_k|+e_k\bigr)
$$

to the radius, in addition to truncation and final midpoint rounding.
The first omitted term is bounded by
$2^E(|V_N|+e_N)(1+R_N)$.

The implementation obtains magnitude bounds from the limbs, using native
doubles with explicit outward margins of $1\pm2^{-47}$. It tracks the
$R_k|V_k|2^E$ contribution in value units so that a large fixed-point integer
does not itself overflow the error calculation. The kernel declines work
above 4500 bits, outside its dyadic exponent and shift limits, beyond the
stack capacity, or when required magnitude and error bounds are unusable.
These are kernel selection limits; the caller can still use Ball summation.

## Kummer's U and asymptotic remainders

The internal U evaluator supplies the large-argument path for `hyp1f1`.
It computes a finite expansion of the normalized function

$$
z^aU(a,b,z)=\sum_{k=0}^{N-1}t_k+R_N,\qquad
t_k=\frac{(a)_k(a-b+1)_k}{k!}(-z)^{-k}.
$$

This is generally a divergent asymptotic series. The planner seeks a small
omitted term before eventual growth makes the expansion unsuitable. A
small term alone is insufficient: `_u_expansion` widens the sum using
Olver's remainder bound from [DLMF 13.7(ii)](https://dlmf.nist.gov/13.7#ii),
in normalized form,

$$
|R_N|\le 2\alpha C_N|t_N|
\exp\left(\frac{2\alpha\rho C_1}{|z|}\right).
$$

Only two argument regions are needed: $z=y>0$ and the two limits
$z=ye^{\pm i\pi}$ on the negative real cut. Set

$$
\sigma=\frac{|b-2a|}{y},\qquad
\nu=\left(\frac{1+\sqrt{1-4\sigma^2}}{2}\right)^{-1/2},\qquad
\chi(N)=\frac{\sqrt\pi\,\Gamma(N/2+1)}{\Gamma(N/2+1/2)}.
$$

The implementation requires $\sigma\le1/2$ in both regions and uses

$$
\begin{array}{c|cc}
 & C_N & \tau\\ \hline
z=y & 1 & \sigma\\
z=ye^{\pm i\pi} & (\chi(N)+\sigma\nu^2N)\nu^N & \nu\sigma
\end{array}
$$

with

$$
\alpha=\frac{1}{1-\tau},\qquad
\rho=\frac{|2a^2-2ab+b|}{2}
     +\frac{\tau(1+\tau/4)}{(1-\tau)^2}.
$$

The cut lies in the closures of DLMF's $R_3$ regions, so the factor $\nu^N$
is required. To avoid a gamma quotient in the error bound, the kernel uses
$\chi(N)<\sqrt{\pi(N+2)/2}$, from
[Gautschi's inequality](https://dlmf.nist.gov/5.6#E4), and $\chi(1)=\pi/2$
for $C_1$. Interval evaluation uses an upper bound on $\sigma$ and a lower
bound on $y$.

On the cut, the finite coefficients are real after the phase normalization;
the bound also covers the real part of the complex remainder. A terminating
expansion needs no remainder. If the region test or asymptotic plan fails,
the caller tries the convergent series.

## hyp1f1

`hyp1f1(a, b, x)` evaluates the unregularized function

$$
M(a,b,x)={}_1F_1(a;b;x)
=\sum_{k=0}^{\infty}\frac{(a)_k}{(b)_k}\frac{x^k}{k!}.
$$

For a strictly negative argument, Kummer's transformation

$$
M(a,b,x)=e^xM(b-a,b,-x)
$$

moves the series to a positive argument and removes the alternating sign
caused by $x^k$. Parameter factors can still introduce cancellation. A Ball
argument that straddles zero goes directly through the series.

The dispatcher normally plans the series first and tries the U expansions
if the estimate exceeds roughly $w$ terms. For arguments already large
relative to $w+|a|+|b|$, it tries the expansions before paying for that
planning pass. Both tests choose work; the remainder bounds decide the
quality of the enclosure.

### Combining the two U expansions

For $x>0$, taking the real part of
[DLMF 13.2.41](https://dlmf.nist.gov/13.2#E41) gives

$$
\begin{aligned}
M(a,b,x)=\Gamma(b)\biggl[
 &\frac{e^x x^{a-b}}{\Gamma(a)}(S_1+E_1)\\
+{}&\frac{\cos(\pi a)x^{-a}}{\Gamma(b-a)}(S_2+E_2)
\biggr],
\end{aligned}
$$

where

$$
S_1=\sum_{k<N_1}\frac{(b-a)_k(1-a)_k}{k!x^k},\qquad
S_2=\sum_{k<N_2}\frac{(a)_k(a-b+1)_k}{k!(-x)^k}.
$$

The first remainder uses the cut bound for $U(b-a,b,xe^{\pm i\pi})$;
the second uses the positive-axis bound for $U(a,b,x)$. The phases in the
first contribution cancel, leaving a real enclosure. For negative input,
Kummer's outer exponential cancels the growing exponential algebraically,
before either is evaluated.

Gamma denominators are evaluated as reciprocal gamma. At a non-positive
integer its value is exactly zero; elsewhere below zero the kernel uses
$1/\Gamma(t)=\sin(\pi t)\Gamma(1-t)/\pi$. Exact arguments use exact reduction
for $\sin(\pi t)$ and $\cos(\pi t)$. This avoids introducing a gamma pole
where the connection coefficient actually vanishes.

### Recognizing rational answers

Besides the terminating polynomial for $a=-n$, the exact dispatcher
recognizes zeros of Kummer's transformed polynomial when $b-a$ is a
non-positive integer. For integers $0<a<b$, it also checks whether the
exponential contribution vanishes. With
$q(t)=t^{a-1}(1-t)^{b-a-1}$, repeated integration by parts gives

$$
M(a,b,x)=\frac{\Gamma(b)}{\Gamma(a)\Gamma(b-a)}
\left[e^x A(x)-B(x)\right],
$$

$$
A(x)=\sum_{j=0}^{b-2}\frac{(-1)^jq^{(j)}(1)}{x^{j+1}},\qquad
B(x)=\sum_{j=0}^{b-2}\frac{(-1)^jq^{(j)}(0)}{x^{j+1}},\quad x\ne0.
$$

The coefficients are rational and are checked exactly. If $A(x)=0$, the
remaining rational value can be rounded directly; $M(2,4,2)=3$ is one such
case. Polynomial handling also preserves the defined value
$M(-n,-n,x)=\sum_{k=0}^{n}x^k/k!$ at coincident negative integer parameters.

## gammainc and gammaincc

For $a>0$ and $x>0$, the two regularized functions are

$$
P(a,x)=\frac{1}{\Gamma(a)}\int_0^x t^{a-1}e^{-t}\,dt,\qquad
Q(a,x)=\frac{1}{\Gamma(a)}\int_x^{\infty}t^{a-1}e^{-t}\,dt,
\qquad P+Q=1.
$$

The lower function uses a positive-term series,

$$
P(a,x)=\frac{\exp(a\log x-x)}{\Gamma(a+1)}
M(1,a+1,x),\qquad
M(1,a+1,x)=\sum_{k=0}^{\infty}\frac{x^k}{(a+1)_k}.
$$

For $x\ge a+1$, the dispatcher first tries the upper expansion

$$
Q(a,x)=\frac{\exp((a-1)\log x-x)}{\Gamma(a)}
\left[\sum_{k=0}^{N-1}\frac{u_k}{x^k}+R_N\right],\qquad
u_k=\prod_{j=1}^{k}(a-j).
$$

For real $a$ and $x>0$, choosing $N\ge a-1$ bounds the remainder by
$|u_N/x^N|$. The kernel enforces this index condition even if the planner
suggests fewer terms. Positive integer $a$ makes the expansion terminate.
The bound comes from [DLMF 8.11.2–3](https://dlmf.nist.gov/8.11#i).

If the expansion has no usable plan, the implementation computes $P$ by
its convergent series. When $Q=1-P$ is requested in the upper region, it
adds working bits based on the estimate

$$
-\log_2 Q\approx
-\frac{(a-1)\log x-x-\log\Gamma(a)}{\log 2}.
$$

The estimate uses a short double-precision Stirling calculation. The
subtraction itself remains a Ball operation. The switch at $a+1$ is a
method-selection rule, not a proof that the directly evaluated function
is below one half.

### Enclosing parameter intervals

$P$ increases with $x$ and decreases with $a$; $Q$ has the opposite
monotonicity. For a valid parameter box, the extrema therefore occur at

$$
\begin{aligned}
P_{\min}&=P(a_{\max},x_{\min}),&
P_{\max}&=P(a_{\min},x_{\max}),\\
Q_{\min}&=Q(a_{\min},x_{\max}),&
Q_{\max}&=Q(a_{\max},x_{\min}).
\end{aligned}
$$

The Ball evaluator encloses these two exact corners, forms their hull, and
intersects it with $[0,1]$. Boundary cases such as $x=0$ are handled before
the interior formulas. This avoids propagating parameter uncertainty
through every term of the incomplete gamma series.

## hyp2f1

`hyp2f1(a, b, c, x)` evaluates

$$
F(a,b;c;x)={}_2F_1(a,b;c;x)
=\sum_{k=0}^{\infty}\frac{(a)_k(b)_k}{(c)_k}\frac{x^k}{k!}.
$$

The real implementation uses the series near zero and transformations for
more distant arguments. Terminating polynomials are evaluated directly;
non-polynomial evaluation is restricted to $x\le1$.

| Argument | Evaluation path |
|---|---|
| $-1/2\le x\le1/2$ | Direct series |
| $x<-1/2$ | Pfaff transformation to $y=x/(x-1)$, then the same dispatch at $y$ |
| $1/2<x<1$ | Connection to series at $1-x$, with the direct series as fallback |
| $x=1$ | Exact cases or Gauss's gamma quotient when $c-a-b>0$ |

The Pfaff transformation is

$$
F(a,b;c;x)=(1-x)^{-a}
F\left(a,c-b;c;\frac{x}{x-1}\right).
$$

The implementation exchanges $a$ and $b$ when that gives a smaller absolute
exponent in the prefactor. For interval arguments, the required region
conditions must hold throughout the interval. The new argument is formed as
$1-1/(1-x)$, which uses $x$ once: as $x/(x-1)$, ball arithmetic would treat
the two occurrences of $x$ as independent, and $x\in[-4.87,-3.79]$ would give
$[0.58,1.02]$ instead of $[0.79,0.83]$, past 1, where neither the series nor
the connection applies. Exact identities such as
$F(a,b;a;x)=(1-x)^{-b}$ bypass the series.

### Connection near one

Set $s=c-a-b$ and $t=1-x$. For noninteger $s$, the connection formula is

$$
\begin{aligned}
F(a,b;c;x)=\Gamma(c)\biggl[
 &\frac{\Gamma(s)}{\Gamma(c-a)\Gamma(c-b)}F(a,b;1-s;t)\\
+{}&t^s\frac{\Gamma(-s)}{\Gamma(a)\Gamma(b)}
 F(c-a,c-b;1+s;t)\biggr].
\end{aligned}
$$

Both inner series have a small argument when $x$ is close to one. Of
$\Gamma(s)$ and $\Gamma(-s)$, only the positive-argument gamma is evaluated
directly; the other follows from

$$
\Gamma(s)\Gamma(-s)=-\frac{\pi}{s\sin(\pi s)}.
$$

Near an integer $s$, the two connection terms can be much larger than their
sum, requiring additional precision in the Float retry loop. If a Ball for
$s$ contains an integer without being that exact integer, the connection
path declines and the evaluator falls back to the convergent series.
Pfaff's and the connection formulas are given in
[DLMF 15.8.1 and 15.8.4](https://dlmf.nist.gov/15.8#i).

### The logarithmic series for integer differences

At $s=m\ge0$, the apparent poles in the connection formula are removed by
its limiting form, [DLMF 15.8.10](https://dlmf.nist.gov/15.8#E10):

$$
\begin{aligned}
\frac{F(a,b;a+b+m;1-t)}{\Gamma(a+b+m)}
={}&\frac{1}{\Gamma(a+m)\Gamma(b+m)}
\sum_{k=0}^{m-1}\frac{(a)_k(b)_k(m-k-1)!}{k!}(-t)^k\\
&-\frac{(-t)^m}{\Gamma(a)\Gamma(b)}\sum_{k=0}^{\infty}A_kL_k,
\end{aligned}
$$

$$
A_k=\frac{(a+m)_k(b+m)_k}{k!(k+m)!}t^k,\qquad
L_k=\log t-\psi(k+1)-\psi(k+m+1)+\psi(a+k+m)+\psi(b+k+m).
$$

The finite sum is empty when $m=0$. For negative integer $s$, Euler's
transformation $F=(1-x)^sF(c-a,c-b;c;x)$ first makes the difference
nonnegative.

Only the initial digamma values need full evaluations. With
$a_k=a+m+k$ and $b_k=b+m+k$, the recurrence

$$
L_{k+1}=L_k+\frac{a_k+b_k}{a_kb_k}
-\frac{2k+m+2}{(k+1)(k+m+1)}
$$

shares $a_kb_k$ with the coefficient update. The implementation advances
these factors in place and sums $A_kL_k$ with the shared Ball dot product.

To bound the tail, use $\psi'(v)<h(v)=1/v+1/v^2$ for $v>0$. The mean value
theorem bounds each digamma difference, giving, for every $k\ge N$,

$$
|L_k|\le B_N=|\log t|
+|a+m-1|h\bigl(N+\min(a+m,1)\bigr)
+|b-1|h\bigl(N+m+\min(b,1)\bigr).
$$

Once both arguments of $h$ are positive and the shared ratio bound gives
$D_N<1$, the remaining sum satisfies

$$
\left|\sum_{k=N}^{\infty}A_kL_k\right|
\le\frac{B_N|A_N|}{1-D_N}.
$$

Ball evaluation replaces the quantities in this bound by the appropriate
upper or lower endpoints. If these conditions cannot be established, the
connection path declines and the ordinary series remains available.

### Values at one and exact transformations

At $x=1$ and $s>0$, Gauss's formula gives

$$
F(a,b;c;1)=\frac{\Gamma(c)\Gamma(s)}{\Gamma(c-a)\Gamma(c-b)}.
$$

The exact dispatcher recognizes terminating polynomials, rational powers
from $c=a$ or $c=b$, Euler-transformed polynomials, and selected rational
cases of this gamma quotient. Positive integer upper parameters can turn
the quotient into a ratio of rising factorials. It also handles the
Chu–Vandermonde cases at one. These checks do not classify every algebraic
or rational value of $F$; unrecognized boundary values can exhaust the
rounding budget.

## betainc

For $a,b>0$ and $0<x<1$, the regularized incomplete beta function is

$$
I_x(a,b)=\frac{1}{B(a,b)}\int_0^x t^{a-1}(1-t)^{b-1}\,dt.
$$

The numerical path uses the representation

$$
I_x(a,b)=\exp\bigl(a\log x+b\log(1-x)\bigr)
\frac{\Gamma(a+b)}{\Gamma(a+1)\Gamma(b)}
F(a+b,1;a+1;x).
$$

The prefactor uses `log1p(-x)` for $\log(1-x)$. The inner series has positive
terms with ratio $x(a+b+k)/(a+1+k)$. When
$x\le(a+1)/(a+b+2)$, the first ratio is below one; all later ratios are
below one as well. Above this threshold, the evaluator uses the symmetry
$I_x(a,b)=1-I_{1-x}(b,a)$. These identities are
[DLMF 8.17.4 and 8.17.8](https://dlmf.nist.gov/8.17).

The switch is chosen from midpoint estimates, so it controls efficiency;
the full Ball computation still carries uncertainty and subtraction error.
It does not establish a universal one-bit bound on cancellation in the
complement. For an exact dyadic $x\in[1/2,1)$, $1-x$ can be formed exactly
at sufficient precision by Sterbenz's lemma; a general Ball difference
still has a radius.

For positive integer shapes, the exact path evaluates

$$
I_x(a,b)=\sum_{j=a}^{a+b-1}\binom{a+b-1}{j}
x^j(1-x)^{a+b-1-j}.
$$

When only $b$ is a positive integer, it instead tries

$$
I_x(a,b)=\frac{x^a}{aB(a,b)}F(a,1-b;a+1;x),
$$

whose hypergeometric factor terminates. This produces an exact rational
answer when $x^a$ is rational; symmetry covers integer $a$ in the same
way. Both paths enforce degree and exact-size limits.

For interval inputs, $I$ increases with $x$ and $b$ and decreases with $a$.
The Ball evaluator therefore uses just the two corners

$$
I_{\min}=I_{x_{\min}}(a_{\max},b_{\min}),\qquad
I_{\max}=I_{x_{\max}}(a_{\min},b_{\max}),
$$

then intersects their hull with $[0,1]$. When $a$ and $b$ are exact, the
two corners share one enclosure of $1/B(a,b)$. The direct and reflected
prefactors then divide that shared value by $a$ and $b$, respectively.

## Extreme values and implementation limits

A very small result, or a result extremely close to one, need not be
constructed at its full exponent before rounding. The incomplete gamma
front end uses bounds

$$
P(a,x)\le\frac{x^a}{\Gamma(a+1)},\qquad
Q(a,x)\le\frac{2x^{a-1}e^{-x}}{\Gamma(a)}\quad(x>2a).
$$

The second follows by writing the upper integral with $t=x+u$: for $a>1$,
$(1+u/x)^{a-1}\le e^{(a-1)u/x}$, and for $0<a\le1$ that factor is at most
one. These bounds yield integer upper bounds on binary exponents, which
can decide underflow or rounding just below one. `_round_near` uses both
the magnitude bound and the known side of the exact value, preserving
directed rounding and status flags.

For $x\ge2^{56}$ and $0<a<2^{40}$, the gamma enclosure uses the coarser
bound $0<Q(a,x)<2^{-2^{55}}$, avoiding an exponential outside its working
range. This can suffice to round a result near zero or one, but cannot
resolve $Q$ in every user-defined Float format.

The beta front end uses a double estimate only to decide whether to try
such a shortcut. A separate 64-bit enclosure of the directly evaluated
side must establish the bound before the shortcut returns a result.
For very large negative `hyp1f1` arguments, an exponentially small
contribution can similarly be retained as a power-of-two radius instead
of forming the exponential itself.

The current algorithms have practical limits:

- The shared planner and convergent tail retry stop at $2^{22}$ terms.
  Large parameters or arguments close to a convergence boundary can reach
  this limit even when the mathematical series converges.
- Large-parameter transition regions use the existing series paths. There
  are no uniform asymptotic expansions or numerical-integration fallbacks
  for incomplete gamma near $x\approx a$, incomplete beta near its switch
  point, or large hypergeometric parameters.
- Incomplete beta uses the hypergeometric series, without a continued
  fraction path. A ratio close to one can make the sum expensive.
- Extreme exponential arguments have additional range checks and
  conservative bounds. A wide user-supplied Float exponent range does not
  guarantee that every internal enclosure can be refined to that range.
- Ball inputs that cross an unsupported singularity or domain boundary
  can produce an indeterminate enclosure. Float calls that cannot certify
  a finite answer within their precision budget raise rather than return
  an uncertified approximation.

The [Float](../reference/float.md) and [Ball](../reference/ball.md) references
specify public domains, boundary values, and errors. The algorithms here
share the enclosure and rounding contracts described in
[Ball arithmetic](balls.md) and [Correct rounding](certified.md).
