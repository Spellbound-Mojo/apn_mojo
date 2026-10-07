"""Integer factorization within a budget, a prime sieve and next_prime.

`factor` divides out the primes below a bound, reduces perfect powers, then
splits what remains with Pollard and Brent's rho method, `x -> x**2 + c` for
`c = 1, 2, 3, ...`, taking a gcd every 128 steps and backtracking when a gcd
collapses to the whole number. Parts wait on an explicit stack, never in a
recursion. Every final part is classified by `is_prime`; a part the budget did
not split is reported as not prime, so a composite is never called prime and
`is_complete()` says whether the factorization is final.
"""

from ..integer.value import Integer
from .number_theory import trial_division, gcd, remove_factor
from .powers import perfect_power
from .primality import is_prime


struct FactorBudget(ImplicitlyCopyable, Writable):
    """How much work `factor` may spend: the trial-division bound and the
    total number of rho iterations over all parts."""

    var trial_bound: Int
    var rho_iterations: Int
    var ecm_curves: Int

    def __init__(out self, *, trial_bound: Int = 1 << 16, rho_iterations: Int = 1 << 24, ecm_curves: Int = 0):
        """A budget; every argument is optional. `factor` checks it.

        Args:
            trial_bound: Divide out every prime below this bound, at least 2.
            rho_iterations: The rho steps shared by all parts, at least 0;
                2**24 by default, enough for products of four 40-bit primes,
                while a composite rho cannot split gives up after about 8 s at
                256 bits.
            ecm_curves: Elliptic-curve trials; only 0 is available.
        """
        self.trial_bound = trial_bound
        self.rho_iterations = rho_iterations
        self.ecm_curves = ecm_curves

    def write_to(self, mut writer: Some[Writer]):
        """Write the settings, as `print` does.

        Args:
            writer: The destination.
        """
        writer.write("FactorBudget(trial_bound=", self.trial_bound, ", rho_iterations=", self.rho_iterations, ")")


@fieldwise_init
struct FactorTerm(ImplicitlyCopyable, Writable):
    """One part of a factorization: `base**multiplicity`, with whether the
    base is prime (proven below 2**64, Baillie-PSW above)."""

    var base: Integer
    var multiplicity: Int
    var is_prime: Bool

    def write_to(self, mut writer: Some[Writer]):
        """Write `base^multiplicity`, marked `?` when the base may be composite.

        Args:
            writer: The destination.
        """
        writer.write(self.base)
        if self.multiplicity != 1:
            writer.write("^", self.multiplicity)
        if not self.is_prime:
            writer.write("?")


struct Factorization(ImplicitlyCopyable, Writable):
    """The sign and the parts of an integer, ascending by base."""

    var _sign: Int
    var _terms: List[FactorTerm]

    def __init__(out self, *, _sign: Int, var _terms: List[FactorTerm]):
        self._sign = _sign
        self._terms = _terms^

    def __init__(out self, *, copy: Self):
        self._sign = copy._sign
        self._terms = copy._terms.copy()

    def sign(self) -> Int:
        """The sign of the factored integer.

        Returns:
            -1, 0 or 1.
        """
        return self._sign

    def terms(self) -> List[FactorTerm]:
        """The parts, ascending by base, equal bases merged.

        Returns:
            The terms.
        """
        return self._terms.copy()

    def is_complete(self) -> Bool:
        """Whether every base is prime.

        Returns:
            False when the budget left a composite part.
        """
        for term in self._terms:
            if not term.is_prime:
                return False
        return True

    def write_to(self, mut writer: Some[Writer]):
        """Write the factorization as `-1 * 2^3 * 5`, `?` marking parts that
        may be composite.

        Args:
            writer: The destination.
        """
        var first = True
        if self._sign < 0:
            writer.write("-1")
            first = False
        for term in self._terms:
            if not first:
                writer.write(" * ")
            writer.write(term)
            first = False
        if first:
            writer.write("1")


def _brent_rho(n: Integer, mut budget: Int) raises -> Optional[Integer]:
    """A nontrivial factor of an odd composite n, or None when the budget ends.

    Every evaluation of `x**2 + c` counts against the budget. A batch whose
    gcd is n is replayed one step at a time from its start, at most its 128
    steps."""
    var c = 1
    while budget > 0:
        var y = Integer(2)
        var x = y
        var saved = y
        var g = Integer(1)
        var q = Integer(1)
        var r = 1
        var batch = 0
        while g == 1 and budget > 0:
            x = y
            var advance = min(r, budget)
            for _ in range(advance):
                y = (y * y + c) % n
            budget -= advance
            var k = 0
            while k < r and g == 1 and budget > 0:
                saved = y
                batch = min(128, r - k, budget)
                for _ in range(batch):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                budget -= batch
                g = gcd(q, n)
                k += 128
            r *= 2
        if g == n:
            g = Integer(1)
            for _ in range(batch):
                saved = (saved * saved + c) % n
                g = gcd(abs(x - saved), n)
                if g != 1:
                    break
        if g > 1 and g < n:
            return g
        c += 1
    return None


def factor(n: Integer, *, budget: FactorBudget = FactorBudget()) raises -> Factorization:
    """The prime factorization of `n`, as far as the budget allows.

    Trial division by the primes below `budget.trial_bound`, then
    perfect-power reduction, then Pollard-Brent rho on what remains. A part
    the budget did not split is reported with `is_prime = False`, and the
    factorization is then incomplete; composites are never reported as prime.
    `factor(2**64 + 1)` is `274177 * 67280421310721`.

    Args:
        n: A nonzero integer.
        budget: The trial bound and the rho iterations.

    Returns:
        The sign and the parts, ascending by base.

    Raises:
        For `n = 0`, a trial bound below 2, negative rho iterations, or
        elliptic-curve trials, which are not available.
    """
    if not n:
        raise Error("Cannot factor 0; every integer divides it.")
    if budget.trial_bound < 2 or budget.rho_iterations < 0:
        raise Error("Cannot factor with a trial bound below 2 or negative rho iterations.")
    if budget.ecm_curves != 0:
        raise Error("The elliptic-curve method is not available; pass ecm_curves=0 and a larger rho budget.")
    var sign = n.sign()
    var m = abs(n)
    var terms = List[FactorTerm]()
    if m == 1:
        return Factorization(_sign=sign, _terms=terms^)
    var bound = Integer(budget.trial_bound)
    var trial = trial_division(m, bound)
    for pair in trial.factors():
        terms.append(FactorTerm(pair[0], Int(pair[1]), True))
    var remaining = budget.rho_iterations
    # Primes found by rho, divided out of every later part before it is split.
    var found_primes = List[Integer]()
    var stack = List[Tuple[Integer, Int]]()
    stack.append((trial.cofactor(), 1))
    while len(stack):
        var item = stack.pop()
        var v = item[0]
        var k = item[1]
        for prime in found_primes:
            var removed = remove_factor(v, prime)
            if removed[1] > 0:
                terms.append(FactorTerm(prime, k * Int(removed[1]), True))
                v = removed[0]
        if v == 1:
            continue
        # No prime below the bound divides v, so v < bound**2 is prime.
        if v < bound * bound or is_prime(v):
            terms.append(FactorTerm(v, k, True))
            found_primes.append(v)
            continue
        var power = perfect_power(v)
        if power[1] > 1:
            stack.append((power[0], k * Int(power[1])))
            continue
        var d = _brent_rho(v, remaining)
        if d:
            # The factor goes on top, so its primes are known before the cofactor's turn.
            var found = d.take()
            stack.append((v // found, k))
            stack.append((found, k))
        else:
            terms.append(FactorTerm(v, k, False))
    # Merge equal bases and sort ascending (insertion sort; the list is short).
    var merged = List[FactorTerm]()
    for term in terms:
        var placed = False
        for i in range(len(merged)):
            if merged[i].base == term.base:
                merged[i].multiplicity += term.multiplicity
                placed = True
                break
        if not placed:
            merged.append(term)
    for i in range(1, len(merged)):
        var j = i
        while j > 0 and merged[j - 1].base > merged[j].base:
            var swap = merged[j - 1]
            merged[j - 1] = merged[j]
            merged[j] = swap
            j -= 1
    return Factorization(_sign=sign, _terms=merged^)


def primes_below(bound: Int) raises -> List[Int]:
    """Every prime below `bound`, by a segmented sieve of Eratosthenes.

    Args:
        bound: The exclusive limit.

    Returns:
        The primes, ascending.

    Raises:
        Only on a checked size error.
    """
    var result = List[Int]()
    if bound <= 2:
        return result^
    var limit = 1
    while (limit + 1) * (limit + 1) < bound:
        limit += 1
    # Base primes up to sqrt(bound) by a plain sieve.
    var small = List[Bool](length=limit + 1, fill=True)
    var base = List[Int]()
    for i in range(2, limit + 1):
        if small[i]:
            base.append(i)
            var j = i * i
            while j <= limit:
                small[j] = False
                j += i
    comptime segment = 1 << 15
    var low = 2
    var flags = List[Bool](length=segment, fill=True)
    while low < bound:
        var high = min(low + segment, bound)
        for i in range(high - low):
            flags[i] = True
        for p in base:
            var start = max(p * p, ((low + p - 1) // p) * p)
            var j = start
            while j < high:
                flags[j - low] = False
                j += p
        for i in range(high - low):
            if flags[i]:
                result.append(low + i)
        low = high
    return result^


def next_prime(n: Integer) raises -> Integer:
    """The least prime above `n`, testing a wheel of 30 with `is_prime`.

    Args:
        n: Any integer.

    Returns:
        The next prime; 2 for every `n < 2`.

    Raises:
        Only on a checked size error.
    """
    if n < 2:
        return Integer(2)
    if n < 3:
        return Integer(3)
    if n < 5:
        return Integer(5)
    var offsets: List[Int] = [1, 7, 11, 13, 17, 19, 23, 29]
    var block = (n // 30) * 30
    while True:
        for d in offsets:
            var candidate = block + d
            if candidate > n and is_prime(candidate):
                return candidate
        block += 30
