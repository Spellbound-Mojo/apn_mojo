"""Binary splitting of series with rational terms (Haible and Papanikolaou).

The one engine for every such series: the constants, and series of exact
rational arguments. A series is

    S = sum over k in [start, end) of  a(k)/b(k) * p(start)...p(k) / (q(start)...q(k))

with integer `p`, `q`, `a` and `b`. Over an interval the engine keeps four
integers, `P = prod p`, `Q = prod q`, `B = prod b` and

    T = B Q sum a(k)/b(k) * p(start)...p(k) / (q(start)...q(k)),

so `S = T / (B Q)`. Two adjacent intervals merge with
`P = Pl Pr`, `Q = Ql Qr`, `B = Bl Br` and `T = Br Qr Tl + Bl Pl Tr`.

The leaves merge like a binary counter: each new leaf merges with the top of
an explicit stack while the two cover equal numbers of terms. The products
then pair operands of equal size, as a recursive split would, and the stack
holds at most `log2(n) + 1` intervals.
"""

from ..integer.value import Integer


trait _SeriesTerms:
    """The integer sequences of a series; `b` is 1 for most series."""

    def p(self, k: Int) raises -> Integer:
        ...

    def q(self, k: Int) raises -> Integer:
        ...

    def a(self, k: Int) raises -> Integer:
        ...

    def b(self, k: Int) raises -> Integer:
        ...


@fieldwise_init
struct _Split(ImplicitlyCopyable):
    """`P`, `Q`, `B` and `T` over an interval of terms."""

    var p: Integer
    var q: Integer
    var b: Integer
    var t: Integer
    var count: Int


def _merge(left: _Split, right: _Split) raises -> _Split:
    var t: Integer
    var b: Integer
    if left.b._is_one() and right.b._is_one():
        b = Integer(1)
        t = right.q * left.t + left.p * right.t
    else:
        b = left.b * right.b
        t = right.b * right.q * left.t + left.b * left.p * right.t
    return _Split(left.p * right.p, left.q * right.q, b, t, left.count + right.count)


def _binary_split[S: _SeriesTerms](series: S, start: Int, end: Int) raises -> _Split:
    """`P`, `Q`, `B` and `T` of the terms in `[start, end)`.

    Args:
        series: The term sequences.
        start: The first term.
        end: One past the last term, at least `start + 1`.

    Returns:
        The interval's integers.

    Raises:
        On an empty interval.
    """
    if end <= start:
        raise Error("Binary splitting needs at least one term.")
    var stack = List[_Split]()
    for k in range(start, end):
        var p = series.p(k)
        var leaf = _Split(p, series.q(k), series.b(k), series.a(k) * p, 1)
        while len(stack) and stack[len(stack) - 1].count == leaf.count:
            leaf = _merge(stack.pop(), leaf)
        stack.append(leaf^)
    var result = stack.pop()
    while len(stack):
        result = _merge(stack.pop(), result)
    return result^
