"""The number of threads that batch operations use."""

from ._threads import _parallel_workers, _set_threads


def set_num_threads(n: Int) raises:
    """Set how many threads long batch operations and mappings use, the calling
    thread included.

    Results do not depend on it: a parallel run gives the results and errors of
    a sequential loop. The default is the `APN_MOJO_NUM_THREADS` environment
    variable when it is a positive integer, else one thread per usable physical
    core. `n` may exceed the core count; growing the pool restarts its workers.
    It never waits: during a running operation, even from inside a mapped
    function, the count takes effect from the next operation.

    Args:
        n: The number of threads, at least 1; 1 runs everything on the calling
            thread.

    Raises:
        When `n` is below 1.
    """
    if n < 1:
        raise Error(String("set_num_threads needs at least one thread; got ", n, "."))
    _set_threads(n)


def get_num_threads() -> Int:
    """The number of threads batch operations use, the calling thread included;
    after `set_num_threads` during an operation, the count the next one uses.

    Returns:
        1 when work runs only on the calling thread, as on a platform without
        POSIX threads.
    """
    return _parallel_workers()
