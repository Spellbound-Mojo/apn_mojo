"""A persistent worker pool for elementwise batch work.

Platform calls (start and join a thread, one mutex, two condition variables,
the usable CPU count) are confined to the OS layer below. Linux and macOS use
POSIX threads; on any other target `_parallel_for` runs every job on the
calling thread, so results are the same everywhere and a Windows backend only
replaces the OS layer.

Workers are created on the first parallel call and reached through one named
global: `APN_MOJO_NUM_THREADS` threads, the caller included, or one per usable
physical core. `_set_threads` limits how many join a job, or restarts the
workers at a larger count between jobs; called during a job, even from inside
it, it records the count for the next job instead of waiting. A job is a plain
function pointer and an untyped context, split into
chunks claimed through an atomic counter; the calling thread works too. After
a job, workers poll for the next one briefly before sleeping, so back-to-back
calls start in microseconds. A call that finds the pool busy (a mapped
function calling a parallel batch operation, or several user threads at once)
runs inline, so nesting cannot deadlock.
"""

from std.atomic import Atomic, Ordering
from std.ffi import external_call, c_size_t, _get_global
from std.bit import pop_count
from std.collections import Array
from std.sys import size_of, llvm_intrinsic
from std.time import perf_counter_ns
from std.sys.info import CompilationTarget, num_physical_cores
from std.os import getenv

comptime _Context = Optional[Pointer[NoneType, MutUntrackedOrigin]]
# A task runs [begin, end) as participant `worker`: 0 is the calling thread.
comptime _Task = def(_Context, Int, Int, Int) thin -> None

# ------------------------------------------------------------------ OS layer

comptime _THREADS = CompilationTarget.is_linux() or CompilationTarget.is_macos()
# Opaque pthread_mutex_t / pthread_cond_t storage: 40 to 64 bytes on every
# supported platform.
comptime _SYNC_BYTES = 128


@always_inline
def _raw(address: Int) -> _Context:
    return _Context(Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=address))


def _zeroed(bytes: Int) -> Int:
    var memory = external_call["calloc", _Context](c_size_t(1), c_size_t(bytes))
    return Int(memory.value()) if memory else 0


def _usable_cpus() -> Int:
    """Physical cores, limited to the CPUs this process may run on.

    Polling workers sharing one CPU with the caller would only slow it down, so
    a process pinned to one CPU gets no pool.
    """
    var cores = num_physical_cores()
    comptime if CompilationTarget.is_linux():
        var mask = Array[UInt64, 16](fill=0)
        if external_call["sched_getaffinity", Int32](Int32(0), c_size_t(128), Pointer(to=mask)) == 0:
            var allowed = 0
            for i in range(16):
                allowed += Int(pop_count(mask[i]))
            return max(1, min(cores, allowed))
    return max(1, cores)


@always_inline
def _lock(mutex: Int):
    _ = external_call["pthread_mutex_lock", Int32](_raw(mutex))


@always_inline
def _unlock(mutex: Int):
    _ = external_call["pthread_mutex_unlock", Int32](_raw(mutex))


@always_inline
def _wait(condition: Int, mutex: Int):
    _ = external_call["pthread_cond_wait", Int32](_raw(condition), _raw(mutex))


@always_inline
def _broadcast(condition: Int):
    _ = external_call["pthread_cond_broadcast", Int32](_raw(condition))


# ---------------------------------------------------------------------- pool

# How long a worker polls for the next job, or the caller for the last worker,
# before blocking, so a back-to-back job pays no wake-up. A time, not a count:
# each poll pauses the core, and a pause lasts from 10 to 140 cycles by CPU.
comptime _SPIN_NS = 20_000


@always_inline
def _relax():
    """Pause the core between polls: less power, the core to a hyperthread
    sibling, and no flood of loads on the polled line."""
    comptime if CompilationTarget.is_x86():
        llvm_intrinsic["llvm.x86.sse2.pause", NoneType]()
    elif CompilationTarget.is_apple_silicon() or CompilationTarget.has_neon():
        llvm_intrinsic["llvm.aarch64.hint", NoneType](Int32(1))


@always_inline
def _spun_out(spins: Int, start: Int) -> Bool:
    """Whether a poll loop has spun its time: the clock is read every 16 polls."""
    return spins & 15 == 0 and perf_counter_ns() - start > _SPIN_NS


@fieldwise_init
struct _Pool(Movable):
    var workers: Int
    var threads: Int  # address of `workers` pthread_t handles
    var mutex: Int
    var wake: Int
    var done: Int
    var busy: Atomic[Int64]
    var generation: Atomic[Int64]
    var active: Atomic[Int64]
    var sleepers: Int
    var waiting: Bool
    var shutdown: Bool
    var task: _Task
    var context: _Context
    var count: Int
    var chunk: Int
    var next: Atomic[Int64]
    # Every participant runs the task once, with its own index (_parallel_each).
    var each: Bool
    # Address of one _Seat per worker thread.
    var seats: Int
    # Threads that take part in a chunked job, the caller included: at most
    # workers + 1. The others skip its chunks but still run _parallel_each.
    var participants: Int
    # A thread count set while the pool was busy, applied by the next job to
    # reserve it; 0 when none is pending.
    var requested: Atomic[Int64]


@fieldwise_init
struct _Seat(Movable):
    """What a worker thread starts with: its pool, its fixed index, and the
    job generation it has already seen (a restarted pool's is not zero)."""
    var pool: Int
    var index: Int
    var generation: Int64


def _no_task(context: _Context, begin: Int, end: Int, worker: Int):
    pass


def _claim(pool: Pointer[_Pool, MutUntrackedOrigin], worker: Int):
    var task = pool[].task
    var context = pool[].context
    if pool[].each:
        task(context, worker, worker + 1, worker)
        return
    var count = pool[].count
    var chunk = pool[].chunk
    while True:
        var begin = Int(pool[].next.fetch_add[ordering=Ordering.RELAXED](Int64(chunk)))
        if begin >= count:
            return
        task(context, begin, min(begin + chunk, count), worker)


def _worker(argument: _Context) -> _Context:
    ref seat = argument.value().unsafe_bitcast[_Seat]()[]
    var pool = Pointer[_Pool, MutUntrackedOrigin](unsafe_from_address=seat.pool)
    var index = seat.index
    var seen = seat.generation
    while True:
        # Poll briefly: a back-to-back job then starts without a kernel wake-up.
        var spins = 0
        var start = perf_counter_ns()
        while pool[].generation.load[ordering=Ordering.ACQUIRE]() == seen:
            _relax()
            spins += 1
            if _spun_out(spins, start):
                break
        if pool[].generation.load[ordering=Ordering.ACQUIRE]() == seen:
            _lock(pool[].mutex)
            pool[].sleepers += 1
            while pool[].generation.load[ordering=Ordering.ACQUIRE]() == seen and not pool[].shutdown:
                _wait(pool[].wake, pool[].mutex)
            pool[].sleepers -= 1
            var stop = pool[].shutdown
            _unlock(pool[].mutex)
            if stop:
                return None
        seen = pool[].generation.load[ordering=Ordering.ACQUIRE]()
        # Shutdown also advances the generation; a polling worker must not run
        # the previous job again (a broadcast job has no count to stop it).
        if pool[].shutdown:
            return None
        if pool[].each or index < pool[].participants:
            _claim(pool, index)
        if pool[].active.fetch_sub[ordering=Ordering.ACQUIRE_RELEASE](1) == 1:
            _lock(pool[].mutex)
            if pool[].waiting:
                _broadcast(pool[].done)
            _unlock(pool[].mutex)


def _requested_threads() -> Int:
    """APN_MOJO_NUM_THREADS when it is a positive integer, else the usable cores."""
    var text = getenv("APN_MOJO_NUM_THREADS")
    if text:
        try:
            var n = Int(text)
            if n >= 1:
                return n
        except:
            pass
    return _usable_cpus()


def _start_pool() -> _Context:
    # The record exists even without workers, so the global caches the answer
    # instead of probing the CPUs again on every call.
    var address = _zeroed(size_of[_Pool]())
    if not address:
        return None
    var pool = Pointer[_Pool, MutUntrackedOrigin](unsafe_from_address=address)
    pool.unsafe_write(_Pool(
        0, 0, 0, 0, 0,
        Atomic(Int64(0)), Atomic(Int64(0)), Atomic(Int64(0)), 0, False, False,
        _no_task, None, 0, 1, Atomic(Int64(0)), False, 0, 1, Atomic(Int64(0)),
    ))
    _spawn(pool, _requested_threads() - 1)
    return _raw(address)


def _spawn(pool: Pointer[_Pool, MutUntrackedOrigin], workers: Int):
    """Start `workers` threads in a pool that has none, creating its mutex and
    condition variables the first time; participants become all of them."""
    comptime if _THREADS:
        if workers <= 0:
            return
        if not pool[].mutex:
            var mutex = _zeroed(_SYNC_BYTES)
            var wake = _zeroed(_SYNC_BYTES)
            var done = _zeroed(_SYNC_BYTES)
            if not (mutex and wake and done):
                return
            pool[].mutex = mutex
            pool[].wake = wake
            pool[].done = done
            _ = external_call["pthread_mutex_init", Int32](_raw(mutex), Int(0))
            _ = external_call["pthread_cond_init", Int32](_raw(wake), Int(0))
            _ = external_call["pthread_cond_init", Int32](_raw(done), Int(0))
        var threads = _zeroed(8 * workers)
        var seats = _zeroed(size_of[_Seat]() * workers)
        if not (threads and seats):
            return
        pool[].seats = seats
        pool[].threads = threads
        var generation = pool[].generation.load[ordering=Ordering.ACQUIRE]()
        for _ in range(workers):
            # A thread that fails to start leaves the pool smaller, never broken.
            var seat = seats + size_of[_Seat]() * pool[].workers
            Pointer[_Seat, MutUntrackedOrigin](unsafe_from_address=seat).unsafe_write(
                _Seat(Int(pool), pool[].workers + 1, generation)
            )
            if external_call["pthread_create", Int32](
                _raw(threads + 8 * pool[].workers), Int(0), _worker, _raw(seat),
            ) == 0:
                pool[].workers += 1
    pool[].participants = pool[].workers + 1


def _join_workers(pool: Pointer[_Pool, MutUntrackedOrigin]):
    """Stop and join every worker and free their handles and seats; the mutex
    and condition variables stay for a restart."""
    if not pool[].workers:
        return
    _lock(pool[].mutex)
    pool[].shutdown = True
    pool[].task = _no_task
    pool[].each = False
    pool[].count = 0
    _ = pool[].generation.fetch_add[ordering=Ordering.RELEASE](1)
    _broadcast(pool[].wake)
    _unlock(pool[].mutex)
    for w in range(pool[].workers):
        var handle = Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=pool[].threads + 8 * w)[]
        _ = external_call["pthread_join", Int32](handle, Int(0))
    for address in [pool[].seats, pool[].threads]:
        external_call["free", NoneType](Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=address))
    pool[].seats = 0
    pool[].threads = 0
    pool[].workers = 0
    pool[].sleepers = 0
    pool[].shutdown = False
    pool[].participants = 1


def _stop_pool(argument: _Context):
    """At process teardown: stop and join every worker, then free the pool."""
    if not argument:
        return
    var pool = argument.value().unsafe_bitcast[_Pool]()
    if not pool[].mutex:
        external_call["free", NoneType](Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(pool)))
        return
    _lock(pool[].mutex)
    pool[].shutdown = True
    pool[].task = _no_task
    pool[].each = False
    pool[].count = 0
    _ = pool[].generation.fetch_add[ordering=Ordering.RELEASE](1)
    _broadcast(pool[].wake)
    _unlock(pool[].mutex)
    for w in range(pool[].workers):
        var handle = Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=pool[].threads + 8 * w)[]
        _ = external_call["pthread_join", Int32](handle, Int(0))
    _ = external_call["pthread_cond_destroy", Int32](_raw(pool[].done))
    _ = external_call["pthread_cond_destroy", Int32](_raw(pool[].wake))
    _ = external_call["pthread_mutex_destroy", Int32](_raw(pool[].mutex))
    for address in [pool[].seats, pool[].done, pool[].wake, pool[].mutex, pool[].threads, Int(pool)]:
        # The same signature as every other free call in the package.
        external_call["free", NoneType](Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=address))


def _acquire() -> Optional[Pointer[_Pool, MutUntrackedOrigin]]:
    """The pool, reserved for one job; None when there is none to use."""
    var found = _get_global["apn_mojo.threads", _start_pool, _stop_pool]()
    if not found:
        return None
    var pool = found.value().unsafe_bitcast[_Pool]()
    var pending = pool[].requested.load[ordering=Ordering.RELAXED]()
    if not pending and (not pool[].workers or pool[].participants <= 1):
        return None
    var expected = Int64(0)
    if not pool[].busy.compare_exchange[success_ordering=Ordering.ACQUIRE, failure_ordering=Ordering.RELAXED](expected, 1):
        return None
    _apply_requested(pool)
    if not pool[].workers or pool[].participants <= 1:
        pool[].busy.store[ordering=Ordering.RELEASE](0)
        return None
    return pool


def _run_job(
    pool: Pointer[_Pool, MutUntrackedOrigin], task: _Task, context: _Context,
    count: Int, chunk: Int, each: Bool,
):
    pool[].task = task
    pool[].context = context
    pool[].count = count
    pool[].chunk = chunk
    pool[].each = each
    pool[].next.store[ordering=Ordering.RELAXED](0)
    pool[].active.store[ordering=Ordering.RELAXED](Int64(pool[].workers))
    # Publishing the generation releases the job fields to every worker.
    _ = pool[].generation.fetch_add[ordering=Ordering.RELEASE](1)
    _lock(pool[].mutex)
    if pool[].sleepers:
        _broadcast(pool[].wake)
    _unlock(pool[].mutex)
    _claim(pool, 0)
    var spins = 0
    var start = perf_counter_ns()
    while pool[].active.load[ordering=Ordering.ACQUIRE]() > 0:
        _relax()
        spins += 1
        if _spun_out(spins, start):
            break
    if pool[].active.load[ordering=Ordering.ACQUIRE]() > 0:
        _lock(pool[].mutex)
        pool[].waiting = True
        while pool[].active.load[ordering=Ordering.ACQUIRE]() > 0:
            _wait(pool[].done, pool[].mutex)
        pool[].waiting = False
        _unlock(pool[].mutex)
    pool[].busy.store[ordering=Ordering.RELEASE](0)


def _parallel_for(task: _Task, context: _Context, count: Int, chunk: Int):
    """Run task over [0, count) in chunks of `chunk`, in parallel when possible.

    Every index runs exactly once; each call learns its participant index.
    Inline (as participant 0) when the pool is busy or absent or the work fits
    in one chunk.
    """
    if count <= chunk:
        task(context, 0, count, 0)
        return
    var pool = _acquire()
    if not pool:
        task(context, 0, count, 0)
        return
    _run_job(pool.value(), task, context, count, chunk, False)


def _parallel_each(task: _Task, context: _Context):
    """Run task once for every participant index, as (worker, worker + 1, worker).

    With the pool, each worker thread runs its own index, so it can release
    what it allocated; otherwise the calling thread runs every index in turn.
    """
    var pool = _acquire()
    if not pool:
        for worker in range(_pool_threads()):
            task(context, worker, worker + 1, worker)
        return
    _run_job(pool.value(), task, context, pool.value()[].workers + 1, 1, True)


def _pool() -> Optional[Pointer[_Pool, MutUntrackedOrigin]]:
    var found = _get_global["apn_mojo.threads", _start_pool, _stop_pool]()
    if not found:
        return None
    return found.value().unsafe_bitcast[_Pool]()


def _parallel_workers() -> Int:
    """Threads a parallel call uses, the caller included, counting a pending
    request; 1 without a pool."""
    var pool = _pool()
    if not pool:
        return 1
    var pending = Int(pool.value()[].requested.load[ordering=Ordering.ACQUIRE]())
    return pending if pending else pool.value()[].participants


def _pool_threads() -> Int:
    """Every thread that can hold job results, the caller included."""
    var pool = _pool()
    return pool.value()[].workers + 1 if pool else 1


def _set_threads(n: Int):
    """Let `n` threads, the caller included, take part in each job: fewer than
    exist join no chunked job, and more restart the workers at n - 1. Never
    waits: while a job runs, even one that called this, the count is recorded
    and the next job applies it before it starts."""
    var found = _pool()
    if not found:
        return
    var pool = found.value()
    pool[].requested.store[ordering=Ordering.RELEASE](Int64(n))
    var expected = Int64(0)
    if pool[].busy.compare_exchange[success_ordering=Ordering.ACQUIRE, failure_ordering=Ordering.RELAXED](expected, 1):
        _apply_requested(pool)
        pool[].busy.store[ordering=Ordering.RELEASE](0)


def _apply_requested(pool: Pointer[_Pool, MutUntrackedOrigin]):
    """Apply a pending thread count; the caller holds the pool's busy flag, so
    no job is running."""
    # Take the request: a failed exchange loads the newer value into `pending`.
    var pending = pool[].requested.load[ordering=Ordering.ACQUIRE]()
    while pending and not pool[].requested.compare_exchange[
        success_ordering=Ordering.ACQUIRE_RELEASE, failure_ordering=Ordering.ACQUIRE
    ](pending, 0):
        pass
    var n = Int(pending)
    if not n:
        return
    if n - 1 > pool[].workers:
        _join_workers(pool)
        _spawn(pool, n - 1)
    pool[].participants = max(1, min(n, pool[].workers + 1))
