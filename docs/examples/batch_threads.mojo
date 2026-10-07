"""Choose the thread count without changing numerical results."""

from apn_mojo import Integer, batch, get_num_threads, set_num_threads


def main() raises:
    set_num_threads(2)
    print("configured threads:", get_num_threads())
    var values = batch.arange[Integer](1000)
    var total = batch.sum(values)
    set_num_threads(1)
    print("calling thread only:", get_num_threads())
    print("same total:", batch.sum(values) == total)
