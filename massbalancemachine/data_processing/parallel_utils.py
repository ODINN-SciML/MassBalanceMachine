"""Deciding how many worker processes the preprocessing may run at once.

Generating gridded features is memory bound rather than CPU bound: a worker holds
the whole regional ERA5 file - close to a gigabyte for the Alps - plus the arrays
it derives from it for the glacier it is working on. Running one worker per core
therefore exhausts the memory of an ordinary workstation long before it saturates
its cores, and a preprocessing run lasting hours is exactly the situation in which
the machine is also being used for something else.

So the number of workers is derived from the memory actually free when a pool is
created, keeping a reserve for the rest of the system, instead of being a constant
chosen once for one machine.
"""

import multiprocessing
import os
from typing import Optional

import psutil

from data_processing.get_climate_data import (
    climate_memory_footprint,
    warm_climate_cache,
)

GIB = 1024**3

# Memory a worker needs on top of the climate data it caches. Measured as the peak
# resident size of one glacier-year task, minus the interpreter baseline and the
# climate arrays: about 0.8 GiB for a small glacier of ~1300 grid cells. It grows
# with the number of cells, since the climate selection and its conversion to a
# dataframe are proportional to it, hence a value on the generous side.
DEFAULT_WORKER_SLACK = 1.5 * GIB

# What a worker costs when the climate data cannot be measured, because it has not
# been downloaded yet.
DEFAULT_WORKER_MEMORY = 2.5 * GIB

# Share of the physical memory left to the rest of the machine, and the floor that
# share may not go under.
RESERVE_FRACTION = 0.2
MIN_RESERVE = 3 * GIB


def available_memory() -> int:
    """Bytes that can be handed out without swapping.

    "Available" rather than "free": memory held by the page cache is reclaimable
    and does count as usable.
    """
    return psutil.virtual_memory().available


def total_memory() -> int:
    """Bytes of physical memory on the machine."""
    return psutil.virtual_memory().total


def default_reserve() -> int:
    """Memory left untouched for the rest of the machine.

    A preprocessing run takes hours, and whoever started it will open a browser or
    an editor in the meantime. A fixed reserve of a couple of gigabytes is too thin
    for that on a large machine and too fat on a small one, so it is a fraction of
    the physical memory with a floor.
    """
    return int(max(MIN_RESERVE, RESERVE_FRACTION * total_memory()))


def worker_count(
    per_worker_bytes: Optional[float] = None,
    requested: Optional[int] = None,
    reserve_bytes: Optional[float] = None,
    max_workers: Optional[int] = None,
    min_workers: int = 1,
    available_bytes: Optional[int] = None,
) -> int:
    """How many workers fit in the memory that is free right now.

    Args:
        per_worker_bytes: memory one worker is expected to hold. Defaults to
            `DEFAULT_WORKER_MEMORY`.
        requested: an explicit number of workers. When given it is returned as is,
            so that a caller who knows their machine keeps the last word.
        reserve_bytes: memory to leave for the rest of the system. Defaults to
            `default_reserve()`.
        max_workers: never return more than this. Defaults to the number of cores,
            beyond which more workers only add memory pressure.
        min_workers: never return less than this. One worker is always allowed even
            when memory is short, so that the work still progresses.
        available_bytes: memory considered free. Defaults to what the system
            reports; passing it is how the decision is exercised in tests.

    Returns:
        int: the number of workers, at least `min_workers`.
    """
    if requested is not None:
        return max(min_workers, int(requested))

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if per_worker_bytes is None:
        per_worker_bytes = DEFAULT_WORKER_MEMORY
    if reserve_bytes is None:
        reserve_bytes = default_reserve()

    available = available_memory() if available_bytes is None else available_bytes
    budget = available - reserve_bytes
    fitting = int(budget // per_worker_bytes)
    return int(max(min_workers, min(fitting, max_workers)))


def describe_worker_count(n_workers: int, per_worker_bytes: float) -> str:
    """One line saying what was decided and on what grounds, for the run log."""
    return (
        f"Using {n_workers} worker(s): "
        f"{available_memory() / GIB:.1f} GiB of memory available, "
        f"{per_worker_bytes / GIB:.1f} GiB expected per worker, "
        f"{default_reserve() / GIB:.1f} GiB reserved for the rest of the system."
    )


_last_worker_count = None


def gridded_worker_count(
    region_id, requested: Optional[int] = None, max_workers: Optional[int] = None
) -> int:
    """How many workers to generate gridded features for `region_id` with.

    Loads the regional climate data into this process first. On a platform where
    `multiprocessing` forks - Linux and, with the default settings, macOS before
    3.8 - the workers then inherit those arrays instead of each reading their own
    copy of the file: they only ever read them, so the pages stay shared and the
    gigabyte is paid once rather than once per worker. Measured on six glacier-years
    with three workers, that is 2.3 GiB of physical memory instead of 5.2 GiB, and
    6.6 s of work instead of 10.5 s. Where processes are spawned instead, each
    worker does load its own copy, and it is counted as such below.

    Called afresh for every pool rather than once per run, so that a preprocessing
    lasting hours backs off when something else is started on the machine in the
    meantime.
    """
    global _last_worker_count

    # Before reading how much memory is free, so that what this process is about to
    # hold is already accounted for.
    warm_climate_cache(region_id)

    climate_bytes = climate_memory_footprint(region_id)
    if climate_bytes is None:
        # The climate files have not been downloaded yet, so their size cannot be
        # read from them.
        per_worker = DEFAULT_WORKER_MEMORY
    elif multiprocessing.get_start_method() == "fork":
        # Inherited from this process, not allocated again by each worker.
        per_worker = DEFAULT_WORKER_SLACK
    else:
        per_worker = climate_bytes + DEFAULT_WORKER_SLACK

    n_workers = worker_count(
        per_worker_bytes=per_worker, requested=requested, max_workers=max_workers
    )
    if n_workers != _last_worker_count:
        # Only when it changes: one line per glacier would drown the progress bars.
        print(describe_worker_count(n_workers, per_worker))
        _last_worker_count = n_workers
    return n_workers
