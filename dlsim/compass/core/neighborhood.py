"""
Neighborhood rule (paper Sec. 4.4): N_a is the k_nb pool members with the smallest RTT L_{i,a} among those with
L_{i,a} < L_max (ties broken by node id). A candidate with fewer than n reachable members declines.
"""
import math
from typing import Dict, Hashable, List, Optional, Sequence


def neighborhood(aggregator: Hashable, pool: Sequence[Hashable], rtt: Dict[Hashable, float], L_max: float,
                 k_nb: int, exclude: Sequence[Hashable] = ()) -> List[Hashable]:
    """
    Return N_a ordered by (RTT, id). Members without a finite RTT are unreachable.
    """
    excluded = set(exclude)
    reachable = []
    for node in pool:
        if node == aggregator or node in excluded:
            continue
        latency = rtt.get(node)
        if latency is None or not math.isfinite(latency) or latency < 0:
            continue
        if latency < L_max:
            reachable.append((latency, node))
    reachable.sort(key=lambda item: (item[0], item[1]))
    return [node for _, node in reachable[:k_nb]]


def neighborhood_or_decline(aggregator: Hashable, pool: Sequence[Hashable], rtt: Dict[Hashable, float], L_max: float,
                            k_nb: int, n: int, exclude: Sequence[Hashable] = ()) -> Optional[List[Hashable]]:
    """
    N_a, or None if |N_a| < n (the candidate declines).
    """
    members = neighborhood(aggregator, pool, rtt, L_max, k_nb, exclude)
    return members if len(members) >= n else None
