import itertools
import math
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .prefix_index import GlobalPrefixIndex

THETA = 15
I_MAX = 10
SMALL_SCOPE_ORDERS = 60000
ILP_SECONDS = 60


def reuse_scopes(rows):
    parent = list(range(len(rows)))

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    first = {}
    for q, row in enumerate(rows):
        for e in row:
            parent[find(q)] = find(first.setdefault(e, q))
    scopes = defaultdict(list)
    for q in range(len(rows)):
        scopes[find(q)].append(q)
    return list(scopes.values())


def greedy_grouping(rows, lengths):
    orders = [None] * len(rows)

    def group(members, prefix):
        used = set(prefix)
        rest = list(members)
        while rest:
            count = Counter(e for q in rest for e in rows[q] if e not in used)
            shared = [e for e, c in count.items() if c >= 2]
            if not shared:
                for q in rest:
                    orders[q] = tuple(prefix) + tuple(e for e in rows[q] if e not in used)
                return
            best = max(shared, key=lambda e: ((count[e] - 1) * lengths[e], -e))
            group([q for q in rest if best in rows[q]], prefix + [best])
            rest = [q for q in rest if best not in rows[q]]

    group(range(len(rows)), [])
    return orders


def solve_ilp(evidence_sets, lengths):
    """Eq. (1) with one order choice per distinct evidence set."""
    uses = Counter(e for row in evidence_sets for e in row)
    shared = [[e for e in row if uses[e] > 1] for row in evidence_sets]
    private = [tuple(e for e in row if uses[e] == 1) for row in evidence_sets]
    if sum(math.factorial(len(s)) for s in shared) > SMALL_SCOPE_ORDERS:
        return None
    choices = [list(itertools.permutations(s)) for s in shared]
    prefixes = {}
    x_of = []
    for i, orders in enumerate(choices):
        for order in orders:
            x_of.append((i, order))
            for j in range(1, len(order) + 1):
                prefixes.setdefault(order[:j], len(prefixes))
    if not prefixes:
        return None
    nx = len(x_of)
    cost = np.zeros(nx + len(prefixes))
    for t, y in prefixes.items():
        cost[nx + y] = lengths[t[-1]]
    rows, cols, vals, lower, upper = [], [], [], [], []
    start = 0
    for orders in choices:
        row = len(lower)
        for j in range(len(orders)):
            rows.append(row)
            cols.append(start + j)
            vals.append(1.0)
        lower.append(1)
        upper.append(1)
        below = defaultdict(list)
        for j, order in enumerate(orders):
            for size in range(1, len(order) + 1):
                below[order[:size]].append(start + j)
        for t, xs in below.items():
            row = len(lower)
            rows.extend([row] * (len(xs) + 1))
            cols.extend(xs + [nx + prefixes[t]])
            vals.extend([1.0] * len(xs) + [-1.0])
            lower.append(-np.inf)
            upper.append(0)
        start += len(orders)
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), nx + len(prefixes))).tocsr()
    result = milp(
        cost,
        constraints=LinearConstraint(matrix, lower, upper),
        integrality=np.ones(nx + len(prefixes)),
        bounds=Bounds(0, 1),
        options=dict(time_limit=ILP_SECONDS, disp=False),
    )
    if result.x is None:
        return None
    chosen = {}
    for (i, order), value in zip(x_of, result.x[:nx]):
        if value > 0.5:
            chosen[evidence_sets[i]] = order + private[i]
    return chosen


class LayoutSearch:
    """Algorithm 1 on one workload, driven through the global prefix index."""

    def __init__(self, rows, lengths):
        self.rows = rows
        self.lengths = lengths
        self.index = GlobalPrefixIndex(lengths, greedy_grouping(rows, lengths))
        self.sets_of = [frozenset(row) for row in rows]

    def try_change(self, change):
        index = self.index
        change = {r: o for r, o in change.items() if o != index.orders[r]}
        if not change:
            return False
        delta = index.compute_delta_weight(change)
        if delta[0] <= 0:
            return False
        index.apply_change(change, delta)
        return True

    def solve_small_scope(self, scope, distinct):
        chosen = solve_ilp(sorted(distinct, key=sorted), self.lengths)
        if chosen is not None:
            self.try_change({r: chosen[self.sets_of[r]] for r in scope})

    def rank_candidates(self, scope):
        table = defaultdict(list)
        for r in scope:
            for items in self.index.subsets(self.sets_of[r]):
                if items:
                    table[items].append(r)
        table = {items: rs for items, rs in table.items() if len(rs) >= 2}
        return sorted(
            table.items(),
            key=lambda c: (-(len(c[1]) - 1) * sum(self.lengths[e] for e in c[0]), sorted(c[0])),
        )

    def try_merge(self, items, requests):
        index = self.index
        order = index.order_for_merge(items, requests)
        return self.try_change(
            {r: order + tuple(e for e in index.orders[r] if e not in items) for r in requests}
        )

    def merge(self, scope):
        index = self.index
        ranked = self.rank_candidates(scope)
        for _ in range(I_MAX):
            before = index.weight
            for items, requests in ranked:
                self.try_merge(items, requests)
                parts = defaultdict(list)
                for r in requests:
                    parts[index.orders[r][0]].append(r)
                if len(parts) > 1:
                    for part in parts.values():
                        if len(part) > 1:
                            self.try_merge(items, part)
            for r in scope:
                order = index.best_order(r)
                if order is not None:
                    self.try_change({r: order})
            if index.weight == before:
                break

    def snapshot_groups(self, scope):
        index = self.index
        through = defaultdict(list)
        for r in scope:
            for v in index.path(index.orders[r]):
                through[v].append(r)
        heads = {}
        saving = Counter()
        for v, requests in through.items():
            if len(requests) >= 2:
                head = v
                while index.parent[head] and index.counts[index.parent[head]] == len(requests):
                    head = index.parent[head]
                heads[head] = through[head]
                saving[head] += (len(requests) - 1) * index.weights[v]
        return [(head, heads[head]) for head in sorted(heads, key=lambda h: (-saving[h], h))]

    def dissolve(self, scope):
        index = self.index
        for _ in range(I_MAX):
            before = index.weight
            gained = set()
            for first_node, members in self.snapshot_groups(scope):
                if first_node in gained:
                    continue
                members = [r for r in members if first_node in index.path(index.orders[r])]
                if len(members) < 2:
                    continue
                change = index.plan_for_dissolve(first_node, members)
                if change is None:
                    continue
                old = {r: index.path(index.orders[r]) for r in change}
                if self.try_change(change):
                    for r, order in change.items():
                        gained.update(set(index.path(order)) - set(old[r]))
            if index.weight == before:
                break


def plan_layout(rows, lengths):
    timing = {}
    started = time.perf_counter()
    search = LayoutSearch(rows, lengths)
    index = search.index
    scopes = reuse_scopes(rows)
    greedy_weight = index.weight
    timing["greedy_seconds"] = time.perf_counter() - started
    small = []
    large = []
    for scope in scopes:
        distinct = {search.sets_of[r] for r in scope}
        if len(distinct) > THETA:
            large.append(scope)
        elif len(distinct) > 1:
            small.append((scope, distinct))
    started = time.perf_counter()
    for scope, distinct in small:
        search.solve_small_scope(scope, distinct)
    small_weight = index.weight
    timing["small_scope_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    for scope in large:
        search.merge(scope)
    merge_weight = index.weight
    timing["merge_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    for scope in large:
        search.dissolve(scope)
    timing["dissolve_seconds"] = time.perf_counter() - started
    if index.weight != layout_weight(index.orders, lengths):
        raise RuntimeError("The index disagrees with the weight of the prefix tree")
    return index.orders, dict(
        scopes=len(scopes),
        small_scopes=len(small),
        large_scopes=len(large),
        greedy_weight=greedy_weight,
        small_scope_weight=small_weight,
        merge_weight=merge_weight,
        final_weight=index.weight,
        index_nodes=len(index.item),
        **timing,
    )


def layout_weight(orders, lengths):
    prefixes = {tuple(o[:i]) for o in orders for i in range(1, len(o) + 1)}
    return sum(lengths[t[-1]] for t in prefixes)
