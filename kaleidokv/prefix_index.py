import array
import hashlib
import itertools
from collections import Counter, defaultdict


class GlobalPrefixIndex:
    """The prefix tree over evidence items with the tables of Algorithm 2."""

    def __init__(self, lengths, orders):
        self.lengths = lengths
        self.orders = [tuple(o) for o in orders]
        self.item = [None]
        self.parent = [None]
        self.children = [{}]
        self.key = [frozenset()]
        self.weights = [0]
        self.counts = [0]
        self.path_table = {}
        # Search counts, equal to the reference counts outside a search.
        self.visible = [1]
        self.sets = defaultdict(set, {frozenset(): {0}})
        self.children_of_set = defaultdict(set)
        self.subset_table = {}
        self.hidden = []
        self.weight = 0
        for order in self.orders:
            for v in self.path(order):
                self.counts[v] += 1
                self.visible[v] += 1
                if self.counts[v] == 1:
                    self.weight += self.weights[v]
                    self.sets[self.key[v]].add(v)
                    self.children_of_set[self.key[self.parent[v]], self.item[v]].add(v)

    def child(self, v, item):
        c = self.children[v].get(item)
        if c is None:
            c = len(self.item)
            self.item.append(item)
            self.parent.append(v)
            self.children.append({})
            self.key.append(self.key[v] | {item})
            self.weights.append(self.lengths[item])
            self.counts.append(0)
            self.visible.append(0)
            self.children[v][item] = c
        return c

    def path(self, order):
        nodes = self.path_table.get(order)
        if nodes is None:
            v = 0
            nodes = []
            for item in order:
                v = self.child(v, item)
                nodes.append(v)
            nodes = tuple(nodes)
            self.path_table[order] = nodes
        return nodes

    def order_of(self, v):
        order = []
        while v:
            order.append(self.item[v])
            v = self.parent[v]
        return tuple(reversed(order))

    def subsets(self, items):
        """All subsets of an evidence set, the longest in tokens first."""
        table = self.subset_table.get(items)
        if table is None:
            table = sorted(
                (
                    frozenset(c)
                    for size in range(len(items) + 1)
                    for c in itertools.combinations(sorted(items), size)
                ),
                key=lambda s: (-sum(self.lengths[e] for e in s), sorted(s)),
            )
            self.subset_table[items] = table
        return table

    def compute_delta_weight(self, change):
        """Lemma 5: the reduction of W, read from the old and new paths only."""
        diff = Counter()
        for r, order in change.items():
            diff.subtract(self.path(self.orders[r]))
            diff.update(self.path(order))
        gain = 0
        for v, d in diff.items():
            if d:
                before = self.counts[v]
                gain += ((before > 0) - (before + d > 0)) * self.weights[v]
        return gain, diff

    def apply_change(self, change, delta):
        gain, diff = delta
        for v, d in diff.items():
            if d:
                self.counts[v] += d
                self.shift(v, d)
        for r, order in change.items():
            self.orders[r] = tuple(order)
        self.weight -= gain

    def shift(self, v, d):
        before = self.visible[v]
        after = before + d
        self.visible[v] = after
        if before > 0 and after <= 0:
            self.sets[self.key[v]].discard(v)
            self.children_of_set[self.key[self.parent[v]], self.item[v]].discard(v)
        elif before <= 0 and after > 0:
            self.sets[self.key[v]].add(v)
            self.children_of_set[self.key[self.parent[v]], self.item[v]].add(v)

    def hide(self, requests):
        for r in requests:
            for v in self.path(self.orders[r]):
                self.shift(v, -1)
                self.hidden.append(v)

    def insert(self, order):
        for v in self.path(order):
            self.shift(v, 1)
            self.hidden.append(~v)

    def restore(self):
        for v in reversed(self.hidden):
            if v < 0:
                self.shift(~v, -1)
            else:
                self.shift(v, 1)
        self.hidden.clear()

    def order_for_merge(self, items, requests):
        """Lemma 6: the order of a set of items that reuses the most visible tokens."""
        self.hide(requests)
        nodes = self.sets[items]
        if not nodes:
            for reused in self.subsets(items):
                nodes = self.sets[reused]
                if nodes:
                    order = self.order_of(min(nodes)) + tuple(sorted(items - reused))
                    break
        else:
            residual = [[e for e in self.orders[r] if e not in items] for r in requests]
            order = self.order_of(self.most_overlap(items, nodes, residual))
        self.restore()
        return order

    def most_overlap(self, items, nodes, residual):
        firsts = {rest[0] if rest else None for rest in residual}
        if len(firsts) == 1 and None not in firsts:
            # Only the nodes with a visible child on the common first item can overlap.
            children = self.children_of_set[items, next(iter(firsts))]
            if not children:
                return min(nodes)
            nodes = {self.parent[c] for c in children}
        tree = {}
        for rest in residual:
            level = tree
            for e in rest:
                level = level.setdefault(e, {})
        return max(sorted(nodes), key=lambda v: self.overlap(v, tree))

    def overlap(self, v, tree):
        total = 0
        stack = [(v, tree)]
        while stack:
            v, level = stack.pop()
            for e, below in level.items():
                c = self.children[v].get(e)
                if c is not None and self.visible[c] > 0:
                    total += self.weights[c]
                    stack.append((c, below))
        return total

    def place(self, order, avoid=None):
        """Lemma 7: the order that reuses the longest visible set of items and avoids a node."""
        items = frozenset(order)
        for reused in self.subsets(items):
            for v in sorted(self.sets[reused]):
                rest = [e for e in order if e not in reused]
                if not rest:
                    return self.order_of(v)
                for e in rest:
                    c = self.children[v].get(e)
                    if c is None or (c != avoid and self.visible[c] <= 0):
                        return self.order_of(v) + (e,) + tuple(x for x in rest if x != e)
        return None

    def plan_for_dissolve(self, first_node, members):
        self.hide(members)
        change = {}
        for r in members:
            order = self.place(self.orders[r], first_node)
            if order is None:
                self.restore()
                return None
            change[r] = order
            self.insert(order)
        self.restore()
        return change

    def best_order(self, r):
        self.hide([r])
        order = self.place(self.orders[r])
        self.restore()
        return order


def fingerprint(tokens):
    return hashlib.sha256(array.array("I", tokens).tobytes()).hexdigest()


def build(prompts, fast_edges=True):
    edges = [[]]
    children = [{}]
    parents = [-1]
    leaves = []

    def node(edge, parent):
        i = len(edges)
        edges.append(edge)
        children.append({})
        parents.append(parent)
        return i

    for prompt in prompts:
        tokens = prompt[:-1]
        p = 0
        while tokens:
            c = children[p].get(tokens[0])
            if c is None:
                c = node(tokens, p)
                children[p][tokens[0]] = c
                p = c
                break
            edge = edges[c]
            limit = min(len(tokens), len(edge))
            k = 0
            if fast_edges and tokens[:limit] == edge[:limit]:
                k = limit
            else:
                while k < limit and tokens[k] == edge[k]:
                    k += 1
            if k < len(edge):
                m = node(edge[:k], p)
                children[p][tokens[0]] = m
                edges[c] = edge[k:]
                children[m][edges[c][0]] = c
                parents[c] = m
                p = m
            else:
                p = c
            tokens = tokens[k:]
        leaves.append(p)
    paths = []
    counts = [0] * len(edges)
    for q, leaf in enumerate(leaves):
        chain = []
        while leaf:
            chain.append(leaf)
            leaf = parents[leaf]
        path = []
        pos = 0
        for n in reversed(chain):
            path.append([n, pos, pos + len(edges[n])])
            pos += len(edges[n])
            counts[n] += 1
        path.append([len(edges) + q, pos, pos + 1])
        paths.append(path)
    return dict(
        paths=paths,
        counts=counts + [1] * len(prompts),
        ideal_P=sum(map(len, edges)) + len(prompts),
        fingerprints=[fingerprint(t) for t in prompts],
        lengths=list(map(len, prompts)),
    )


def summarize_prompts(prompts, sibling_order="first"):
    edges = [()]
    children = [{}]
    first = [-1]
    terminals = [[]]

    def node(edge, q):
        i = len(edges)
        edges.append(edge)
        children.append({})
        first.append(q)
        terminals.append([])
        return i

    for q, prompt in enumerate(prompts):
        assert prompt
        tokens = tuple(prompt[:-1])
        current = 0
        while tokens:
            child = children[current].get(tokens[0])
            if child is None:
                child = node(tokens, q)
                children[current][tokens[0]] = child
                current = child
                break
            edge = edges[child]
            limit = min(len(edge), len(tokens))
            k = 0
            if tokens[:limit] == edge[:limit]:
                k = limit
            else:
                while k < limit and edge[k] == tokens[k]:
                    k += 1
            if k < len(edge):
                middle = node(edge[:k], first[child])
                children[current][tokens[0]] = middle
                edges[child] = edge[k:]
                children[middle][edge[k]] = child
                current = middle
            else:
                current = child
            tokens = tokens[k:]
        terminals[current].append(q)
    key = {}
    if sibling_order == "cost":
        cost = [0] * len(edges)
        cnt = [0] * len(edges)
        post = []
        todo = [0]
        while todo:
            i = todo.pop()
            post.append(i)
            todo.extend(children[i].values())
        for i in reversed(post):
            cost[i] = (
                len(edges[i]) + len(terminals[i]) + sum((cost[c] for c in children[i].values()))
            )
            cnt[i] = len(terminals[i]) + sum((cnt[c] for c in children[i].values()))
        key = dict(cost=cost, cnt=cnt)
    else:
        assert sibling_order == "first", sibling_order
    order = []
    stack = [(False, 0)]
    while stack:
        terminal, i = stack.pop()
        if terminal:
            order.append(i)
            continue
        if key:
            branches = [
                ((key["cost"][c] / key["cnt"][c], first[c]), False, c) for c in children[i].values()
            ] + [((1.0, q), True, q) for q in terminals[i]]
        else:
            branches = [(first[c], False, c) for c in children[i].values()] + [
                (q, True, q) for q in terminals[i]
            ]
        stack.extend(((terminal, c) for _, terminal, c in sorted(branches, reverse=True)))
    assert sorted(order) == list(range(len(prompts)))
    return dict(
        order=order, ideal_P=sum(map(len, edges)) + len(prompts), compressed_nodes=len(edges)
    )
