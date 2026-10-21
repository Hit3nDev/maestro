"""
Chord Pathfinder: BFS, DFS, UCS and A* over a harmonic graph.
Course mapping: Unit 2 (uninformed and informed search, heuristic functions).
"""
import heapq
import itertools
import math
from collections import deque

from .theory import NOTE_NAMES, chord_pcs, chord_display

QUALITIES = ["maj", "min", "dom7"]
FIFTHS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"]
RING = {"maj": 3.0, "min": 2.05, "dom7": 1.2}


def pc_dist(a, b):
    d = abs(a - b) % 12
    return min(d, 12 - d)


def voice_leading_cost(A, B):
    """Total semitone movement needed to morph chord A into chord B (max of both directions)."""
    A, B = set(A), set(B)
    to_b = sum(min(pc_dist(a, b) for b in B) for a in A)
    to_a = sum(min(pc_dist(b, a) for a in A) for b in B)
    return max(to_b, to_a)


def build_chord_graph(max_cost=4):
    """Returns (nodes, adj, meta): pitch classes per chord, weighted adjacency, (root, quality) per chord."""
    nodes, meta = {}, {}
    for r in NOTE_NAMES:
        for q in QUALITIES:
            name = chord_display(r, q)
            nodes[name] = chord_pcs(r, q)
            meta[name] = (r, q)
    adj = {n: {} for n in nodes}
    for a, b in itertools.combinations(nodes, 2):
        c = voice_leading_cost(nodes[a], nodes[b])
        if 0 < c <= max_cost:
            adj[a][b] = c
            adj[b][a] = c
    return nodes, adj, meta


def positions(meta):
    """Lay chords on concentric circles of fifths (major outer, minor middle, dom7 inner)."""
    pos = {}
    for name, (root, q) in meta.items():
        theta = math.pi / 2 - 2 * math.pi * FIFTHS.index(root) / 12
        r = RING[q]
        pos[name] = (r * math.cos(theta), r * math.sin(theta))
    return pos


def heuristic(nodes, n, goal):
    """Notes that still differ from the goal chord. Each must move at least 1 semitone, so h <= true cost."""
    a, g = set(nodes[n]), set(nodes[goal])
    return max(len(a - g), len(g - a))


# ---------- helpers ----------
def _reconstruct(parent, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1]

def path_cost(adj, path):
    return sum(adj[a][b] for a, b in zip(path, path[1:]))

def _result(path, cost, order, frontier_max):
    return {"path": path, "cost": cost, "hops": len(path) - 1,
            "expanded": len(order), "order": order, "frontier_max": frontier_max}


# ---------- algorithms ----------
def bfs(adj, start, goal, **_):
    if start == goal:
        return _result([start], 0, [start], 1)
    parent, q, order, fmax = {start: None}, deque([start]), [], 1
    while q:
        n = q.popleft()
        order.append(n)
        for m in sorted(adj[n]):
            if m not in parent:
                parent[m] = n
                if m == goal:
                    p = _reconstruct(parent, start, goal)
                    return _result(p, path_cost(adj, p), order, fmax)
                q.append(m)
                fmax = max(fmax, len(q))
    return None


def dfs(adj, start, goal, **_):
    stack, parent, visited, order, fmax = [start], {start: None}, set(), [], 1
    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)
        order.append(n)
        if n == goal:
            p = _reconstruct(parent, start, goal)
            return _result(p, path_cost(adj, p), order, fmax)
        for m in sorted(adj[n], reverse=True):
            if m not in visited and m not in parent:
                parent[m] = n
                stack.append(m)
        fmax = max(fmax, len(stack))
    return None


def _best_first(adj, start, goal, h):
    frontier = [(h(start), 0, 0, start)]  # (f, g, tie, node)
    tie, best_g, parent, closed, order, fmax = 1, {start: 0}, {start: None}, set(), [], 1
    while frontier:
        f, g, _, n = heapq.heappop(frontier)
        if n in closed:
            continue
        closed.add(n)
        order.append(n)
        if n == goal:
            return _result(_reconstruct(parent, start, goal), g, order, fmax)
        for m, w in adj[n].items():
            ng = g + w
            if m not in closed and ng < best_g.get(m, math.inf):
                best_g[m] = ng
                parent[m] = n
                heapq.heappush(frontier, (ng + h(m), ng, tie, m))
                tie += 1
        fmax = max(fmax, len(frontier))
    return None


def ucs(adj, start, goal, **_):
    return _best_first(adj, start, goal, lambda n: 0)


def astar(adj, start, goal, nodes=None, **_):
    return _best_first(adj, start, goal, lambda n: heuristic(nodes, n, goal))


ALGORITHMS = {
    "Breadth-First Search": bfs,
    "Depth-First Search": dfs,
    "Uniform-Cost Search": ucs,
    "A* Search": astar,
}

PROPERTIES = [
    {"Strategy": "Breadth-First", "Type": "Uninformed", "Complete": "Yes", "Optimal": "Only for unit costs",
     "Time": "O(b^d)", "Space": "O(b^d)", "Explores": "Fewest hops first"},
    {"Strategy": "Depth-First", "Type": "Uninformed", "Complete": "Yes (finite graph, visited set)", "Optimal": "No",
     "Time": "O(b^m)", "Space": "O(b·m)", "Explores": "Deepest branch first"},
    {"Strategy": "Uniform-Cost", "Type": "Uninformed", "Complete": "Yes", "Optimal": "Yes",
     "Time": "O(b^(1+C*/ε))", "Space": "O(b^(1+C*/ε))", "Explores": "Cheapest path cost first"},
    {"Strategy": "A*", "Type": "Informed", "Complete": "Yes", "Optimal": "Yes, if h is admissible",
     "Time": "Exponential worst case, far less in practice", "Space": "Keeps all nodes", "Explores": "g(n) + h(n)"},
]