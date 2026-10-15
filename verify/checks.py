"""Exact finite checks for "Finitely presented infinite p-groups for every prime".

Every computation is in integer arithmetic modulo p; nothing is floating point.
Statement numbers refer to the paper. Run:  python checks.py
The checks support the proofs in the paper; they do not replace them.
"""
import itertools
import random

import numpy as np


# ---------------------------------------------------------------- matrices mod p

def eye(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def mul(A, B, p):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B))) % p for j in range(len(B[0]))]
            for i in range(len(A))]


def prod(Ms, p):
    out = eye(len(Ms[0]))
    for M in Ms:
        out = mul(out, M, p)
    return out


def e(n, i, j, a, p):
    """Elementary matrix e_ij(a), 0-based indices."""
    M = eye(n)
    M[i][j] = a % p
    return M


def W(n, a, b, u, p):
    """W_ab(u) = e_ab(u) e_ba(-u^{-1}) e_ab(u)."""
    ui = pow(u % p, p - 2, p)
    return prod([e(n, a, b, u, p), e(n, b, a, -ui, p), e(n, a, b, u, p)], p)


def det(A, p):
    A = [row[:] for row in A]
    n, d = len(A), 1
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] % p), None)
        if piv is None:
            return 0
        if piv != c:
            A[c], A[piv] = A[piv], A[c]
            d = -d
        d = d * A[c][c] % p
        iv = pow(A[c][c], p - 2, p)
        for r in range(c + 1, n):
            f = A[r][c] * iv % p
            A[r] = [(A[r][t] - f * A[c][t]) % p for t in range(n)]
    return d % p


def inv(A, p):
    n = len(A)
    M = [A[i][:] + eye(n)[i] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] % p)
        M[c], M[piv] = M[piv], M[c]
        iv = pow(M[c][c], p - 2, p)
        M[c] = [x * iv % p for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [(M[r][t] - f * M[c][t]) % p for t in range(2 * n)]
    return [row[n:] for row in M]


def diag(entries, p):
    n = len(entries)
    return [[entries[i] % p if i == j else 0 for j in range(n)] for i in range(n)]


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    assert ok, name


# ---------------------------------------------------------------- Lemma 3.1

def symsum(Xs, p):
    """sum over sigma in S_m of X_sigma(1)...X_sigma(m) mod p, by an exact subset recursion."""
    m, d = len(Xs), Xs[0].shape[0]
    f = np.zeros((1 << m, d, d), dtype=np.int64)
    f[0] = np.eye(d, dtype=np.int64)
    pc = np.array([bin(s).count("1") for s in range(1 << m)])
    for k in range(1, m + 1):
        layer = np.nonzero(pc == k)[0]
        acc = np.zeros((len(layer), d, d), dtype=np.int64)
        for i in range(m):
            has = (layer >> i) & 1 == 1
            S = layer[has]
            acc[has] = (acc[has] + np.einsum('sab,bc->sac', f[S ^ (1 << i)], Xs[i])) % p
        f[layer] = acc
    return f[(1 << m) - 1] % p


def symsum_brute(Xs, p):
    d = Xs[0].shape[0]
    tot = np.zeros((d, d), dtype=np.int64)
    for perm in itertools.permutations(range(len(Xs))):
        M = np.eye(d, dtype=np.int64)
        for i in perm:
            M = (M @ Xs[i]) % p
        tot = (tot + M) % p
    return tot


def matrix_units(d):
    units = []
    for i in range(d):
        for j in range(d):
            Y = np.zeros((d, d), dtype=np.int64)
            Y[i, j] = 1
            units.append(Y)
    return units


def lemma_3_1():
    print("Lemma 3.1: the symmetric sum vanishes once m > (p-1) d^2")
    rng = random.Random(1)
    for (p, d, m) in [(3, 2, 5), (5, 2, 6), (2, 2, 6)]:
        Xs = [np.array([[rng.randrange(p) for _ in range(d)] for _ in range(d)], dtype=np.int64)
              for _ in range(m)]
        check(f"  recursion equals brute force, p={p} d={d} m={m}",
              (symsum(Xs, p) == symsum_brute(Xs, p)).all())
    # By multilinearity it suffices to check tuples of matrix units; this is exhaustive.
    for (p, d) in [(2, 1), (3, 1), (5, 1), (7, 1), (2, 2), (3, 2)]:
        m = (p - 1) * d * d + 1
        units = matrix_units(d)
        ok = all(not symsum([units[i] for i in combo], p).any()
                 for combo in itertools.combinations_with_replacement(range(d * d), m))
        check(f"  all tuples of matrix units, p={p} d={d} m={m}", ok)
    for (p, d, m) in [(5, 2, 17), (3, 3, 19)]:
        ok = True
        for s in range(5):
            r = random.Random(100 * p + s)
            Xs = [np.array([[r.randrange(p) for _ in range(d)] for _ in range(d)], dtype=np.int64)
                  for _ in range(m)]
            ok &= not symsum(Xs, p).any()
        check(f"  five random tuples, p={p} d={d} m={m}", ok)
    # Zalesskii's degree dp (remark after Lemma 3.1): the sum vanishes at m = dp and not at m = dp - 1.
    for (p, d) in [(2, 2), (3, 2), (5, 2), (2, 3)]:
        units = matrix_units(d)
        m = d * p
        zero_at = all(not symsum([units[i] for i in combo], p).any()
                      for combo in itertools.combinations_with_replacement(range(d * d), m))
        nonzero_below = any(symsum([units[i] for i in combo], p).any()
                            for combo in itertools.combinations_with_replacement(range(d * d), m - 1))
        check(f"  Zalesskii, p={p} d={d}: zero for all matrix-unit tuples at m=dp={m}, nonzero for some at m={m - 1}",
              zero_at and nonzero_below)
    # The characteristic-two bound m > d^2 fails over F_3.
    p, d = 3, 2
    nonzero = 0
    for s in range(20):
        r = random.Random(1000 + s)
        Xs = [np.array([[r.randrange(p) for _ in range(d)] for _ in range(d)], dtype=np.int64)
              for _ in range(5)]
        nonzero += int(symsum(Xs, p).any())
    check(f"  over F_3 with d=2, m=d^2+1=5: nonzero for {nonzero}/20 random tuples", nonzero > 0)


# ---------------------------------------------------------------- Proposition 6.5

def prop_6_5():
    print("Proposition 6.5: the unordered-frame stabilizer over odd fields")
    for p in (3, 5, 7):
        for m in (3, 6):
            r = m - 2
            H = prod([W(m, 0, m - 1, -1, p), inv(W(m, 0, m - 1, 1, p), p)], p)
            check(f"  p={p} m={m}: H_1m = diag(-1,1,...,1,-1)",
                  H == diag([-1] + [1] * (m - 2) + [-1], p))
            h = mul(H, W(m, m - 2, m - 1, 1, p), p)
            target = diag([-1] + [1] * (m - 1), p)
            target[m - 2][m - 2] = target[m - 1][m - 1] = 0
            target[m - 2][m - 1] = target[m - 1][m - 2] = 1
            check(f"  p={p} m={m}: H_1m W_(m-1,m) = diag(A_0, P_2)", h == target)
            A0 = [row[:r] for row in h[:r]]
            check(f"  p={p} m={m}: det A_0 = -1, so A_0 is not in E_r", det(A0, p) == p - 1)


# ---------------------------------------------------------------- Theorem 6.8 (face comparison)

def face_comparison():
    print("Theorem 6.8, face comparison: W = W_(m,m-1) fixes the complementary block")
    for p in (3, 5, 7, 11):
        m = 7
        Wm = W(m, m - 1, m - 2, 1, p)
        e_ = lambda k: [int(i == k) for i in range(m)]
        col = lambda A, j: [A[i][j] for i in range(m)]
        check(f"  p={p}: W e_(m-1) = e_m", col(Wm, m - 2) == e_(m - 1))
        check(f"  p={p}: W e_m = -e_(m-1)", col(Wm, m - 1) == [(-x) % p for x in e_(m - 2)])
        rng = random.Random(p)
        ok = True
        for _ in range(20):
            A = [[rng.randrange(p) for _ in range(m - 2)] for _ in range(m - 2)]
            D = eye(m)
            for i in range(m - 2):
                D[i][:m - 2] = A[i]
            ok &= mul(Wm, D, p) == mul(D, Wm, p)
        check(f"  p={p}: W commutes with diag(A, 1_2) for 20 random A", ok)


# ---------------------------------------------------------------- Section 7

def section_7():
    print("Section 7: coordinate changes and the triangularization of C_b")
    for p in (3, 5):
        s = prod([e(2, 0, 1, 1, p), e(2, 1, 0, 1, p), e(2, 0, 1, 1, p)], p)
        check(f"  p={p}: x_ab(1)x_ba(1)x_ab(1) has matrix [[2,3],[1,2]] mod p, not a transposition",
              s == [[2, 3 % p], [1, 2]])
        wab, wbc = W(3, 0, 1, 1, p), W(3, 1, 2, 1, p)
        g = prod([wbc, wbc, wab, wbc], p)
        check(f"  p={p}: w_bc^2 w_ab w_bc is the unsigned 3-cycle e_a->e_b->e_c->e_a",
              g == [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    # Lemma 7.4: diag(v, v^{-1}) = W(v) W(1)^{-1}
    for p in (3, 5, 7):
        ok = all(prod([W(2, 0, 1, v, p), inv(W(2, 0, 1, 1, p), p)], p) == diag([v, pow(v, p - 2, p)], p)
                 for v in range(1, p))
        check(f"  p={p}: W_12(v) W_12(1)^-1 = diag(v, v^-1) for every unit v (Lemma 7.4)", ok)
    # Lemma 7.5: the relabelling matrix g_0 has determinant one
    for p in (3, 5):
        ok = True
        for perm in itertools.permutations(range(4)):
            sign = 1
            for i in range(4):
                for j in range(i + 1, 4):
                    if perm[i] > perm[j]:
                        sign = -sign
            g0 = [[0] * 5 for _ in range(5)]
            for k in range(4):
                g0[perm[k]][k] = 1
            g0[4][4] = sign % p
            ok &= det(g0, p) == 1
        check(f"  p={p}: the relabelling matrix with sign on c has determinant one, all of S_4 (Lemma 7.5)", ok)
    # Proof of Theorem 7.1: P^{-1} C_b P unitriangular, and Q = cP has det 1 with the same conjugate
    for (p, a) in [(3, 1), (3, 2), (5, 1), (7, 1)]:
        b = p ** a
        C = [[int(j == (i + 1) % b) for j in range(b)] for i in range(b)]
        N = [[(C[i][j] - int(i == j)) % p for j in range(b)] for i in range(b)]
        Nb = eye(b)
        for _ in range(b):
            Nb = mul(Nb, N, p)
        check(f"  p={p} b={b}: (C_b - 1)^b = 0", all(x == 0 for row in Nb for x in row))
        v = [int(i == 0) for i in range(b)]
        chain = [v]
        for _ in range(b - 1):
            v = [sum(N[i][t] * v[t] for t in range(b)) % p for i in range(b)]
            chain.append(v)
        basis = list(reversed(chain))
        P = [[basis[j][i] for j in range(b)] for i in range(b)]
        U = mul(mul(inv(P, p), C, p), P, p)
        check(f"  p={p} b={b}: P^-1 C_b P is upper unitriangular",
              all(U[i][j] == 0 for i in range(b) for j in range(i)) and all(U[i][i] == 1 for i in range(b)))
        # force a determinant different from one, to exercise the correction
        P2 = mul(P, diag([2] + [1] * (b - 1), p), p)
        c = pow(det(P2, p), p - 2, p)
        Q = [[c * x % p for x in row] for row in P2]
        U2 = mul(mul(inv(P2, p), C, p), P2, p)
        check(f"  p={p} b={b}: Q = cP has det 1 and Q^-1 C_b Q = P^-1 C_b P",
              det(Q, p) == 1 and mul(mul(inv(Q, p), C, p), Q, p) == U2)


# ---------------------------------------------------------------- Lemma 8.6

def lemma_8_6():
    print("Lemma 8.6: v_p(2^n+1) <= C + v_p(n), and the parts of L_0...L_d prime to p are unbounded")

    def vp(n, p):
        v = 0
        while n % p == 0:
            n //= p
            v += 1
        return v

    for p in (3, 5, 11, 13, 17, 19, 29, 41):
        o = 1
        while pow(2, o, p) != 1:
            o += 1
        if o % 2:
            check(f"  p={p}: the order of 2 is odd, so p divides no 2^n+1",
                  all((pow(2, n, p) + 1) % p for n in range(1, 400)))
            continue
        C = vp(pow(2, o // 2) + 1, p)
        ok = all(vp(pow(2, n) + 1, p) <= C + vp(n, p) for n in range(1, 2000))
        check(f"  p={p}: v_p(2^n+1) <= {C} + v_p(n) for n < 2000", ok)
    for (p, h0, D) in [(3, 4, 2), (3, 6, 4), (5, 1, 5), (11, 4, 11), (13, 5, 13)]:
        Lprod, sizes, divides = 1, [], True
        for j in range(5):
            L = 2 * 2 ** (h0 * D ** j) + 1
            divides &= L % p == 0
            Lprod *= L
            part = Lprod // p ** vp(Lprod, p)
            sizes.append(part.bit_length())
        check(f"  p={p} h0={h0} D={D}: p divides every L_j, and the prime-to-p parts have bit lengths {sizes}",
              divides and all(sizes[i] < sizes[i + 1] for i in range(4)))


if __name__ == "__main__":
    lemma_3_1()
    prop_6_5()
    face_comparison()
    section_7()
    lemma_8_6()
    print("All checks passed.")
