import torch

from myna.trunk import MynaTrunk, gla_chunked, gla_parallel, gla_recurrent


def _rand(B, H, T, dk, dv, seed, gate_lo=0.5):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, dk, generator=g, dtype=torch.float64) * 0.3
    k = torch.randn(B, H, T, dk, generator=g, dtype=torch.float64) * 0.3
    v = torch.randn(B, H, T, dv, generator=g, dtype=torch.float64) * 0.3
    gate = gate_lo + torch.rand(B, H, T, dk, generator=g, dtype=torch.float64) * (1 - gate_lo)
    return q, k, v, gate


def test_recurrent_matches_parallel():
    for seed, gate_lo in [(0, 0.5), (1, 0.9), (2, 0.99), (3, 0.2)]:
        q, k, v, gate = _rand(2, 3, 120, 16, 16, seed, gate_lo)
        o_par = gla_parallel(q, k, v, gate)
        o_rec, _ = gla_recurrent(q, k, v, gate)
        assert torch.allclose(o_par, o_rec, atol=1e-8), (seed, gate_lo)


def test_chunked_matches_recurrent_with_init():
    """The training path (chunked) must equal the streaming path (recurrent)
    even when resuming from a carried state."""
    for chunk in (8, 16, 32, 64, 128):
        q, k, v, gate = _rand(2, 3, 137, 16, 16, 11, gate_lo=0.7)
        init = torch.randn(2, 3, 16, 16, dtype=torch.float64) * 0.1
        o_rec, S_rec = gla_recurrent(q, k, v, gate, init_S=init)
        o_ch, S_ch = gla_chunked(q, k, v, gate, init_S=init, chunk=chunk)
        assert torch.allclose(o_rec, o_ch, atol=1e-8), chunk
        assert torch.allclose(S_rec, S_ch, atol=1e-8), chunk


def test_prefix_cache_continuation_is_exact():
    """Encoding a prefix, keeping S, and continuing with a suffix must equal
    one pass over the concatenation — the delta-encoding claim."""
    q, k, v, gate = _rand(1, 4, 200, 16, 16, 7)
    full_o, full_S = gla_recurrent(q, k, v, gate)

    cut = 130
    _, S_pre = gla_recurrent(q[:, :, :cut], k[:, :, :cut], v[:, :, :cut], gate[:, :, :cut])
    suffix_o, S_end = gla_recurrent(q[:, :, cut:], k[:, :, cut:], v[:, :, cut:], gate[:, :, cut:], init_S=S_pre)

    assert torch.allclose(full_o[:, :, cut:], suffix_o, atol=1e-8)
    assert torch.allclose(full_S, S_end, atol=1e-8)


def test_state_is_fixed_size_independent_of_length():
    _, s_short = gla_recurrent(*_rand(1, 2, 50, 8, 8, 4))
    _, s_long = gla_recurrent(*_rand(1, 2, 400, 8, 8, 5))
    assert s_short.shape == s_long.shape == (1, 2, 8, 8)


def test_trunk_append_matches_full_encode():
    """The headline claim, end to end: append_state(delta) produces the same
    hidden states and layer states as encoding the full sequence at once."""
    torch.manual_seed(0)
    trunk = MynaTrunk(vocab=64, d_model=64, n_layers=3, n_heads=4, d_k=16, d_v=16, d_ff=128).double().eval()

    a = torch.randint(0, 64, (1, 90), dtype=torch.int64)
    b = torch.randint(0, 64, (1, 37), dtype=torch.int64)
    full = torch.cat([a, b], dim=1)

    with torch.no_grad():
        h_full, S_full = trunk.encode_state(full)
        _, S_pre = trunk.encode_state(a)
        delta_h, new_S = trunk.append_state(b, S_pre, a.shape[1])

        for i, (S_f, S_n) in enumerate(zip(S_full, new_S)):
            assert torch.allclose(S_f, S_n, atol=1e-8), f"layer {i} state diverged"
        # hidden states of the appended span match a full re-encode
        assert torch.allclose(h_full[:, -b.shape[1] :], trunk.norm(delta_h[-1]), atol=1e-7)


def test_question_branches_are_isolated():
    """A question's hidden states must not change when other questions are
    added, removed, or reordered — isolation by architecture."""
    torch.manual_seed(0)
    trunk = MynaTrunk(vocab=64, d_model=64, n_layers=3, n_heads=4, d_k=16, d_v=16, d_ff=128).double().eval()
    state = torch.randint(0, 64, (1, 80), dtype=torch.int64)
    qa = torch.randint(0, 64, (1, 12), dtype=torch.int64)
    qb = torch.randint(0, 64, (1, 15), dtype=torch.int64)

    with torch.no_grad():
        _, S = trunk.encode_state(state)
        (only_a,) = trunk.encode_questions([qa], S, state.shape[1])
        a_again, b_out = trunk.encode_questions([qa, qb], S, state.shape[1])
        b_first, a_last = trunk.encode_questions([qb, qa], S, state.shape[1])

    assert torch.allclose(only_a, a_again, atol=1e-9)
    assert torch.allclose(only_a, a_last, atol=1e-9)
    assert torch.allclose(b_out, b_first, atol=1e-9)
