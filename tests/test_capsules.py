import torch

from capsnet_pcam.models.capsules import PrimaryCapsules, RoutingCapsules, margin_loss, squash


def test_squash_keeps_direction_and_bounds_length():
    s = torch.nn.functional.normalize(torch.randn(64, 16), dim=-1) * torch.logspace(-3, 3, 64).unsqueeze(1)
    v = squash(s)
    lengths = v.norm(dim=-1)
    assert torch.all(lengths <= 1 + 1e-6)  # < 1 mathematically; rounds to 1.0 in float32 for huge |s|
    assert torch.allclose(torch.cosine_similarity(v, s, dim=-1), torch.ones(64), atol=1e-5)
    # Length is |s|^2 / (1 + |s|^2), which increases monotonically with |s|.
    assert torch.all(lengths[1:] >= lengths[:-1])


def test_squash_is_finite_at_zero():
    v = squash(torch.zeros(3, 8, requires_grad=True))
    v.sum().backward()
    assert torch.all(v == 0)


def test_primary_capsules_shape():
    caps = PrimaryCapsules(in_channels=16, num_types=4, capsule_dim=8, kernel_size=9, stride=2)
    out = caps(torch.randn(2, 16, 24, 24))
    assert out.shape == (2, 4 * 8 * 8, 8)
    assert torch.all(out.norm(dim=-1) < 1)


def test_routing_output_shape_and_gradient():
    layer = RoutingCapsules(in_caps=50, in_dim=8, out_caps=2, out_dim=16, routings=3)
    u = squash(torch.randn(4, 50, 8))
    v = layer(u)
    assert v.shape == (4, 2, 16)
    assert torch.all(v.norm(dim=-1) < 1)
    v.norm(dim=-1).sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_single_routing_iteration_is_uniform_average():
    # With one iteration the coupling coefficients are uniform (softmax of zeros = 1/out_caps),
    # so output j is squash(1/out_caps * sum_i u_hat_{j|i}).
    layer = RoutingCapsules(in_caps=10, in_dim=4, out_caps=2, out_dim=3, routings=1)
    u = torch.randn(1, 10, 4)
    u_hat = torch.einsum("jiod,bid->bjio", layer.weight, u)
    expected = squash(0.5 * u_hat.sum(dim=2))
    assert torch.allclose(layer(u), expected, atol=1e-6)


def test_routing_increases_agreement():
    # Routing should move each input's coupling towards the output its prediction agrees with.
    torch.manual_seed(0)
    one = RoutingCapsules(in_caps=32, in_dim=8, out_caps=2, out_dim=16, routings=1)
    three = RoutingCapsules(in_caps=32, in_dim=8, out_caps=2, out_dim=16, routings=3)
    three.weight.data.copy_(one.weight.data)
    u = squash(torch.randn(8, 32, 8))
    # Routing concentrates the votes, which makes the winning capsule longer.
    assert three(u).norm(dim=-1).max(dim=1).values.mean() >= one(u).norm(dim=-1).max(dim=1).values.mean()


def test_margin_loss():
    target = torch.tensor([0, 1])
    perfect = torch.tensor([[0.95, 0.05], [0.02, 0.99]])
    wrong = torch.tensor([[0.05, 0.95], [0.99, 0.02]])
    assert margin_loss(perfect, target) == 0
    assert margin_loss(wrong, target) > 0.5
