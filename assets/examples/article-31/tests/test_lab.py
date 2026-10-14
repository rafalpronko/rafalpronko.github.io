import copy
import pickle
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import lab

torch.set_num_threads(1)  # tiny tensors; avoid CPU oversubscription


@pytest.fixture
def batch():
    return lab.make_batch(torch.Generator().manual_seed(7))


def test_environment_matches_article():
    assert torch.__version__.split("+")[0] == "2.2.2"


def test_no_grad_does_not_switch_mode(batch):
    model, _ = lab.build(seed=0)
    with torch.no_grad():
        assert model.training
        assert not torch.equal(model(batch[0]), model(batch[0]))


def test_eval_does_not_disable_autograd(batch):
    model, _ = lab.build(seed=0)
    model.eval()
    assert model(batch[0]).requires_grad


def test_train_mode_forward_updates_batchnorm_buffers(batch):
    model, _ = lab.build(seed=0)
    before = model.norm.running_mean.clone()
    with torch.no_grad():
        model(batch[0])
    assert model.norm.num_batches_tracked.item() == 1
    assert not torch.equal(before, model.norm.running_mean)


def test_eval_forward_keeps_batchnorm_buffers(batch):
    model, _ = lab.build(seed=0)
    model.eval()
    before = copy.deepcopy(model.norm.state_dict())
    with torch.no_grad():
        model(batch[0])
    for key, value in before.items():
        assert torch.equal(value, model.norm.state_dict()[key])


def test_eval_mode_step_still_changes_weights(batch):
    model, opt = lab.build(seed=0)
    model.eval()
    weight = model.out.weight.detach().clone()
    loss = F.cross_entropy(model(batch[0]), batch[1])
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert not torch.equal(weight, model.out.weight)


def test_batches_contain_padding_and_real_tokens(batch):
    tokens, targets = batch
    assert (tokens == lab.PAD).any()
    assert ((tokens != lab.PAD).sum(dim=1) >= 2).all()
    assert set(targets.tolist()) <= {0, 1}


def test_plain_mean_depends_on_padding_masked_mean_does_not():
    model, _ = lab.build(seed=0)
    emb = model.embed
    short = torch.tensor([[3, 7, lab.PAD, lab.PAD]])
    long = torch.tensor([[3, 7] + [lab.PAD] * 6])
    with torch.no_grad():
        assert not torch.allclose(emb(short).mean(1), emb(long).mean(1))
        assert torch.allclose(lab.masked_mean(emb(short), short != lab.PAD),
                              lab.masked_mean(emb(long), long != lab.PAD))


def test_model_logits_ignore_extra_padding():
    model, _ = lab.build(seed=0)
    model.eval()
    short = torch.tensor([[3, 7, 9, lab.PAD]])
    long = torch.tensor([[3, 7, 9] + [lab.PAD] * 5])
    with torch.no_grad():
        assert torch.allclose(model(short), model(long))


def test_all_padding_sequence_is_rejected():
    model, _ = lab.build(seed=0)
    tokens = torch.tensor([[4, 5, lab.PAD], [lab.PAD, lab.PAD, lab.PAD]])
    with pytest.raises(ValueError, match="only padding"):
        model(tokens)


def test_padding_row_starts_at_zero_and_gets_zero_gradient(batch):
    model, _ = lab.build(seed=0)
    assert not model.embed.weight[lab.PAD].any()
    F.cross_entropy(model(batch[0]), batch[1]).backward()
    assert not model.embed.weight.grad[lab.PAD].any()
    assert model.embed.weight.grad[1:].abs().sum() > 0


def test_ignore_index_mean_uses_non_ignored_targets():
    logits = torch.randn(6, 5)
    labels = torch.tensor([1, 2, -100, 3, -100, -100])
    keep = labels != -100
    expected = F.cross_entropy(logits[keep], labels[keep])
    actual = F.cross_entropy(logits, labels, ignore_index=-100)
    assert torch.allclose(actual, expected)


def test_state_dict_is_an_alias_deepcopy_is_a_snapshot():
    model, opt = lab.build(seed=0)
    alias = model.state_dict()
    snapshot = copy.deepcopy(model.state_dict())
    lab.train(model, opt, torch.Generator().manual_seed(2), 1)
    assert alias["out.weight"].data_ptr() == model.out.weight.data_ptr()
    assert torch.equal(alias["out.weight"], model.out.weight)
    assert not torch.equal(snapshot["out.weight"], model.out.weight)


def test_weights_only_round_trip_gives_same_cpu_logits(tmp_path, batch):
    model, opt = lab.build(seed=0)
    gen = torch.Generator().manual_seed(3)
    lab.train(model, opt, gen, 4)
    path = tmp_path / "ckpt.pt"
    lab.save_checkpoint(path, model, opt, gen, step=4)
    restored, restored_opt = lab.build(seed=99)
    gen2 = torch.Generator()
    assert lab.load_checkpoint(path, restored, restored_opt, gen2) == 4
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model(batch[0]), restored(batch[0]))
    saved_state = opt.state_dict()["state"]
    assert restored_opt.state_dict()["state"].keys() == saved_state.keys()


@dataclass
class RunConfig:
    lr: float


def test_weights_only_refuses_arbitrary_objects(tmp_path):
    path = tmp_path / "with_object.pt"
    torch.save({"config": RunConfig(lr=0.01)}, path)
    with pytest.raises(pickle.UnpicklingError):
        torch.load(path, weights_only=True)


def test_resume_with_optimizer_and_rng_matches_straight_run():
    assert lab.resume_matches(restore_rng=True)


def test_resume_without_rng_state_diverges():
    assert not lab.resume_matches(restore_rng=False)


def test_seed_before_construction_decides_init():
    a, _ = lab.build(seed=0)
    b, _ = lab.build(seed=0)
    c, _ = lab.build(seed=1)
    assert lab.same_state(a, b)
    assert not torch.equal(a.out.weight, c.out.weight)


def test_seed_after_construction_does_not_change_weights():
    model, _ = lab.build(seed=0)
    weights = copy.deepcopy(model.state_dict())
    torch.manual_seed(0)
    for key, value in weights.items():
        assert torch.equal(value, model.state_dict()[key])


def test_lab_main_runs(capsys):
    lab.main()
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert out.count("seed=") == 3
