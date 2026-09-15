from typing import Any
import types
import numpy as np
import pytest

# Ensure missing packages can be stubbed if running in lightweight local environment
def _get_policy_class():
    import sys
    if "jax" not in sys.modules:
        jax = types.ModuleType("jax")
        jax.numpy = np
        jax.jit = lambda fn: fn
        jax.device_get = lambda value: value
        jax.random = types.SimpleNamespace(
            key=lambda seed: f"key_{seed}",
            split=lambda k: (f"{k}_next", f"{k}_sample"),
        )
        jax.tree = types.SimpleNamespace(
            map=lambda fn, tree: {k: fn(v) for k, v in tree.items()} if isinstance(tree, dict) else tree
        )
        sys.modules["jax"] = jax
        sys.modules["jax.numpy"] = np

    if "flax.traverse_util" not in sys.modules:
        traverse_util = types.ModuleType("flax.traverse_util")
        traverse_util.flatten_dict = lambda d, *args, **kwargs: d
        traverse_util.unflatten_dict = lambda d, *args, **kwargs: d
        sys.modules["flax.traverse_util"] = traverse_util
        if "flax" in sys.modules:
            sys.modules["flax"].traverse_util = traverse_util

    if "flax" not in sys.modules:
        flax = types.ModuleType("flax")
        flax.struct = types.SimpleNamespace(dataclass=lambda cls: cls)
        nnx = types.ModuleType("flax.nnx")
        nnx_bridge = types.ModuleType("flax.nnx.bridge")
        nnx.bridge = nnx_bridge
        flax.nnx = nnx
        flax.traverse_util = sys.modules["flax.traverse_util"]
        sys.modules["flax"] = flax
        sys.modules["flax.struct"] = flax.struct
        sys.modules["flax.nnx"] = nnx
        sys.modules["flax.nnx.bridge"] = nnx_bridge

    if "einops" not in sys.modules:
        einops = types.ModuleType("einops")
        einops.rearrange = lambda *args, **kwargs: None
        sys.modules["einops"] = einops

    if "cv2" not in sys.modules:
        sys.modules["cv2"] = types.ModuleType("cv2")


    if "PIL" not in sys.modules:
        sys.modules["PIL"] = types.ModuleType("PIL")
        sys.modules["PIL.Image"] = types.ModuleType("PIL.Image")

    if "openpi.transforms" not in sys.modules:
        openpi = sys.modules.get("openpi") or types.ModuleType("openpi")
        transforms = types.ModuleType("openpi.transforms")
        transforms.compose = lambda fns: (lambda x: x)
        transforms.DataTransformFn = Any
        transforms.NormStats = Any
        openpi.transforms = transforms
        
        shared = types.ModuleType("openpi.shared")
        array_typing = types.ModuleType("openpi.shared.array_typing")
        class _Subscriptable:
            def __getitem__(self, item):
                return Any
        array_typing.typecheck = lambda fn_or_cls: fn_or_cls
        array_typing.Float = _Subscriptable()
        array_typing.Bool = _Subscriptable()
        array_typing.Int = _Subscriptable()
        array_typing.Array = np.ndarray
        array_typing.KeyArrayLike = Any
        array_typing.PyTree = dict
        shared.array_typing = array_typing
        
        nnx_utils = types.ModuleType("openpi.shared.nnx_utils")
        nnx_utils.module_jit = lambda fn: fn
        shared.nnx_utils = nnx_utils

        image_tools = types.ModuleType("openpi.shared.image_tools")
        image_tools.resize_with_pad = lambda *args, **kwargs: None
        shared.image_tools = image_tools

        models = types.ModuleType("openpi.models")
        model = types.ModuleType("openpi.models.model")
        model.ArrayT = np.ndarray
        model.preprocess_observation = lambda *args, **kwargs: None
        class Observation:
            images = {}
            image_masks = {}
            state = None
            tokenized_prompt = None
            tokenized_prompt_mask = None
            token_ar_mask = None
            token_loss_mask = None
            @classmethod
            def from_dict(cls, data):
                return cls()
        model.Observation = Observation
        models.model = model

        sys.modules["openpi"] = openpi
        sys.modules["openpi.transforms"] = transforms
        sys.modules["openpi.shared"] = shared
        sys.modules["openpi.shared.array_typing"] = array_typing
        sys.modules["openpi.shared.nnx_utils"] = nnx_utils
        sys.modules["openpi.shared.image_tools"] = image_tools
        sys.modules["openpi.models"] = models
        sys.modules["openpi.models.model"] = model

    if "mme_vla_suite.models.integration.history_observation" not in sys.modules:
        hist_obs_mod = types.ModuleType("mme_vla_suite.models.integration.history_observation")
        class HistAugObservation:
            def __init__(self, state=None):
                self.state = state if state is not None else np.zeros((1, 8))
            @classmethod
            def from_dict(cls, data):
                return cls(state=data.get("state") if isinstance(data, dict) else np.zeros((1, 8)))
        hist_obs_mod.HistAugObservation = HistAugObservation
        sys.modules["mme_vla_suite.models.integration.history_observation"] = hist_obs_mod

    if "mme_vla_suite.models.integration.history_pi0" not in sys.modules:
        hist_pi0_mod = types.ModuleType("mme_vla_suite.models.integration.history_pi0")
        hist_pi0_mod.HistoryPi0 = object
        sys.modules["mme_vla_suite.models.integration.history_pi0"] = hist_pi0_mod

    from mme_vla_suite.policies.policy import MME_VLA_Policy
    return MME_VLA_Policy


def test_infer_preserves_caller_observation_dictionary():
    PolicyClass = _get_policy_class()

    # Create a mock policy instance
    policy = object.__new__(PolicyClass)
    policy._seed = 42
    policy._rng = "key_42"
    policy.step_idx = 0
    policy.config = types.SimpleNamespace(
        representation_type="perceptual",
        budget=64,
        token_per_image=16,
        perceptual_memory=types.SimpleNamespace(type="frame_sampling"),
    )
    policy.mem_buffer = types.SimpleNamespace(
        _history_feats={0: True},
        default_history_feats_gather_fn=lambda *args: {},
        prepare_frame_sampling=lambda *args, **kwargs: (
            np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.ones(1, dtype=bool)
        ),
    )
    policy.state_norm_stats = types.SimpleNamespace(mean=0.0, std=1.0)
    policy.use_quantiles = False
    policy._input_transform = lambda x: x
    policy._output_transform = lambda x: x
    policy._sample_kwargs = {}
    policy._sample_actions = lambda rng, obs, **kwargs: np.zeros((1, 16, 8))

    caller_obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "reset_rng": True,
        "temporal_pos_override": {"enabled": False},
    }

    # Calling infer must not mutate caller_obs
    out = policy.infer(caller_obs)
    assert "reset_rng" in caller_obs
    assert "temporal_pos_override" in caller_obs
    assert caller_obs["reset_rng"] is True
    assert caller_obs["temporal_pos_override"] == {"enabled": False}
    assert "actions" in out


def test_infer_validates_reset_rng_and_temporal_override_types():
    PolicyClass = _get_policy_class()

    policy = object.__new__(PolicyClass)
    policy._seed = 42
    policy._rng = "key_42"
    policy.config = types.SimpleNamespace(
        representation_type="perceptual",
        perceptual_memory=types.SimpleNamespace(type="frame_sampling"),
    )
    policy.mem_buffer = types.SimpleNamespace(_history_feats={0: True})

    # reset_rng must be bool
    with pytest.raises(TypeError, match="reset_rng must be a bool"):
        policy.infer({"reset_rng": "not_a_bool"})

    # temporal_pos_override must be dict
    with pytest.raises(TypeError, match="temporal_pos_override must be a dict"):
        policy.infer({"temporal_pos_override": "not_a_dict"})


def test_infer_fails_closed_when_policy_is_not_framesamp():
    PolicyClass = _get_policy_class()

    # Case 1: symbolic policy
    policy_sym = object.__new__(PolicyClass)
    policy_sym._seed = 42
    policy_sym._rng = "key_42"
    policy_sym.config = types.SimpleNamespace(representation_type="symbolic")
    policy_sym.mem_buffer = None

    with pytest.raises(ValueError, match="only supported for perceptual FrameSamp policies"):
        policy_sym.infer({"temporal_pos_override": {"enabled": False}})

    # Case 2: perceptual token_dropping policy
    policy_td = object.__new__(PolicyClass)
    policy_td._seed = 42
    policy_td._rng = "key_42"
    policy_td.config = types.SimpleNamespace(
        representation_type="perceptual",
        perceptual_memory=types.SimpleNamespace(type="token_dropping"),
    )
    policy_td.mem_buffer = types.SimpleNamespace(_history_feats={0: True})

    with pytest.raises(ValueError, match="only supported for perceptual FrameSamp policies"):
        policy_td.infer({"temporal_pos_override": {"enabled": False}})

def test_invalid_enabled_values_raise_typeerror():
    """enabled field in temporal_pos_override must be a Python bool, not a truthy stand-in."""
    from mme_vla_suite.shared.mem_buffer import MemoryBuffer

    buf = MemoryBuffer(
        num_views=1, img_emb_dim=2, pos_emb_dim=6,
        state_emb_dim=4, prepare_buffer=False,
    )
    # Populate minimal history so _load_emb doesn't fail before validation.
    for idx in range(6):
        buf._history_feats[idx] = {
            "image_emb_4x4": np.full((1, 16, 2), idx, dtype=np.float32),
            "pos_emb_4x4": np.full((1, 16, 6), idx + 0.25, dtype=np.float32),
            "state_emb": np.arange(4, dtype=np.float32) + idx,
        }

    indices = [0, 1, 2, 3, 4, 5]
    history_feats = buf._history_feats

    invalid_values = ["false", "true", 0, 1]
    for val in invalid_values:
        override = {"enabled": val, "step_idx": 5, "position_map": []}
        with pytest.raises(TypeError, match="'enabled' must be a bool"):
            buf._prepare_frame_sampling(
                history_feats, indices,
                token_budget=96, token_per_image=16,
                step_idx=5, temporal_pos_override=override,
            )

