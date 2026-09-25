"""Source-compatible AgentSpec adapters for frozen AeroWall skill actors."""

from copy import copy
from types import SimpleNamespace


def agent_spec_for_observation_dim(agent_spec, observation_dim):
    """Build a lightweight source-policy spec with a chosen observation width."""
    if agent_spec.observation_spec.shape[-1] == observation_dim:
        return agent_spec
    env = agent_spec._env
    if env is None:
        raise ValueError("source policy spec needs an environment-backed AgentSpec")
    observation_spec = env.observation_spec.clone()
    from torchrl.data import UnboundedContinuousTensorSpec
    current = agent_spec.observation_spec
    observation_shape = (*current.shape[:-1], observation_dim)
    observation_spec[agent_spec.observation_key] = UnboundedContinuousTensorSpec(
        observation_shape, device=current.device, dtype=current.dtype,
    )
    proxy = SimpleNamespace(
        observation_spec=observation_spec,
        action_spec=env.action_spec,
        reward_spec=env.reward_spec,
    )
    for name in ("done_spec", "input_spec", "output_spec"):
        if hasattr(env, name):
            setattr(proxy, name, getattr(env, name))
    source_spec = copy(agent_spec)
    source_spec._env = proxy
    return source_spec
