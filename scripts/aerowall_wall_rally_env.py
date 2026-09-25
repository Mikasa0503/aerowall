"""Single-Iris wall rally task using the released HCSP model and PRT actions."""
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec

from hcsp.envs.isaac_env import AgentSpec
from hcsp.envs.low_level_skill.volley_env import VolleyEnv
from hcsp.robots.drone import MultirotorBase
from hcsp.utils.torch import quat_rotate
from omni.isaac.core.materials import PhysicsMaterial, PreviewSurface
from omni.isaac.core.objects import FixedCuboid
from pxr import Gf, Sdf, UsdGeom
from aerowall_wall_reward_logic import center_band_wall_reward
from aerowall.wall_rl.observations import (
    build_aerowall_intercept_target_retention_v1_observation,
    build_aerowall_goal_observation,
    build_relative_observation,
    build_relative_observation_v2,
)
from aerowall.wall_rl.curriculum import (
    case_is_feasible,
    sample_hit_starts,
    validate_handoff_case,
)
from aerowall.wall_rl.rewards import (
    compute_skill_reward,
    compute_aerowall_causal_v3_reward_terms,
    compute_aerowall_causal_v4_reward_terms,
    compute_aerowall_causal_v5_reward_terms,
    compute_aerowall_causal_v6_reward_terms,
    intercept_potential,
)
from aerowall.wall_rl.skill_fsm import Skill, select_skill_batch
from aerowall.wall_rl.trajectory import (
    intercept_target_geometry_valid_batch,
    outbound_quality,
    outbound_quality_v2,
    predict_intercept_batch,
    target_quality,
)


class AeroWallSingleWallRallyEnv(VolleyEnv):
    """Finite-rally embodied task; event scoring never writes physical state."""

    wall_center_x = 0.0
    wall_size = (0.2, 6.0, 8.0)
    wall_front = 0.1
    racket_radius = 0.2
    wall_visual_rgb = (0.18, 0.27, 0.34)

    def _design_scene(self):
        paths = super()._design_scene()
        # Place assets clear of the wall before IsaacEnv's initial sim.reset.
        # Teleporting the articulation only at episode reset is too late: the
        # upstream x=0 placeholder has already intersected the new x=0 wall.
        from omni.isaac.core.prims import XFormPrim
        import numpy as np
        XFormPrim(f'/World/envs/env_0/{self.drone.name}_0').set_world_pose(position=np.array([1.5, 0., 2.]))
        XFormPrim('/World/envs/env_0/ball').set_world_pose(position=np.array([1.5, 0., 4.8]))
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        # The ground-plane USD adds a rendered sphere emitter at
        # /World/defaultGroundPlane/SphereLight. It appears through the wall as
        # the circular hotspot in the camera. Author invisibility in the root
        # layer so the referenced emitter is hidden while its light contribution
        # and the broad ambient DistantLight remain active.
        for light_path in (
            '/World/defaultGroundPlane/SphereLight',
            '/World/Light/GreySphere',
            '/World/Light/WhiteSphere',
        ):
            prim = stage.GetPrimAtPath(light_path)
            if prim.IsValid():
                imageable = UsdGeom.Imageable(prim)
                if imageable:
                    imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        ambient_prim = stage.GetPrimAtPath('/World/defaultGroundPlane/AmbientLight')
        ambient_intensity = ambient_prim.GetAttribute('inputs:intensity')
        if ambient_intensity.IsValid():
            # Keep the original HCSP illumination readable after hiding the
            # visible emitter; a modest lift preserves the dark grid-floor look.
            ambient_intensity.Set(900.0)
        material = PhysicsMaterial(
            "/World/Physics_Materials/hcsp_single_wall",
            restitution=self.wall_restitution,
            static_friction=0.0,
            dynamic_friction=0.0,
        )
        # A flat, unlit slate blue-grey fits HCSP's grid floor while keeping the
        # red ball, green trajectory, and dark Iris readable. Emission is used
        # only to make the wall a true single color with no lighting gradient.
        wall_visual = PreviewSurface(
            "/World/Looks/hcsp_flat_wall",
            color=np.array([0.0, 0.0, 0.0]),
            roughness=1.0,
            metallic=0.0,
        )
        wall_shader = wall_visual.shaders_list[0]
        wall_shader.CreateInput(
            "emissiveColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(*self.wall_visual_rgb))
        wall_shader.CreateInput(
            "useSpecularWorkflow", Sdf.ValueTypeNames.Int
        ).Set(1)
        wall_shader.CreateInput(
            "specularColor", Sdf.ValueTypeNames.Color3f
        ).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        wall_shader.CreateInput(
            "ior", Sdf.ValueTypeNames.Float
        ).Set(1.0)
        FixedCuboid(
            "/World/envs/env_0/single_wall",
            position=torch.tensor([self.wall_center_x, 0.0, 4.0]).cpu().numpy(),
            scale=torch.tensor(self.wall_size).cpu().numpy(),
            visual_material=wall_visual,
            physics_material=material,
        )
        return paths

    def __init__(self, cfg, headless):
        self.curriculum_stage = str(cfg.task.get("wall_curriculum_stage", "C")).upper()
        self.draw_ball_trajectory = bool(cfg.task.get("wall_draw_ball_trajectory", False))
        self.recenter_reward_enabled = bool(cfg.task.get("wall_recenter_reward", False))
        self.wall_reward_variant = str(cfg.task.get("wall_reward_variant", "constant"))
        self.wall_observation_version = str(cfg.task.get("wall_observation_version", "legacy"))
        self.wall_reward_design = str(cfg.task.get("wall_reward_design", "legacy"))
        self.wall_case_mode = str(cfg.task.get("wall_case_mode", "fixed"))
        self.wall_hit_fixed_incoming = bool(cfg.task.get("wall_hit_fixed_incoming", False))
        self.wall_goal_y_bounds = tuple(cfg.task.get("wall_goal_y_bounds", (-0.3, 0.3)))
        self.wall_goal_z_bounds = tuple(cfg.task.get("wall_goal_z_bounds", (3.8, 4.2)))
        self.wall_restitution = float(cfg.task.get("wall_restitution", 0.8))
        self.wall_contact_height = float(cfg.task.get("wall_contact_height", 2.18))
        self.wall_y_bounds = tuple(cfg.task.get("wall_y_bounds", (-2.8, 2.8)))
        self.wall_z_bounds = tuple(cfg.task.get("wall_z_bounds", (0.3, 7.7)))
        if len(self.wall_y_bounds) != 2 or self.wall_y_bounds[0] >= self.wall_y_bounds[1]:
            raise ValueError("wall_y_bounds must be an increasing pair")
        if len(self.wall_z_bounds) != 2 or self.wall_z_bounds[0] >= self.wall_z_bounds[1]:
            raise ValueError("wall_z_bounds must be an increasing pair")
        if self.wall_observation_version not in ("legacy", "relative_v2", "relative_v3", "aerowall_goal_v1"):
            raise ValueError(f"Unknown wall_observation_version: {self.wall_observation_version}")
        if len(self.wall_goal_y_bounds) != 2 or self.wall_goal_y_bounds[0] >= self.wall_goal_y_bounds[1]:
            raise ValueError("wall_goal_y_bounds must be an increasing pair")
        if len(self.wall_goal_z_bounds) != 2 or self.wall_goal_z_bounds[0] >= self.wall_goal_z_bounds[1]:
            raise ValueError("wall_goal_z_bounds must be an increasing pair")
        if self.wall_reward_design not in ("legacy", "causal_v1", "causal_v2", "aerowall_causal_v3", "aerowall_causal_v4", "aerowall_causal_v5", "aerowall_causal_v6"):
            raise ValueError(f"Unknown wall_reward_design: {self.wall_reward_design}")
        if self.wall_reward_design in ("causal_v1", "causal_v2") and self.wall_observation_version != "relative_v2":
            raise ValueError(f"{self.wall_reward_design} requires relative_v2 observations")
        if self.wall_reward_design in ("aerowall_causal_v3", "aerowall_causal_v4", "aerowall_causal_v5", "aerowall_causal_v6") and self.wall_observation_version not in ("relative_v3", "aerowall_goal_v1"):
            raise ValueError(f"{self.wall_reward_design} requires relative_v3-compatible observations")
        if self.wall_case_mode not in ("fixed", "train", "bank"):
            raise ValueError(f"Unknown wall_case_mode: {self.wall_case_mode}")
        self.wall_train_skill = str(cfg.task.get("wall_train_skill", ""))
        self.wall_launch_distribution = str(cfg.task.get("wall_launch_distribution", "standard"))
        self.wall_launch_y_bounds = tuple(cfg.task.get("wall_launch_y_bounds_m", (-0.4, 0.4)))
        self.wall_launch_vy_bounds = tuple(cfg.task.get("wall_launch_vy_bounds_mps", (-0.3, 0.3)))
        if self.wall_launch_distribution not in ("standard", "AeroWallLateralInterceptV1"):
            raise ValueError(f"Unknown AeroWall launch distribution: {self.wall_launch_distribution}")
        if self.wall_launch_distribution != "standard" and self.wall_train_skill != "intercept":
            raise ValueError("AeroWallLateralInterceptV1 is reserved for Intercept-skill training")
        if (len(self.wall_launch_y_bounds) != 2 or self.wall_launch_y_bounds[0] >= self.wall_launch_y_bounds[1]
                or len(self.wall_launch_vy_bounds) != 2 or self.wall_launch_vy_bounds[0] >= self.wall_launch_vy_bounds[1]):
            raise ValueError("AeroWall launch distribution ranges must be increasing pairs")
        self.wall_hit_artificial_ratio = float(cfg.task.get("wall_hit_artificial_ratio", 0.0))
        self.wall_hit_window = float(cfg.task.get("wall_hit_enter_seconds", cfg.task.get("wall_hit_window", 0.18)))
        self.wall_hit_exit_seconds = float(cfg.task.get("wall_hit_exit_seconds", self.wall_hit_window))
        self.wall_min_dwell_steps = int(cfg.task.get("wall_min_dwell_steps", 2))
        if not 0.0 <= self.wall_hit_artificial_ratio <= 1.0:
            raise ValueError("wall_hit_artificial_ratio must be in [0, 1]")
        if not 0.05 <= self.wall_hit_window <= 1.0:
            raise ValueError("wall_hit_window must be between 0.05 and 1.0 seconds")
        if not self.wall_hit_window <= self.wall_hit_exit_seconds <= 1.0:
            raise ValueError("wall_hit_exit_seconds must be between hit_enter_seconds and 1.0 seconds")
        if self.wall_min_dwell_steps < 0:
            raise ValueError("wall_min_dwell_steps must be nonnegative")
        self.initial_cases = None
        if self.wall_reward_variant not in ("constant", "center_band"):
            raise ValueError(f"Unknown wall_reward_variant: {self.wall_reward_variant}")
        self.perturb_mode = str(cfg.task.get("wall_perturb_mode", "none"))
        self.perturb_light = float(cfg.task.get("wall_perturb_light", 0.0))
        self.perturb_medium = float(cfg.task.get("wall_perturb_medium", 0.0))
        self.record_events = bool(cfg.task.get("wall_record_events", True))
        self.wall_end_on_ball_drop = bool(cfg.task.get("wall_end_on_ball_drop", False))
        super().__init__(cfg, headless)
        assert self.drone.n == 1
        n = self.num_envs
        self.phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self.rallies = torch.zeros(n, dtype=torch.long, device=self.device)
        self.streak = torch.zeros(n, dtype=torch.long, device=self.device)
        self.max_streak = torch.zeros(n, dtype=torch.long, device=self.device)
        self.caps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.walls = torch.zeros(n, dtype=torch.long, device=self.device)
        self.failure = torch.zeros(n, dtype=torch.long, device=self.device)
        self.failure_reason_bits = torch.zeros(n, dtype=torch.long, device=self.device)
        self.prev_ball_vel = torch.zeros(n, 3, device=self.device)
        self.prev_action = torch.zeros(n, 1, 4, device=self.device)
        self.action_before = torch.zeros_like(self.prev_action)
        self.wall_target = torch.tensor([0.0, 4.0], device=self.device).expand(n, 2).clone()
        self.skill_id = torch.zeros(n, dtype=torch.long, device=self.device)
        self.skill_held_steps = torch.zeros_like(self.skill_id)
        self.executed_skill_id = torch.zeros_like(self.skill_id)
        self.hit_artificial_start = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.hit_reset_count = 0
        self.hit_artificial_reset_count = 0
        self.potential_before = torch.zeros(n, device=self.device)
        self.current_potential = torch.zeros(n, device=self.device)
        self.potential_before_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.current_potential_valid = torch.zeros_like(self.potential_before_valid)
        self.step_outbound_quality = torch.zeros(n, device=self.device)
        self.step_wall_quality = torch.zeros(n, device=self.device)
        self.step_cap = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.step_wall = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.step_rally = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.step_illegal = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.step_phase0_illegal = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.step_phase2_illegal = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.step_ground = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.step_center_error = torch.ones(n, device=self.device)
        self.step_launch_quality = torch.zeros(n, device=self.device)
        self.step_launch_progress = torch.zeros(n, device=self.device)
        self.step_recenter_bonus = torch.zeros(n, device=self.device)
        self.step_wall_reward = torch.zeros(n, device=self.device)
        self.perturb_case_bank = None
        self.perturb_update = 1
        self.perturb_probability = 0.0
        self.perturb_medium_fraction = 0.0
        self.perturb_wanted = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.perturb_applied = torch.zeros_like(self.perturb_wanted)
        self.perturb_missed = torch.zeros_like(self.perturb_wanted)
        self.perturb_delta = torch.zeros(n, device=self.device)
        self.perturb_delay = torch.zeros(n, device=self.device)
        self.perturb_trigger_at = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.perturb_remaining = torch.zeros(n, dtype=torch.long, device=self.device)
        self.perturb_start_at = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.perturb_end_at = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.perturb_force_dv = torch.zeros(n, 3, device=self.device)
        self.perturb_events = []
        self.episode_id = torch.zeros(n, dtype=torch.long, device=self.device)
        self.current_substep = -1
        self.events = []

    def set_curriculum_update(self, update):
        self.perturb_update = int(update)
        if self.perturb_mode != "curriculum" or update <= 50:
            self.perturb_probability = self.perturb_medium_fraction = 0.0
        elif update <= 100:
            self.perturb_probability, self.perturb_medium_fraction = 0.25, 0.0
        elif update <= 200:
            self.perturb_probability, self.perturb_medium_fraction = 0.5, 0.0
        else:
            self.perturb_probability, self.perturb_medium_fraction = 0.5, 0.5

    def set_perturbation_cases(self, cases):
        assert len(cases) == self.num_envs
        self.perturb_case_bank = list(cases)
        self.perturb_mode = "cases"

    def set_initial_cases(self, cases):
        """Install a frozen, per-environment bank before the first reset."""
        assert len(cases) == self.num_envs
        handoff = ["handoff_state_version" in case for case in cases]
        if any(handoff) and not all(handoff):
            raise ValueError("Initial case bank cannot mix natural and handoff states")
        if all(handoff):
            for case in cases:
                validate_handoff_case(case)
        elif not all(case_is_feasible(case) for case in cases):
            raise ValueError("Initial case bank contains an infeasible state")
        if not all(self.wall_y_bounds[0] <= case['wall_target'][0] <= self.wall_y_bounds[1]
                   and self.wall_z_bounds[0] <= case['wall_target'][1] <= self.wall_z_bounds[1]
                   for case in cases):
            raise ValueError("Wall target outside the finite wall")
        self.initial_cases = list(cases)
        self.wall_case_mode = "bank"

    def _set_specs(self):
        state_dim = self.drone.state_spec.shape[-1]
        # A0-A2 stay exactly compatible with HCSP Goto, including its 26-wide
        # LayerNorm. Wall/return stages append ball and phase observations.
        obs_dim = 26 if self.curriculum_stage in ("A0", "A", "A1", "A2") else 46
        if obs_dim == 46 and self.wall_observation_version == "aerowall_goal_v1":
            obs_dim = 48
        agent_observations = {"observation": UnboundedContinuousTensorSpec((1, obs_dim))}
        if obs_dim in (46, 48):
            for version in ("legacy", "relative_v2", "relative_v3"):
                agent_observations[f"{version}_observation"] = UnboundedContinuousTensorSpec((1, 46))
            agent_observations["aerowall_goal_v1_observation"] = UnboundedContinuousTensorSpec((1, 48))
        reward_components = (
            "approach_shaping", "legal_contact", "outbound_quality", "wall_quality",
            "next_contact", "illegal_penalty", "crash_penalty", "out_penalty",
            "action_smoothness", "phase0_intercept_safety_penalty",
            "phase2_return_safety_penalty", "phase0_intercept_safety_penalty_v6",
        )
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec(agent_observations),
            "stats": CompositeSpec({
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "rallies": UnboundedContinuousTensorSpec(1),
                "max_streak": UnboundedContinuousTensorSpec(1),
                "caps": UnboundedContinuousTensorSpec(1),
                "walls": UnboundedContinuousTensorSpec(1),
                "failure": UnboundedContinuousTensorSpec(1),
                "failure_reason_bits": UnboundedContinuousTensorSpec(1),
                **{f"reward_{name}": UnboundedContinuousTensorSpec(1) for name in reward_components},
            }),
            "info": CompositeSpec({
                "drone_state": UnboundedContinuousTensorSpec((1, 13)),
                "skill_id": UnboundedContinuousTensorSpec((1,)),
            }),
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({"action": torch.stack([self.drone.action_spec], dim=0)})
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({"reward": UnboundedContinuousTensorSpec((1, 1))})
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        self.stats = self.observation_spec["stats"].zero()
        self.info = self.observation_spec["info"].zero()

    def _reset_idx(self, env_ids):
        self.drone._reset_idx(env_ids, self.training)
        count = len(env_ids)
        artificial_hit = torch.zeros(count, dtype=torch.bool, device=self.device)
        if (self.curriculum_stage == "RALLY" and self.wall_case_mode == "train"
                and self.wall_train_skill == "hit" and self.wall_hit_artificial_ratio > 0.0
                and hasattr(self, "phase")):
            artificial_hit = torch.rand(count, device=self.device) < self.wall_hit_artificial_ratio
            if self.training:
                self.hit_reset_count += count
                self.hit_artificial_reset_count += int(artificial_hit.sum().item())
        pos = torch.tensor([1.5, 0.0, 2.0], device=self.device).expand(count, 1, 3).clone()
        quat = torch.zeros(count, 1, 4, device=self.device); quat[..., 0] = 1.0
        drone_velocity = torch.zeros(count, 6, device=self.device)
        self.drone.set_world_poses(pos + self.envs_positions[env_ids, None], quat, env_ids)
        self.drone.set_velocities(torch.zeros(count, 6, device=self.device), env_ids)
        ball = torch.tensor([1.5, 0.0, 4.8], device=self.device).expand(count, 1, 3).clone()
        ball_quat = torch.zeros(count, 4, device=self.device); ball_quat[..., 0] = 1.0
        ball_velocity = torch.zeros(count, 6, device=self.device)
        handoff_throttle = []
        if self.wall_case_mode == "train" and self.curriculum_stage in ("RALLY", "INTERCEPT"):
            # The standard ranges remain separate from case banks. Named AeroWall
            # training distributions are generated independently below.
            fixed_hit_prefix = self.wall_train_skill == "hit" and self.wall_hit_fixed_incoming
            if not fixed_hit_prefix:
                ball[..., 0] = 1.35 + 0.30 * torch.rand(count, 1, device=self.device)
                ball[..., 1] = -0.4 + 0.80 * torch.rand(count, 1, device=self.device)
                ball[..., 2] = 4.3 + 0.70 * torch.rand(count, 1, device=self.device)
                ball_velocity[:, 0] = -0.15 + 0.30 * torch.rand(count, device=self.device)
                ball_velocity[:, 1] = -0.30 + 0.60 * torch.rand(count, device=self.device)
                if self.wall_launch_distribution == "AeroWallLateralInterceptV1":
                    y0, y1 = self.wall_launch_y_bounds
                    vy0, vy1 = self.wall_launch_vy_bounds
                    ball[..., 1] = y0 + (y1 - y0) * torch.rand(count, 1, device=self.device)
                    ball_velocity[:, 1] = vy0 + (vy1 - vy0) * torch.rand(count, device=self.device)
                ball_velocity[:, 2] = -0.25 * torch.rand(count, device=self.device)
                pos[..., 0] = 1.35 + 0.30 * torch.rand(count, 1, device=self.device)
                pos[..., 1] = -0.15 + 0.30 * torch.rand(count, 1, device=self.device)
                self.drone.set_world_poses(pos + self.envs_positions[env_ids, None], quat, env_ids)
            if fixed_hit_prefix:
                y_bounds, z_bounds = self.wall_goal_y_bounds, self.wall_goal_z_bounds
                self.wall_target[env_ids, 0] = y_bounds[0] + (y_bounds[1] - y_bounds[0]) * torch.rand(count, device=self.device)
                self.wall_target[env_ids, 1] = z_bounds[0] + (z_bounds[1] - z_bounds[0]) * torch.rand(count, device=self.device)
            else:
                self.wall_target[env_ids, 0] = -0.6 + 1.2 * torch.rand(count, device=self.device)
                self.wall_target[env_ids, 1] = 3.5 + torch.rand(count, device=self.device)
            if bool(artificial_hit.any()):
                rows = torch.nonzero(artificial_hit, as_tuple=False).squeeze(-1)
                starts = sample_hit_starts(pos[rows, 0], plane_z=self.wall_contact_height)
                ball[rows, 0] = starts["ball_position"]
                ball_velocity[rows, :3] = starts["ball_velocity"]
        elif self.wall_case_mode == "bank" and self.curriculum_stage in ("RALLY", "INTERCEPT"):
            assert self.initial_cases is not None, "set_initial_cases before reset"
            for offset, index in enumerate(env_ids.cpu().tolist()):
                case = self.initial_cases[index]
                pos[offset, 0] = torch.tensor(case["drone_position"], device=self.device)
                ball[offset, 0] = torch.tensor(case["ball_position"], device=self.device)
                ball_velocity[offset, :3] = torch.tensor(case["ball_velocity"], device=self.device)
                self.wall_target[index] = torch.tensor(case["wall_target"], device=self.device)
                if "handoff_state_version" in case:
                    quat[offset, 0] = torch.tensor(case["drone_orientation"], device=self.device)
                    drone_velocity[offset] = torch.tensor(case["drone_velocity"], device=self.device)
                    ball_quat[offset] = torch.tensor(case["ball_orientation"], device=self.device)
                    ball_velocity[offset, 3:] = torch.tensor(case["ball_angular_velocity"], device=self.device)
                    handoff_throttle.append(case["motor_throttle"])
            self.drone.set_world_poses(pos + self.envs_positions[env_ids, None], quat, env_ids)
            self.drone.set_velocities(drone_velocity, env_ids)
            if handoff_throttle:
                throttle = torch.tensor(
                    handoff_throttle,
                    dtype=self.drone.throttle.dtype, device=self.device,
                ).reshape(count, 1, 4)
                self.drone.throttle[env_ids] = throttle
        if self.curriculum_stage == "A0":
            ball[:] = torch.tensor([100.0, 0.0, 100.0], device=self.device)
        elif self.curriculum_stage in ("A", "A1"):
            ball[..., 2] = 2.8
            ball_velocity[:, 2] = -0.25
        elif self.curriculum_stage == "A2":
            ball[..., :2] += (torch.rand(count, 1, 2, device=self.device) - 0.5) * 0.3
            ball[..., 2] = 2.8 + torch.rand(count, 1, device=self.device) * 2.0
            ball_velocity[:, 2] = -torch.rand(count, device=self.device) * 0.75
        self.ball.set_world_poses(ball + self.envs_positions[env_ids, None], ball_quat, env_ids)
        self.ball.set_velocities(ball_velocity, env_ids)
        self.ball.set_masses(torch.full((count,), self.ball_mass, device=self.device), env_ids)
        if hasattr(self, "phase"):
            self.episode_id[env_ids] += 1
            for value in (self.phase, self.rallies, self.streak, self.max_streak,
                          self.caps, self.walls, self.failure, self.failure_reason_bits):
                value[env_ids] = 0
            self.prev_ball_vel[env_ids] = ball_velocity[:, :3]
            self.prev_action[env_ids] = 0
            self.action_before[env_ids] = 0
            self.skill_id[env_ids] = 0
            self.hit_artificial_start[env_ids] = artificial_hit
            self.skill_id[env_ids[artificial_hit]] = int(Skill.HIT)
            self.executed_skill_id[env_ids] = self.skill_id[env_ids]
            self.skill_held_steps[env_ids] = 0
            self.potential_before[env_ids] = 0
            self.current_potential[env_ids] = 0
            self.potential_before_valid[env_ids] = False
            self.current_potential_valid[env_ids] = False
            self.step_outbound_quality[env_ids] = 0
            self.stats[env_ids] = 0.0
            self.step_recenter_bonus[env_ids] = 0.0
            self.step_wall_reward[env_ids] = 0.0
            self.step_wall_quality[env_ids] = 0.0
            self.perturb_wanted[env_ids] = False
            self.perturb_applied[env_ids] = False
            self.perturb_missed[env_ids] = False
            self.perturb_trigger_at[env_ids] = -1
            self.perturb_remaining[env_ids] = 0
            self.perturb_start_at[env_ids] = -1
            self.perturb_end_at[env_ids] = -1
            self.perturb_force_dv[env_ids] = 0.0
            if self.wall_case_mode == "bank" and self.initial_cases is not None:
                for offset, index in enumerate(env_ids.cpu().tolist()):
                    case = self.initial_cases[index]
                    if "handoff_state_version" not in case:
                        continue
                    self.phase[index] = int(case["phase"])
                    self.caps[index] = int(case["caps"])
                    self.walls[index] = int(case["walls"])
                    self.rallies[index] = int(case["rallies"])
                    self.streak[index] = int(case["streak"])
                    self.max_streak[index] = int(case["max_streak"])
                    self.skill_id[index] = int(case["skill_id"])
                    self.executed_skill_id[index] = int(case["skill_id"])
                    self.skill_held_steps[index] = int(case["skill_held_steps"])
                    self.prev_ball_vel[index] = torch.tensor(case["prev_ball_velocity"], device=self.device)
                    self.prev_action[index, 0] = torch.tensor(case["prev_action"], device=self.device)
                    self.action_before[index, 0] = torch.tensor(case["action_before"], device=self.device)
            if self.perturb_mode == "cases":
                assert self.perturb_case_bank is not None
                for index in env_ids.cpu().tolist():
                    case = self.perturb_case_bank[index]
                    self.perturb_wanted[index] = True
                    self.perturb_delta[index] = float(case["delta_vy"])
                    self.perturb_delay[index] = float(case["delay_seconds"])
            elif self.perturb_mode == "curriculum" and self.perturb_probability > 0.0:
                wanted = torch.rand(count, device=self.device) < self.perturb_probability
                signs = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
                medium = torch.rand(count, device=self.device) < self.perturb_medium_fraction
                magnitude = torch.where(medium, self.perturb_medium, self.perturb_light)
                self.perturb_wanted[env_ids] = wanted
                self.perturb_delta[env_ids] = signs * magnitude
                self.perturb_delay[env_ids] = 0.12 + 0.08 * torch.rand(count, device=self.device)
            else:
                self.perturb_delta[env_ids] = 0.0
                self.perturb_delay[env_ids] = 0.0
        if self.draw_ball_trajectory and hasattr(self, "ball_traj_vis"):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()

    def set_probe_states(self):
        """Initialization-only fixtures for contact contract validation."""
        assert self.num_envs == 5 and not bool((self.progress_buf > 0).any())
        ids = torch.arange(5, device=self.device)
        q = torch.zeros(5, 4, device=self.device); q[:, 0] = 1
        ball = torch.tensor([
            [1.50, 0.00, 2.32],
            [1.61, 0.00, 2.32],
            [1.79, 0.00, 2.32],
            [0.38, 0.00, 2.50],
            [3.00, 0.00, 0.25],
        ], device=self.device)
        vel = torch.tensor([
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -2.0],
            [-3.0, 0.0, 0.0],
            [0.0, 0.0, -2.0],
        ], device=self.device)
        self.ball.set_world_poses(ball + self.envs_positions, q, ids)
        full_vel = torch.zeros(5, 6, device=self.device); full_vel[:, :3] = vel
        self.ball.set_velocities(full_vel, ids)
        self.prev_ball_vel[:] = vel
        self.phase[3] = 1

    def _kinematics(self):
        root = self.drone.get_state()
        ball_pos, _ = self.get_env_poses(self.ball.get_world_poses())
        ball_vel = self.ball.get_velocities()
        return root, ball_pos, ball_vel

    def _legal_racket_region(self, root, ball_pos):
        drone_pos = root[:, 0, :3]
        quat = root[:, 0, 3:7]
        up = quat_rotate(quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        rel = ball_pos[:, 0] - drone_pos
        axial = (rel * up).sum(-1)
        radial = (rel - axial[:, None] * up).norm(dim=-1)
        legal = (axial >= 0.0) & (axial <= 2.0 * self.ball_radius) & (radial <= self.racket_radius)
        return legal, radial, axial, up

    def _observe_substep(self, substep):
        root, bp, bv = self._kinematics()
        linear = bv[:, 0, :3]
        delta = linear - self.prev_ball_vel - self.perturb_force_dv
        legal_region, radial, axial, up = self._legal_racket_region(root, bp)
        near_body = (bp[:, 0] - root[:, 0, :3]).norm(dim=-1) < 0.45
        body = (delta.norm(dim=-1) > 0.25) & near_body
        wall = (self.prev_ball_vel[:, 0] < -0.1) & (linear[:, 0] > 0.1) & (bp[:, 0, 0] < self.wall_front + self.ball_radius + 0.08)
        ground = (self.prev_ball_vel[:, 2] < -0.1) & (linear[:, 2] > 0.1) & (bp[:, 0, 2] < self.ball_radius + 0.08) & ~near_body
        ambiguous = body & wall
        legal_cap = body & legal_region & ~ambiguous
        invalid_body = body & ~legal_region & ~ambiguous
        expected_cap = legal_cap & ((self.phase == 0) | (self.phase == 2))
        return_cap = expected_cap & (self.phase == 2)
        unexpected_cap = legal_cap & (self.phase == 1)
        expected_wall = wall & (self.phase == 1) & ~ambiguous
        unexpected_wall = wall & (self.phase != 1) & ~ambiguous
        wall_reward = (center_band_wall_reward(bp[:, 0, 1]) if self.wall_reward_variant == "center_band"
                       else torch.full_like(bp[:, 0, 1], 10.0))
        self.step_wall_reward = torch.where(expected_wall, wall_reward, self.step_wall_reward)
        if self.record_events:
            for i in torch.nonzero(body | wall).flatten().cpu().tolist():
                self.events.append({
                    "env": i, "episode_id": int(self.episode_id[i]),
                    "control_step": int(self.progress_buf[i]), "substep": substep,
                    "body": bool(body[i]), "legal_cap": bool(legal_cap[i]), "wall": bool(wall[i]),
                    "expected_cap": bool(expected_cap[i]), "phase_before": int(self.phase[i]),
                    "expected_wall": bool(expected_wall[i]),
                    "wall_reward": float(wall_reward[i]) if bool(expected_wall[i]) else 0.0,
                    "radial_error": float(radial[i]), "axial_distance": float(axial[i]),
                    "racket_radius": self.racket_radius, "axial_limit": 2.0 * self.ball_radius,
                    "drone_position": root[i, 0, :3].cpu().tolist(),
                    "drone_orientation_wxyz": root[i, 0, 3:7].cpu().tolist(),
                    "drone_up_vector": up[i].cpu().tolist(),
                    "drone_linear_velocity": root[i, 0, 7:10].cpu().tolist(),
                    "drone_angular_velocity": root[i, 0, 10:13].cpu().tolist(),
                    "ball_relative_position_world": (bp[i, 0] - root[i, 0, :3]).cpu().tolist(),
                    "executed_skill_id": int(self.executed_skill_id[i].item()),
                    "fsm_skill_id": int(self.skill_id[i].item()),
                    "policy_action": self.prev_action[i, 0].cpu().tolist(),
                    "motor_throttle": self.drone.throttle[i, 0].cpu().tolist(),
                    "motor_thrust_local_z_n": self.drone.thrusts[i, 0, :, 2].cpu().tolist(),
                    "ball_position": bp[i, 0].cpu().tolist(),
                    "ball_velocity_before": self.prev_ball_vel[i].cpu().tolist(), "ball_velocity_after": linear[i].cpu().tolist(),
                    "known_external_delta_v": self.perturb_force_dv[i].cpu().tolist(),
                })
        self.step_cap |= expected_cap
        self.step_wall |= expected_wall
        self.step_rally |= return_cap
        illegal_event = invalid_body | unexpected_cap | unexpected_wall | ambiguous
        self.step_illegal |= illegal_event
        self.step_phase0_illegal |= illegal_event & (self.phase == 0)
        self.step_phase2_illegal |= illegal_event & (self.phase == 2)
        self.step_ground |= ground
        self.step_center_error = torch.where(expected_cap, radial, self.step_center_error)
        if self.wall_reward_design in ("causal_v1", "causal_v2", "aerowall_causal_v3", "aerowall_causal_v4", "aerowall_causal_v5", "aerowall_causal_v6"):
            quality_fn = outbound_quality if self.wall_reward_design == "causal_v1" else outbound_quality_v2
            quality = quality_fn(bp[:, 0], linear, self.wall_target,
                                 root[:, 0, :3], wall_x=self.wall_front + self.ball_radius,
                                 wall_y_bounds=self.wall_y_bounds,
                                 wall_z_bounds=self.wall_z_bounds,
                                 restitution=self.wall_restitution)
            self.step_outbound_quality = torch.where(expected_cap, quality, self.step_outbound_quality)
        if self.wall_reward_design in ("causal_v2", "aerowall_causal_v3", "aerowall_causal_v4", "aerowall_causal_v5", "aerowall_causal_v6"):
            quality = target_quality(bp[:, 0], self.wall_target)
            self.step_wall_quality = torch.where(expected_wall, quality, self.step_wall_quality)
        lateral_error = linear[:, 1]
        if self.recenter_reward_enabled:
            z, vx, vy, vz = bp[:, 0, 2], linear[:, 0], linear[:, 1], linear[:, 2]
            discriminant = vz.square() + 19.62 * (z - 2.18)
            time_to_return = (vz + discriminant.clamp_min(0.0).sqrt()) / 9.81
            time_to_wall = (self.wall_front + self.ball_radius - bp[:, 0, 0]) / torch.where(vx.abs() > 0.05, vx, torch.full_like(vx, -0.05))
            valid = return_cap & (discriminant >= 0.0) & (vx < -0.05) & (time_to_return > 0.0) & (time_to_return <= 2.0) & (time_to_wall > 0.0) & (time_to_wall < time_to_return)
            desired_vy = (-bp[:, 0, 1] / time_to_return.clamp_min(0.001)).clamp(-1.2, 1.2)
            lateral_error = torch.where(valid, vy - desired_vy, vy)
            return_y = bp[:, 0, 1] + vy * time_to_return
            wall_y = bp[:, 0, 1] + vy * time_to_wall
            bonus = 10.0 * torch.exp(-(return_y / 0.75).square()) + 5.0 * torch.exp(-(wall_y / 0.75).square())
            self.step_recenter_bonus = torch.where(valid, bonus, self.step_recenter_bonus)
        launch_quality = torch.exp(-((linear[:, 0] + 2.3) / 1.2).square() - ((linear[:, 2] - 6.5) / 2.5).square() - (lateral_error / 0.8).square())
        self.step_launch_quality = torch.where(expected_cap, launch_quality, self.step_launch_quality)
        progress = (1. - 0.2 * (linear[:, 0] + 2.3).abs() - 0.15 * lateral_error.abs()
                    - 0.1 * (linear[:, 2] - 6.5).abs()).clamp(-1., 1.)
        self.step_launch_progress = torch.where(return_cap, progress, self.step_launch_progress)
        self.caps += expected_cap.long(); self.walls += expected_wall.long()
        self.rallies += return_cap.long(); self.streak += return_cap.long()
        self.max_streak = torch.maximum(self.max_streak, self.streak)
        self.phase = torch.where(expected_cap, torch.ones_like(self.phase), self.phase)
        self.phase = torch.where(expected_wall, torch.full_like(self.phase, 2), self.phase)
        just_second = return_cap & (self.rallies == 2) & self.perturb_wanted
        absolute_substep = self.progress_buf * self.substeps + substep
        delay_substeps = (self.perturb_delay / float(self.cfg.sim.dt)).ceil().long()
        self.perturb_trigger_at = torch.where(just_second, absolute_substep + delay_substeps, self.perturb_trigger_at)
        self.perturb_missed |= expected_wall & self.perturb_wanted & (self.rallies >= 2) & ~self.perturb_applied
        self.prev_ball_vel[:] = linear

    def _apply_perturbation(self, substep):
        self.perturb_force_dv.zero_()
        if self.perturb_mode == "none":
            return
        absolute_substep = self.progress_buf * self.substeps + substep
        waiting = self.perturb_wanted & ~self.perturb_applied & ~self.perturb_missed & (self.perturb_trigger_at >= 0) & (absolute_substep >= self.perturb_trigger_at)
        if bool(waiting.any()):
            root, bp, bv = self._kinematics()
            x = bp[:, 0, 0]
            vx = bv[:, 0, 0]
            wall_time = (self.wall_front + self.ball_radius - x) / vx.clamp_max(-0.05)
            safe = ((bp[:, 0] - root[:, 0, :3]).norm(dim=-1) > 0.5) & (x > 0.5) & (vx < -0.05) & (wall_time > 0.15) & (self.phase == 1)
            starting = waiting & safe
            self.perturb_remaining = torch.where(starting, torch.full_like(self.perturb_remaining, 40), self.perturb_remaining)
            self.perturb_applied |= starting
            self.perturb_start_at = torch.where(starting, absolute_substep, self.perturb_start_at)
        active = self.perturb_remaining > 0
        force_y = torch.where(active, self.ball_mass * self.perturb_delta / 0.1, torch.zeros_like(self.perturb_delta))
        forces = torch.stack([torch.zeros_like(force_y), force_y, torch.zeros_like(force_y)], -1)
        self.ball.apply_forces(forces, is_global=True)
        self.perturb_force_dv[:, 1] = force_y * float(self.cfg.sim.dt) / self.ball_mass
        self.perturb_remaining -= active.long()
        finished = active & (self.perturb_remaining == 0)
        self.perturb_end_at = torch.where(finished, absolute_substep, self.perturb_end_at)
        if self.record_events:
            for i in torch.nonzero(active).flatten().cpu().tolist():
                self.perturb_events.append({"env": i, "control_step": int(self.progress_buf[i]), "substep": substep,
                                            "force_y": float(force_y[i]), "known_delta_vy": float(self.perturb_force_dv[i, 1])})

    def _step(self, tensordict):
        self.executed_skill_id[:] = self.skill_id
        self.step_launch_progress.zero_()
        self.step_recenter_bonus.zero_()
        self.step_wall_reward.zero_()
        self.step_wall_quality.zero_()
        self.step_outbound_quality.zero_()
        self.step_cap.zero_(); self.step_wall.zero_(); self.step_rally.zero_(); self.step_illegal.zero_(); self.step_phase0_illegal.zero_(); self.step_phase2_illegal.zero_(); self.step_ground.zero_(); self.step_center_error.fill_(1.0); self.step_launch_quality.zero_()
        self.potential_before[:] = self.current_potential
        self.potential_before_valid[:] = self.current_potential_valid
        for substep in range(self.substeps):
            self.current_substep = substep
            if substep == 0:
                # HCSP's rotor lag coefficient is defined at the 50 Hz policy
                # rate. Update it once, then reapply the resulting thrust over
                # the remaining fine physics steps without advancing the motor
                # filter another seven times.
                self._pre_sim_step(tensordict)
            else:
                held_action = self.drone.throttle.square().mul(2.0).sub(1.0)
                self.drone.apply_action(held_action)
            self._apply_perturbation(substep)
            self.sim.step(self._should_render(substep))
            self._observe_substep(substep)
        self.current_substep = -1
        self._post_sim_step(tensordict)
        self.progress_buf += 1
        if (self.curriculum_stage == "RALLY"
                and self.wall_observation_version in ("relative_v2", "relative_v3", "aerowall_goal_v1")):
            self._advance_skill_state()
        out = TensorDict({}, self.batch_size, device=self.device)
        out.update(self._compute_state_and_obs())
        out.update(self._compute_reward_and_done())
        return out

    def _pre_sim_step(self, tensordict):
        self.action_before[:] = self.prev_action
        self.prev_action[:] = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(self.prev_action)

    def _advance_skill_state(self):
        """Advance the FSM once after a physical control transition."""
        root, bp, bv = self._kinematics()
        prediction_options = {
            "wall_y_bounds": self.wall_y_bounds,
            "wall_z_bounds": self.wall_z_bounds,
            "restitution": self.wall_restitution,
        }
        if self.wall_observation_version == "relative_v2":
            # Reproduce the historical infinite-plane route for v2 checkpoints.
            prediction_options = {
                "wall_y_bounds": (-float("inf"), float("inf")),
                "wall_z_bounds": (-float("inf"), float("inf")),
                "restitution": self.wall_restitution,
            }
        _, contact_time, feasible = predict_intercept_batch(
            bp[:, 0], bv[:, 0, :3], root[:, 0, :3],
            plane_z=self.wall_contact_height, wall_x=self.wall_front + self.ball_radius,
            **prediction_options,
        )
        completed_steps = self.skill_held_steps + 1
        next_skill = select_skill_batch(
            self.skill_id, self.caps, self.phase, contact_time, feasible,
            completed_steps, hit_enter_seconds=self.wall_hit_window,
            hit_exit_seconds=self.wall_hit_exit_seconds,
            min_dwell_steps=self.wall_min_dwell_steps,
            artificial_hit=self.hit_artificial_start,
        )
        changed = next_skill != self.skill_id
        self.skill_id[:] = next_skill
        self.skill_held_steps[:] = torch.where(
            changed, torch.zeros_like(completed_steps), completed_steps,
        )

    def _compute_state_and_obs(self):
        root, bp, bv = self._kinematics()
        self.root_state = root; self.ball_pos = bp; self.ball_vel = bv
        if self.draw_ball_trajectory:
            ball_plot_pos = (bp[0] + self.envs_positions[0]).tolist()
            # Match HCSP's own trajectory visualization exactly.
            if len(self.ball_traj_vis) > 1:
                self.draw.draw_lines(
                    self.ball_traj_vis[-1], ball_plot_pos,
                    [(.1, 1., .1, 1.)], [1.5],
                )
            self.ball_traj_vis.append(ball_plot_pos)
        rel = bp - root[..., :3]
        fixed_target = torch.tensor([1.5, 0.0, 2.0], device=self.device).expand(self.num_envs, 3)
        tracking_target = torch.stack([bp[:, 0, 0], bp[:, 0, 1], torch.full_like(bp[:, 0, 2], 2.0)], -1)
        target = fixed_target if self.curriculum_stage == "A0" else torch.where((self.phase == 0)[:, None], tracking_target, fixed_target)
        if self.curriculum_stage in ('RETURN', 'RALLY'):
            target = torch.where((self.caps > 0)[:, None], self._return_target(bp[:, 0], bv[:, 0, :3]), target)
        heading_target = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        goto_prefix = torch.cat([
            target[:, None] - root[..., :3],
            root[..., 3:],
            heading_target[:, None] - root[..., 13:16],
        ], -1)
        phase = F.one_hot(self.phase, 3).float()[:, None]
        vx = bv[:, 0, 0]
        wall_t = torch.where(vx < -0.05, (self.wall_front + self.ball_radius - bp[:, 0, 0]) / vx, torch.zeros_like(vx)).clamp(0, 3)
        return_t = torch.where(vx > 0.05, (1.5 - bp[:, 0, 0]) / vx, torch.zeros_like(vx)).clamp(0, 3)
        t = torch.where(self.phase == 1, wall_t, return_t)
        pred_z = bp[:, 0, 2] + bv[:, 0, 2] * t - 4.905 * t.square()
        pred = torch.stack([t, torch.full_like(t, 1.5), torch.zeros_like(t), pred_z], -1)[:, None]
        agents = {}
        if self.curriculum_stage in ("A0", "A", "A1", "A2"):
            obs = goto_prefix
        else:
            legacy_obs = torch.cat([goto_prefix, bp, rel, bv[..., :3], phase, pred, self.prev_action], -1)
            old_infinite_bounds = (-float("inf"), float("inf"))
            intercept_v2, flight_v2, feasible_v2 = predict_intercept_batch(
                bp[:, 0], bv[:, 0, :3], root[:, 0, :3],
                plane_z=self.wall_contact_height, wall_x=self.wall_front + self.ball_radius,
                wall_y_bounds=old_infinite_bounds,
                wall_z_bounds=old_infinite_bounds,
                restitution=self.wall_restitution,
            )
            intercept_v3, flight_v3, feasible_v3 = predict_intercept_batch(
                bp[:, 0], bv[:, 0, :3], root[:, 0, :3],
                plane_z=self.wall_contact_height, wall_x=self.wall_front + self.ball_radius,
                wall_y_bounds=self.wall_y_bounds,
                wall_z_bounds=self.wall_z_bounds,
                restitution=self.wall_restitution,
            )
            def relative_view(intercept, flight_t, feasible, *, version_two):
                drone_target = torch.where(
                    feasible[:, None], intercept - torch.tensor([0., 0., .18], device=self.device), target
                )
                prefix = torch.cat((
                    drone_target[:, None] - root[..., :3], root[..., 3:],
                    heading_target[:, None] - root[..., 13:16],
                ), -1)
                builder = build_relative_observation_v2 if version_two else build_relative_observation
                return builder(
                    prefix, root, bp, bv[..., :3], intercept[:, None],
                    flight_t, feasible, self.phase, self.wall_target, self.prev_action,
                )

            relative_v2 = relative_view(intercept_v2, flight_v2, feasible_v2, version_two=True)
            relative_v3 = relative_view(intercept_v3, flight_v3, feasible_v3, version_two=False)
            intercept_geometry_valid_v3 = intercept_target_geometry_valid_batch(
                bp[:, 0], bv[:, 0, :3], intercept_v3,
                plane_z=self.wall_contact_height,
            )
            aerowall_intercept_target_retention_v1 = (
                build_aerowall_intercept_target_retention_v1_observation(
                    relative_v3, root[:, 0, :3], intercept_v3,
                    intercept_geometry_valid_v3, self.phase,
                )
            )
            aerowall_goal_v1 = build_aerowall_goal_observation(
                legacy_obs, self.wall_target, self.wall_goal_y_bounds, self.wall_goal_z_bounds,
            )
            agents.update({
                "legacy_observation": legacy_obs,
                "relative_v2_observation": relative_v2,
                "relative_v3_observation": relative_v3,
                "aerowall_intercept_target_retention_v1_observation": (
                    aerowall_intercept_target_retention_v1
                ),
                "aerowall_goal_v1_observation": aerowall_goal_v1,
            })
            if self.wall_observation_version == "relative_v2":
                obs = relative_v2
                intercept, feasible = intercept_v2, feasible_v2
            elif self.wall_observation_version == "relative_v3":
                obs = relative_v3
                intercept, feasible = intercept_v3, feasible_v3
            elif self.wall_observation_version == "aerowall_goal_v1":
                obs = aerowall_goal_v1
                intercept, feasible = intercept_v3, feasible_v3
            else:
                obs = legacy_obs
                intercept, feasible = intercept_v3, feasible_v3
            self.current_potential[:] = intercept_potential(root[:, 0, :3], intercept, feasible)
            self.current_potential_valid[:] = feasible
        self.info["drone_state"][:] = root[..., :13]
        self.info["skill_id"][:] = self.skill_id[:, None].float()
        agents["observation"] = obs
        return TensorDict({"agents": agents, "stats": self.stats, "info": self.info}, self.num_envs)

    def _return_target(self, ball_pos, ball_vel):
        if self.wall_observation_version in ("relative_v3", "aerowall_goal_v1"):
            intercept, _, feasible = predict_intercept_batch(
                ball_pos, ball_vel, self.root_state[:, 0, :3],
                plane_z=self.wall_contact_height, wall_x=self.wall_front + self.ball_radius,
                restitution=self.wall_restitution,
                wall_y_bounds=self.wall_y_bounds,
                wall_z_bounds=self.wall_z_bounds,
            )
            target = torch.cat((
                intercept[:, :2], torch.full_like(intercept[:, 2:3], 2.0),
            ), -1)
            safe_home = torch.tensor([1.5, 0.0, 2.0], device=self.device).expand_as(target)
            return torch.where(feasible[:, None], target, safe_home)

        # Historical legacy/relative_v2 target for exact checkpoint replay.
        # Descending crossing of z=2.18m, with a racket base at z=2m.
        discriminant = (ball_vel[:, 2].square() + 19.62 * (ball_pos[:, 2] - 2.18)).clamp_min(0.)
        time = ((ball_vel[:, 2] + discriminant.sqrt()) / 9.81).clamp(0., 2.)
        xy = ball_pos[:, :2] + ball_vel[:, :2] * time[:, None]
        vx = ball_vel[:, 0]
        plane = self.wall_front + self.ball_radius
        wall_time = (plane - ball_pos[:, 0]) / torch.where(vx.abs() > .05, vx, torch.full_like(vx, -.05))
        reflected_x = plane - 0.8 * vx * (time - wall_time)
        xy[:, 0] = torch.where((vx < -.05) & (wall_time >= 0.) & (wall_time < time), reflected_x, xy[:, 0])
        return torch.cat([xy, torch.full_like(time[:, None], 2.)], dim=-1)

    def _compute_reward_and_done(self):
        bp = self.ball_pos[:, 0]; dp = self.root_state[:, 0, :3]
        distance = (bp - dp).norm(dim=-1)
        # HCSP-style pose shaping makes the first skill learnable before the
        # sparse cap event: remain upright and stationary under the drop point.
        fixed_target = torch.tensor([1.5, 0.0, 2.0], device=self.device).expand(self.num_envs, 3)
        tracking_target = torch.stack([bp[:, 0], bp[:, 1], torch.full_like(bp[:, 2], 2.0)], -1)
        hover_target = fixed_target if self.curriculum_stage == "A0" else torch.where((self.phase == 0)[:, None], tracking_target, fixed_target)
        if self.curriculum_stage in ('RETURN', 'RALLY'):
            hover_target = torch.where((self.caps > 0)[:, None], self._return_target(bp, self.ball_vel[:, 0, :3]), hover_target)
        heading_target = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
        pose_distance = torch.cat([dp - hover_target, self.root_state[:, 0, 13:16] - heading_target], -1).norm(dim=-1)
        reward_pose = 1.0 / (1.0 + (1.2 * pose_distance).square())
        reward_up = ((self.root_state[:, 0, 18] + 1.0) / 2.0).square()
        reward_spin = 1.0 / (1.0 + self.root_state[:, 0, 12].square().square())
        reward_hover = 3.0 * (reward_pose + reward_pose * (reward_up + reward_spin))
        horizontal_error = (dp[:, :2] - hover_target[:, :2]).norm(dim=-1)
        vertical_error = (dp[:, 2] - hover_target[:, 2]).abs()
        precision = torch.exp(-horizontal_error.square() / 0.0225 - vertical_error.square() / 0.0225)
        dense = reward_hover + 2.0 * precision + 0.01 / (1.0 + 4.0 * distance.square())
        center = torch.where(self.step_cap, 2.0 * (1.0 - self.step_center_error / self.racket_radius).clamp(0, 1), torch.zeros_like(dense))
        reward = dense + self.step_cap.float() + 2.0 * self.step_wall.float() + 10.0 * self.step_rally.float() + center + 2.0 * self.step_launch_quality
        reward -= 10.0 * self.step_illegal.float()
        if self.curriculum_stage == 'WALL':
            # Short launch skill: tiny hover shaping, event-dominated return.
            # A wall hit ends the skill; full rally stages keep running.
            wall_height_quality = torch.exp(-((bp[:, 2] - 4.) / 1.5).square())
            reward = 0.02 * reward_hover + 2. * self.step_cap.float() + 20. * self.step_launch_quality
            reward += self.step_wall.float() * (20. + 10. * wall_height_quality)
            reward -= 10. * self.step_illegal.float()
        elif self.curriculum_stage == 'INTERCEPT':
            # A single contact must outweigh the whole approach. Repeating
            # hover reward at 50 Hz would dominate this one-shot skill event.
            reward = 0.5 * (0.995 * self.current_potential - self.potential_before)
            reward += 20.0 * self.step_cap.float() - 10.0 * self.step_illegal.float()
            reward -= 0.002 * (self.prev_action[:, 0] - self.action_before[:, 0]).square().sum(-1)
        elif self.curriculum_stage == 'RETURN':
            reward = (self.caps > 0).float() * (0.1 * reward_hover + precision)
            reward += 50. * self.step_rally.float() - 10. * self.step_illegal.float()
        elif self.curriculum_stage == 'RALLY':
            reward = (self.caps > 0).float() * (0.02 * reward_hover + 0.2 * precision)
            reward += 30. * self.step_rally.float() + self.step_wall_reward
            reward += 20. * self.step_launch_quality - 10. * self.step_illegal.float()
            reward += 30. * self.step_launch_progress
            reward += self.step_recenter_bonus
        ball_ground = (bp[:, 2] <= self.ball_radius) | self.step_ground
        drone_ground = dp[:, 2] < 0.3
        out = (bp[:, 0] < -0.2) | (bp[:, 0] > 6.0) | (bp[:, 1].abs() > 3.0) | (bp[:, 2] > 8.0)
        drone_wall = dp[:, 0] < 0.5
        if self.curriculum_stage == 'INTERCEPT':
            reward -= 20.0 * (drone_ground | drone_wall).float()
            reward -= 10.0 * (ball_ground | out).float()
        if self.curriculum_stage == "RALLY" and self.wall_reward_design in ("causal_v1", "causal_v2"):
            reward = compute_skill_reward(
                potential_before=self.potential_before,
                potential_after=self.current_potential,
                approach_mask=(self.executed_skill_id != int(Skill.HIT)),
                legal_contact=self.step_cap,
                next_contact=self.step_rally,
                legal_wall=self.step_wall,
                outbound_quality=self.step_outbound_quality,
                wall_quality=self.step_wall_quality if self.wall_reward_design == "causal_v2" else None,
                illegal=self.step_illegal,
                crash=drone_ground | drone_wall,
                out=out | ball_ground,
                action=self.prev_action[:, 0],
                previous_action=self.action_before[:, 0],
            )
        elif self.curriculum_stage == "RALLY" and self.wall_reward_design in ("aerowall_causal_v3", "aerowall_causal_v4", "aerowall_causal_v5", "aerowall_causal_v6"):
            truncated_for_reward = self.progress_buf >= self.max_episode_length
            terminal_for_reward = (ball_ground if self.wall_end_on_ball_drop else
                                   ball_ground | drone_ground | out | drone_wall | self.step_illegal)
            terminal_for_reward |= truncated_for_reward
            reward_fn = {
                "aerowall_causal_v3": compute_aerowall_causal_v3_reward_terms,
                "aerowall_causal_v4": compute_aerowall_causal_v4_reward_terms,
                "aerowall_causal_v5": compute_aerowall_causal_v5_reward_terms,
                "aerowall_causal_v6": compute_aerowall_causal_v6_reward_terms,
            }[self.wall_reward_design]
            reward_kwargs = dict(
                potential_before=self.potential_before,
                potential_after=self.current_potential,
                potential_before_valid=self.potential_before_valid,
                potential_after_valid=self.current_potential_valid,
                approach_mask=(self.executed_skill_id != int(Skill.HIT)),
                terminal=terminal_for_reward,
                legal_contact=self.step_cap,
                next_contact=self.step_rally,
                legal_wall=self.step_wall,
                outbound_quality=self.step_outbound_quality,
                wall_quality=self.step_wall_quality,
                illegal=self.step_illegal,
                crash=drone_ground | drone_wall,
                out=out | ball_ground,
                action=self.prev_action[:, 0],
                previous_action=self.action_before[:, 0],
            )
            if self.wall_reward_design in ("aerowall_causal_v4", "aerowall_causal_v5", "aerowall_causal_v6"):
                reward_kwargs["phase0_illegal"] = self.step_phase0_illegal
            if self.wall_reward_design in ("aerowall_causal_v5", "aerowall_causal_v6"):
                reward_kwargs["phase2_illegal"] = self.step_phase2_illegal
            terms = reward_fn(**reward_kwargs)
            reward = terms["total"]
            for name, value in terms.items():
                if name != "total":
                    self.stats[f"reward_{name}"] += value[:, None]
        if self.curriculum_stage == "A0":
            terminal = drone_ground | drone_wall
        elif self.curriculum_stage in ("A", "A1", "A2"):
            terminal = ball_ground | drone_ground | out | drone_wall | self.step_illegal | self.step_cap
        elif self.curriculum_stage == 'INTERCEPT':
            terminal = ball_ground | drone_ground | out | drone_wall | self.step_illegal | self.step_cap
        elif self.curriculum_stage == 'WALL':
            terminal = ball_ground | drone_ground | out | drone_wall | self.step_illegal | self.step_wall
        elif self.curriculum_stage == 'RETURN':
            terminal = ball_ground | drone_ground | out | drone_wall | self.step_illegal | self.step_rally
        elif self.wall_end_on_ball_drop and self.curriculum_stage == "RALLY":
            terminal = ball_ground
        else:
            terminal = ball_ground | drone_ground | out | drone_wall | self.step_illegal
        if self.curriculum_stage != "A0":
            self.failure = torch.where(ball_ground, torch.ones_like(self.failure), self.failure)
            self.failure = torch.where(out, torch.full_like(self.failure, 3), self.failure)
            self.failure = torch.where(self.step_illegal, torch.full_like(self.failure, 4), self.failure)
        if not self.wall_end_on_ball_drop:
            self.failure = torch.where(drone_ground, torch.full_like(self.failure, 2), self.failure)
            self.failure = torch.where(drone_wall, torch.full_like(self.failure, 5), self.failure)
        reasons = (ball_ground.long()
                   | (drone_ground.long() << 1)
                   | (out.long() << 2)
                   | (self.step_illegal.long() << 3)
                   | (drone_wall.long() << 4))
        self.failure_reason_bits |= reasons
        truncated = self.progress_buf >= self.max_episode_length
        done = terminal | truncated
        self.stats["return"] += reward[:, None]
        self.stats["episode_len"] = self.progress_buf[:, None]
        for key, value in [("rallies", self.rallies), ("max_streak", self.max_streak),
                           ("caps", self.caps), ("walls", self.walls),
                           ("failure", self.failure),
                           ("failure_reason_bits", self.failure_reason_bits)]:
            self.stats[key] = value[:, None].float()
        return TensorDict({
            "agents": {"reward": reward[:, None, None]},
            "done": done[:, None], "terminated": terminal[:, None], "truncated": truncated[:, None],
        }, self.num_envs)
