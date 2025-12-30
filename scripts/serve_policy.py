import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""
    PI0_BASE = "pi0_base"
    PI0_FAST_BASE = "pi0_fast_base"
    PI0_BASE_DROID = "pi0_base_droid"
    PI0_FAST_DROID = "pi0_fast_droid"
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    XARM = "xarm"
    XARM_R2_FULL = "xarm_r2_full"
    # ------------------------
    # AGRIVLA ROUND 3
    # ------------------------
    AGRIVLA_ALL = "pi0_lora_xarm6_agrivla_pi0_all"
    AGRIVLA_TOMATOES_ONLY = "pi0_lora_xarm6_agrivla_tomatoes_only"
    AGRIVLA_TOMATOES_PLUS_10 = "pi0_lora_xarm6_agrivla_tomatoes_plus_10"
    AGRIVLA_TOMATOES_PLUS_20 = "pi0_lora_xarm6_agrivla_tomatoes_plus_20"
    AGRIVLA_TOMATOES_PLUS_50 = "pi0_lora_xarm6_agrivla_tomatoes_plus_50"
    AGRIVLA_TOMATOES_PLUS_100 = "pi0_lora_xarm6_agrivla_tomatoes_plus_100"
    AGRIVLA_TOMATOES_PLUS_200 = "pi0_lora_xarm6_agrivla_tomatoes_plus_200"
    AGRIVLA_CHILLIS_ONLY = "pi0_lora_xarm6_agrivla_chillis_only"

    DEMO = "demo"
    DEMO_SERVER = "demo_server"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.PI0_BASE: Checkpoint(
        config="pi0_base",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.PI0_FAST_BASE: Checkpoint(
        config="pi0_fast_base",
        dir="s3://openpi-assets/checkpoints/pi0_fast_base",
    ),
    EnvMode.PI0_BASE_DROID: Checkpoint(
        config="pi0_base",
        dir="s3://openpi-assets/checkpoints/pi0_droid",
    ),
    EnvMode.PI0_FAST_DROID: Checkpoint(
        config="pi0_fast_base",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
    EnvMode.XARM: Checkpoint(
        config="pi0_xarm6_low_mem_finetune",
        dir="checkpoints/pi0_xarm6_low_mem_finetune/pi0_xarm6_lora_pickTomatoes_noFrozenFrames/29999",
    ),
    EnvMode.XARM_R2_FULL: Checkpoint(
        config="pi0base_lora_xarm6_round2_fulldataset",
        dir="checkpoints/pi0base_lora_xarm6_round2_fulldataset/pi0base_lora_xarm6_round2_fulldataset/29999",
    ),

    # ------------------------
    # AGRIVLA ROUND 3
    # ------------------------
    EnvMode.AGRIVLA_ALL: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_all",
        dir="checkpoints/pi0_lora_xarm6_agrivla_pi0_all/pi0_lora_xarm6_agrivla_pi0_all_20251012_1322/29999",
    ),
    EnvMode.AGRIVLA_TOMATOES_ONLY: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_tomatoes_only",
        dir="checkpoints/pi0_lora_xarm6_agrivla_pi0_tomatoes_only/pi0_lora_xarm6_agrivla_pi0_tomatoes_only_20251012_1322/29999",
    ),
    EnvMode.AGRIVLA_TOMATOES_PLUS_10: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_10",
        dir="checkpoints/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_10/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_10_20251012_1322/29999",
    ),
    EnvMode.AGRIVLA_TOMATOES_PLUS_20: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_20",
        dir="checkpoints/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_20/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_20_20251012_1322/29999",
    ),
    EnvMode.AGRIVLA_TOMATOES_PLUS_50: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_50",
        dir="checkpoints/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_50/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_50_20251012_1322/29999",
    ),
    EnvMode.AGRIVLA_TOMATOES_PLUS_100: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_100",
        dir="checkpoints/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_100/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_100_20251012_1322/29999",
    ),
    EnvMode.AGRIVLA_TOMATOES_PLUS_200: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_200",
        dir="checkpoints/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_200/pi0_lora_xarm6_agrivla_pi0_tomatoes_plus_200_20251012_1322/29999",
    ),
    EnvMode.AGRIVLA_CHILLIS_ONLY: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_chillis_only",
        dir="checkpoints/pi0_lora_xarm6_agrivla_pi0_chillis_only/pi0_lora_xarm6_agrivla_pi0_chillis_only_20251012_1322/29999",
    ),

    EnvMode.DEMO: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_all",
        dir="/media/acrv/DanielsSSD/Thesis/pi0/checkpoints/29999/pi0_lora_xarm6_agrivla_pi0_all/29999",
    ),
    EnvMode.DEMO_SERVER: Checkpoint(
        config="pi0_lora_xarm6_agrivla_pi0_all",
        dir="checkpoints/AgriVLA/29999",
    ),
}

def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
