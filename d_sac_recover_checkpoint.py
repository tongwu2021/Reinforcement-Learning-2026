# 1회용 복구 스크립트
#
# 정책이 학습 도중 망가졌을 때(예: Episode 7,600 부근에서 정책 붕괴),
# 체크포인트의 policy 가중치만 이전에 "Solved"로 저장된 정책 파일로 교체하고,
# (붕괴 구간의 그래디언트 모멘텀이 남아있는) policy optimizer는 초기화한다.
#
# Critic(q_network/target_q_network/optimizer), alpha, replay buffer, n_episode,
# time_steps 등은 기존 체크포인트 값을 그대로 유지한다.
#
# 사용법: 이 파일을 sac 폴더(Drive의 a_sac_models.py 등과 같은 위치)에 올려두고
#         한 번만 실행한 뒤, 평소처럼 b_sac_train.main()을 실행해 이어서 학습한다.
#         (다시 실행하면 이후 진행된 학습이 다시 이 정책으로 덮어써지므로 주의)
import os
import shutil

import gymnasium as gym
import torch
import torch.optim as optim

from a_sac_models import MODEL_DIR, CHECKPOINT_DIR, GaussianPolicy, DEVICE


def main() -> None:
    ENV_NAME = "BipedalWalkerHardcore-v3"
    LEARNING_RATE = 0.0003

    # 정책 붕괴 이전, "Solved" 시점에 저장된 정책 가중치 파일
    WARM_START_POLICY_FILE = "sac_BipedalWalkerHardcore-v3_302.6_2026-06-09_15-40-02.pth"

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "sac_{0}_checkpoint.pth".format(ENV_NAME))
    warm_start_policy_path = os.path.join(MODEL_DIR, WARM_START_POLICY_FILE)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다: {0}".format(checkpoint_path))
    if not os.path.exists(warm_start_policy_path):
        raise FileNotFoundError("정책 파일을 찾을 수 없습니다: {0}".format(warm_start_policy_path))

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    print("[Before] Episode: {0:,}, Time Steps: {1:,}, Alpha: {2:.4f}, Replay Buffer: {3:,}".format(
        checkpoint["n_episode"], checkpoint["time_steps"], checkpoint["alpha"], len(checkpoint["replay_buffer"])
    ))

    # 1) 정책 가중치를 "Solved" 시점 가중치로 교체
    env = gym.make(ENV_NAME)
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
    policy.load_state_dict(torch.load(warm_start_policy_path, map_location=DEVICE, weights_only=True))
    env.close()

    checkpoint["policy_state_dict"] = policy.state_dict()

    # 2) 붕괴 구간의 그래디언트 모멘텀이 남아있는 policy optimizer는 초기화
    fresh_policy_optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    checkpoint["policy_optimizer_state_dict"] = fresh_policy_optimizer.state_dict()

    # 원본은 만일을 대비해 별도 파일로 백업
    backup_path = os.path.join(CHECKPOINT_DIR, "sac_{0}_checkpoint_before_recovery.pth".format(ENV_NAME))
    shutil.copy2(checkpoint_path, backup_path)
    print("[Backup] Original checkpoint saved to {0}".format(backup_path))

    tmp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)

    print("[After] Policy <- {0} (optimizer reset). Critic / Alpha / Replay Buffer 유지됨.".format(WARM_START_POLICY_FILE))
    print("[Done] {0} 에 패치된 체크포인트를 저장했습니다. 이제 학습을 다시 시작하면 이어서 진행됩니다.".format(checkpoint_path))


if __name__ == "__main__":
    main()
