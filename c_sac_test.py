# https://gymnasium.farama.org/environments/box2d/bipedal_walker/
import os

import gymnasium as gym
import numpy as np
import torch

from a_sac_models import MODEL_DIR, CHECKPOINT_DIR, GaussianPolicy, DEVICE


def test(env: gym.Env, actor: GaussianPolicy, num_episodes: int) -> list:
    episode_rewards = []

    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = actor.get_action(observation, exploration=False)

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(i, episode_steps, episode_reward))

    return episode_rewards


def print_report(episode_rewards: list, num_episodes: int) -> None:
    rewards = np.array(episode_rewards)
    success = rewards >= 300
    fail = rewards < 0

    print()
    print("=" * 60)
    print("Episodes: {0}".format(num_episodes))
    print("Average Reward: {0:7.2f} (+/- {1:.2f})".format(rewards.mean(), rewards.std()))
    print("Min / Max:      {0:7.2f} / {1:.2f}".format(rewards.min(), rewards.max()))
    print("Success (>= 300): {0:3d} / {1} ({2:.1f}%)".format(success.sum(), num_episodes, 100 * success.mean()))
    print("Fail (< 0):       {0:3d} / {1} ({2:.1f}%)".format(fail.sum(), num_episodes, 100 * fail.mean()))
    print("=" * 60)


def main_play_model(num_episodes: int, env_name: str, model_filename: str, render_mode: str = None) -> None:
    # models/ 폴더의 policy 가중치 파일(latest.pth 또는 고득점 스냅샷 sac_{env}_{avg}_{time}.pth)로 테스트
    env = gym.make(env_name, render_mode=render_mode)

    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
    model_params = torch.load(os.path.join(MODEL_DIR, model_filename), map_location=DEVICE, weights_only=True)
    policy.load_state_dict(model_params)
    policy.eval()

    print("[Model] {0}".format(model_filename))

    episode_rewards = test(env, policy, num_episodes=num_episodes)

    env.close()

    print_report(episode_rewards, num_episodes)


def main_play(num_episodes: int, env_name: str, render_mode: str = None) -> None:
    main_play_model(num_episodes, env_name, "sac_{0}_latest.pth".format(env_name), render_mode=render_mode)


def main_play_checkpoint(num_episodes: int, env_name: str, checkpoint_filename: str) -> None:
    # checkpoints/ 폴더의 체크포인트(현재: sac_{env}_checkpoint.pth, 백업: sac_{env}_checkpoint_backup_0/1/2.pth)
    # 안의 policy_state_dict로 정책을 복원해 테스트
    env = gym.make(env_name, render_mode=None)

    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, checkpoint_filename), map_location=DEVICE, weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    print("[Checkpoint] {0} - Episode: {1:,}, Time Steps: {2:,}, Alpha: {3:.4f}".format(
        checkpoint_filename, checkpoint["n_episode"], checkpoint["time_steps"], checkpoint["alpha"]
    ))

    episode_rewards = test(env, policy, num_episodes=num_episodes)

    env.close()

    print_report(episode_rewards, num_episodes)


if __name__ == "__main__":
    NUM_EPISODES = 3
    ENV_NAME = "BipedalWalkerHardcore-v3"

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME, render_mode="human")
