# https://gymnasium.farama.org/environments/box2d/bipedal_walker/
import os
import shutil
import time
from datetime import datetime
import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a_sac_models import MODEL_DIR, CHECKPOINT_DIR, GaussianPolicy, SoftQNetwork, ReplayBuffer, Transition, DEVICE

import wandb


class SAC:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="sac_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.validation_time_steps_interval = config["validation_time_steps_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.high_score_save_threshold = config["high_score_save_threshold"]
        self.steps_between_train = config["steps_between_train"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]
        self.alpha_automatic_tuning = config["alpha_automatic_tuning"]
        self.alpha = config["alpha"]

        n_features = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        self.policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.q_network = SoftQNetwork(n_features=n_features, n_actions=n_actions)
        self.target_q_network = SoftQNetwork(n_features=n_features, n_actions=n_actions)

        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        if self.alpha_automatic_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(DEVICE)).item()
            print("TARGET ENTROPY: {0}".format(self.target_entropy))
            self.log_alpha = torch.tensor(math.log(self.alpha), requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
            self.alpha = self.log_alpha.exp().item()

        self.time_steps = 0
        self.training_time_steps = 0

        self.max_alpha = 5.0

        self.total_train_start_time = None

        # Checkpoint: 장시간(수 시간~수일) 학습 도중 끊겨도 이어서 학습할 수 있도록
        # 모델/옵티마이저/리플레이 버퍼/진행 상태를 주기적으로 저장하고, 시작 시 있으면 자동으로 복구
        self.checkpoint_path = os.path.join(CHECKPOINT_DIR, "sac_{0}_checkpoint.pth".format(self.env_name))
        self.start_episode = 1
        self.n_episode = 1

        # Checkpoint Backup: 정책이 학습 도중 망가지더라도 직전 정상 시점으로 되돌릴 수 있도록,
        # 체크포인트 저장 N회마다 별도 슬롯에 순환 백업본을 보관
        self.checkpoint_backup_interval = config["checkpoint_backup_interval"]
        self.checkpoint_backup_slots = config["checkpoint_backup_slots"]
        self.checkpoint_save_count = 0

        self.load_checkpoint()

        # 체크포인트의 optimizer state_dict에는 이전 learning_rate가 그대로 저장되어 있으므로,
        # 재개 시점에 config의 learning_rate로 강제 동기화 (lr을 바꿔서 이어서 학습하고 싶을 때 사용)
        for param_group in self.policy_optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        for param_group in self.q_network_optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        if self.alpha_automatic_tuning:
            for param_group in self.alpha_optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()

        validation_episode_reward_avg = -1500
        policy_loss = q_1_td_loss = q_2_td_loss = alpha_loss = mu = entropy = 0.0

        is_terminated = False

        for n_episode in range(self.start_episode, self.max_num_episodes + 1):
            self.n_episode = n_episode
            episode_reward = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                if self.time_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    policy_loss, q_1_td_loss, q_2_td_loss, alpha_loss, mu, entropy = self.train()

                if self.time_steps % self.validation_time_steps_interval == 0:
                    validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                    self.save_checkpoint()
                    self.save_latest_model()

                    # 발표/테스트용 후보를 놓치지 않도록, 검증 평균이 일정 기준 이상이면
                    # 해당 시점의 정책 가중치를 별도 파일로 저장 (체크포인트와 달리 덮어써지지 않음)
                    if validation_episode_reward_avg >= self.high_score_save_threshold:
                        self.model_save(validation_episode_reward_avg)

                        if validation_episode_reward_avg > self.episode_reward_avg_solved:
                            print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                            is_terminated = True

                    if self.use_wandb:
                        self.log_wandb(
                            validation_episode_reward_avg,
                            episode_reward,
                            policy_loss,
                            q_1_td_loss, q_2_td_loss,
                            alpha_loss,
                            mu,
                            entropy,
                            n_episode,
                        )

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Epi. {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Epi. Reward: {:>9.3f},".format(episode_reward),
                    "Policy L.: {:>7.3f},".format(policy_loss),
                    "Critic L.: {:>7.3f}, {:>7.3f}".format(q_1_td_loss, q_2_td_loss),
                    "Alpha L.: {:>7.3f},".format(alpha_loss),
                    "Alpha: {:>7.3f},".format(self.alpha),
                    "Entropy: {:>7.3f},".format(entropy),
                    "Train Steps: {:5,}".format(self.training_time_steps),
                )

            if is_terminated:
                if self.wandb:
                    for _ in range(5):
                        self.log_wandb(
                            validation_episode_reward_avg,
                            episode_reward,
                            policy_loss,
                            q_1_td_loss, q_2_td_loss,
                            alpha_loss,
                            mu,
                            entropy,
                            n_episode,
                        )
                break

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(
        self,
        validation_episode_reward_avg: float,
        episode_reward: float,
        policy_loss: float,
        q_1_td_loss: float, q_2_td_loss: float,
        alpha_loss: float,
        mu: float,
        entropy: float,
        n_episode: float,
    ) -> None:
        self.wandb.log(
            {
                "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                    self.validation_num_episodes
                ): validation_episode_reward_avg,
                "[TRAIN] episode reward": episode_reward,
                "[TRAIN] policy loss": policy_loss,
                "[TRAIN] critic 1 loss": q_1_td_loss,
                "[TRAIN] critic 2 loss": q_2_td_loss,
                "[TRAIN] alpha loss": alpha_loss,
                "[TRAIN] alpha": self.alpha,
                "[TRAIN] mu": mu,
                "[TRAIN] entropy": entropy,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "training episode": n_episode,
                "training steps": self.training_time_steps,
            }
        )

    def train(self):
        self.training_time_steps += 1

        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        ####################
        # Q NETWORK UPDATE #
        ####################
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_observations)
            qf1_next_target, qf2_next_target = self.target_q_network(next_observations, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target[dones] = 0.0
            target_values = rewards + self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.q_network(observations, actions)
        qf1_loss = F.mse_loss(qf1, target_values)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, target_values)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.q_network_optimizer.zero_grad()
        qf_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 3.0)
        self.q_network_optimizer.step()

        #################
        # Policy UPDATE #
        #################
        sample_actions, log_pi, mu, entropy = self.policy.sample(observations, reparameterization_trick=True)

        qf1_pi, qf2_pi = self.q_network(observations, sample_actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = -1.0 * (min_qf_pi - self.alpha * log_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 3.0)
        self.policy_optimizer.step()

        #################
        # Alpha UPDATE #
        #################
        if self.alpha_automatic_tuning:
            with torch.no_grad():
                _, log_pi, _, _ = self.policy.sample(observations)

            alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.log_alpha], 3.0)
            self.alpha_optimizer.step()

            # log_alpha를 기반으로 알파 값 계산 (상한선을 설정)
            with torch.no_grad():
                self.alpha = self.log_alpha.exp().item()

                # 클램핑 기법을 사용해 알파 값이 상한선을 넘지 않도록 제한
                if self.alpha > self.max_alpha:
                    self.log_alpha.data = torch.log(torch.tensor(self.max_alpha))
        else:
            alpha_loss = torch.tensor(0.).to(DEVICE)

        # sync, TAU: 0.995
        self.soft_synchronize_models(
            source_model=self.q_network, target_model=self.target_q_network, tau=self.soft_update_tau
        )

        return policy_loss.item(), qf1_loss.item(), qf2_loss.item(), alpha_loss.item(), mu.mean().item(), entropy.item()

    def soft_synchronize_models(self, source_model, target_model, tau):
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = tau * target_model_state[k] + (1.0 - tau) * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "sac_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.policy.state_dict(), os.path.join(MODEL_DIR, filename))
        print("[Model Saved] {0} (Validation Average: {1:.3f})".format(filename, validation_episode_reward_avg))

    def save_latest_model(self) -> None:
        torch.save(self.policy.state_dict(), os.path.join(MODEL_DIR, "sac_{0}_latest.pth".format(self.env_name)))

    def save_checkpoint(self) -> None:
        checkpoint = {
            "n_episode": self.n_episode,
            "time_steps": self.time_steps,
            "training_time_steps": self.training_time_steps,
            "alpha": self.alpha,
            "checkpoint_save_count": self.checkpoint_save_count + 1,
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "q_network_state_dict": self.q_network.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "q_network_optimizer_state_dict": self.q_network_optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer.buffer),
        }
        if self.alpha_automatic_tuning:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        # 임시 파일에 먼저 쓰고 교체 -> 저장 도중 프로세스가 끊겨도 체크포인트 파일이 손상되지 않도록 함
        tmp_path = self.checkpoint_path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, self.checkpoint_path)

        self.checkpoint_save_count += 1

        print("[Checkpoint Saved] Episode: {0:,}, Time Steps: {1:,}, Replay Buffer: {2:,}".format(
            self.n_episode, self.time_steps, self.replay_buffer.size()
        ))

        # N회마다 별도 슬롯에 순환 백업 -> 학습이 중간에 망가져도 직전 정상 시점으로 되돌릴 수 있음
        if self.checkpoint_save_count % self.checkpoint_backup_interval == 0:
            slot = (self.checkpoint_save_count // self.checkpoint_backup_interval - 1) % self.checkpoint_backup_slots
            backup_path = os.path.join(
                CHECKPOINT_DIR, "sac_{0}_checkpoint_backup_{1}.pth".format(self.env_name, slot)
            )
            shutil.copy2(self.checkpoint_path, backup_path)
            print("[Checkpoint Backup] Slot {0} <- Episode: {1:,}, Time Steps: {2:,} ({3})".format(
                slot, self.n_episode, self.time_steps, backup_path
            ))

    def load_checkpoint(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            return

        checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)

        self.start_episode = checkpoint["n_episode"] + 1
        self.n_episode = checkpoint["n_episode"]
        self.time_steps = checkpoint["time_steps"]
        self.training_time_steps = checkpoint["training_time_steps"]
        self.alpha = checkpoint["alpha"]

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.q_network_optimizer.load_state_dict(checkpoint["q_network_optimizer_state_dict"])

        if self.alpha_automatic_tuning and "log_alpha" in checkpoint:
            self.log_alpha = checkpoint["log_alpha"].clone().to(DEVICE).requires_grad_(True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])

        for transition in checkpoint["replay_buffer"]:
            self.replay_buffer.append(transition)

        self.checkpoint_save_count = checkpoint.get("checkpoint_save_count", 0)

        print("[Checkpoint Loaded] Resuming from Episode {0:,}, Time Steps {1:,}, Replay Buffer {2:,}".format(
            self.start_episode, self.time_steps, self.replay_buffer.size()
        ))

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.test_env.reset()

            done = False

            while not done:
                action = self.policy.get_action(observation, exploration=False)

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

            if self.use_wandb:
                self.wandb.log({"[VALIDATION] Episode Reward": episode_reward})

        episode_reward_avg = np.average(episode_reward_lst)

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

        print(
            "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                episode_reward_lst, episode_reward_avg, total_training_time
            )
        )
        return episode_reward_lst, episode_reward_avg


def main() -> None:
    print("TORCH VERSION:", torch.__version__)

    ENV_NAME = "BipedalWalkerHardcore-v3"

    # env
    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    config = {
        "env_name": ENV_NAME,                               # 환경의 이름
        "max_num_episodes": 100_000,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "steps_between_train": 4,                           # 훈련 사이의 환경 스텝 수 (Colab GPU 여유를 활용해 update-to-data 비율 ↑)
        "replay_buffer_size": 1_000_000,                    # 리플레이 버퍼 사이즈
        # 7,000K~11,150K 구간에서 같은 plateau(약 230~250)를 oscillation하며 유지 -> 업데이트 폭을 줄여
        # 안정성 위주로 fine-tuning 하기 위해 낮춤 (재개 시 옵티마이저 lr도 이 값으로 강제 동기화됨)
        "learning_rate": 0.0001,                            # 학습율
        "gamma": 0.99,                                      # 감가율
        "soft_update_tau": 0.995,                           # Soft Update Tau
        "print_episode_interval": 10,                       # Episode 통계 출력에 관한 에피소드 간격
        "validation_time_steps_interval": 50_000,           # 검증 사이 마다 각 훈련 time steps 간격
        "validation_num_episodes": 3,                       # 검증에 수행하는 에피소드 횟수
        "checkpoint_backup_interval": 5,                    # 체크포인트 저장 N회(=N*50,000 time steps)마다 백업
        "checkpoint_backup_slots": 3,                       # 백업을 순환 보관할 슬롯 개수 (최근 N*3회 분량 롤백 가능)
        # 300 목표는 7,000K 스텝에서 validation 평균 302.6으로 이미 달성(checkpoint에 저장됨).
        # 추가 학습으로 정책을 더 안정화하기 위해 사실상 도달 불가능한 값으로 올려 자동 종료를 방지
        "episode_reward_avg_solved": 320,                   # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        # 검증 평균이 이 값 이상인 시점마다 정책 가중치를 models/ 에 별도 파일로 저장 (발표/테스트용 후보 확보)
        "high_score_save_threshold": 290,                   # 고득점 정책 스냅샷 저장 기준
        "learning_starts": 10_000,                          # 충분한 경험 데이터 수집 (에피소드 길이가 길어 늘림)
        "alpha_automatic_tuning": True,                     # Alpha Auto Tuning
        "alpha": 0.2                                        # Alpha
    }

    use_wandb = True
    sac = SAC(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    try:
        sac.train_loop()
    except KeyboardInterrupt:
        # Colab 런타임 중단, Ctrl+C 등으로 학습이 갑자기 끊겨도 그 시점까지의 진행 상황을 저장
        print("\n[Interrupted] Saving checkpoint before exit...")
        sac.save_checkpoint()
        raise


if __name__ == "__main__":
    main()
