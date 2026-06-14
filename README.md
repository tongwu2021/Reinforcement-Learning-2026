# BipedalWalkerHardcore-v3 SAC

Gymnasium의 `BipedalWalkerHardcore-v3` 환경을 SAC(Soft Actor-Critic)로 학습한 강화학습 과제 코드입니다.

## 환경 (Environment)
- **Env**: `BipedalWalkerHardcore-v3` (Box2D, continuous control)
- **Observation**: 24차원 연속 벡터
- **Action**: 4차원 연속 벡터 (각 관절 토크), 범위 `[-1, 1]`
- **Solved 기준**: validation 평균 리워드 300 이상

## 알고리즘 (SAC)
- Gaussian Policy + 2개의 Soft Q-Network (Double Q-learning)
- Target Q-Network, Soft Update (`tau=0.995`)
- 자동 alpha(entropy coefficient) 튜닝
- Replay Buffer (최대 1,000,000 transition)
- 장시간 학습 도중 끊겨도 이어서 학습할 수 있도록 체크포인트 자동 저장/복구 (백업 슬롯 포함)

## 파일 구조
| 파일 | 설명 |
|---|---|
| `a_sac_models.py` | `GaussianPolicy`, `SoftQNetwork`, `ReplayBuffer` 등 모델/유틸 정의 |
| `b_sac_train.py` | SAC 학습 루프 (체크포인트 저장/복구, wandb 로깅) |
| `c_sac_test.py` | 학습된 모델/체크포인트로 정책 평가 (렌더링 가능) |
| `d_sac_recover_checkpoint.py` | 정책 붕괴 시 이전 정상 가중치로 복구하는 1회용 스크립트 |
| `_wait_for_steps.py` | wandb API로 학습 진행 step 수를 polling하는 보조 스크립트 |
| `colab_train.ipynb` | Google Colab에서 학습을 실행하기 위한 노트북 |
| `colab_test.ipynb` | Google Colab에서 학습된 모델을 평가하기 위한 노트북 |
| `models/sac_BipedalWalkerHardcore-v3_latest.pth` | 최신 학습된 정책(policy) 가중치 |

## 환경 설치

```bash
conda create -n walker python=3.13 -y
conda activate walker

# Box2D 빌드에 필요한 swig
conda install -c anaconda swig -y   # Windows/Linux
# brew install swig                 # Mac

pip install "gymnasium[box2d]" torch numpy wandb
```

## 실행 방법

### 로컬에서 학습
```bash
python b_sac_train.py
```
- 학습 중간에 끊겨도 `checkpoints/`에 저장된 체크포인트로 자동 이어서 학습합니다.
- wandb 로그인이 필요합니다 (`wandb login`).

### 로컬에서 테스트 (학습된 모델을 화면에 렌더링, 3 episodes)
```bash
python c_sac_test.py
```

### Google Colab에서 실행
- `colab_train.ipynb`: Google Drive에 `sac/` 폴더를 업로드한 뒤 학습 실행 (체크포인트는 Drive에 저장되어 세션이 끊겨도 이어서 학습 가능)
- `colab_test.ipynb`: 학습된 모델/체크포인트를 불러와 평가 (학습용 노트북과 별도 런타임에서 동시 실행 가능)

## 결과
학습 약 7,000K time steps 시점에 validation 평균 리워드 **302.6**을 달성하여 Solved 기준(300)을 넘었습니다. 이후 추가 학습은 정책 안정화를 위해 계속 진행되었으며, `models/sac_BipedalWalkerHardcore-v3_latest.pth`에 최신 정책 가중치가 저장되어 있습니다.
