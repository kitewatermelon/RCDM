# RCDM (Representation Conditional Diffusion Model) 분석 리포트

## 1. 프로젝트 개요

**논문:** [High Fidelity Visualization of What Your Self-Supervised Representation Knows About](https://arxiv.org/abs/2112.09164)  
**저자:** Florian Bordes, Randall Balestriero, Pascal Vincent (Meta AI)  
**기반 레포지토리:** OpenAI [guided-diffusion](https://github.com/openai/guided-diffusion) (Diffusion Models Beat GANs on Image Synthesis)

RCDM의 핵심 아이디어는, 기존 guided-diffusion이 분류기의 **gradient**로 샘플링을 유도하는 것과 달리, **SSL(Self-Supervised Learning) 모델이 추출한 표현 벡터(feature vector)** 를 직접 조건으로 주입하여 이미지를 생성하는 것입니다.

---

## 2. 사용하는 디퓨전 모델

### 2.1 기본 아키텍처: DDPM (Denoising Diffusion Probabilistic Models)

`guided_diffusion_rcdm/gaussian_diffusion.py`에서 확인할 수 있듯이, Ho et al.의 DDPM을 PyTorch로 포팅한 구현을 기반으로 합니다.

**Forward Process (노이즈 추가):**
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t} x_0,\ (1 - \bar\alpha_t)\mathbf{I})$$

- 노이즈 스케줄: **linear** (기본값) 또는 cosine
- 디퓨전 스텝 수: **T = 1000**

**Backward Process (노이즈 제거):**
- 모델이 예측하는 값: `EPSILON` (노이즈 ε) 또는 `START_X` (원본 x₀) 중 선택
- 분산: `LEARNED_RANGE`를 사용해 모델이 고정 상한/하한 사이의 분산을 학습

**샘플링:**
- 기본: `p_sample_loop` (DDPM 역방향 샘플링)
- 가속 옵션: `ddim_sample_loop` (DDIM, 100 스텝 등 단축 가능)
- `SpacedDiffusion`을 통해 timestep respacing 지원 (`--timestep_respacing 100` 등)

### 2.2 디노이저: 조건부 UNet

`guided_diffusion_rcdm/unet.py`의 `UNetModel`이 실제 노이즈 예측기입니다.

**기본 구조:**
```
입력 x_t (노이즈 이미지)
    ↓
Input Blocks (Encoder): ResBlock + AttentionBlock × N levels
    ↓
Middle Block: ResBlock → AttentionBlock → ResBlock
    ↓
Output Blocks (Decoder): ResBlock + AttentionBlock × N levels (skip connections)
    ↓
출력 ε̂ (예측 노이즈, 3 또는 6 채널)
```

**주요 하이퍼파라미터 (128×128 학습 기준):**
| 파라미터 | 값 |
|---|---|
| `image_size` | 128 |
| `num_channels` | 256 |
| `num_res_blocks` | 2 |
| `channel_mult` | (1, 1, 2, 3, 4) |
| `attention_resolutions` | 32, 16, 8 |
| `num_heads` | 4 |
| `learn_sigma` | True |
| `noise_schedule` | linear |
| `diffusion_steps` | 1000 |

---

## 3. 컨디셔닝 메커니즘

RCDM의 컨디셔닝은 두 단계로 이루어집니다: (1) SSL 인코더로 특징 벡터 추출, (2) 해당 벡터를 UNet의 timestep embedding에 더하기.

### 3.1 Step 1: SSL 인코더로 특징 추출

`guided_diffusion_rcdm/get_ssl_models.py` 참조.

모든 SSL 모델은 **ResNet-50 backbone** 기반이며, trunk(2048-dim) 또는 head 출력을 조건으로 사용합니다.

| SSL 모델 | Trunk 차원 | Head 구조 | Head 출력 차원 |
|---|---|---|---|
| **Supervised** | 2048 | 없음 (fc=Identity) | 2048 |
| **DINO** | 2048 | MLP (2048→4096→256→60000), 2 layers | 60000 |
| **SimCLR** | 2048 | MLP (2048→2048→128) + ReLU | 128 |
| **Barlow Twins** | 2048 | MLP (2048→8192→8192→8192) + BN + ReLU | 8192 |
| **VICReg** | 2048 | Projector MLP (2048→8192→8192→8192) + BN + ReLU | 8192 |

`RCDM.py`에서 SSL 모델을 초기화할 때:
```python
self.ssl_model = get_model(args.type_model, args.use_head).cuda().eval()
self.ssl_dim = self.ssl_model(th.zeros(1, 3, 224, 224).cuda()).size(1)
```
— SSL 모델은 freeze되어 있으며 (gradient 없음), 조건 벡터만 제공합니다.

### 3.2 Step 2: 특징 벡터 → timestep embedding에 가산 (FiLM-style)

`unet.py:695–726` (`UNetModel.forward`)의 핵심 컨디셔닝 코드:

```python
def forward(self, x, timesteps, y=None, feat=None):
    # 1. timestep embedding 계산
    emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
    #    timestep t → sinusoidal embedding → 2-layer MLP → time_embed_dim (= 4 × model_channels)

    # 2. 클래스 레이블 컨디셔닝 (선택적)
    if self.num_classes is not None:
        emb = emb + self.label_emb(y)

    # 3. SSL 특징 컨디셔닝 ← 핵심 RCDM 로직
    if feat is not None:
        emb = emb + self.ssl_emb(feat)
    #    ssl_emb: Linear(ssl_dim → time_embed_dim) 1개 레이어

    # 4. 모든 ResBlock에 emb를 전달
    for module in self.input_blocks:
        h = module(h, emb)
    ...
```

**`ssl_emb` 레이어 초기화** (`unet.py:512`):
```python
if instance_cond:  # feat_cond=True일 때
    self.ssl_emb = nn.Linear(self.ssl_dim, time_embed_dim)
```
— `ssl_dim`(예: 2048)을 `time_embed_dim`(= `num_channels × 4` = 1024)으로 선형 투영합니다.

### 3.3 ResBlock에서의 컨디셔닝 적용

`unet.py:258–269` (`ResBlock._forward`):

```python
emb_out = self.emb_layers(emb).type(h.dtype)
# emb_layers: SiLU → Linear(time_embed_dim → 2×out_channels)

if self.use_scale_shift_norm:  # 기본값: True
    scale, shift = th.chunk(emb_out, 2, dim=1)
    h = out_norm(h) * (1 + scale) + shift  # FiLM (Feature-wise Linear Modulation)
else:
    h = h + emb_out
```

**`use_scale_shift_norm=True`(기본값)** 일 때, `emb`(시간 + SSL 특징 합산 임베딩)에서 `scale`과 `shift`를 예측하여 각 ResBlock의 정규화 레이어에 FiLM 방식으로 적용합니다. 이로써 SSL 특징 정보가 UNet 전 계층에 걸쳐 채널별로 스케일·시프트를 통해 주입됩니다.

### 3.4 샘플링 시 컨디셔닝 전달 경로

`RCDM.py:54–63`:
```python
feat = self.ssl_model(batch).detach()    # SSL 인코더로 특징 추출
model_kwargs["feat"] = feat              # model_kwargs에 저장
sample = self.sample_fn(
    self.rcdm_model,
    (num_samples, 3, image_size, image_size),
    clip_denoised=True,
    model_kwargs=model_kwargs,           # 디퓨전 루프 내부로 전달
)
```

`gaussian_diffusion.py`의 `p_mean_variance`에서 `model_kwargs`가 매 timestep마다 모델에 `**model_kwargs`로 전달되어, 역방향 디퓨전 전 과정에서 동일한 SSL 특징이 조건으로 사용됩니다.

---

## 4. 전체 컨디셔닝 흐름 요약

```
입력 이미지
    │
    ▼
[SSL 인코더 (ResNet-50, freeze)]
    │  → feat 벡터 (예: 2048-dim trunk or 128-dim simclr head)
    ▼
[ssl_emb: Linear(ssl_dim → time_embed_dim)]
    │
    ├── + [time_embed: sinusoidal → MLP]  ← timestep t
    │
    ▼
emb (통합 임베딩, time_embed_dim = 4 × model_channels)
    │
    ▼ (매 ResBlock마다)
[emb_layers: SiLU → Linear → scale, shift]
    │
    ▼
FiLM: h = GroupNorm(h) × (1 + scale) + shift
    │
    ▼
예측 노이즈 ε̂ → DDPM/DDIM 역방향 샘플링
    │
    ▼
생성 이미지 (SSL 표현이 담고 있는 시맨틱 정보 반영)
```

---

## 5. 확장 기능

### 5.1 표현 보간 (Interpolation)
`scripts/image_sample_interpolation.py`: 두 이미지의 SSL 표현 벡터를 선형 보간하여 중간 시맨틱의 이미지를 생성합니다.

### 5.2 표현 조작 (Manipulation)
`scripts/image_sample_manipulation.py`: 최근접 이웃(NN) 탐색 기반으로 특정 속성을 추출하여 타깃 이미지의 표현에 더하는 방식으로 속성 편집을 수행합니다.

---

## 6. 기술적 특징 정리

| 항목 | 내용 |
|---|---|
| 디퓨전 모델 유형 | DDPM (Ho et al. 2020) |
| 디노이저 아키텍처 | 조건부 UNet (ResBlock + Self-Attention) |
| 노이즈 스케줄 | Linear (T=1000) |
| 분산 학습 | LEARNED_RANGE |
| 컨디셔닝 방식 | SSL 특징 벡터 → Linear 투영 후 timestep emb에 가산 → FiLM |
| SSL 백본 | ResNet-50 (DINO / SimCLR / Barlow Twins / VICReg / Supervised) |
| 컨디셔닝 위치 | 모든 ResBlock (FiLM: scale + shift) |
| 분류기 가이던스 | 사용하지 않음 (대신 SSL 표현을 직접 조건으로 사용) |
| 학습 해상도 | 128×128 (기본 제공 pretrained) |
| 샘플링 가속 | DDIM 지원 (`--use_ddim`, `--timestep_respacing`) |
