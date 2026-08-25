# limo_delivery_rl_v2

Nav2가 **전역 경로만** 계산하고, PPO 정책이 `(v, ω)`를 직접 결정해 `/cmd_vel`에 발행하는
경유지 주행 강화학습 환경. 지도에 없는 정적·동적 장애물은 LiDAR 관측만으로 회피한다.

AgileX LIMO 기반 커스텀 4륜 스키드 조향 로봇, ROS 2 Humble + Gazebo Classic 대상.

| | |
|---|---|
| 관측 | 112차원 (LiDAR 24 bin × 4 frame, 경로 lookahead 5점, 경유지 거리/heading, 속도, 이전 action) |
| 행동 | 연속 2차원 — `v ∈ [0, 0.42] m/s`, `ω ∈ [-0.9, 0.9] rad/s` |
| 알고리즘 | Stable-Baselines3 PPO (MlpPolicy, CPU) |
| 제어 주기 | 20 Hz (`/scan`이 루프를 페이싱) |
| 전역 경로 | Nav2 `ComputePathThroughPoses` (planner-only, 에피소드당 1회) |
| 테스트 | 185 passed |

**검증 상태** — 단위 테스트, `colcon build`, 오프라인 백엔드 전 구간 완주, PPO 학습 루프와
TensorBoard 기록은 확인됨. Gazebo 실기 리셋 시퀀스와 `ComputePathThroughPoses` 왕복은
테스트 더블로만 검증되어 있으므로, 첫 실행 시 3절의 사전 점검을 먼저 돌릴 것.

---

## 1. 요구사항과 설치

### 요구사항

| | |
|---|---|
| OS | Ubuntu 22.04 |
| ROS 2 | Humble |
| Python | 3.10 |
| 시뮬레이터 | Gazebo Classic |
| RMW | `rmw_cyclonedds_cpp` |
| Python 패키지 | `gymnasium`, `numpy`, `stable-baselines3`, `tensorboard` |
| ROS 패키지 | `nav2_map_server`, `nav2_planner`, `nav2_lifecycle_manager`, `nav2_navfn_planner`, `tf2_ros`, `gazebo_ros` |

로봇 모델(`limo_car`)과 bringup은 이 저장소에 포함되어 있지 않다.
[AgileX limo_ros2](https://github.com/agilexrobotics/limo_ros2)에서 받아 같은 워크스페이스에
배치한다. 이 프로젝트는 `limo_car`의 `final_map_diff_with_sensor.xacro`
(720빔 / 240° / 0.2–8.0 m / 20 Hz LiDAR, `odometry_source=1`)와
`final_map.world`를 전제로 한다.

### 설치

```bash
mkdir -p ~/limo_ws/src && cd ~/limo_ws/src
git clone <이 저장소> limo_delivery_rl_v2
git clone https://github.com/agilexrobotics/limo_ros2.git

# 맵을 워크스페이스 루트에 배치 (코드가 /home/kim/limo_ws/map.yaml 을 참조)
cp limo_delivery_rl_v2/map/map.yaml limo_delivery_rl_v2/map/map.pgm ~/limo_ws/

cd ~/limo_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

> **경로 주의** — `state.py`, `launch/delivery_v2_rl.launch.py`,
> `config/nav2_planner_only_rl.yaml` 세 곳에 `/home/kim/limo_ws/map.yaml`이 하드코딩되어
> 있다. 다른 경로에 클론했다면 이 셋을 함께 고칠 것. `map/`의 사본은 재현용이며
> 런타임이 읽는 파일이 아니다.

### 테스트

ROS를 source하지 않아도 전부 돈다 (ROS 의존 테스트는 자동 skip).

```bash
cd ~/limo_ws/src/limo_delivery_rl_v2
python3 -m pytest -q test
```

---

## 2. 아키텍처 A 선택 근거 (로컬 컨트롤러 대체)

| # | 근거 |
|---|---|
| 1 | 이 프로젝트는 이미 Nav2 controller 없이 planner-only 구성으로 동작한다. |
| 2 | 기존 조향 구조도 `delta_final = delta_ppo`였고, Nav2 조향을 합산하지 않았다. |
| 3 | 장애물은 정적 맵에 넣지 않으므로 전역 경로가 바뀌지 않는다. 회피 판단은 `/scan`만으로 이뤄져야 한다. |
| 4 | 속도까지 action에 포함해야 거리 기반 자동 감속 없이 정책이 상황별 적정 속도를 학습한다. |
| 5 | Nav2 controller와 RL controller가 `/cmd_vel`을 두고 충돌할 여지를 원천 제거한다. |

**절대 금지**

```python
delta_final = delta_nav2 + delta_ppo   # 금지
```

**최종 제어**

```python
v_final = v_ppo
omega_final = omega_ppo
```

`safety_controller.apply_safety_limits()`에는 거리 비례 감속이 존재하지 않는다.
남아 있는 것은 하드 정지 조건, 절대 속도 clipping, 가·감속 제한뿐이다.

---

## 3. 패키지 구조

```
limo_delivery_rl_v2/
├── config/
│   └── nav2_planner_only_rl.yaml     # planner-only Nav2 파라미터 (obstacle_layer 없음)
├── map/                              # 재현용 정적 맵 사본 (런타임은 ~/limo_ws/map.yaml)
│   ├── map.yaml
│   └── map.pgm
├── launch/
│   └── delivery_v2_rl.launch.py      # map_server + planner_server + map→odom static TF
├── limo_delivery_rl_v2/
│   ├── state.py                      # 모든 설정 dataclass, StopReason, SafetyMode
│   ├── geometry.py                   # wrap_angle, map↔base_link 변환
│   ├── lidar.py                      # 720빔 → 24 bin 각도 기반 다운샘플
│   ├── path_tracker.py               # PathTracker (단조 진행 인덱스, lookahead, cross-track)
│   ├── waypoint_manager.py           # WaypointManager (5 step 유지, 보너스 1회)
│   ├── action_scaler.py              # ActionScaler, RateLimiter, sanitize_action
│   ├── safety_controller.py          # 하드 세이프티 게이트
│   ├── reward.py                     # 보상 항목 (상·하한 고정)
│   ├── termination.py                # 종료 조건 + 우선순위
│   ├── observation.py                # 112차원 관측, LidarFrameStack
│   ├── metrics.py                    # EpisodeMetrics
│   ├── map_utils.py                  # 정적 맵 존재 검증
│   ├── env_backend.py                # EnvBackend 프로토콜 + OfflineBackend
│   ├── ros_bridge.py                 # RosBridgeNode, TfPoseProvider, Nav2ThroughPosesPathProvider
│   ├── gazebo_reset.py               # GazeboResetManager, ObstacleManager
│   ├── ros_backend.py                # RosBackend (리셋 순서 + step 사이클)
│   ├── delivery_env.py               # LimoWaypointRLEnv
│   ├── tb_callback.py                # EpisodeMetricCallback
│   ├── train_ppo.py                  # 학습 스크립트
│   └── evaluate_ppo.py               # 평가 스크립트
├── test/                             # 179개 단위 테스트
└── legacy_v1/                        # v1 잔재 (삭제 대상, 아래 참고)
```

---

## 4. 프레임 규약

* 전역 경로 · 경유지 · 로봇 pose는 모두 `map` 프레임에서 계산한다.
* 로봇 pose는 **`tf2`의 `map→base_link` lookup 결과만** 사용한다.
* `/odom`은 twist(선속도·각속도) 취득 전용이다. **pose 필드는 어디에도 사용하지 않는다.**
* `ComputePathThroughPoses`가 반환한 Path의 `header.frame_id`가 `map`이 아니면 즉시 `FrameContractError`.
* tf lookup은 `rclpy.time.Time()`(latest available) + simulation clock으로 수행한다.
* tf lookup 실패는 이전 값으로 조용히 대체하지 않고 `tf_age`로 보고되어 timeout 경로를 탄다.

### Localization

학습 시 AMCL을 실행하지 않는다. `map→odom`은 `static_transform_publisher`가 발행하며 **기본값은 identity**다.

> 검증됨: `final_map.world`의 벽 모델은 map 좌표를 그대로 쓴다.
> 예) 월드 벽 `(-6.9, 10.005)`는 `origin=[-30.2,-4.72]`, `resolution=0.05` 기준으로
> `map.pgm`의 occupied 픽셀과 정확히 일치한다. `gazebo_ros_diff_drive`도
> `<odometry_source>1</odometry_source>`(ground truth)이므로 `odom` 역시 world 프레임과 같다.
> 따라서 map origin 오프셋을 변환에 넣으면 **안 된다.**

AMCL 제외 근거:

1. 학습 장애물이 `map.pgm`에 없으므로 scan matching이 오염되고, `map→odom`이 흔들리면
   lookahead 상대좌표와 cross-track error가 장애물 유무에 따라 달라진다.
2. 리셋마다 재수렴 대기 시간이 그대로 처리량 손실이 된다.
3. Gazebo odometry가 이미 ground truth다.

실기 배포 시에만 `use_amcl:=true`로 전환하며, 이때만 `/set_initial_pose`와
`/request_nomotion_update` 경로가 활성화된다.

---

## 5. 관측 · 행동 · 보상 요약

### 관측 (112차원, 전부 `[-1, 1]`)

| 구간 | 내용 | 정규화 |
|---|---|---|
| `[0:96)` | LiDAR 24 bin × 4 frame (오래된 것 먼저) | `range / 8.0` |
| `[96:106)` | 전방 경로 5점 × `(x, y)` (base_link) | `clip(±3.0) / 3.0` |
| `[106:108)` | 다음 경유지 거리, heading error | `/20.0`, `/π` |
| `[108:110)` | 실측 선속도, 각속도 | `/0.42`, `/0.9` |
| `[110:112)` | 직전 normalized action | 그대로 |

LiDAR bin은 배열 인덱스가 아니라 `angle_min` / `angle_increment`로 계산한다.
720빔 / 240° / 24 bin이면 bin당 정확히 30빔이고, 정면(0 rad)은 bin 11과 12의 경계다.

### 행동

```python
v_target     = (action[0] + 1.0) * 0.5 * 0.42     # [0.0, 0.42] m/s
omega_target = action[1] * 0.9                    # [-0.9, 0.9] rad/s
```

가속도 제한: `|Δv| ≤ 0.05 m/s`, `|Δω| ≤ 0.09 rad/s` per step (20 Hz).

### 보상 (스텝당 상·하한 고정)

| 항목 | 범위 / 값 |
|---|---|
| `progress` | `10 × clip(Δd, ±0.05)` → `±0.5` |
| `waypoint` | `+20.0` (경유지당 1회) |
| `success` | `+100.0` |
| `collision` | `-100.0` |
| `stuck` | `-80.0` |
| `danger` | `[-0.10, 0]` (0.80 m 이하) |
| `deviation` | `[-0.02, 0]` |
| `time` | `-0.01` |
| `smoothness` | `[-0.04, 0]` |

`progress_at_waypoint_switch`는 **측정 전용**이며 총보상에 다시 더하지 않는다.
경유지 전환 스텝에서는 이 값이 하한 `-0.5`에 고정되는데, 1차 학습에서는 수정하지 않고
`episode/waypoint_switch_progress_sum`으로 계측만 한다.

### 종료 조건 우선순위

`COLLISION` → `SUCCESS` → `PATH_FAILED` → `SENSOR_TIMEOUT` → `PATH_DEVIATION` → `STUCK` → `TIMEOUT`

`COLLISION`과 `SUCCESS`만 `terminated=True`이고 나머지는 `truncated=True`다.
하드 세이프티가 로봇을 강제 정지시킨 스텝에서는 stuck 카운터를 올리지 않는다.

---

## 6. 실행 순서

터미널 4개를 사용한다. 모든 노드는 `use_sim_time:=true`다.

```bash
# 0) 공통
cd /home/kim/limo_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

```bash
# 1) Gazebo (GUI 없음)
ros2 launch limo_car final_map_diff_gazebo.launch.py gui:=false use_sim_time:=true
```

```bash
# 2) Nav2 planner-only + map→odom static TF (amcl 없음)
ros2 launch limo_delivery_rl_v2 delivery_v2_rl.launch.py use_amcl:=false
```

```bash
# 3) 사전 점검
ros2 topic info /cmd_vel --verbose | grep -A1 "Node name"   # RL 브릿지 외 publisher 없어야 함
ros2 run tf2_ros tf2_echo map base_link
ros2 action list | grep compute_path                        # 서버는 정확히 하나
ros2 run limo_delivery_rl_v2 delivery_v2_smoke --steps 20
```

```bash
# 4) 학습
ros2 run limo_delivery_rl_v2 delivery_v2_train_ppo \
  --total-timesteps 1000000 \
  --tensorboard-log runs/limo_delivery_rl_v2/tensorboard

# 5) TensorBoard
tensorboard --logdir runs/limo_delivery_rl_v2/tensorboard
```

```bash
# 6) 평가 (deterministic)
ros2 run limo_delivery_rl_v2 delivery_v2_eval_ppo \
  runs/limo_delivery_rl_v2/final_model.zip \
  --episodes 20 \
  --json-out runs/limo_delivery_rl_v2/eval.json \
  --csv-out  runs/limo_delivery_rl_v2/eval.csv
```

ROS 없이 로직만 확인하려면 어느 스크립트든 `--no-ros`를 붙인다 (결정론적 unicycle 백엔드).

### 검증 명령

```bash
cd /home/kim/limo_ws/src/limo_delivery_rl_v2
python3 -m pytest -q test

cd /home/kim/limo_ws
source /opt/ros/humble/setup.bash
colcon build \
  --base-paths src/limo_delivery_rl_v2 \
  --packages-select limo_delivery_rl_v2 \
  --symlink-install
source install/setup.bash
```

---

## 7. 에피소드 리셋 순서

1. `(0, 0)` 반복 발행
2. `/pause_physics`
3. 이전 학습 장애물 `/delete_entity`
4. `/set_entity_state`로 pose + twist 동시 초기화
5. `/unpause_physics`
6. `(0, 0)` 반복 발행
7. pause → pose 재설정 → unpause (안정화)
8. `map→odom` static transform 존재 확인 (`use_amcl:=true`면 AMCL seed + nomotion update)
9. **실행 중인** costmap만 clear (local costmap은 대기 대상이 아님)
10. `map→base_link` TF가 리셋 pose와 허용치(0.10 m) 내로 일치할 때까지 대기
11. 리셋 이전 timestamp의 scan/odom 폐기
12. 새 scan · odom 각각 최소 1회 수신
13. `ComputePathThroughPoses` 완료 대기
14. Path `frame_id == "map"` 검증
15. 장애물 spawn
16. 장애물이 반영된 새 LiDAR 수신 후 첫 observation 반환

`/reset_world`는 사용하지 않는다. simulation clock을 되돌리면 tf2 버퍼가 전부 무효화된다.

---

## 8. 수렴 실패 체크리스트

증상별로 위에서부터 확인한다.

### A. 전혀 전진하지 않는다 / 즉시 stuck

1. `ros2 topic echo /cmd_vel` — 명령이 나가는가? 안 나가면 `info["safety_mode"]`를 본다.
2. `safety_mode == "sensor_timeout"` → `/scan`·`/odom` 발행 확인, `use_sim_time` 확인.
3. `safety_mode == "tf_timeout"` → `ros2 run tf2_ros tf2_echo map base_link`. static TF가 죽었는지 확인.
4. `safety_mode == "imminent_collision"` → 리셋 pose에서 이미 벽에 붙어 있다. 시작 좌표 확인.
5. `log_std_init=-2.0`은 탐색이 매우 좁다. 초반 학습이 멈추면 `-1.0`으로 올려본다.

### B. 장애물 앞에서 정지하는 정책으로 수렴

1. `reward/danger`가 `reward/progress`를 압도하는지 확인. 정상이면 progress가 훨씬 크다.
2. `episode/stuck_count`가 오르는데 `reward/stuck`이 0이면, 하드 세이프티가 정지시킨 상태다.
   `safety_mode` 분포를 먼저 확인한다 (stuck 페널티는 강제 정지 스텝에서 부과되지 않는다).
3. `COLLISION_PENALTY(-100)`가 `SUCCESS_REWARD(+100)`보다 훨씬 자주 발생하면 장애물 난이도를 낮춘다.

### C. 경유지 전환마다 보상이 급락한다

1. `reward/progress_at_waypoint_switch`가 전환마다 `-0.5`에 고정되어 있는지 확인한다.
2. `episode/waypoint_switch_progress_sum ≈ -0.5 × (경유지 수 - 1)`이면 설계대로다.
3. 이 값이 학습을 방해한다고 판단되면 그때 보상 재설계를 검토한다 (1차 학습에서는 계측만).

### D. 경로를 벗어나 truncate된다 (`path_deviation`)

1. `episode/max_cross_track_error`가 2.5 m에 닿는지 확인.
2. `deviation` 페널티는 스텝당 `-0.02`가 상한이라 회피를 막지 못한다. 실제 원인은 대개 조향 학습 부족이다.
3. `observation[96:106]`(lookahead)이 전부 0에 가까우면 경로가 비었거나 진행 인덱스가 멈춘 것이다.

### E. 관측이 이상하다

1. `env.observation_space.contains(obs)` — False면 정규화 기준이 어긋난 것이다.
2. LiDAR가 항상 1.0이면 `/scan`이 안 들어오거나 `lidar_max_range`가 센서와 불일치한다 (반드시 8.0).
3. 정면 장애물이 bin 11/12에 안 잡히면 `angle_min`이 xacro와 다른 것이다.

### F. 프레임 문제

1. RViz에서 LaserScan이 맵 벽과 어긋나면 `map→odom`이 identity가 아니거나 world와 map이 어긋난 것이다.
2. `FrameContractError`가 뜨면 Nav2 global_frame이 `map`이 아니다.
3. `/odom` pose를 다시 쓰기 시작하면 안 된다. `frame.pose`(tf) 외의 경로가 생겼는지 확인한다.

### G. Nav2 경로가 장애물을 우회한다

`config/nav2_planner_only_rl.yaml`의 `global_costmap.plugins`에 `obstacle_layer`가
들어가 있지 않은지 확인한다. `["static_layer", "inflation_layer"]`만 있어야 한다.

### H. `/cmd_vel` 충돌

`RuntimeError: Forbidden /cmd_vel publishers active: ...`가 뜨면
`controller_server`, `velocity_smoother`, `behavior_server`, `waypoint_follower`,
`bt_navigator` 중 하나가 살아 있다. planner-only launch만 사용한다.

---

## 9. 학습 커리큘럼

관측 112차원에 경유지 *개수*가 들어가지 않는다 — "다음 경유지까지 거리/heading" 2개뿐이다.
따라서 경유지 1개로 배운 정책이 2개·3개 환경에 그대로 load된다. `--resume`으로 단계를
이어붙이는 것이 이 설계의 전제다.

### 경유지 단계

| 단계 | 경유지 | 도달 판정 | 목표 | 전환 기준 |
|---|---|---|---|---|
| A | 1개 (3.0) | 반경 0.9, hold 1, 통과 평면 1.0 | 장애물 회피 | 성공률 ~70% |
| B | 1개 (3.0) | 반경 0.60, hold 1 | 정밀 도달 | 성공률 ~70% |
| C | 2개 (3.0, 6.0) | 반경 0.60, hold 1 | 연속 전환 | 성공률 ~60% |
| D | 3개 (3.0, 6.0, 9.5) | 반경 0.60, hold 1 | 최종 | — |

`--waypoint-hold-steps 1`은 체류(dwell)를 없앤다. 스펙 기본값 5 step은 경유지에
0.25초 머물기를 요구하는데, 이는 "통과해서 계속 가기"와 상충하고 통과 속도를 깎는다.
`0`은 반경 밖에서도 도달로 판정되므로 거부된다 (최소 1).

A단계를 90%까지 밀지 말 것. 경유지 1개에서 "도착해서 멈추기"를 과학습하면
C·D단계의 "통과해서 계속 가기"와 충돌한다.

```bash
# A: 장애물 회피 (성공 보상이 9.5 m가 아니라 3 m 앞에 있다)
ros2 run limo_delivery_rl_v2 delivery_v2_train_ppo \
  --waypoints 1 --waypoint-radius 0.9 --waypoint-hold-steps 1 --waypoint-capture-width 1.0 \
  --total-timesteps 200000 --log-std-init -1.5 \
  --save-path runs/limo_delivery_rl_v2/stage_a --tb-log-name stage_a

# B: 도달 판정을 스펙대로 조인다
ros2 run limo_delivery_rl_v2 delivery_v2_train_ppo \
  --waypoints 1 --waypoint-hold-steps 1 --resume runs/limo_delivery_rl_v2/stage_a.zip \
  --total-timesteps 150000 \
  --save-path runs/limo_delivery_rl_v2/stage_b --tb-log-name stage_b

# C: 경유지 2개
ros2 run limo_delivery_rl_v2 delivery_v2_train_ppo \
  --waypoints 2 --waypoint-hold-steps 1 --resume runs/limo_delivery_rl_v2/stage_b.zip \
  --total-timesteps 200000 \
  --save-path runs/limo_delivery_rl_v2/stage_c --tb-log-name stage_c

# D: 전체
ros2 run limo_delivery_rl_v2 delivery_v2_train_ppo \
  --waypoints 3 --waypoint-hold-steps 1 --resume runs/limo_delivery_rl_v2/stage_c.zip \
  --total-timesteps 500000 \
  --save-path runs/limo_delivery_rl_v2/final_model --tb-log-name stage_d
```

평가할 때는 학습한 단계와 같은 인자를 준다.

```bash
ros2 run limo_delivery_rl_v2 delivery_v2_eval_ppo \
  runs/limo_delivery_rl_v2/stage_a.zip --waypoints 1 --waypoint-radius 0.9 \
  --waypoint-hold-steps 1 --waypoint-capture-width 1.0 --episodes 20
```

### 통과 평면 판정

반경 판정만으로는 **경유지 옆을 반경 밖으로 스쳐 지나간 경우를 영영 못 잡는다.**
원 안에 한 번도 안 들어오므로 도달 판정이 나지 않고 에피소드가 계속 달린다.
`--waypoint-capture-width W`를 주면 다음 조건이 OR로 추가된다.

```
(로봇 - 경유지) · 진행방향 > 0   AND   |측면 오차| <= W
```

진행방향은 이전 경유지(첫 경유지는 시작 pose)에서 현재 경유지로 향하는 방향이다.
기본값 `0.0`은 이 판정을 끄고 스펙의 반경 + 5 step 규칙만 남긴다.

### 장애물 단계

1. 장애물 없는 경로 추종 (`ObstacleConfig.enabled = False`) — 경로 추종이 이미 되면 생략 가능
2. 고정 장애물 1개 `(2.07, -0.18)`
3. 검증된 free-space 내 정적 장애물 위치 랜덤화
4. 정적 장애물 다수
5. 동적 장애물 추가 — **이동 구간과 속도는 아직 미정이다.**
   3단계가 성공한 뒤 시작점 · 종료점 · 속도를 확정하고 진행한다.

현재 장애물의 기하는 다음과 같다. 중심선 직진은 약 3 cm 물리적으로 겹치므로
회피가 필수다.

| | |
|---|---|
| 장애물 | x ∈ [1.945, 2.195], y ∈ [-0.305, -0.055] |
| 로봇 폭 (바퀴 포함) | y ∈ [-0.0875, +0.0875] |
| 0.25 m 여유 통과에 필요한 횡변위 | 중심 y ≈ +0.2 |

경유지 구간(`x ∈ [0, 12]`)의 충돌 없는 y 범위는 대략 `[-1.4, +1.05]` m다
(`map.pgm` 실측). 이 범위 밖에 장애물을 두면 안 된다.

## 10. 기존 체크포인트 / v1 잔재

* v1 체크포인트(`ppo_steering_*.zip`)는 action 공간이 `(1,)`이라 **재사용 불가**다.
  새 `(2,)` 공간으로 처음부터 학습한다.
* `runs/`는 `.gitignore` 대상이다. 체크포인트·TensorBoard 이벤트·평가 리포트는
  저장소에 포함되지 않는다.
* v1 모듈(잔차 조향, 30차원 관측, 거리 기반 자동 감속)은 전부 제거되었다.
