# 🧠 Pong DQN - 환경 설정 및 오류 해결 요약

---

## ✅ 환경 생성

```bash
conda create -n pong-dqn-env python=3.10 -y
conda activate pong-dqn-env
```

---

## ✅ 패키지 설치 순서

```bash
pip install numpy==1.26.4
pip install scipy==1.13.0
pip install pillow==10.3.0
pip install imageio==2.34.0
pip install protobuf==4.25.3
pip install opencv-python==4.9.0.80
pip install tensorflow==2.15.0
pip install matplotlib==3.8.4
pip install "gymnasium[atari]"
pip install autorom
AutoROM --accept-license
```

---

## ❗ 오류 및 해결 요약

### 🔸 오류: `Namespace ALE not found`
```text
gymnasium.error.NamespaceNotFound: Namespace ALE not found.
```
→ `pip install "gymnasium[atari]"`로 해결

---

### 🔸 오류: `Environment PongDeterministic doesn't exist`
```text
gymnasium.error.NameNotFound: Environment PongDeterministic doesn't exist in namespace ALE.
```
→ 환경 이름을 `"ALE/Pong-v5"`로 변경

---

### 🔸 오류: `Agent.__init__() missing learn_rate`
```text
TypeError: Agent.__init__() missing 1 required positional argument: 'learn_rate'
```
→ `Agent(...)` 생성 시 `learn_rate=0.00025` 추가

---

### 🔸 오류: `We're Unable to find the game "Pong"`
```text
gymnasium.error.Error: We're Unable to find the game "Pong".
```
→ ROM 설치 필요 → `AutoROM --accept-license` 실행

---

### 🔸 오류: `cd`가 작동하지 않음
```text
지정된 경로를 찾을 수 없습니다.
```
→ 드라이브 먼저 이동: `D:` → 그 다음 `cd` 명령

---

## ✅ 실행 예시

```bash
conda activate pong-dqn-env
D:
cd Projects\Python\ReinforcementLearning\project_2\pongGame
python main.py
```


