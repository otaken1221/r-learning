# Reinforcement Learning

## CartPole
### Q-Table
- 0~1の間で初期化
### 方策
- epsilon-greedy法
- step数が大きくなるごとに行動決定のランダム性を減衰させていき最適行動を取るようにする
### 報酬
- 各stepで報酬を与える
    - Poleが立っている場合 +1
    <!-- - Episode成功規定step数まで立っている場合 +1 -->
    - Episode成功規定step数より小さいstep数でPoleが倒れた場合 -100
### Episode成功条件
- max_steps数-5よりも先のstepでPoleが立ち続けていた場合
### パラメータ
- $\varepsilon$-greedy法の $\varepsilon$値
    - 0.5
- 最大エピソード数 max_episodes
    - 500
- 最大ステップ数 max_steps
    - 200
- 学習率 $\alpha$
    - 0.1
- 割引率 $\gamma$
    - 0.99

