# Reinforcement Learning

## CartPole
### Q-Table
- 0~1の間で初期化
### 方策
- epsilon-greedy法
- episode数が大きくなるごとに行動決定のランダム性を減衰させていき最適行動を取るようにする
### 報酬
- 各stepで報酬を与える
    - Poleが立っている場合 +1
    <!-- - Episode成功規定step数まで立っている場合 +1 -->
    - Episode成功規定step数より小さいstep数でPoleが倒れた場合 -100
### Episode成功条件
- (max_steps)stepでPoleが立っていた場合
### パラメータ
| パラメータ名 | 値 |
| ---- | ---- |
| $\varepsilon$-greedy法の $\varepsilon$値 | 0.5 |
| 最大エピソード数 max_episodes | 2000 |
| 最大ステップ数 max_steps | env.spec.max_episode_steps |
| 学習率 $\alpha$ | 0.5 |
| 割引率 $\gamma$ | 0.99 |