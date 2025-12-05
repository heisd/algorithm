"""
Q-Learning 强化学习算法 Demo
经典的迷宫寻路问题

Author: Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib.animation as animation
# from IPython.display import HTML  # Optional for Jupyter

# 设置
np.random.seed(42)


class MazeEnvironment:
    """
    迷宫环境
    0 = 空地
    1 = 墙
    2 = 起点
    3 = 终点 (奖励 +100)
    4 = 陷阱 (奖励 -50)
    """
    
    def __init__(self):
        self.maze = np.array([
            [2, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 4, 0, 3]
        ])
        # 元组解包特性
        self.rows, self.cols = self.maze.shape
       
        self.start = (0, 0)
        self.goal = (4, 4)
        self.trap = (4, 2)
        self.state = self.start
        
        # 动作: 0=上, 1=下, 2=左, 3=右
        self.actions = ['up', 'down', 'left', 'right']
        self.action_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
    def reset(self):
        """重置环境"""
        self.state = self.start
        return self.state
    
    def step(self, action):
        """
        执行动作
        返回: (新状态, 奖励, 是否结束)
        """
        action_name = self.actions[action]
        dr, dc = self.action_deltas[action_name]
        
        new_row = self.state[0] + dr
        new_col = self.state[1] + dc
        
        # 检查边界和墙
        if (0 <= new_row < self.rows and 
            0 <= new_col < self.cols and 
            self.maze[new_row, new_col] != 1):
            self.state = (new_row, new_col)
        
        # 计算奖励
        cell = self.maze[self.state[0], self.state[1]]
        
        if cell == 3:  # 终点
            return self.state, 100, True
        elif cell == 4:  # 陷阱
            return self.state, -50, True
        else:
            return self.state, -1, False  # 每步小惩罚
    
    def get_valid_actions(self, state):
        """获取有效动作"""
        valid = []
        for i, action in enumerate(self.actions):
            dr, dc = self.action_deltas[action]
            new_row = state[0] + dr
            new_col = state[1] + dc
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols and 
                self.maze[new_row, new_col] != 1):
                valid.append(i)
        return valid if valid else list(range(4))


class QLearningAgent:
    """
    Q-Learning 智能体
    """
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.3):
        self.env = env
        self.lr = learning_rate      # 学习率 α
        self.gamma = discount_factor  # 折扣因子 γ
        self.epsilon = epsilon        # 探索率 ε
        
        # 初始化 Q 表
        self.q_table = {}
        for r in range(env.rows):
            for c in range(env.cols):
                self.q_table[(r, c)] = np.zeros(4)
        
        # 训练历史
        self.episode_rewards = []
        self.episode_steps = []
        
    def select_action(self, state, training=True):
        """
        ε-贪婪策略选择动作
        """
        if training and np.random.random() < self.epsilon:
            # 探索: 随机选择
            return np.random.randint(4)
        else:
            # 利用: 选择 Q 值最大的动作
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """
        Q-Learning 更新公式:
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        
        # TD 更新
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        
        self.q_table[state][action] += self.lr * td_error
        
        return td_error
    
    def train_episode(self):
        """
        训练一个回合
        """
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = self.select_action(state)
            next_state, reward, done = self.env.step(action)
            
            self.update(state, action, reward, next_state)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done or steps > 100:
                break
        
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        
        return total_reward, steps
    
    def train(self, episodes=500, verbose=True):
        """
        训练多个回合
        """
        for ep in range(episodes):
            reward, steps = self.train_episode()
            
            if verbose and (ep + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                print(f"Episode {ep+1:4d} | Avg Reward: {avg_reward:7.2f} | Avg Steps: {avg_steps:.1f}")
        
        return self.episode_rewards
    
    def get_optimal_path(self):
        """
        获取最优路径
        """
        state = self.env.reset()
        path = [state]
        
        for _ in range(50):
            action = self.select_action(state, training=False)
            next_state, _, done = self.env.step(action)
            path.append(next_state)
            state = next_state
            
            if done:
                break
        
        return path
    
    def get_policy(self):
        """
        获取策略 (每个状态的最佳动作)
        """
        policy = {}
        for state in self.q_table:
            policy[state] = np.argmax(self.q_table[state])
        return policy


def visualize_q_table(agent, env):
    """
    可视化 Q 表和策略
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 颜色映射
    cell_colors = {
        0: '#e2e8f0',  # 空地
        1: '#475569',  # 墙
        2: '#60a5fa',  # 起点
        3: '#4ade80',  # 终点
        4: '#f87171',  # 陷阱
    }
    
    arrows = {
        0: (0, 0.3),   # 上
        1: (0, -0.3),  # 下
        2: (-0.3, 0),  # 左
        3: (0.3, 0),   # 右
    }
    
    # === 左图: Q 值热力图 ===
    ax1 = axes[0]
    ax1.set_title('Q-Table Values\n(Numbers show Q values for each action)', fontsize=12)
    
    for r in range(env.rows):
        for c in range(env.cols):
            cell = env.maze[r, c]
            color = cell_colors[cell]
            
            rect = Rectangle((c, env.rows - 1 - r), 1, 1, 
                            facecolor=color, edgecolor='black', linewidth=2)
            ax1.add_patch(rect)
            
            # 显示 Q 值
            if cell not in [1, 3, 4]:
                q_values = agent.q_table[(r, c)]
                # 上
                ax1.text(c + 0.5, env.rows - r - 0.15, f'{q_values[0]:.1f}', 
                        ha='center', va='center', fontsize=7, color='blue')
                # 下
                ax1.text(c + 0.5, env.rows - r - 0.85, f'{q_values[1]:.1f}', 
                        ha='center', va='center', fontsize=7, color='blue')
                # 左
                ax1.text(c + 0.15, env.rows - r - 0.5, f'{q_values[2]:.1f}', 
                        ha='center', va='center', fontsize=7, color='blue')
                # 右
                ax1.text(c + 0.85, env.rows - r - 0.5, f'{q_values[3]:.1f}', 
                        ha='center', va='center', fontsize=7, color='blue')
            
            # 特殊标记
            if cell == 2:
                ax1.text(c + 0.5, env.rows - r - 0.5, 'S', 
                        ha='center', va='center', fontsize=14, fontweight='bold')
            elif cell == 3:
                ax1.text(c + 0.5, env.rows - r - 0.5, 'G', 
                        ha='center', va='center', fontsize=14, fontweight='bold')
            elif cell == 4:
                ax1.text(c + 0.5, env.rows - r - 0.5, 'X', 
                        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    ax1.set_xlim(0, env.cols)
    ax1.set_ylim(0, env.rows)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # === 右图: 最优策略 ===
    ax2 = axes[1]
    ax2.set_title('Learned Policy\n(Arrows show best action at each state)', fontsize=12)
    
    policy = agent.get_policy()
    optimal_path = agent.get_optimal_path()
    path_set = set(optimal_path)
    
    for r in range(env.rows):
        for c in range(env.cols):
            cell = env.maze[r, c]
            color = cell_colors[cell]
            
            # 高亮最优路径
            if (r, c) in path_set and cell not in [3, 4]:
                color = '#fef08a'  # 黄色高亮
            
            rect = Rectangle((c, env.rows - 1 - r), 1, 1, 
                            facecolor=color, edgecolor='black', linewidth=2)
            ax2.add_patch(rect)
            
            # 绘制策略箭头
            if cell not in [1, 3, 4]:
                best_action = policy[(r, c)]
                dx, dy = arrows[best_action]
                ax2.arrow(c + 0.5 - dx/2, env.rows - r - 0.5 - dy/2, 
                         dx * 0.8, dy * 0.8,
                         head_width=0.15, head_length=0.1, 
                         fc='#1e40af', ec='#1e40af')
            
            # 特殊标记
            if cell == 2:
                ax2.text(c + 0.5, env.rows - r - 0.5, 'S', 
                        ha='center', va='center', fontsize=14, fontweight='bold')
            elif cell == 3:
                ax2.text(c + 0.5, env.rows - r - 0.5, 'G', 
                        ha='center', va='center', fontsize=14, fontweight='bold')
            elif cell == 4:
                ax2.text(c + 0.5, env.rows - r - 0.5, 'X', 
                        ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    ax2.set_xlim(0, env.cols)
    ax2.set_ylim(0, env.rows)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def plot_learning_curve(agent):
    """
    绘制学习曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 奖励曲线
    ax1 = axes[0]
    rewards = agent.episode_rewards
    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    ax1.plot(rewards, alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(range(window-1, len(rewards)), smoothed, color='red', linewidth=2, label=f'Moving Avg ({window})')
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Curve: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 步数曲线
    ax2 = axes[1]
    steps = agent.episode_steps
    smoothed_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
    
    ax2.plot(steps, alpha=0.3, color='orange', label='Episode Steps')
    ax2.plot(range(window-1, len(steps)), smoothed_steps, color='red', linewidth=2, label=f'Moving Avg ({window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Learning Curve: Steps per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Q-Learning 强化学习 Demo")
    print("=" * 60)
    
    # 创建环境和智能体
    env = MazeEnvironment()
    agent = QLearningAgent(
        env,
        learning_rate=0.1,    # α: 学习率
        discount_factor=0.95,  # γ: 折扣因子
        epsilon=0.2           # ε: 探索率
    )
    
    print("\n迷宫环境:")
    print("S = 起点, G = 终点, X = 陷阱, # = 墙")
    print(env.maze)

    
    print("\n开始训练...")
    print("-" * 60)
    
    # 训练
    agent.train(episodes=500, verbose=True)
    
    print("-" * 60)
    print("\n训练完成!")
    
    # 获取最优路径
    optimal_path = agent.get_optimal_path()
    print(f"\n最优路径 ({len(optimal_path)} 步):")
    print(" -> ".join([f"({r},{c})" for r, c in optimal_path]))
    
    # 显示最终 Q 表
    print("\n最终 Q 表 (部分):")
    print("-" * 40)
    for state in [(0, 0), (0, 2), (2, 2), (4, 3)]:
        print(f"State {state}: {agent.q_table[state].round(2)}")
    
    # 可视化
    fig1 = visualize_q_table(agent, env)
    fig1.savefig('qlearning_policy.png', dpi=150, bbox_inches='tight')
    print("\n策略可视化已保存到 qlearning_policy.png")
    
    fig2 = plot_learning_curve(agent)
    fig2.savefig('qlearning_curve.png', dpi=150, bbox_inches='tight')
    print("学习曲线已保存到 qlearning_curve.png")
    
    # 核心算法说明
    print("\n" + "=" * 60)
    print("Q-Learning 核心算法")
    print("=" * 60)
    print("""
    Q-Learning 更新公式:
    
    Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
                             └──────────┬──────────┘
                                    TD Target
    
    其中:
    - Q(s, a): 状态 s 下采取动作 a 的价值
    - α (学习率): 新信息的权重 (0~1)
    - r: 即时奖励
    - γ (折扣因子): 未来奖励的重要性 (0~1)
    - max Q(s', a'): 下一状态的最大 Q 值
    
    ε-贪婪策略:
    - 以概率 ε 随机探索
    - 以概率 1-ε 选择最优动作 (利用)
    """)
    
    plt.show()