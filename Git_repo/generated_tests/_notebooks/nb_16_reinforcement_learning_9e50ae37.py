# --- notebook cell 6 ---
import numpy as np
import os
import sklearn
import sys

try:
    # %tensorflow_version only exists in Colab.
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# --- notebook cell 10 ---
import gym

# --- notebook cell 12 ---
env = gym.make('MsPacman-v0')

# --- notebook cell 14 ---
env.seed(42)
obs = env.reset()

# --- notebook cell 16 ---
obs.shape

# --- notebook cell 19 ---
try:
    import pyvirtualdisplay
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
except ImportError:
    pass

# --- notebook cell 20 ---
env.render()

# --- notebook cell 22 ---
img = env.render(mode="rgb_array")
img.shape

# --- notebook cell 24 ---
plt.figure(figsize=(5,4))
plt.imshow(img)
plt.axis("off")
save_fig("MsPacman")
plt.show()

# --- notebook cell 27 ---
(img == obs).all()

# --- notebook cell 29 ---
def plot_environment(env, figsize=(5,4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")

# --- notebook cell 31 ---
env.action_space

# --- notebook cell 34 ---
env.seed(42)
env.reset()
for step in range(110):
    env.step(3) #left
for step in range(40):
    env.step(8) #lower-left

# --- notebook cell 36 ---
plot_environment(env)
plt.show()

# --- notebook cell 38 ---
obs, reward, done, info = env.step(0)

# --- notebook cell 40 ---
obs.shape

# --- notebook cell 42 ---
reward

# --- notebook cell 44 ---
done

# --- notebook cell 46 ---
info

# --- notebook cell 48 ---
frames = []

n_max_steps = 1000
n_change_steps = 10

env.seed(42)
obs = env.reset()
for step in range(n_max_steps):
    img = env.render(mode="rgb_array")
    frames.append(img)
    if step % n_change_steps == 0:
        action = env.action_space.sample() # play randomly
    obs, reward, done, info = env.step(action)
    if done:
        break

# --- notebook cell 50 ---
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

# --- notebook cell 51 ---
plot_animation(frames)

# --- notebook cell 53 ---
env.close()

# --- notebook cell 57 ---
env = gym.make("CartPole-v0")

# --- notebook cell 58 ---
env.seed(42)
obs = env.reset()

# --- notebook cell 59 ---
obs

# --- notebook cell 61 ---
plot_environment(env)
plt.show()

# --- notebook cell 63 ---
env.action_space

# --- notebook cell 65 ---
env.seed(42)
obs = env.reset()
while True:
    obs, reward, done, info = env.step(0)
    if done:
        break

# --- notebook cell 66 ---
plot_environment(env)
save_fig("cart_pole_plot")

# --- notebook cell 67 ---
img.shape

# --- notebook cell 69 ---
obs = env.reset()
while True:
    obs, reward, done, info = env.step(1)
    if done:
        break

# --- notebook cell 70 ---
plot_environment(env)
plt.show()

# --- notebook cell 74 ---
frames = []

n_max_steps = 1000
n_change_steps = 10

env.seed(42)
obs = env.reset()
for step in range(n_max_steps):
    img = env.render(mode="rgb_array")
    frames.append(img)

    # hard-coded policy
    position, velocity, angle, angular_velocity = obs
    if angle < 0:
        action = 0
    else:
        action = 1

    obs, reward, done, info = env.step(action)
    if done:
        break

# --- notebook cell 75 ---
plot_animation(frames)

# --- notebook cell 80 ---
import tensorflow as tf

# 1. Specify the network architecture
n_inputs = 4  # == env.observation_space.shape[0]
n_hidden = 4  # it's a simple task, we don't need more than this
n_outputs = 1 # only outputs the probability of accelerating left
initializer = tf.variance_scaling_initializer()

# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                         kernel_initializer=initializer)
outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid,
                          kernel_initializer=initializer)

# 3. Select a random action based on the estimated probabilities
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

init = tf.global_variables_initializer()

# --- notebook cell 84 ---
n_max_steps = 1000
frames = []

with tf.Session() as sess:
    init.run()
    env.seed(42)
    obs = env.reset()
    for step in range(n_max_steps):
        img = env.render(mode="rgb_array")
        frames.append(img)
        action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
        obs, reward, done, info = env.step(action_val[0][0])
        if done:
            break

env.close()

# --- notebook cell 86 ---
plot_animation(frames)

# --- notebook cell 88 ---
import tensorflow as tf

reset_graph()

n_inputs = 4
n_hidden = 4
n_outputs = 1

learning_rate = 0.01

initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.float32, shape=[None, n_outputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits) # probability of action 0 (left)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# --- notebook cell 90 ---
n_environments = 10
n_iterations = 1000

envs = [gym.make("CartPole-v0") for _ in range(n_environments)]
observations = [env.reset() for env in envs]

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations]) # if angle<0 we want proba(left)=1., or else proba(left)=0.
        action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(observations), y: target_probas})
        for env_index, env in enumerate(envs):
            obs, reward, done, info = env.step(action_val[env_index][0])
            observations[env_index] = obs if not done else env.reset()
    saver.save(sess, "./my_policy_net_basic.ckpt")

for env in envs:
    env.close()

# --- notebook cell 91 ---
def render_policy_net(model_path, action, X, n_max_steps=1000):
    frames = []
    env = gym.make("CartPole-v0")
    obs = env.reset()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            img = env.render(mode="rgb_array")
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break
    env.close()
    return frames        

# --- notebook cell 92 ---
frames = render_policy_net("./my_policy_net_basic.ckpt", action, X)
plot_animation(frames)

# --- notebook cell 96 ---
import tensorflow as tf

reset_graph()

n_inputs = 4
n_hidden = 4
n_outputs = 1

learning_rate = 0.01

initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# --- notebook cell 97 ---
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

# --- notebook cell 98 ---
discount_rewards([10, 0, -50], discount_rate=0.8)

# --- notebook cell 99 ---
discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8)

# --- notebook cell 100 ---
env = gym.make("CartPole-v0")

n_games_per_update = 10
n_max_steps = 1000
n_iterations = 250
save_iterations = 10
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration), end="")
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./my_policy_net_pg.ckpt")

# --- notebook cell 101 ---
env.close()

# --- notebook cell 102 ---
frames = render_policy_net("./my_policy_net_pg.ckpt", action, X, n_max_steps=1000)
plot_animation(frames)

# --- notebook cell 104 ---
transition_probabilities = [
        [0.7, 0.2, 0.0, 0.1],  # from s0 to s0, s1, s2, s3
        [0.0, 0.0, 0.9, 0.1],  # from s1 to ...
        [0.0, 1.0, 0.0, 0.0],  # from s2 to ...
        [0.0, 0.0, 0.0, 1.0],  # from s3 to ...
    ]

n_max_steps = 50

def print_sequence(start_state=0):
    current_state = start_state
    print("States:", end=" ")
    for step in range(n_max_steps):
        print(current_state, end=" ")
        if current_state == 3:
            break
        current_state = np.random.choice(range(4), p=transition_probabilities[current_state])
    else:
        print("...", end="")
    print()

for _ in range(10):
    print_sequence()

# --- notebook cell 106 ---
transition_probabilities = [
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], # in s0, if action a0 then proba 0.7 to state s0 and 0.3 to state s1, etc.
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
        [None, [0.8, 0.1, 0.1], None],
    ]

rewards = [
        [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
        [[0, 0, 0], [+40, 0, 0], [0, 0, 0]],
    ]

possible_actions = [[0, 1, 2], [0, 2], [1]]

def policy_fire(state):
    return [0, 2, 1][state]

def policy_random(state):
    return np.random.choice(possible_actions[state])

def policy_safe(state):
    return [0, 0, 1][state]

class MDPEnvironment(object):
    def __init__(self, start_state=0):
        self.start_state=start_state
        self.reset()
    def reset(self):
        self.total_rewards = 0
        self.state = self.start_state
    def step(self, action):
        next_state = np.random.choice(range(3), p=transition_probabilities[self.state][action])
        reward = rewards[self.state][action][next_state]
        self.state = next_state
        self.total_rewards += reward
        return self.state, reward

def run_episode(policy, n_steps, start_state=0, display=True):
    env = MDPEnvironment()
    if display:
        print("States (+rewards):", end=" ")
    for step in range(n_steps):
        if display:
            if step == 10:
                print("...", end=" ")
            elif step < 10:
                print(env.state, end=" ")
        action = policy(env.state)
        state, reward = env.step(action)
        if display and step < 10:
            if reward:
                print("({})".format(reward), end=" ")
    if display:
        print("Total rewards =", env.total_rewards)
    return env.total_rewards

for policy in (policy_fire, policy_random, policy_safe):
    all_totals = []
    print(policy.__name__)
    for episode in range(1000):
        all_totals.append(run_episode(policy, n_steps=100, display=(episode<5)))
    print("Summary: mean={:.1f}, std={:1f}, min={}, max={}".format(np.mean(all_totals), np.std(all_totals), np.min(all_totals), np.max(all_totals)))
    print()

# --- notebook cell 109 ---
n_states = 3
n_actions = 3
n_steps = 20000
alpha = 0.01
gamma = 0.99
exploration_policy = policy_random
q_values = np.full((n_states, n_actions), -np.inf)
for state, actions in enumerate(possible_actions):
    q_values[state][actions]=0

env = MDPEnvironment()
for step in range(n_steps):
    action = exploration_policy(env.state)
    state = env.state
    next_state, reward = env.step(action)
    next_value = np.max(q_values[next_state]) # greedy policy
    q_values[state, action] = (1-alpha)*q_values[state, action] + alpha*(reward + gamma * next_value)

# --- notebook cell 110 ---
def optimal_policy(state):
    return np.argmax(q_values[state])

# --- notebook cell 111 ---
q_values

# --- notebook cell 112 ---
all_totals = []
for episode in range(1000):
    all_totals.append(run_episode(optimal_policy, n_steps=100, display=(episode<5)))
print("Summary: mean={:.1f}, std={:1f}, min={}, max={}".format(np.mean(all_totals), np.std(all_totals), np.min(all_totals), np.max(all_totals)))
print()

# --- notebook cell 116 ---
env = gym.make("MsPacman-v0")
obs = env.reset()
obs.shape

# --- notebook cell 117 ---
env.action_space

# --- notebook cell 120 ---
mspacman_color = 210 + 164 + 74

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)

img = preprocess_observation(obs)

# --- notebook cell 122 ---
plt.figure(figsize=(11, 7))
plt.subplot(121)
plt.title("Original observation (160×210 RGB)")
plt.imshow(obs)
plt.axis("off")
plt.subplot(122)
plt.title("Preprocessed observation (88×80 greyscale)")
plt.imshow(img.reshape(88, 80), interpolation="nearest", cmap="gray")
plt.axis("off")
save_fig("preprocessing_plot")
plt.show()

# --- notebook cell 125 ---
reset_graph()

input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.variance_scaling_initializer()

def q_network(X_state, name):
    prev_layer = X_state / 128.0 # scale pixel intensities to the [-1.0, 1.0] range.
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

# --- notebook cell 126 ---
X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# --- notebook cell 127 ---
online_vars

# --- notebook cell 128 ---
learning_rate = 0.001
momentum = 0.95

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keepdims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# --- notebook cell 131 ---
class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0
        
    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen
    
    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size) # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

# --- notebook cell 132 ---
replay_memory_size = 500000
replay_memory = ReplayMemory(replay_memory_size)

# --- notebook cell 133 ---
def sample_memories(batch_size):
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

# --- notebook cell 134 ---
eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

# --- notebook cell 135 ---
n_steps = 4000000  # total number of training steps
training_start = 10000  # start training after 10,000 game iterations
training_interval = 4  # run a training step every 4 game iterations
save_steps = 1000  # save the model every 1,000 training steps
copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.99
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
checkpoint_path = "./my_dqn.ckpt"
done = True # env needs to be reset

# --- notebook cell 137 ---
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0

# --- notebook cell 139 ---
with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(
            iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
        if done: # game over, start again
            obs = env.reset()
            for skip in range(skip_start): # skip the start of each game
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        # Compute statistics for tracking progress (not shown in the book)
        total_max_q += q_values.max()
        game_length += 1
        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if iteration < training_start or iteration % training_interval != 0:
            continue # only train after warmup period and at regular intervals
        
        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})

        # Regularly copy the online DQN to the target DQN
        if step % copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)

# --- notebook cell 141 ---
frames = []
n_max_steps = 10000

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    obs = env.reset()
    for step in range(n_max_steps):
        state = preprocess_observation(obs)

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = np.argmax(q_values)

        # Online DQN plays
        obs, reward, done, info = env.step(action)

        img = env.render(mode="rgb_array")
        frames.append(img)

        if done:
            break

# --- notebook cell 142 ---
plot_animation(frames)

# --- notebook cell 146 ---
def preprocess_observation(obs):
    img = obs[34:194:2, ::2] # crop and downsize
    return np.mean(img, axis=2).reshape(80, 80) / 255.0

# --- notebook cell 147 ---
env = gym.make("Breakout-v0")
obs = env.reset()
for step in range(10):
    obs, _, _, _ = env.step(1)

img = preprocess_observation(obs)

# --- notebook cell 148 ---
plt.figure(figsize=(11, 7))
plt.subplot(121)
plt.title("Original observation (160×210 RGB)")
plt.imshow(obs)
plt.axis("off")
plt.subplot(122)
plt.title("Preprocessed observation (80×80 grayscale)")
plt.imshow(img, interpolation="nearest", cmap="gray")
plt.axis("off")
plt.show()

# --- notebook cell 150 ---
from collections import deque

def combine_observations_multichannel(preprocessed_observations):
    return np.array(preprocessed_observations).transpose([1, 2, 0])

def combine_observations_singlechannel(preprocessed_observations, dim_factor=0.5):
    dimmed_observations = [obs * dim_factor**index
                           for index, obs in enumerate(reversed(preprocessed_observations))]
    return np.max(np.array(dimmed_observations), axis=0)

n_observations_per_state = 3
preprocessed_observations = deque([], maxlen=n_observations_per_state)

obs = env.reset()
for step in range(10):
    obs, _, _, _ = env.step(1)
    preprocessed_observations.append(preprocess_observation(obs))

# --- notebook cell 151 ---
img1 = combine_observations_multichannel(preprocessed_observations)
img2 = combine_observations_singlechannel(preprocessed_observations)

plt.figure(figsize=(11, 7))
plt.subplot(121)
plt.title("Multichannel state")
plt.imshow(img1, interpolation="nearest")
plt.axis("off")
plt.subplot(122)
plt.title("Singlechannel state")
plt.imshow(img2, interpolation="nearest", cmap="gray")
plt.axis("off")
plt.show()

# --- notebook cell 157 ---
import gym

# --- notebook cell 158 ---
env = gym.make("BipedalWalker-v3")

# --- notebook cell 159 ---
obs = env.reset()

# --- notebook cell 160 ---
img = env.render(mode="rgb_array")

# --- notebook cell 161 ---
plt.imshow(img)
plt.axis("off")
plt.show()

# --- notebook cell 162 ---
obs

# --- notebook cell 164 ---
env.action_space

# --- notebook cell 165 ---
env.action_space.low

# --- notebook cell 166 ---
env.action_space.high

# --- notebook cell 168 ---
from itertools import product

# --- notebook cell 169 ---
possible_torques = np.array([-1.0, 0.0, 1.0])
possible_actions = np.array(list(product(possible_torques, possible_torques, possible_torques, possible_torques)))
possible_actions.shape

# --- notebook cell 170 ---
tf.reset_default_graph()

# 1. Specify the network architecture
n_inputs = env.observation_space.shape[0]  # == 24
n_hidden = 10
n_outputs = len(possible_actions) # == 625
initializer = tf.variance_scaling_initializer()

# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.selu,
                         kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs,
                         kernel_initializer=initializer)
outputs = tf.nn.softmax(logits)

# 3. Select a random action based on the estimated probabilities
action_index = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=-1)

# 4. Training
learning_rate = 0.01

y = tf.one_hot(action_index, depth=len(possible_actions))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# --- notebook cell 172 ---
def run_bipedal_walker(model_path=None, n_max_steps = 1000):
    env = gym.make("BipedalWalker-v3")
    frames = []
    with tf.Session() as sess:
        if model_path is None:
            init.run()
        else:
            saver.restore(sess, model_path)
        obs = env.reset()
        for step in range(n_max_steps):
            img = env.render(mode="rgb_array")
            frames.append(img)
            action_index_val = action_index.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            action = possible_actions[action_index_val]
            obs, reward, done, info = env.step(action[0])
            if done:
                break
    env.close()
    return frames

# --- notebook cell 173 ---
frames = run_bipedal_walker()
plot_animation(frames)

# --- notebook cell 175 ---
n_games_per_update = 10
n_max_steps = 1000
n_iterations = 1000
save_iterations = 10
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}/{}".format(iteration + 1, n_iterations), end="")
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_index_val, gradients_val = sess.run([action_index, gradients],
                                                           feed_dict={X: obs.reshape(1, n_inputs)})
                action = possible_actions[action_index_val]
                obs, reward, done, info = env.step(action[0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./my_bipedal_walker_pg.ckpt")

# --- notebook cell 176 ---
frames = run_bipedal_walker("./my_bipedal_walker_pg.ckpt")
plot_animation(frames)

# --- notebook cell 180 ---
import gym

env = gym.make('Pong-v0')
obs = env.reset()

# --- notebook cell 181 ---
obs.shape

# --- notebook cell 182 ---
env.action_space

# --- notebook cell 184 ---
# A helper function to run an episode of Pong. It's first argument should be a
# function which takes the observation of the environment and the current
# iteration and produces an action for the agent to take.

def run_episode(policy, n_max_steps=1000, frames_per_action=1):
    obs = env.reset()
    frames = []
    for i in range(n_max_steps):
        obs, reward, done, info = env.step(policy(obs, i))
        frames.append(env.render(mode='rgb_array'))
        if done:
            break
    return plot_animation(frames)

# --- notebook cell 185 ---
run_episode(lambda obs, i: np.random.randint(0, 5))

# --- notebook cell 187 ---
green_paddle_color = (92, 186, 92)
red_paddle_color = (213, 130, 74)
background_color = (144, 72, 17)
ball_color = (236, 236, 236)

def preprocess_observation(obs):
    img = obs[34:194:2, ::2].reshape(-1, 3)
    tmp = np.full(shape=(80 * 80), fill_value=0.0, dtype=np.float32)
    for i, c in enumerate(img):
        c = tuple(c)
        if c in {green_paddle_color, red_paddle_color, ball_color}:
            tmp[i] = 1.0
        else:
            tmp[i] = 0.0
    return tmp.reshape(80, 80)

# --- notebook cell 188 ---
obs = env.reset()
for _ in range(25):
    obs, _, _, _ = env.step(0)

plt.figure(figsize=(11, 7))
plt.subplot(121)
plt.title('Original Observation (160 x 210 RGB)')
plt.imshow(obs)
plt.axis('off')
plt.subplot(122)
plt.title('Preprocessed Observation (80 x 80 Grayscale)')
plt.imshow(preprocess_observation(obs), interpolation='nearest', cmap='gray')
plt.axis('off')
plt.show()

# --- notebook cell 189 ---
def combine_observations(preprocess_observations, dim_factor=0.75):
    dimmed = [obs * (dim_factor ** idx)
              for idx, obs in enumerate(reversed(preprocess_observations))]
    return np.max(np.array(dimmed), axis=0)

# --- notebook cell 190 ---
n_observations_per_state = 3

obs = env.reset()
for _ in range(20):
    obs, _, _, _ = env.step(0)

preprocess_observations = []
for _ in range(n_observations_per_state):
    obs, _, _, _ = env.step(2)
    preprocess_observations.append(preprocess_observation(obs))

img = combine_observations(preprocess_observations)

plt.figure(figsize=(6, 6))
plt.title('Combined Observations as a Single State')
plt.imshow(img, interpolation='nearest', cmap='gray')
plt.axis('off')
plt.show()

# --- notebook cell 192 ---
reset_graph()

input_width = 80
input_height = 80
input_channels = 1

conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [9, 5, 3]
conv_kernel_strides = [4, 2, 1]
conv_paddings = ['VALID'] * 3
conv_activation = [tf.nn.relu] * 3

n_hidden_in = 5 * 5 * 64
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n

he_init = tf.contrib.layers.variance_scaling_initializer()

# --- notebook cell 194 ---
def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
            conv_n_maps, conv_kernel_sizes, conv_kernel_strides, conv_paddings,
            conv_activation):
            prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps,
                                          kernel_size=kernel_size,
                                          strides=strides, padding=padding,
                                          activation=activation,
                                          kernel_initializer=he_init)
        flattened = tf.reshape(prev_layer, [-1, n_hidden_in])
        hidden = tf.layers.dense(flattened, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=he_init)
        outputs = tf.layers.dense(hidden, n_outputs, kernel_initializer=he_init)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

# --- notebook cell 195 ---
# Starting the DQN definition.

X_state = tf.placeholder(tf.float32, shape=(None, input_height, input_width,
                                            input_channels))
online_q_values, online_vars = q_network(X_state, 'q_networks/online')
target_q_values, target_vars = q_network(X_state, 'q_networks/target')
copy_ops = [var.assign(online_vars[name]) for name, var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# --- notebook cell 196 ---
# Defining the training objective.

learning_rate = 1e-3
momentum = 0.95

with tf.variable_scope('training') as scope:
    X_action = tf.placeholder(tf.int32, shape=(None,))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    Q_target = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                             axis=1, keepdims=True)
    error = tf.abs(y - Q_target)
    loss = tf.reduce_mean(tf.square(error))

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum,
                                           use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

# --- notebook cell 197 ---
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# --- notebook cell 199 ---
class ReplayMemory(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.index += 1
        self.index %= self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def sample(self, batch_size):
        return self.buf[np.random.randint(self.length, size=batch_size)]

# --- notebook cell 200 ---
replay_size = 200000
replay_memory = ReplayMemory(replay_size)

# --- notebook cell 201 ---
def sample_memories(batch_size):
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], \
         cols[4].reshape(-1, 1)

# --- notebook cell 203 ---
eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 6000000

def epsilon_greedy(q_values, step):
    epsilon = min(eps_min,
                  eps_max - ((eps_max - eps_min) * (step / eps_decay_steps)))
    if np.random.random() < epsilon:
        return np.random.randint(n_outputs)
    return np.argmax(q_values)

# --- notebook cell 205 ---
n_steps = 10000000
training_start = 100000
training_interval = 4
save_steps = 1000
copy_steps = 10000
discount_rate = 0.95
skip_start = 20
batch_size = 50
iteration = 0
done = True  # To reset the environment at the start.

loss_val = np.infty
game_length = 0
total_max_q = 0.0
mean_max_q = 0.0

checkpoint_path = "./pong_dqn.ckpt"

# --- notebook cell 206 ---
# Utility function to get the environment state for the model.

def perform_action(action):
    preprocess_observations = []
    total_reward = 0.0
    for i in range(3):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            for _ in range(i, 3):
                preprocess_observations.append(preprocess_observation(obs))
            break
        else:
            preprocess_observations.append(preprocess_observation(obs))
    return combine_observations(preprocess_observations).reshape(80, 80, 1), \
        total_reward, done

# --- notebook cell 207 ---
# Main training loop

with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + '.index'):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        print('\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}'
              '\tMean Max-Q {:5f}   '.format(
                  iteration, step, n_steps, 100 * step / n_steps, loss_val,
                  mean_max_q),
              end='')
        if done:
            obs = env.reset()
            for _ in range(skip_start):
                obs, reward, done, info = env.step(0)
            state, reward, done = perform_action(0)

        # Evaluate the next action for the agent.
        q_values = online_q_values.eval(
            feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # The online DQN plays the game.
        next_state, reward, done = perform_action(action)

        # Save the result in the ReplayMemory.
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        # Compute statistics which help us monitor how training is going.
        total_max_q += q_values.max()
        game_length += 1
        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        # Only train after the warmup rounds and only every few rounds.
        if iteration < training_start or iteration % training_interval != 0:
            continue

        # Sample memories from the reply memory.
        X_state_val, X_action_val, rewards, X_next_state_val, continues = \
            sample_memories(batch_size)
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN.
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val,
            X_action: X_action_val,
            y: y_val,
        })

        # Regularly copy the online DQN to the target DQN.
        if step % copy_steps == 0:
            copy_online_to_target.run()

        # Regularly save the model.
        if step and step % save_steps == 0:
            saver.save(sess, checkpoint_path)

# --- notebook cell 208 ---
preprocess_observations = []

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    def dqn_policy(obs, i):
        if len(preprocess_observations) < 3:
            preprocess_observations.append(preprocess_observation(obs))
            if len(preprocess_observations) == 3:
                state = combine_observations(preprocess_observations)
                q_values = online_q_values.eval(
                    feed_dict={X_state: [state.reshape(80, 80, 1)]})
                dqn_policy.cur_action = np.argmax(q_values)
            return dqn_policy.cur_action
        preprocess_observations[i % 3] = preprocess_observation(obs)
        if i % 3 == 2:
            state = combine_observations(preprocess_observations)
            q_values = online_q_values.eval(
                feed_dict={X_state: [state.reshape(80, 80, 1)]})
            dqn_policy.cur_action = np.argmax(q_values)
        return dqn_policy.cur_action
    dqn_policy.cur_action = 0

    html = run_episode(dqn_policy, n_max_steps=10000)
html