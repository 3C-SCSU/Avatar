import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax
import time
import pandas as pd


MASTER_KEY = random.PRNGKey(42)
LEARNING_RATE = 0.005
L2_REG_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 200

# load data 

X = concatenated_df_files.drop(columns=['label']).values.astype('float32')
Y = concatenated_df_files['label'].values.astype('int32')

num_samples = X.shape[0]
MASTER_KEY, split_key = random.split(MASTER_KEY)
perm = random.permutation(split_key, num_samples)

split_idx = int(0.8 * num_samples)
train_idx, test_idx = perm[:split_idx], perm[split_idx:]

X_train = jnp.array(X[train_idx])
Y_train = jnp.array(Y[train_idx])
X_test = jnp.array(X[test_idx])
Y_test = jnp.array(Y[test_idx])

# Standardize features

mean = jnp.mean(X_train, axis=0)
std = jnp.std(X_train, axis=0) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

num_features = X_train.shape[1]
num_classes = len(jnp.unique(Y_train))

# ONE-HOT ENCODE TARGETS to help the model to tread target value as some
Y_train_onehot = jnp.eye(num_classes)[Y_train]
Y_test_onehot = jnp.eye(num_classes)[Y_test]

print(f" Data Prepared. X_train: {X_train.shape}, num_classes: {num_classes}")

# init params function

def init_params(input_layer, layer1, layer2, output_layer, key):
    keys = jax.random.split(key, 4)

    w1 = jax.random.normal(keys[0], (input_layer, layer1))
    b1 = jnp.zeros((layer1,))
    w2 = jax.random.normal(keys[1], (layer1, layer2))
    b2 = jnp.zeros((layer2,))
    w3 = jax.random.normal(keys[2], (layer2, output_layer))
    b3 = jnp.zeros((output_layer,))
   
    return [(w1, b1), (w2, b2), (w3, b3)]

# number of neuron on each layer

layer_sizes = [num_features, 128, 64, num_classes]
MASTER_KEY, init_key = random.split(MASTER_KEY)

params = init_params(
    layer_sizes[0],
    layer_sizes[1],
    layer_sizes[2],
    layer_sizes[3],
    
    init_key
)

# forward function

def forward(params, x):
    for W, b in params[:-1]:
        x = jax.nn.relu(jnp.dot(x, W) + b)
    W, b = params[-1]
    logits = jnp.dot(x, W) + b
    return logits

# loss function

def loss_fn(params, x, y_onehot):
    logits = forward(params, x)
    probabilities = jax.nn.softmax(logits)

    # Cross-entropy loss
    loss = -jnp.mean(jnp.sum(y_onehot * jnp.log(probabilities + 1e-8), axis=-1))


    l2_loss = L2_REG_RATE * sum(jnp.sum(W**2) for W, b in params)

    return loss + l2_loss

batch_loss_fn = vmap(loss_fn, in_axes=(None, 0, 0))

# optimizer

optimizer = optax.adamw(LEARNING_RATE, weight_decay=L2_REG_RATE)
opt_state = optimizer.init(params)

# training step using git
@jax.jit
def train_step(params, x_batch, y_batch_onehot, opt_state):
    grads = grad(lambda p: jnp.mean(batch_loss_fn(p, x_batch, y_batch_onehot)))(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# accuracy function

def accuracy(params, X, Y_onehot):
    logits = forward(params, X)
    preds = jnp.argmax(logits, axis=1)
    targets = jnp.argmax(Y_onehot, axis=1)
    return jnp.mean(preds == targets)

# batch loader

def batch_loader(X, Y_onehot, batch_size, key):
    perm = jax.random.permutation(key, len(X))
    X_shuffled, Y_shuffled = X[perm], Y_onehot[perm]

    for i in range(0, len(X), batch_size):
        yield X_shuffled[i:i+batch_size], Y_shuffled[i:i+batch_size]

# training loop

print(f" Starting training for {NUM_EPOCHS} epochs...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    MASTER_KEY, shuffle_key = random.split(MASTER_KEY)

    for x_batch, y_batch_onehot in batch_loader(X_train, Y_train_onehot, BATCH_SIZE, shuffle_key):
        params, opt_state = train_step(params, x_batch, y_batch_onehot, opt_state)

    if epoch % 10 == 0:
        train_acc = accuracy(params, X_train, Y_train_onehot)
        test_acc = accuracy(params, X_test, Y_test_onehot)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

end_time = time.time()
print(f"Training Complete in {end_time - start_time:.2f} seconds.")

# final test result
final_test_acc = accuracy(params, X_test, Y_test_onehot)
print(f"Final Test Accuracy: {final_test_acc:.4f}")
