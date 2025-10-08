import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd

# Load  dataset 

X = concatenated_df_files.drop(columns=['label']).values.astype('float32')
Y = concatenated_df_files['label'].values.astype('int32')

X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train = jnp.array(scaler.fit_transform(X_train_np))
X_test = jnp.array(scaler.transform(X_test_np))
Y_train = jnp.array(Y_train_np)
Y_test = jnp.array(Y_test_np)

num_features = X_train.shape[1]
num_classes = len(jnp.unique(Y_train))
print(f"Data Prepared. X_train: {X_train.shape}, num_classes: {num_classes}")


def init_params(layer_sizes, key):
    params = []
    keys = random.split(key, len(layer_sizes))
    for i in range(len(layer_sizes)-1):
        w_key, b_key = random.split(keys[i])
        W = random.normal(w_key, (layer_sizes[i], layer_sizes[i+1])) * jnp.sqrt(2.0 / layer_sizes[i])
        b = jnp.zeros((layer_sizes[i+1],))
        params.append((W, b))
    return params

layer_sizes = [num_features, 128, 64, num_classes]  
key = random.PRNGKey(42)
params = init_params(layer_sizes, key)


# Forward pass

def relu(x): return jnp.maximum(0, x)
def softmax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def forward(params, x):
    for W, b in params[:-1]:
        x = relu(jnp.dot(x, W) + b)
    W, b = params[-1]
    return softmax(jnp.dot(x, W) + b)



def one_hot(y, num_classes):
    return jnp.eye(num_classes)[y]

def loss_fn(params, x, y):
    y_pred = forward(params, x)
    y_true = one_hot(y, num_classes)
    return -jnp.sum(y_true * jnp.log(jnp.clip(y_pred, 0.00000001, 1.0)))

batch_loss_fn = vmap(loss_fn, in_axes=(None, 0, 0))


# Optimizer

learning_rate = 0.001 # leaning rate
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

@jit
def update(params, x, y, opt_state):
    grads = grad(lambda p: jnp.mean(batch_loss_fn(p, x, y)))(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


batch_size = 128
num_batches = X_train.shape[0] // batch_size
num_epochs = 50  # fewer epochs for CPU
best_val_acc = 0.0
patience = 5
patience_counter = 0

start_time = time.time()
for epoch in range(num_epochs):
    # Shuffle data
    perm = jax.random.permutation(key, X_train.shape[0])
    X_train = X_train[perm]
    Y_train = Y_train[perm]
    
    for i in range(num_batches):
        x_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = Y_train[i*batch_size:(i+1)*batch_size]
        params, opt_state = update(params, x_batch, y_batch, opt_state)
    
    # Validation
    y_pred = jnp.argmax(vmap(lambda x: forward(params, x))(X_test), axis=1)
    val_acc = jnp.mean(y_pred == Y_test)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params
        patience_counter = 0
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
    
    if patience_counter >= patience:
        print("Early stopping triggered.")
        params = best_params
        break

end_time = time.time()
print(f"Training complete in {end_time - start_time:.2f} seconds.")


y_pred_final = jnp.argmax(vmap(lambda x: forward(params, x))(X_test), axis=1)
accuracy = jnp.mean(y_pred_final == Y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
