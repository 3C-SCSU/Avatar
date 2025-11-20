from typing import Any, Dict, Tuple
import numpy as np
import tensorflow as tf
import pickle


class GaussianNBTF:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = float(var_smoothing)
        self.class_prior_ = None  # (C,)
        self.theta_ = None        # (C,D)
        self.var_ = None          # (C,D)
        self.n_classes_ = None
        self.n_features_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y, dtype=tf.int32)
        n_samples = tf.shape(X_tf)[0]
        n_features = tf.shape(X_tf)[1]
        num_classes = tf.cast(tf.reduce_max(y_tf) + 1, tf.int32)

        counts = tf.math.bincount(y_tf, minlength=num_classes, maxlength=num_classes, dtype=tf.int32)
        counts_f = tf.cast(tf.maximum(counts, 1), tf.float32)  # avoid div by zero

        sums = tf.math.unsorted_segment_sum(X_tf, y_tf, num_classes)              # (C,D)
        sums2 = tf.math.unsorted_segment_sum(tf.square(X_tf), y_tf, num_classes)  # (C,D)

        means = sums / tf.expand_dims(counts_f, 1)
        vars_ = (sums2 / tf.expand_dims(counts_f, 1)) - tf.square(means)
        vars_ = tf.maximum(vars_, self.var_smoothing)

        priors = tf.cast(counts, tf.float32) / tf.cast(n_samples, tf.float32)

        self.theta_ = means.numpy()
        self.var_ = vars_.numpy()
        self.class_prior_ = priors.numpy()
        self.n_classes_ = int(num_classes.numpy())
        self.n_features_ = int(n_features.numpy())
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)  # (N,D)
        mu = tf.convert_to_tensor(self.theta_, dtype=tf.float32)  # (C,D)
        var = tf.convert_to_tensor(self.var_, dtype=tf.float32)    # (C,D)
        log_prior = tf.math.log(tf.convert_to_tensor(self.class_prior_, dtype=tf.float32) + 1e-12)  # (C,)

        # log P(x|y=c) = -0.5 * sum( log(2*pi*var_c) + (x-mu_c)^2 / var_c )
        const_term = -0.5 * tf.reduce_sum(tf.math.log(2.0 * np.pi * var), axis=1)  # (C,)
        diff = X_tf[:, tf.newaxis, :] - mu[tf.newaxis, :, :]                      # (N,C,D)
        quad = -0.5 * tf.reduce_sum(tf.square(diff) / var[tf.newaxis, :, :], axis=2)  # (N,C)
        log_lik = quad + const_term[tf.newaxis, :]                                 # (N,C)
        return (log_lik + log_prior[tf.newaxis, :]).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logp = self.predict_log_proba(X)
        m = np.max(logp, axis=1, keepdims=True)
        p = np.exp(logp - m)
        p /= np.sum(p, axis=1, keepdims=True)
        return p

    def predict(self, X: np.ndarray) -> np.ndarray:
        logp = self.predict_log_proba(X)
        return np.argmax(logp, axis=1).astype(np.int32)


def save_model(path: str, model: GaussianNBTF, meta: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {
        'model': {
            'var_smoothing': float(model.var_smoothing),
            'class_prior_': model.class_prior_,
            'theta_': model.theta_,
            'var_': model.var_,
            'n_classes_': model.n_classes_,
            'n_features_': model.n_features_,
        },
        'meta': meta or {},
        'format': 'GaussianNBTF-pickle-v1'
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str) -> Tuple[GaussianNBTF, Dict[str, Any]]:
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    m = GaussianNBTF(var_smoothing=payload['model'].get('var_smoothing', 1e-9))
    m.class_prior_ = np.array(payload['model']['class_prior_'], dtype=np.float32)
    m.theta_ = np.array(payload['model']['theta_'], dtype=np.float32)
    m.var_ = np.array(payload['model']['var_'], dtype=np.float32)
    m.n_classes_ = int(payload['model']['n_classes_'])
    m.n_features_ = int(payload['model']['n_features_'])
    return m, payload.get('meta', {})

