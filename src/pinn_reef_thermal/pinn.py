"""
Production PINN for reef thermal reconstruction.

Key features:
  1. Hard BC with real satellite SST: T(z,t) = SST(t) + (z/z_max)·T_scale·f_nn(z,t)
  2. Modified MLP with multiplicative gating (Wang et al. 2021)
  3. Decomposed time encoding: diurnal (hour) + seasonal (day_of_year) + trend
  4. UTC-corrected solar source term
  5. Non-dimensionalised PDE residual
  6. Mini-batch training with cosine LR annealing
"""

import time as time_module

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap


def init_params(layers, key):
    """Xavier initialization for MLP parameters."""
    params = []
    for i in range(len(layers) - 1):
        key, subkey = random.split(key)
        w = random.normal(subkey, (layers[i], layers[i + 1])) * jnp.sqrt(2.0 / layers[i])
        b = jnp.zeros(layers[i + 1])
        params.append((w, b))
    return params


def init_modified_mlp_params(input_dim, hidden_dim, n_hidden, key):
    """Initialise Modified MLP params (Wang et al. 2021)."""
    params = {}
    # Encoding layers U and V
    key, k1, k2 = random.split(key, 3)
    params['Wu'] = random.normal(k1, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim)
    params['bu'] = jnp.zeros(hidden_dim)
    params['Wv'] = random.normal(k2, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim)
    params['bv'] = jnp.zeros(hidden_dim)
    # Hidden layers
    layers = []
    # First hidden layer: input_dim -> hidden_dim
    key, subkey = random.split(key)
    layers.append((
        random.normal(subkey, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim),
        jnp.zeros(hidden_dim)
    ))
    # Remaining hidden layers: hidden_dim -> hidden_dim
    for _ in range(n_hidden - 1):
        key, subkey = random.split(key)
        layers.append((
            random.normal(subkey, (hidden_dim, hidden_dim)) * jnp.sqrt(2.0 / hidden_dim),
            jnp.zeros(hidden_dim)
        ))
    params['layers'] = layers
    # Output layer
    key, subkey = random.split(key)
    params['Wout'] = random.normal(subkey, (hidden_dim, 1)) * jnp.sqrt(2.0 / hidden_dim)
    params['bout'] = jnp.zeros(1)
    return params


def _get_activation(name):
    """Return activation function by name."""
    if name == 'swish':
        return jax.nn.swish
    return jnp.tanh


def modified_mlp_forward(params, x, activation='tanh'):
    """Modified MLP with multiplicative interactions (Wang et al. 2021)."""
    act = _get_activation(activation)
    U = act(x @ params['Wu'] + params['bu'])
    V = act(x @ params['Wv'] + params['bv'])
    H = x
    for w, b in params['layers']:
        H = act(H @ w + b)
        H = (1 - H) * U + H * V
    H = H @ params['Wout'] + params['bout']
    return H


def mlp_forward(params, x, activation='tanh'):
    """Standard MLP forward pass."""
    act = _get_activation(activation)
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = act(x)
    return x


class ReefPINN:
    """
    Physics-Informed Neural Network for reef thermal reconstruction.

    Hard BC enforcement:
      T(z,t) = T_surface(t) + (z/z_max) · T_scale · f_nn(z,t)
      This guarantees T(0,t) = T_surface(t) exactly.
    """

    def __init__(self, physics, layers=None, learn_kappa=False, learn_Kd=False,
                 n_fourier=4, use_hard_bc=True, seed=42,
                 init_kappa=1e-4, init_Kd=0.2, pde_chunk_size=2000,
                 kappa_mode='constant', pde_depth_scale=None,
                 w_bc_bottom=0.0, learn_w=False,
                 t_sst=None, T_sst=None,
                 use_modified_mlp=False, hidden_dim=128, n_hidden=4,
                 t_start_utc_hour=0.0, start_day_of_year=0.0,
                 T_scale=2.0, activation='tanh', grad_clip=0.0):
        self.physics = physics
        self.learn_kappa = learn_kappa
        self.learn_Kd = learn_Kd
        self.learn_w = learn_w
        self.use_hard_bc = use_hard_bc
        self.n_fourier = n_fourier
        self.pde_chunk_size = pde_chunk_size
        self.kappa_mode = kappa_mode
        self.pde_depth_scale = pde_depth_scale
        self.w_bc_bottom = w_bc_bottom
        self.use_modified_mlp = use_modified_mlp
        self.activation = activation
        self.grad_clip = grad_clip
        self.t_start_utc_hour = t_start_utc_hour
        self.start_doy = start_day_of_year

        self.z_max = physics.z_max
        self.t_max = physics.t_max
        self.T_mean = physics.T_mean
        self.T_scale = T_scale

        # Real SST data for hard BC interpolation
        if t_sst is not None and T_sst is not None and len(t_sst) > 0:
            self.t_sst = jnp.array(t_sst, dtype=jnp.float32)
            self.T_sst = jnp.array(T_sst, dtype=jnp.float32)
            self.has_sst = True
        else:
            self.t_sst = None
            self.T_sst = None
            self.has_sst = False

        # Input: z_norm + t_norm + diurnal (hour-based) + annual (day-of-year-based)
        n_annual = 3  # 3 annual harmonics
        self.n_annual = n_annual
        input_dim = 1 + 1 + 2 * n_fourier + 2 * n_annual  # z_norm + t_norm + diurnal + annual

        key = random.PRNGKey(seed)

        if use_modified_mlp:
            self.nn_params = init_modified_mlp_params(input_dim, hidden_dim, n_hidden, key)
            # Count "layers" for pack/unpack (Modified MLP is a single dict)
            self.n_nn_layers = 1  # Modified MLP is one param item
        else:
            if layers is None:
                layers = [input_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, 1]
            else:
                layers = [input_dim] + layers[1:]
            self.layer_sizes = layers
            self.n_nn_layers = len(layers) - 1
            self.nn_params = init_params(layers, key)

        # Learnable physics parameters
        self.log_kappa = jnp.log(jnp.array(init_kappa)) if learn_kappa else None
        self.log_Kd = jnp.log(jnp.array(init_Kd)) if learn_Kd else None
        self.alpha = jnp.array(0.0) if kappa_mode == 'log_linear' else None
        self.w_vel = jnp.array(0.0) if learn_w else None

        self.trained_params = None

    def _pack_params(self):
        """Pack all trainable parameters into a flat list."""
        if self.use_modified_mlp:
            p = [self.nn_params]  # Single dict item
        else:
            p = list(self.nn_params)
        if self.learn_kappa:
            p.append(self.log_kappa)
        if self.learn_Kd:
            p.append(self.log_Kd)
        if self.kappa_mode == 'log_linear':
            p.append(self.alpha)
        if self.learn_w:
            p.append(self.w_vel)
        return p

    def _unpack_params(self, params):
        """Unpack parameter list. Returns (nn_params, log_k, log_kd, alpha, w_vel)."""
        if self.use_modified_mlp:
            nn = params[0]  # Single dict
            idx = 1
        else:
            nn = list(params[:self.n_nn_layers])
            idx = self.n_nn_layers
        log_k = params[idx] if self.learn_kappa else None
        if self.learn_kappa:
            idx += 1
        log_kd = params[idx] if self.learn_Kd else None
        if self.learn_Kd:
            idx += 1
        alpha = params[idx] if self.kappa_mode == 'log_linear' else None
        if self.kappa_mode == 'log_linear':
            idx += 1
        w_vel = params[idx] if self.learn_w else None
        return nn, log_k, log_kd, alpha, w_vel

    def _encode_input(self, z, t):
        """Encode (z, t) with decomposed time features."""
        z_norm = 2.0 * z / self.z_max - 1.0
        t_norm = 2.0 * t / self.t_max - 1.0

        # Convert to local time for physically meaningful features
        local_hour = ((t / 3600.0) + self.t_start_utc_hour + self.physics.utc_offset) % 24.0
        day_of_year = ((t / 86400.0) + self.start_doy + self.physics.utc_offset / 24.0) % 365.25

        features = [jnp.array([z_norm]), jnp.array([t_norm])]

        # Diurnal harmonics (keyed to local hour, not raw t)
        for k in range(1, self.n_fourier + 1):
            features.append(jnp.array([jnp.sin(2 * jnp.pi * k * local_hour / 24.0)]))
            features.append(jnp.array([jnp.cos(2 * jnp.pi * k * local_hour / 24.0)]))

        # Annual harmonics (keyed to day of year)
        for k in range(1, self.n_annual + 1):
            features.append(jnp.array([jnp.sin(2 * jnp.pi * k * day_of_year / 365.25)]))
            features.append(jnp.array([jnp.cos(2 * jnp.pi * k * day_of_year / 365.25)]))

        return jnp.concatenate(features)

    def _predict_scalar(self, params, z, t):
        """Predict T at a single (z, t) point."""
        nn_params, log_k, log_kd, alpha, w_vel = self._unpack_params(params)
        x = self._encode_input(z, t)

        if self.use_modified_mlp:
            raw = modified_mlp_forward(nn_params, x, self.activation)[0]
        else:
            raw = mlp_forward(nn_params, x, self.activation)[0]

        if self.use_hard_bc:
            if self.has_sst:
                T_surf = jnp.interp(t, self.t_sst, self.T_sst)
            else:
                local_hour = ((t / 3600.0) + self.t_start_utc_hour + self.physics.utc_offset) % 24.0
                T_surf = self.physics.T_mean + self.physics.T_amp * jnp.sin(
                    2 * jnp.pi * (local_hour - 8) / 24)
            return T_surf + (z / self.z_max) * self.T_scale * raw
        else:
            return self.T_mean + self.T_scale * raw

    def _physics_residual(self, params, z, t):
        """PDE residual: ∂T/∂t - κ·∂²T/∂z² - w·∂T/∂z - S(z,t), non-dimensionalised."""
        nn_params, log_k, log_kd, alpha, w_vel = self._unpack_params(params)

        if self.kappa_mode == 'log_linear' and alpha is not None:
            log_k_base = log_k if log_k is not None else jnp.log(jnp.array(self.physics.kappa))
            kappa = jnp.exp(log_k_base + alpha * z)
        else:
            kappa = jnp.exp(log_k) if log_k is not None else self.physics.kappa
        Kd = jnp.exp(log_kd) if log_kd is not None else self.physics.Kd

        pred = lambda z_, t_: self._predict_scalar(params, z_, t_)

        dT_dt = grad(pred, argnums=1)(z, t)
        dT_dz = grad(pred, argnums=0)(z, t)
        d2T_dz2 = grad(grad(pred, argnums=0), argnums=0)(z, t)

        advection = w_vel * dT_dz if w_vel is not None else 0.0

        # Solar source (local time)
        local_hour = ((t / 3600.0) + self.t_start_utc_hour + self.physics.utc_offset) % 24.0
        I_surf = self.physics.Q_max * jnp.maximum(0.0, jnp.sin(jnp.pi * (local_hour - 6.0) / 12.0))
        S = (I_surf * Kd / self.physics.rho_cp) * jnp.exp(-Kd * z)

        kappa_ref = self.physics.kappa
        scale = self.z_max ** 2 / kappa_ref
        return (dT_dt - kappa * d2T_dz2 - advection - S) * scale

    def _loss_fn(self, params, z_pde, t_pde, z_data, t_data, T_data,
                 z_bc, t_bc, T_bc, w_pde, w_data, w_bc):
        """Total loss with PDE, data, and optional BC terms."""
        pred = lambda z_, t_: self._predict_scalar(params, z_, t_)
        res = lambda z_, t_: self._physics_residual(params, z_, t_)

        # PDE loss — chunked (pad to avoid double-counting from dynamic_slice clipping)
        pde_depth_scale = self.pde_depth_scale
        chunk = self.pde_chunk_size
        n_pde = z_pde.shape[0]
        if chunk > 0 and n_pde > chunk:
            remainder = n_pde % chunk
            if remainder > 0:
                pad_n = chunk - remainder
                z_pde_c = jnp.concatenate([z_pde, z_pde[:pad_n]])
                t_pde_c = jnp.concatenate([t_pde, t_pde[:pad_n]])
            else:
                pad_n = 0
                z_pde_c = z_pde
                t_pde_c = t_pde
            n_padded = z_pde_c.shape[0]
            n_chunks = n_padded // chunk
            sum_weighted = jnp.array(0.0)
            sum_weights = jnp.array(0.0)
            for i in range(n_chunks):
                s = i * chunk
                z_ch = jax.lax.dynamic_slice(z_pde_c, (s,), (chunk,))
                t_ch = jax.lax.dynamic_slice(t_pde_c, (s,), (chunk,))
                r_chunk = vmap(res)(z_ch, t_ch)
                if pde_depth_scale is not None:
                    w_depth = jnp.exp(-z_ch / pde_depth_scale)
                else:
                    w_depth = jnp.ones_like(z_ch)
                # Mask out padded points in the last chunk
                if pad_n > 0 and i == n_chunks - 1:
                    real_count = chunk - pad_n
                    mask = jnp.arange(chunk) < real_count
                    w_depth = w_depth * mask
                sum_weighted = sum_weighted + jnp.sum(w_depth * r_chunk ** 2)
                sum_weights = sum_weights + jnp.sum(w_depth)
            pde_loss = sum_weighted / jnp.maximum(sum_weights, 1.0)
        else:
            residuals = vmap(res)(z_pde, t_pde)
            if pde_depth_scale is not None:
                w_depth = jnp.exp(-z_pde / pde_depth_scale)
                pde_loss = jnp.sum(w_depth * residuals ** 2) / jnp.sum(w_depth)
            else:
                pde_loss = jnp.mean(residuals ** 2)

        # Data loss
        T_pred = vmap(pred)(z_data, t_data)
        data_loss = jnp.mean((T_pred - T_data) ** 2)

        # BC loss
        if self.use_hard_bc:
            bc_loss = jnp.array(0.0)
        else:
            T_pred_bc = vmap(pred)(z_bc, t_bc)
            bc_loss = jnp.mean((T_pred_bc - T_bc) ** 2)

        # Bottom Neumann BC — sample random time points (not always the same subset)
        if self.w_bc_bottom > 0:
            dT_dz_fn = grad(lambda z_, t_: self._predict_scalar(params, z_, t_), argnums=0)
            n_bot = min(200, n_pde)
            # Use evenly-spaced indices across full PDE array for diversity
            bot_indices = jnp.linspace(0, n_pde - 1, n_bot).astype(jnp.int32)
            t_bot = t_pde[bot_indices]
            z_bot = jnp.full_like(t_bot, self.z_max * 0.99)
            dTdz_vals = vmap(dT_dz_fn)(z_bot, t_bot)
            bc_bottom_loss = jnp.mean(dTdz_vals ** 2)
        else:
            bc_bottom_loss = jnp.array(0.0)

        total = (w_pde * pde_loss + w_data * data_loss + w_bc * bc_loss
                 + self.w_bc_bottom * bc_bottom_loss)
        return total, (pde_loss, data_loss, bc_loss)

    def train(self, z_pde, t_pde, z_data, t_data, T_data,
              z_bc, t_bc, T_bc, n_epochs=10000, lr=1e-3,
              w_pde=1.0, w_data=10.0, w_bc=10.0, print_every=500,
              resample_pde_every=0, n_pde_points=0,
              lr_physics=None, w_pde_schedule=None,
              batch_size=0, lr_schedule='constant'):
        """Train with Adam optimiser."""
        params = self._pack_params()

        if lr_physics is None:
            lr_physics = lr

        # Build base LR tree (will be scaled by lr_scale each epoch)
        n_nn = 1 if self.use_modified_mlp else self.n_nn_layers
        base_lr_list = []
        for i, p in enumerate(params):
            rate = 1.0 if i < n_nn else (lr_physics / lr if lr > 0 else 1.0)
            base_lr_list.append(jax.tree.map(lambda x: jnp.ones_like(x) * rate, p))

        # Adam state
        m = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        v = jax.tree.map(lambda p: jnp.zeros_like(p), params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def _get_w_pde(epoch):
            if w_pde_schedule is None:
                return w_pde
            for i in range(len(w_pde_schedule) - 1):
                e0, w0 = w_pde_schedule[i]
                e1, w1 = w_pde_schedule[i + 1]
                if e0 <= epoch < e1:
                    frac = (epoch - e0) / (e1 - e0)
                    return w0 + frac * (w1 - w0)
            return w_pde_schedule[-1][1]

        def _get_lr_scale(epoch):
            if lr_schedule == 'cosine':
                return max(0.01, 0.5 * (1 + np.cos(np.pi * epoch / n_epochs)))
            return 1.0

        # Gradient clipping helper
        grad_clip_val = self.grad_clip

        def _clip_grads(grads):
            if grad_clip_val <= 0:
                return grads
            leaves = jax.tree.leaves(grads)
            total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
            scale = jnp.minimum(1.0, grad_clip_val / (total_norm + 1e-8))
            return jax.tree.map(lambda g: g * scale, grads)

        # Step function with data in closure, lr_scale as scalar
        @jit
        def step(params, m, v, epoch, z_pde_, t_pde_, w_pde_cur, lr_scale):
            (loss, aux), grads = jax.value_and_grad(
                self._loss_fn, has_aux=True
            )(params, z_pde_, t_pde_, z_data, t_data, T_data,
              z_bc, t_bc, T_bc, w_pde_cur, w_data, w_bc)
            grads = _clip_grads(grads)

            m_new = jax.tree.map(lambda mi, gi: beta1 * mi + (1 - beta1) * gi, m, grads)
            v_new = jax.tree.map(lambda vi, gi: beta2 * vi + (1 - beta2) * gi ** 2, v, grads)

            m_hat = jax.tree.map(lambda mi: mi / (1 - beta1 ** (epoch + 1)), m_new)
            v_hat = jax.tree.map(lambda vi: vi / (1 - beta2 ** (epoch + 1)), v_new)

            # Scale base_lr_list by lr * lr_scale
            actual_lr = lr * lr_scale
            params_new = jax.tree.map(
                lambda p, mh, vh, lri: p - actual_lr * lri * mh / (jnp.sqrt(vh) + eps),
                params, m_hat, v_hat, base_lr_list
            )
            return params_new, m_new, v_new, loss, aux

        # Mini-batch step (data passed as arguments, different JIT trace)
        @jit
        def step_batch(params, m, v, epoch, z_pde_, t_pde_, w_pde_cur, lr_scale,
                       z_d, t_d, T_d):
            (loss, aux), grads = jax.value_and_grad(
                self._loss_fn, has_aux=True
            )(params, z_pde_, t_pde_, z_d, t_d, T_d,
              z_bc, t_bc, T_bc, w_pde_cur, w_data, w_bc)
            grads = _clip_grads(grads)

            m_new = jax.tree.map(lambda mi, gi: beta1 * mi + (1 - beta1) * gi, m, grads)
            v_new = jax.tree.map(lambda vi, gi: beta2 * vi + (1 - beta2) * gi ** 2, v, grads)

            m_hat = jax.tree.map(lambda mi: mi / (1 - beta1 ** (epoch + 1)), m_new)
            v_hat = jax.tree.map(lambda vi: vi / (1 - beta2 ** (epoch + 1)), v_new)

            actual_lr = lr * lr_scale
            params_new = jax.tree.map(
                lambda p, mh, vh, lri: p - actual_lr * lri * mh / (jnp.sqrt(vh) + eps),
                params, m_hat, v_hat, base_lr_list
            )
            return params_new, m_new, v_new, loss, aux

        history = {"total": [], "pde": [], "data": [], "bc": []}
        if self.learn_kappa:
            history["kappa"] = []
        if self.learn_Kd:
            history["Kd"] = []
        if self.kappa_mode == 'log_linear':
            history["alpha"] = []
        if self.learn_w:
            history["w_vel"] = []

        header = f"{'Epoch':>6} | {'Total':>10} | {'PDE':>10} | {'Data':>10} | {'BC':>10}"
        if self.learn_kappa:
            header += f" | {'κ':>10}"
        if self.learn_Kd:
            header += f" | {'Kd':>10}"
        if self.kappa_mode == 'log_linear':
            header += f" | {'α':>10}"
        if self.learn_w:
            header += f" | {'w':>10}"
        if w_pde_schedule:
            header += f" | {'w_pde':>8}"
        print(header)
        print("-" * len(header))

        t_start = time_module.time()
        rng = np.random.RandomState(123)
        n_data = len(z_data)
        use_batch = batch_size > 0 and batch_size < n_data

        # Pre-allocate fixed-size batch arrays for consistent JIT shapes
        if use_batch:
            z_batch = jnp.zeros(batch_size, dtype=jnp.float32)
            t_batch = jnp.zeros(batch_size, dtype=jnp.float32)
            T_batch = jnp.zeros(batch_size, dtype=jnp.float32)

        for epoch in range(n_epochs):
            w_pde_cur = _get_w_pde(epoch)
            lr_scale = _get_lr_scale(epoch)

            # Resample PDE points
            if resample_pde_every > 0 and epoch > 0 and epoch % resample_pde_every == 0 and n_pde_points > 0:
                z_pde = jnp.array(rng.uniform(0, self.z_max, n_pde_points).astype(np.float32))
                t_pde = jnp.array(rng.uniform(0, self.t_max, n_pde_points).astype(np.float32))

            if use_batch:
                idx = rng.choice(n_data, batch_size, replace=False)
                z_batch = z_data[idx]
                t_batch = t_data[idx]
                T_batch = T_data[idx]
                params, m, v, loss, (pde_l, data_l, bc_l) = step_batch(
                    params, m, v, jnp.array(epoch + 1, dtype=jnp.float32),
                    z_pde, t_pde, jnp.array(w_pde_cur), jnp.array(lr_scale),
                    z_batch, t_batch, T_batch
                )
            else:
                params, m, v, loss, (pde_l, data_l, bc_l) = step(
                    params, m, v, jnp.array(epoch + 1, dtype=jnp.float32),
                    z_pde, t_pde, jnp.array(w_pde_cur), jnp.array(lr_scale)
                )

            history["total"].append(float(loss))
            history["pde"].append(float(pde_l))
            history["data"].append(float(data_l))
            history["bc"].append(float(bc_l))

            if self.learn_kappa or self.learn_Kd or self.kappa_mode == 'log_linear' or self.learn_w:
                nn_p, lk, lkd, al, wv = self._unpack_params(params)
                if self.learn_kappa:
                    history["kappa"].append(float(jnp.exp(lk)))
                if self.learn_Kd:
                    history["Kd"].append(float(jnp.exp(lkd)))
                if self.kappa_mode == 'log_linear':
                    history["alpha"].append(float(al))
                if self.learn_w:
                    history["w_vel"].append(float(wv))

            if epoch % print_every == 0 or epoch == n_epochs - 1:
                msg = (f"{epoch:6d} | {float(loss):10.6f} | {float(pde_l):10.6f} | "
                       f"{float(data_l):10.6f} | {float(bc_l):10.6f}")
                if self.learn_kappa:
                    msg += f" | {history['kappa'][-1]:10.6f}"
                if self.learn_Kd:
                    msg += f" | {history['Kd'][-1]:10.6f}"
                if self.kappa_mode == 'log_linear':
                    msg += f" | {history['alpha'][-1]:10.6f}"
                if self.learn_w:
                    msg += f" | {history['w_vel'][-1]:10.6f}"
                if w_pde_schedule:
                    msg += f" | {w_pde_cur:8.3f}"
                print(msg)

        elapsed = time_module.time() - t_start
        print(f"\nTraining: {n_epochs} epochs in {elapsed:.1f}s ({elapsed/n_epochs*1000:.1f} ms/epoch)")

        self.trained_params = params
        nn_p, lk, lkd, al, wv = self._unpack_params(params)
        self.nn_params = nn_p
        if self.learn_kappa:
            self.log_kappa = lk
        if self.learn_Kd:
            self.log_Kd = lkd
        if self.kappa_mode == 'log_linear':
            self.alpha = al
        if self.learn_w:
            self.w_vel = wv

        return history

    def predict(self, z, t):
        """Predict T at single (z, t). Requires trained_params."""
        return self._predict_scalar(self.trained_params, z, t)

    def predict_field(self, z_grid, t_grid):
        """Predict T on a full z × t grid."""
        pred = jit(lambda z_, t_: self._predict_scalar(self.trained_params, z_, t_))
        zz, tt = np.meshgrid(z_grid, t_grid, indexing="ij")
        T_flat = vmap(pred)(jnp.array(zz.flatten()), jnp.array(tt.flatten()))
        return np.array(T_flat).reshape(len(z_grid), len(t_grid))

    def get_estimated_params(self):
        """Return estimated physical parameters."""
        result = {}
        if self.learn_kappa:
            result["kappa"] = float(jnp.exp(self.log_kappa))
        if self.learn_Kd:
            result["Kd"] = float(jnp.exp(self.log_Kd))
        if self.kappa_mode == 'log_linear' and self.alpha is not None:
            result["alpha"] = float(self.alpha)
        if self.learn_w and self.w_vel is not None:
            result["w_vel"] = float(self.w_vel)
        return result
