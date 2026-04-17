"""High-level PINN training convenience wrapper.

This module exposes :func:`fit_pinn`, a one-call training entry point that
constructs a :class:`~pinn_reef_thermal.ReefPINN` with defaults matching the
paper's production configuration, samples PDE collocation points stratified
around the logger depths, and trains the model. It is intended for users who
want to get a trained model from a prepared ``reef_data`` dict without touching
the full :class:`~pinn_reef_thermal.ReefPINN` constructor.

For fine-grained control (custom loss schedules, alternative network widths,
multi-seed sweeps), instantiate :class:`~pinn_reef_thermal.ReefPINN` directly
and call its :meth:`~pinn_reef_thermal.ReefPINN.train` method.
"""

from __future__ import annotations

import numpy as np

from .data_loader import generate_stratified_collocation_points
from .physics import ReefThermalPhysics
from .pinn import ReefPINN


def fit_pinn(
    reef_data: dict,
    *,
    n_epochs: int = 15000,
    n_pde: int = 20000,
    lr: float = 1e-3,
    w_pde: float = 1.0,
    w_data: float = 10.0,
    w_bc: float = 10.0,
    hidden_dim: int = 128,
    n_hidden: int = 4,
    seed: int = 42,
    learn_kappa: bool = True,
    learn_Kd: bool = True,
    use_modified_mlp: bool = True,
    use_hard_bc: bool = True,
    activation: str = "tanh",
    grad_clip: float = 1.0,
    kappa_mode: str = "constant",
    print_every: int = 500,
    verbose: bool = True,
) -> tuple[ReefPINN, dict]:
    """Train a reef-thermal PINN on pre-prepared data with sensible defaults.

    Parameters
    ----------
    reef_data : dict
        Output of :func:`~pinn_reef_thermal.prepare_pinn_data` or
        :func:`~pinn_reef_thermal.load_reef_data`. Must contain keys ``z_data``,
        ``t_data``, ``T_data``, ``z_bc``, ``t_bc``, ``T_bc``, ``depths``, and
        ``metadata`` (with ``z_max``, ``t_max``, ``T_mean``, ``T_range``,
        ``t_start_utc_hour``, ``start_day_of_year``).
    n_epochs : int, optional
        Number of Adam optimiser iterations. Default ``15000`` matches the
        paper's full-training setting. Reduce to a few thousand for quick
        exploratory runs.
    n_pde : int, optional
        Number of PDE collocation points sampled (half uniform, half
        concentrated near logger depths). Default ``20000``.
    lr : float, optional
        Base Adam learning rate. Default ``1e-3``.
    w_pde, w_data, w_bc : float, optional
        Loss weights for the PDE residual, data fit, and boundary condition
        terms respectively. The ``w_bc`` term is ignored when the hard
        boundary condition is active.
    hidden_dim : int, optional
        Hidden-layer width. Default ``128``.
    n_hidden : int, optional
        Number of hidden layers. Default ``4``.
    seed : int, optional
        PRNG seed for network initialisation and collocation sampling.
    learn_kappa, learn_Kd : bool, optional
        If ``True`` (default), the corresponding physical parameter is trained
        jointly with the network and exposed via
        :meth:`~pinn_reef_thermal.ReefPINN.get_estimated_params`.
    use_modified_mlp : bool, optional
        Use the Wang et al. (2021) modified MLP architecture with
        multiplicative gating. Default ``True``; improves accuracy over a
        standard MLP on this problem.
    use_hard_bc : bool, optional
        Enforce the surface boundary condition exactly via the output
        parameterisation. Default ``True``. When ``True``, the ``w_bc`` loss
        term is inactive.
    activation : {"tanh", "swish"}, optional
        Hidden-layer activation. Default ``"tanh"``; ``"swish"`` degrades
        accuracy on this problem in our tests.
    grad_clip : float, optional
        Global gradient norm cap. Default ``1.0``. Set to ``0`` to disable.
    kappa_mode : {"constant", "log_linear"}, optional
        Parameterisation of the learned thermal diffusivity. ``"constant"``
        learns a scalar kappa; ``"log_linear"`` learns ``log(kappa) + alpha * z``
        with an additional trainable slope.
    print_every : int, optional
        Epochs between progress prints. Default ``500``.
    verbose : bool, optional
        If ``False``, suppresses the training log header. Default ``True``.

    Returns
    -------
    pinn : ReefPINN
        The trained model. Its ``trained_params`` attribute is set; call
        :meth:`~pinn_reef_thermal.ReefPINN.predict` or
        :meth:`~pinn_reef_thermal.ReefPINN.predict_field` to make predictions.
    history : dict
        Training history with per-epoch ``total``, ``pde``, ``data``, and
        ``bc`` loss values, plus ``kappa`` and ``Kd`` trajectories when the
        corresponding parameters are being learned.

    Notes
    -----
    Defaults reproduce the production configuration reported in the paper's
    holdout validation experiments. Approximate cost on an RTX 4090 is five to
    fifteen minutes per reef at the default ``n_epochs``. For CPU exploration,
    see ``examples/train_on_new_reef.py`` which uses ``n_epochs=2000`` to
    complete within a few minutes on a laptop.

    Examples
    --------
    >>> from pinn_reef_thermal import load_reef_data, fit_pinn  # doctest: +SKIP
    >>> data = load_reef_data("davies_reef")                    # doctest: +SKIP
    >>> pinn, hist = fit_pinn(data, n_epochs=5000)              # doctest: +SKIP
    >>> pinn.get_estimated_params()                              # doctest: +SKIP
    {'kappa': 0.00038, 'Kd': 0.027}
    """
    meta = reef_data["metadata"]
    z_max = float(meta["z_max"])
    t_max = float(meta["t_max"])
    T_mean = float(meta.get("T_mean", float(np.mean(reef_data["T_data"]))))
    T_range = meta.get("T_range", (float(reef_data["T_data"].min()),
                                    float(reef_data["T_data"].max())))
    T_scale = max(1.0, (T_range[1] - T_range[0]) * 0.5)

    physics = ReefThermalPhysics(
        kappa=5e-4, Kd=0.25, Q_max=350.0, rho_cp=4.1e6,
        T_mean=T_mean, T_amp=1.2,
        z_max=z_max, t_days=t_max / 86400.0,
        utc_offset=10.0,
    )

    logger_depths = list(reef_data["depths"].values()) if reef_data.get("depths") else None
    z_pde, t_pde = generate_stratified_collocation_points(
        z_max, t_max, n_pde, logger_depths=logger_depths, seed=seed
    )

    pinn = ReefPINN(
        physics=physics,
        learn_kappa=learn_kappa,
        learn_Kd=learn_Kd,
        use_hard_bc=use_hard_bc,
        n_fourier=4,
        seed=seed,
        init_kappa=1e-3,
        init_Kd=0.2,
        pde_chunk_size=2000,
        pde_depth_scale=20.0,
        w_bc_bottom=0.1,
        kappa_mode=kappa_mode,
        t_sst=reef_data["t_bc"],
        T_sst=reef_data["T_bc"],
        use_modified_mlp=use_modified_mlp,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        T_scale=T_scale,
        activation=activation,
        grad_clip=grad_clip,
        t_start_utc_hour=meta.get("t_start_utc_hour", 0.0),
        start_day_of_year=meta.get("start_day_of_year", 0.0),
    )

    history = pinn.train(
        z_pde=z_pde, t_pde=t_pde,
        z_data=reef_data["z_data"],
        t_data=reef_data["t_data"],
        T_data=reef_data["T_data"],
        z_bc=reef_data["z_bc"],
        t_bc=reef_data["t_bc"],
        T_bc=reef_data["T_bc"],
        n_epochs=n_epochs,
        lr=lr,
        w_pde=w_pde,
        w_data=w_data,
        w_bc=w_bc,
        print_every=print_every if verbose else n_epochs + 1,
    )

    return pinn, history
