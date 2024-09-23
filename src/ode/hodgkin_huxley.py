from typing import Dict

from jax import Array
from jax import numpy as jnp

from src.ode.ode import ODE, ODEBuilder

# cf. https://github.com/berenslab/DiffusionTempering/blob/main/src/models.jl
# gating variables
a_m = lambda V, V_T: -0.32 * (V - V_T - 13.0) / (jnp.exp(-(V - V_T - 13.0) / 4.0) - 1.0)
b_m = lambda V, V_T: 0.28 * (V - V_T - 40.0) / (jnp.exp((V - V_T - 40.0) / 5.0) - 1.0)
a_n = lambda V, V_T: -0.032 * (V - V_T - 15.0) / (jnp.exp(-(V - V_T - 15.0) / 5.0) - 1.0)
b_n = lambda V, V_T: 0.5 * jnp.exp(-(V - V_T - 10.0) / 40.0)
a_h = lambda V, V_T: 0.128 * jnp.exp(-(V - V_T - 17.0) / 18.0)
b_h = lambda V, V_T: 4.0 / (1.0 + jnp.exp(-(V - V_T - 40.0) / 5.0))
a_q = lambda V: 0.055 * (-27.0 - V) / (jnp.exp((-27.0 - V) / 3.8) - 1.0)
b_q = lambda V: 0.94 * jnp.exp((-75.0 - V) / 17.0)
a_r = lambda V: 0.000457 * jnp.exp((-13.0 - V) / 50.0)
b_r = lambda V: 0.0065 / (jnp.exp((-15.0 - V) / 28.0) + 1.0)
tau_p = lambda V, tau_max: tau_max / (
    3.3 * jnp.exp((V + 35.0) / 20.0) + jnp.exp(-(V + 35.0) / 20.0)
)
tau_u = lambda V, V_x: (30.8 + (211.4 + jnp.exp((V + V_x + 113.2) / 5.0))) / (
    3.7 * (1.0 + jnp.exp((V + V_x + 84.0) / 3.2))
)
# initial functions
m_inf = lambda V, V_T: 1.0 / (1.0 + b_m(V, V_T) / a_m(V, V_T))
n_inf = lambda V, V_T: 1.0 / (1.0 + b_n(V, V_T) / a_n(V, V_T))
h_inf = lambda V, V_T: 1.0 / (1.0 + b_h(V, V_T) / a_h(V, V_T))
p_inf = lambda V: 1.0 / (1.0 + jnp.exp(-(V + 35.0) / 10.0))
q_inf = lambda V: 1.0 / (1.0 + b_q(V) / a_q(V))
r_inf = lambda V: 1.0 / (1.0 + b_r(V) / a_r(V))
s_inf = lambda V, V_x: 1.0 / (1.0 + jnp.exp(-(V + V_x + 57.0) / 6.2))
u_inf = lambda V, V_x: 1.0 / (1.0 + jnp.exp(-(V + V_x + 81.0) / 4.0))
# differential functions
f_m = lambda V, m, V_T: a_m(V, V_T) * (1 - m) - b_m(V, V_T) * m
f_h = lambda V, h, V_T: a_h(V, V_T) * (1 - h) - b_h(V, V_T) * h
f_n = lambda V, n, V_T: a_n(V, V_T) * (1 - n) - b_n(V, V_T) * n
f_p = lambda V, p, tau_max: (p_inf(V) - p) / tau_p(V, tau_max)
f_q = lambda V, q: a_q(V) * (1 - q) - b_q(V) * q
f_r = lambda V, r: a_r(V) * (1 - r) - b_r(V) * r
f_u = lambda V, u, V_x: (u_inf(V, V_x) - u) / tau_u(V, V_x)
# potential functions
f_I_Na = lambda V, m, h, g_Na, E_Na: g_Na * m**3 * h * (E_Na - V)
f_I_K = lambda V, n, g_K, E_K: g_K * n**4 * (E_K - V)
f_I_leak = lambda V, g_leak, E_leak: g_leak * (E_leak - V)
f_I_M = lambda V, p, g_M, E_K: g_M * p * (E_K - V)
f_I_L = lambda V, q, r, g_L, E_Ca: g_L * q**2 * r * (E_Ca - V)
f_I_T = lambda V, u, V_x, g_T, E_Ca: g_T * s_inf(V, V_x) ** 2 * u * (E_Ca - V)
f_I_in = lambda t: jnp.where(jnp.logical_and(t >= 10.0, t <= 90.0), 210.0 * 1e-6, 0.0)
f_V = (
    lambda I_Na, I_K, I_leak, I_M, I_L, I_T, I_in, A, C: (
        I_Na + I_K + I_leak + I_M + I_L + I_T + I_in / A
    )
    / C
)


class HodgkinHuxley(ODEBuilder):
    """Hodgkin-Huxley ODE (first-order)."""

    def __init__(
        self,
        model: str = "reduced-1",
        C: float = 1.0,
        A: float = 8.3e-5,
        g_Na: float = 25.0,
        E_Na: float = 53.0,
        g_K: float = 7.0,
        E_K: float = -107.0,
        g_leak: float = 0.1,
        E_leak: float = -70.0,
        V_T: float = -60.0,
        g_M: float = 0.01,
        tau_max: float = 4e3,
        g_L: float = 0.01,
        E_Ca: float = 120.0,
        g_T: float = 0.0,
        V_x: float = 2.0,
    ) -> None:
        """
        Initialization for Hodgkin-Huxley model.

        Args:
            model (str, optional): Model: full/reduced-1/reduced-4. Defaults to "reduced-1".
            C (float, optional): _description_. Defaults to 1.0.
            A (float, optional): _description_. Defaults to 8.3e-5.
            g_Na (float, optional): _description_. Defaults to 25.0.
            E_Na (float, optional): _description_. Defaults to 53.0.
            g_K (float, optional): _description_. Defaults to 7.0.
            E_K (float, optional): _description_. Defaults to -107.0.
            g_leak (float, optional): _description_. Defaults to 0.1.
            E_leak (float, optional): _description_. Defaults to -70.0.
            V_T (float, optional): _description_. Defaults to -60.0.
            g_M (float, optional): _description_. Defaults to 0.01.
            tau_max (float, optional): _description_. Defaults to 4e3.
            g_L (float, optional): _description_. Defaults to 0.01.
            E_Ca (float, optional): _description_. Defaults to 120.0.
            g_T (float, optional): _description_. Defaults to 0.0.
            V_x (float, optional): _description_. Defaults to 2.0.
        """

        super().__init__(
            C=C,
            A=A,
            g_Na=g_Na,
            E_Na=E_Na,
            g_K=g_K,
            E_K=E_K,
            g_leak=g_leak,
            E_leak=E_leak,
            V_T=V_T,
            g_M=g_M,
            tau_max=tau_max,
            g_L=g_L,
            E_Ca=E_Ca,
            g_T=g_T,
            V_x=V_x,
        )

        self.model = model

    def build(self) -> ODE:
        def ode_full(t: Array, x: Array, params: Dict[str, Array]) -> Array:
            """
            RHS of ODE (full model).
            D=8: Latent dimension.
            N=1: ODE order.

            Args:
                t (Array): Time [].
                x (Array): State [N, D].
                params (Dict[str, Array]): Parameters.

            Returns:
                Array: d/dt State [N, D].
            """

            # x: [V, M, H, N, P, Q, R, U]
            #     0, 1, 2, 3, 4, 5, 6, 7

            dm_dt = f_m(x[0, 0], x[0, 1], params["V_T"])
            dh_dt = f_h(x[0, 0], x[0, 2], params["V_T"])
            dn_dt = f_n(x[0, 0], x[0, 3], params["V_T"])
            dp_dt = f_p(x[0, 0], x[0, 4], params["tau_max"])
            dq_dt = f_q(x[0, 0], x[0, 5])
            dr_dt = f_r(x[0, 0], x[0, 6])
            du_dt = f_u(x[0, 0], x[0, 7], params["V_x"])

            I_Na = f_I_Na(x[0, 0], x[0, 1], x[0, 2], params["g_Na"], params["E_Na"])
            I_K = f_I_K(x[0, 0], x[0, 3], params["g_K"], params["E_K"])
            I_leak = f_I_leak(x[0, 0], params["g_leak"], params["E_leak"])
            I_M = f_I_M(x[0, 0], x[0, 4], params["g_M"], params["E_K"])
            I_L = f_I_L(x[0, 0], x[0, 5], x[0, 6], params["g_L"], params["E_Ca"])
            I_T = f_I_T(x[0, 0], x[0, 7], params["V_x"], params["g_T"], params["E_Ca"])
            I_in = f_I_in(t)
            dV_dt = f_V(I_Na, I_K, I_leak, I_M, I_L, I_T, I_in, params["A"], params["C"])

            dx_dt_next = jnp.stack([dV_dt, dm_dt, dh_dt, dn_dt, dp_dt, dq_dt, dr_dt, du_dt])[
                None, :
            ]  # [N, D]

            return dx_dt_next

        def ode_reduced_1(t: Array, x: Array, params: Dict[str, Array]) -> Array:
            """
            RHS of ODE (reduced-1 model).
            D=7: Latent dimension.
            N=1: ODE order.

            Args:
                t (Array): Time [].
                x (Array): State [N, D].
                params (Dict[str, Array]): Parameters.

            Returns:
                Array: d/dt State [N, D].
            """

            # x: [V, M, H, N, P, Q, R, U]
            #     0, 1, 2, 3, 4, 5, 6, 7

            dm_dt = f_m(x[0, 0], x[0, 1], params["V_T"])
            dh_dt = f_h(x[0, 0], x[0, 2], params["V_T"])
            dn_dt = f_n(x[0, 0], x[0, 3], params["V_T"])
            dp_dt = f_p(x[0, 0], x[0, 4], params["tau_max"])
            dq_dt = f_q(x[0, 0], x[0, 5])
            dr_dt = f_r(x[0, 0], x[0, 6])

            I_Na = f_I_Na(x[0, 0], x[0, 1], x[0, 2], params["g_Na"], params["E_Na"])
            I_K = f_I_K(x[0, 0], x[0, 3], params["g_K"], params["E_K"])
            I_leak = f_I_leak(x[0, 0], params["g_leak"], params["E_leak"])
            I_M = f_I_M(x[0, 0], x[0, 4], params["g_M"], params["E_K"])
            I_L = f_I_L(x[0, 0], x[0, 5], x[0, 6], params["g_L"], params["E_Ca"])
            I_T = f_I_T(x[0, 0], x[0, 7], params["V_x"], params["g_T"], params["E_Ca"])
            I_in = f_I_in(t)
            dV_dt = f_V(I_Na, I_K, I_leak, I_M, I_L, I_T, I_in, params["A"], params["C"])

            dx_dt_next = jnp.stack([dV_dt, dm_dt, dh_dt, dn_dt, dp_dt, dq_dt, dr_dt])[
                None, :
            ]  # [N, D]

            return dx_dt_next

        def ode_reduced_4(t: Array, x: Array, params: Dict[str, Array]) -> Array:
            """
            RHS of ODE (reduced-4 model).
            D=4: Latent dimension.
            N=1: ODE order.

            Args:
                t (Array): Time [].
                x (Array): State [N, D].
                params (Dict[str, Array]): Parameters.

            Returns:
                Array: d/dt State [N, D].
            """

            # x: [V, M, H, N, P, Q, R, U]
            #     0, 1, 2, 3, 4, 5, 6, 7

            dm_dt = f_m(x[0, 0], x[0, 1], params["V_T"])
            dh_dt = f_h(x[0, 0], x[0, 2], params["V_T"])
            dn_dt = f_n(x[0, 0], x[0, 3], params["V_T"])

            I_Na = f_I_Na(x[0, 0], x[0, 1], x[0, 2], params["g_Na"], params["E_Na"])
            I_K = f_I_K(x[0, 0], x[0, 3], params["g_K"], params["E_K"])
            I_leak = f_I_leak(x[0, 0], params["g_leak"], params["E_leak"])
            I_M = f_I_M(x[0, 0], x[0, 4], params["g_M"], params["E_K"])
            I_L = f_I_L(x[0, 0], x[0, 5], x[0, 6], params["g_L"], params["E_Ca"])
            I_T = f_I_T(x[0, 0], x[0, 7], params["V_x"], params["g_T"], params["E_Ca"])
            I_in = f_I_in(t)
            dV_dt = f_V(I_Na, I_K, I_leak, I_M, I_L, I_T, I_in, params["A"], params["C"])

            dx_dt_next = jnp.stack([dV_dt, dm_dt, dh_dt, dn_dt])[None, :]  # [N, D]

            return dx_dt_next

        if self.model == "full":
            return ode_full
        elif self.model == "reduced-1":
            return ode_reduced_1
        elif self.model == "reduced-4":
            return ode_reduced_4
        else:
            raise ValueError(f"Unknown model: {self.model}")

    def build_initial_value(self, initial_value: Array, params: Dict[str, Array]) -> Array:
        """
        Builds initial value.
        D=8/7/4: Latent dimension.
        N=1: ODE order.

        Args:
            initial_value (Array): Initial value [N, 1].
            params (Dict[str, Array]): Parameters.

        Returns:
            Array: Built initial value [N, D].
        """

        V0 = initial_value[0, 0]
        M0 = m_inf(V0, params["V_T"])
        H0 = h_inf(V0, params["V_T"])
        N0 = n_inf(V0, params["V_T"])
        P0 = p_inf(V0)
        Q0 = q_inf(V0)
        R0 = r_inf(V0)
        U0 = u_inf(V0, params["V_x"])

        if self.model == "full":
            return jnp.stack([V0, M0, H0, N0, P0, Q0, R0, U0], axis=-1)[None, :]
        elif self.model == "reduced-1":
            return jnp.stack([V0, M0, H0, N0, P0, Q0, R0], axis=-1)[None, :]
        elif self.model == "reduced-4":
            return jnp.stack([V0, M0, H0, N0], axis=-1)[None, :]
        else:
            raise ValueError(f"Unknown model: {self.model}")
