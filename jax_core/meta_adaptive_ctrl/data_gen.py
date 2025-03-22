
if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    # Updated import: now use the new jit-compatible module
    from jax_core.simulator.waves.wave_load_jax_jit import wave_load, WaveLoad
    import pickle
    from jax.experimental.ode import odeint
    from jax_core.utils import spline, random_ragged_spline
    from jax_core.meta_adaptive_ctrl.dynamics import prior_3dof, plant, disturbance  # using updated disturbance & prior functions

    # Seed random numbers
    seed = 0
    key = jax.random.PRNGKey(seed)

    # Generate smooth trajectories
    num_traj = 500
    T = 30
    num_knots = 5
    poly_orders = (9, 9, 6)
    deriv_orders = (4, 4, 2)
    min_step = jnp.array([-0.3, -0.3, -jnp.pi / 12])
    max_step = jnp.array([0.3, 0.3, jnp.pi / 12])
    min_knot = jnp.array([-jnp.inf, -jnp.inf, -jnp.pi / 2])
    max_knot = jnp.array([jnp.inf, jnp.inf, jnp.pi / 2])

    key, *subkeys = jax.random.split(key, 1 + num_traj)
    subkeys = jnp.vstack(subkeys)
    in_axes = (0, None, None, None, None, None, None, None, None)
    t_knots, knots, coefs = jax.vmap(random_ragged_spline, in_axes)(
        subkeys, T, num_knots, poly_orders, deriv_orders,
        min_step, max_step, min_knot, max_knot
    )
    r_knots = jnp.dstack(knots)

    # Define simulate function (now vmapped over the batched wave load)
    @jax.tree_util.Partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def simulate(ts, wl, t_knots, coefs,
                 plant=plant, prior=prior_3dof, disturbance=wave_load):
        def reference(t):
            x_coefs, y_coefs, yaw_coefs = coefs
            x = spline(t, t_knots, x_coefs)
            y = spline(t, t_knots, y_coefs)
            yaw = spline(t, t_knots, yaw_coefs)
            yaw = jnp.clip(yaw, -jnp.pi / 3, jnp.pi / 3)
            r = jnp.array([x, y, yaw])
            return r

        def ref_derivatives(t):
            ref_vel = jax.jacfwd(reference)
            ref_acc = jax.jacfwd(ref_vel)
            # Evaluate at a slightly shifted time to avoid boundary issues.
            t_safe = t + 1e-8
            r = reference(t_safe)
            dr = ref_vel(t_safe)
            ddr = ref_acc(t_safe)
            ddr = jnp.nan_to_num(ddr, nan=0.0, posinf=0.0, neginf=0.0)
            return r, dr, ddr

        def controller(q, dq, r, dr, ddr):
            kp, kd = 0.5, 0.5
            e, de = q - r, dq - dr
            dv = ddr - kp * e - kd * de
            M, D, G, R = prior(q, dq)
            τ = M @ dv + D @ dq + G @ q
            u = jnp.transpose(R) @ τ
            return u, τ

        def ode(x, t, u, wl=wl):
            q, dq = x
            # Use the new jit-compatible wave load function.
            f_ext = disturbance(t, q, wl)
            ddq = plant(q, dq, u, f_ext)
            return (dq, ddq)

        def loop(carry, t):
            t_prev, q_prev, dq_prev, u_prev = carry
            qs, dqs = odeint(ode, (q_prev, dq_prev), jnp.array([t_prev, t]), u_prev)
            q, dq = qs[-1], dqs[-1]
            r, dr, ddr = ref_derivatives(t)
            u, τ = controller(q, dq, r, dr, ddr)
            carry = (t, q, dq, u)
            output_slice = (q, dq, u, τ, r, dr)
            return carry, output_slice

        t0 = ts[0]
        r0, dr0, ddr0 = ref_derivatives(t0)
        q0, dq0 = r0, dr0  # initialize states to reference
        u0, τ0 = controller(q0, dq0, r0, dr0, ddr0)
        carry = (t0, q0, dq0, u0)
        carry, output = jax.lax.scan(loop, carry, ts[1:])
        q, dq, u, τ, r, dr = output
        q = jnp.vstack((q0, q))
        dq = jnp.vstack((dq0, dq))
        u = jnp.vstack((u0, u))
        τ = jnp.vstack((τ0, τ))
        r = jnp.vstack((r0, r))
        dr = jnp.vstack((dr0, dr))
        return q, dq, u, τ, r, dr

    # Define wave parameters
    scale = 1 / 90
    sqrt_scale = jnp.sqrt(scale)
    a = 6
    b = 3
    hs_min = 0.5 * scale
    hs_max = 7.0 * scale
    key, subkey = jax.random.split(key, 2)
    hs = hs_min + (hs_max - hs_min) * jax.random.beta(subkey, a, b, (num_traj,))
    wave_dir = jnp.rint(jax.random.uniform(key, (num_traj,), minval=0, maxval=360)).astype(int)
    tp_min = 7 * sqrt_scale
    tp_max = 20 * sqrt_scale
    tp = tp_min + (tp_max - tp_min) * jax.random.beta(subkey, a, b, (num_traj,))
    wave_parm = (hs, tp, wave_dir)

    # Initialize wave loads for each trajectory.
    wl_list = []
    for i in range(num_traj):
        print( f"Making init waves for number {i}...")
        wave_parm_single = (hs[i], tp[i], wave_dir[i])
        wl = disturbance(wave_parm_single, key)
        wl_list.append(wl)

    # Stack list of WaveLoad PyTrees into a batched WaveLoad using tree_map.
    import jax.tree_util as tu
    wl_batched = tu.tree_map(lambda *x: jnp.stack(x), *wl_list)

    print("Simulating the system...")
    dt = 0.01
    t = jnp.arange(0, T + dt, dt)
    q, dq, u, τ, r, dr = simulate(t, wl_batched, t_knots, coefs)
    # --------------------- Save Training Data ---------------------
    print("Saving training data...")
    # Save training data
    data = {
        'seed': seed, 'prng_key': key,
        't': t, 'q': q, 'dq': dq,
        'u': u, 'r': r, 'dr': dr,
        't_knots': t_knots, 'r_knots': r_knots,
        'wave_parm': wave_parm
    }
    with open('training_data.pkl', 'wb') as file:
        pickle.dump(data, file)

    # --------------------- Plotting Section ---------------------
    print("Plotting the results...")
    import matplotlib.pyplot as plt
    import numpy as np

    t_np = np.array(data['t'])
    q_np = np.array(data['q'])
    dq_np = np.array(data['dq'])
    r_np = np.array(data['r'])
    dr_np = np.array(data['dr'])
    u_np = np.array(data['u'])

    if q_np.ndim == 3:
        q_plot = q_np[1]
        dq_plot = dq_np[1]
        r_plot = r_np[1]
        dr_plot = dr_np[1]
        u_plot = u_np[1]
    else:
        q_plot = q_np
        dq_plot = dq_np
        r_plot = r_np
        dr_plot = dr_np
        u_plot = u_np

    dof_names = ['x', 'y', 'yaw']

    fig_pos, axs_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs_pos[i].plot(t_np, q_plot[:, i], label='Actual Position')
        axs_pos[i].plot(t_np, r_plot[:, i], '--', label='Desired Reference')
        axs_pos[i].set_ylabel(f'{dof_names[i]} Position')
        axs_pos[i].legend()
        axs_pos[i].grid(True)
    axs_pos[-1].set_xlabel('Time [s]')
    fig_pos.suptitle('Position vs Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    fig_vel, axs_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs_vel[i].plot(t_np, dq_plot[:, i], label='Actual Velocity')
        axs_vel[i].plot(t_np, dr_plot[:, i], '--', label='Desired Reference Velocity')
        axs_vel[i].set_ylabel(f'{dof_names[i]} Velocity')
        axs_vel[i].legend()
        axs_vel[i].grid(True)
    axs_vel[-1].set_xlabel('Time [s]')
    fig_vel.suptitle('Velocity vs Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(q_plot[:, 0], q_plot[:, 1], label='Actual Trajectory', linewidth=2)
    plt.plot(r_plot[:, 0], r_plot[:, 1], '--', label='Desired Reference Trajectory', linewidth=2)
    plt.title('X-Y Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    fig_u, axs_u = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axs_u[i].plot(t_np, u_plot[:, i], label='Control Input')
        axs_u[i].set_ylabel(f'{dof_names[i]} Control Input')
        axs_u[i].legend()
        axs_u[i].grid(True)
    axs_u[-1].set_xlabel('Time [s]')
    fig_u.suptitle('Control Input vs Time')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
