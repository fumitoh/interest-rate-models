from modelx.serialize.jsonvalues import *

_formula = None

_bases = []

_allow_none = None

_spaces = []

# ---------------------------------------------------------------------------
# Cells

def A_t_T(i, j):
    t = t_(i)
    P_t = mkt_zcb(i)
    P_T = mkt_zcb(j)
    f_t = mkt_fwd(i)
    B = B_t_T(i, j)

    return P_T / P_t * np.exp(
        B * f_t - sigma**2 / (4*a) * (1 - np.exp(-2*a*t)) * B**2 
    )


def B_t_T(i, j):
    t, T = t_(i), t_(j)
    return (1 / a) * (1 - np.exp(-a * (T-t)))


def E_rt():
    return np.array([E_rt_s(0, i)[0] for i in range(step_size + 1)])


def E_rt_s(i, j):
    s, t = t_(i), t_(j)
    r_s = short_rate(i)
    return r_s * np.exp(-a * (t-s)) + alpha(j) - alpha(i) * np.exp(-a * (t-s))


def P_t_T(i, j):
    return A_t_T(i, j) * np.exp(-B_t_T(i, j) * short_rate(i))


def V_t_T(i, j):
    dt = t_(j) - t_(i)
    return sigma**2 / a**2 * (dt + (2/a)*np.exp(-a*dt) - (1/(2*a))*np.exp(-2*a*dt) - (3/(2*a)))


def Var_rt():
    return np.array([Var_rt_s(0, i) for i in range(step_size + 1)])


def Var_rt_s(i, j):
    s, t = t_(i), t_(j)
    return sigma**2 / (2*a) * (1 - np.exp(-2 * a * (t-s)))


def accum_short_rate(i):
    if i == 0:
        return np.full(scen_size, 0.0)
    else:
        dt = t_(i) - t_(i-1)
        return accum_short_rate(i-1) + short_rate(i-1) * dt


def accum_short_rate2(i):
    if i == 0:
        return np.full(scen_size, 0.0)
    else:
        t, T = t_(i-1), t_(i)
        dt = T - t
        cov = sigma**2/(2*a**2)*(1 + np.exp(-2*a*dt) -2 * np.exp(-a*dt))
        z1 = std_norm_rand(seed1)[:, i-1]
        z2 = std_norm_rand(seed2)[:, i-1]

        rho = cov / (Var_rt_s(i-1, i)**0.5 * V_t_T(i-1, i)**0.5)

        mean = B_t_T(i-1, i) * (short_rate(i-1) - alpha(i-1)) + np.log(mkt_zcb(i-1)/mkt_zcb(i)) + 0.5*(V_t_T(0, i)-V_t_T(0, i-1))
        return accum_short_rate2(i-1) + mean + V_t_T(i-1, i)**0.5 * (rho*z1 + (1-rho**2)**0.5*z2)


def alpha(i):
    t = t_(i)
    return mkt_fwd(i) + 0.5 * sigma**2 / a**2 * (1 - np.exp(-a*t))**2


delta_W = lambda i: delta_time(i)**0.5 * std_norm_rand()[:, i]

delta_time = lambda i: time_len / step_size

def disc_factor(i):
    return np.exp(-accum_short_rate(i))


def disc_factor_paths():
    return np.array([disc_factor(i) for i in range(step_size + 1)]).transpose()


def mean_disc_factor():
    return np.array([np.mean(disc_factor(i)) for i in range(step_size + 1)])


def mean_short_rate():
    return np.array([np.mean(short_rate(i)) for i in range(step_size + 1)])


def mkt_fwd(i):
    return 0.05


def mkt_zcb(i):
    if i == 0:
        return 1.0
    else:
        dt = t_(i) - t_(i-1)
        return mkt_zcb(i-1) * np.exp(-mkt_fwd(i-1)*dt)


def short_rate(i):
    if i == 0:
        return np.full(scen_size, mkt_fwd(0))
    else:
        return E_rt_s(i-1, i) + Var_rt_s(i-1, i)**0.5 * std_norm_rand(seed1)[:, i-1]


def short_rate_paths():
    return np.array([short_rate(i) for i in range(step_size + 1)]).transpose()


def std_norm_rand(seed=1234):

    size = (scen_size, step_size)

    if hasattr(np.random, 'default_rng'):
        gen = np.random.default_rng(seed)
        return gen.standard_normal(size)
    else:
        np.random.seed(seed)
        return np.random.standard_normal(size)


t_ = lambda i: i * time_len / step_size

def var_short_rate():
    return np.array([np.var(short_rate(i)) for i in range(step_size + 1)])


# ---------------------------------------------------------------------------
# References

np = ("Module", "numpy")

step_size = 360

time_len = 30

a = 0.1

scen_size = 500

sigma = 0.1

seed1 = 1234

seed2 = 5678