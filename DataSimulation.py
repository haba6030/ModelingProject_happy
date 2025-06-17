from cmdstanpy import CmdStanModel
from cmdstanpy import install_cmdstan
install_cmdstan()

import numpy as np
import json

import os
from scipy.stats import norm

import random
import numpy as np
import pandas as pd
import arviz as az

from scipy.optimize import minimize

import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

from tqdm import tqdm

def save_stan_outputs_and_evaluation(fit, prefix="model", trace_chunk_size=20):
    # 0. Create output directory
    os.makedirs("outputs/simulation", exist_ok=True)

    # 1. Summary of parameters
    summary_df = fit.summary()
    summary_path = f"outputs/simulation/{prefix}_parameter_summary.csv"
    summary_df.to_csv(summary_path)

    # 2. Posterior predicted values (y_pred)
    y_pred = fit.stan_variable("y_pred")
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    #y_pred_path = f"outputs/simulation/{prefix}_posterior_y_pred.csv"
    #np.savetxt(y_pred_path, y_pred_flat, delimiter=",")

    # 3. Log-likelihood samples
    log_lik = fit.stan_variable("log_lik").copy()
    log_lik[log_lik == -1] = np.nan
    log_lik = log_lik.astype(np.float64)  # ensure proper dtype

    log_lik_mean = np.nanmean(log_lik, axis=0)  # shape: (N, T)
    log_lik_path = f"outputs/simulation/{prefix}_log_lik.csv"
    np.savetxt(log_lik_path, log_lik_mean, delimiter=",")
 
    # 4. ArviZ evaluation
    idata = az.from_cmdstanpy(
        posterior=fit,
        posterior_predictive="y_pred",
        log_likelihood="log_lik"
    )

    waic = az.waic(idata)
    loo = az.loo(idata)
    
    # Pareto-k diagnostics CSV and plot
    pareto_k_vals = loo.pareto_k.values.flatten()
    pareto_k_df = pd.DataFrame({"pareto_k": pareto_k_vals})
    pareto_k_df["observation"] = np.arange(len(pareto_k_vals))
    pareto_k_path = f"outputs/simulation/{prefix}_pareto_k.csv"
    pareto_k_df.to_csv(pareto_k_path, index=False)

    plt.figure(figsize=(10, 4))
    az.plot_khat(loo)
    plt.title("Pareto-k Diagnostic (per observation)")
    plt.tight_layout()
    pareto_k_plot_path = f"outputs/simulation/{prefix}_pareto_k_plot.png"
    plt.savefig(pareto_k_plot_path, dpi=300)
    plt.close()

    # AIC and BIC using log-likelihood
    log_lik_total = np.sum(log_lik, axis=-1)  # sum over trials
    log_lik_mean = np.mean(log_lik_total)
    n = np.prod(log_lik.shape[1:])  # total number of likelihood points
    k = summary_df.shape[0]  # number of parameters
    aic = 2 * k - 2 * log_lik_mean
    bic = k * np.log(n) - 2 * log_lik_mean

    eval_df = pd.DataFrame({
        "elpd_waic": [waic.elpd_waic],
        "waic_se": [waic.se],
        "elpd_loo": [loo.elpd_loo],
        "loo_se": [loo.se],
        "aic": [aic],
        "bic": [bic],
        "log_lik_total_mean": [log_lik_mean]
    })
    eval_path = f"outputs/simulation/{prefix}_evaluation.csv"
    eval_df.to_csv(eval_path, index=False)

    # 5. Generated quantities
    stan_vars = fit.stan_variables().keys()
    gen_quant_vars = [var for var in stan_vars if var.startswith("mu_")]
    gen_quant_stats = {}

    for var in gen_quant_vars:
        if var in fit.stan_variables():
            data = fit.stan_variable(var)
            gen_quant_stats[var] = {
                "mean": np.mean(data),
                "std": np.std(data),
                "median": np.median(data),
                "min": np.min(data),
                "max": np.max(data)
            }
            # Plot histogram
            plt.hist(data, bins=30, alpha=0.7)
            plt.title(f"Posterior of {var}")
            plt.xlabel(var)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"outputs/simulation/{prefix}_posterior_{var}.png", dpi=300)
            plt.close()

    gen_quant_df = pd.DataFrame(gen_quant_stats).T
    gen_quant_csv = f"outputs/simulation/{prefix}_generated_quantities_summary.csv"
    gen_quant_df.to_csv(gen_quant_csv)

    # 8. Traceplot (chunked if too many parameters)
    rhat_mean = summary_df["R_hat"].mean()
    all_params = list(idata.posterior.data_vars.keys())
    traceplot_paths = []

    for i in range(0, len(all_params), trace_chunk_size):
        subset = all_params[i:i+trace_chunk_size]
        az.plot_trace(idata, var_names=subset, compact=True)
        fig = plt.gcf()  # 현재 활성화된 figure 가져오기
        fig.suptitle(f"Trace Plot (Rhat ≈ {rhat_mean:.2f})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        path = f"outputs/simulation/{prefix}_traceplot_part{i//trace_chunk_size+1}.png"
        plt.savefig(path, dpi=300)
        traceplot_paths.append(path)
        plt.close()

    return {
        "summary_csv": summary_path,
        #"y_pred_csv": y_pred_path,
        "log_lik_csv": log_lik_path,
        "eval_csv": eval_path,
        "traceplot_pngs": traceplot_paths,
        "waic": waic.elpd_waic,
        "loo": loo.elpd_loo,
        "aic": aic,
        "bic": bic
    }


# # Common variables
N = 75
T = 160
rate_happy = 0.25  # 25%만 happiness 응답

def minmax_per_participant(matrix):
    normed = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        min_val = np.min(matrix[i])
        max_val = np.max(matrix[i])
        normed[i] = (matrix[i] - min_val) / (max_val - min_val)
    return normed

cr = np.zeros((N, T))
## gain loss
reward_set = [10, 20, 40, 80]

gain = np.zeros((N, T))
loss = np.zeros((N, T))

for i in range(N):
    for t in range(T):
        m1 = np.random.choice(reward_set)
        # mag2는 mag1과 다른 값으로 선택
        m2 = np.random.choice([r for r in reward_set if r != m1])

        gain[i, t] = m1
        loss[i, t] = m2
        
gain = minmax_per_participant(gain)
loss = minmax_per_participant(loss)

## 확률 선택
prob1 = np.zeros((N, T))
prob2 = np.zeros((N, T))

min_block = 8   # 최소 유지 trial 수
max_block = 20  # 최대 유지 trial 수

for i in range(N):
    t = 0
    current_prob = (0.8, 0.2) if np.random.rand() < 0.5 else (0.2, 0.8)

    while t < T:
        block_size = np.random.randint(min_block, max_block + 1)
        end_t = min(t + block_size, T)

        prob1[i, t:end_t] = current_prob[0]
        prob2[i, t:end_t] = current_prob[1]

        # 전환
        current_prob = (0.2, 0.8) if current_prob == (0.8, 0.2) else (0.8, 0.2)
        t = end_t

gamble = np.zeros((N, T)) + 1

# 선택: 0 → option1, 1 → option2
choice = np.random.randint(0, 2, (N, T)) # 실제 절반 가량 선택하였음
outcome = np.zeros((N, T))
winLose = np.zeros((N, T))

for i in range(N):
    for t in range(T):
        if choice[i, t] == 0:
            reward = gain[i, t]
            prob = prob1[i, t]
        else:
            reward = loss[i, t]
            prob = prob2[i, t]

        rand_val = np.random.rand()
        if rand_val < prob:
            outcome[i, t] = reward
            winLose[i, t] = 1
        else:
            outcome[i, t] = 0

# Recover each parameter (75 participants assumed)
def recover_param(param_prefix):
    base = param_df[param_df["name"].str.contains(f"{param_prefix}\\[")]
    base = base.sort_values("name")
    base_vals = base["Mean"].values
    recovered = base_vals
    return recovered

# SD도 함께 복원할 함수
def recover_param_with_sd(param_prefix):
    base = param_df[param_df["name"].str.contains(f"{param_prefix}\\[")]
    base = base.sort_values("name")
    base_vals = base["Mean"].values
    base_stds = base["StdDev"].values

    return base_vals, base_stds


# 1st Raw
# Load newly uploaded file to extract all required parameters
param_df = pd.read_csv("outputs/HBA_raw_bla_1st_parameter_summary.csv")

# Recover w0 to w3
w0_recov = recover_param("w0")
w1_recov = recover_param("w1")
w2_recov = recover_param("w2")
w3_recov = recover_param("w3")
gamma_recov = recover_param("gam")

# Combine into a DataFrame
recovered_df = pd.DataFrame({
    "participant": np.arange(1, len(w0_recov) + 1),
    "w0": w0_recov,
    "w1": w1_recov,
    "w2": w2_recov,
    "w3": w3_recov,
    "gamma": gamma_recov
})

recovered_df.to_csv("outputs/simulation/HBA_raw_bla_1st_original_parameters.csv")

# 파라미터 ground truth
true_w0 = recovered_df["w0"]
true_w1 = recovered_df["w1"]
true_w2 = recovered_df["w2"]
true_w3 = recovered_df["w3"]

true_gamma = recovered_df["gamma"]

i_vec = []
t_vec = []
T_temp = 0

# happy 계산
happy = np.full((N, T), -999.0)

# Calculate happiness
ev = gain * prob1 + loss * prob2
rpe = outcome - ev

cr_d = np.zeros((N, T))
ev_d = np.zeros((N, T))
rpe_d = np.zeros((N, T))

for i in range(N):
    T_temp = 0
    for t in range(T):
        if t == 0:
            cr_d[i, t] = cr[i, t]
            ev_d[i, t] = ev[i, t]
            rpe_d[i, t] = rpe[i, t]
        else:
            cr_d[i, t] = true_gamma[i] * cr_d[i, t - 1] + cr[i, t]
            ev_d[i, t] = true_gamma[i] * ev_d[i, t - 1] + ev[i, t]
            rpe_d[i, t] = true_gamma[i] * rpe_d[i, t - 1] + rpe[i, t]

        if t % 4 == 3:
            i_vec.append(i + 1) #stan 기준
            t_vec.append(t + 1)
            T_temp += 1

            mu = (true_w0[i] 
                  + true_w1[i] * cr_d[i, t]
                  + true_w2[i] * ev_d[i, t] 
                  + true_w3[i] * rpe_d[i, t] 
                 )
            happy[i, t] = np.random.normal(mu, 1.0)

Tsubj = [T for _ in range(N)]

stan_data = {
    "N": N,
    "T": T,
    "Tsubj": Tsubj,
    "gain": gain.tolist(),
    "loss": loss.tolist(),
    "prob1": prob1.tolist(),
    "prob2": prob2.tolist(),
    "cert": cr.tolist(),
    "gamble": gamble.tolist(),
    "outcome": outcome.tolist(),
    "happy": happy.tolist(),
    "happy_num": T_temp,
    "i_vec": i_vec,
    "t_vec": t_vec
}

with open("simulated_blain_1straw_hba.json", "w") as f:
    json.dump(stan_data, f)

model = CmdStanModel(stan_file='happy_2014.stan')
fit = model.sample(data="simulated_blain_1straw_hba.json", chains=4, parallel_chains=4, 
                   iter_warmup=1000, iter_sampling=1000, 
                   seed=2025, adapt_delta=0.95, max_treedepth=15,
                   show_progress=True)
results = save_stan_outputs_and_evaluation(fit, prefix="HBA_raw_bla_1st")

# Load newly uploaded file to extract all required parameters
param_df = pd.read_csv("outputs/simulation/HBA_raw_bla_1st_parameter_summary.csv")

# Recover w0 to w3
w0_mean, w0_sd = recover_param_with_sd("w0")
w1_mean, w1_sd = recover_param_with_sd("w1")
w2_mean, w2_sd = recover_param_with_sd("w2")
w3_mean, w3_sd = recover_param_with_sd("w3")
gamma_mean, gamma_sd = recover_param_with_sd("gam")

recovered_df = pd.DataFrame({
    "participant": np.arange(1, len(w0_mean) + 1),
    "w0": w0_mean, "w0_sd": w0_sd,
    "w1": w1_mean, "w1_sd": w1_sd,
    "w2": w2_mean, "w2_sd": w2_sd,
    "w3": w3_mean, "w3_sd": w3_sd,
    "gamma": gamma_mean, "gamma_sd": gamma_sd,
})
recovered_df.to_csv("outputs/simulation/HBA_raw_bla_1st_simulated_parameters.csv")


# 1st addit
# Load newly uploaded file to extract all required parameters
param_df = pd.read_csv("outputs/HBA_addit_bla_1st_parameter_summary.csv")

# Recover w0 to w3
w0_recov = recover_param("w0")
w1_recov = recover_param("w1")
w2_recov = recover_param("w2")
w3_recov = recover_param("w3")

# Recover gamma with Phi_approx
gamma_recov = recover_param("gam")
alpha_recov = recover_param("alpha")

# Combine into a DataFrame
recovered_df = pd.DataFrame({
    "participant": np.arange(1, len(w0_recov) + 1),
    "w0": w0_recov,
    "w1": w1_recov,
    "w2": w2_recov,
    "w3": w3_recov,
    "gamma": gamma_recov,
    "alpha": alpha_recov
})
recovered_df.to_csv("outputs/simulation/HBA_addit_bla_1st_original_parameters.csv")

# 파라미터 ground truth
true_w0 = recovered_df["w0"]
true_w1 = recovered_df["w1"]
true_w2 = recovered_df["w2"]
true_w3 = recovered_df["w3"]

true_gamma = recovered_df["gamma"]
true_alpha = recovered_df["alpha"]

i_vec = []
t_vec = []
T_temp = 0

# happy 계산
happy = np.full((N, T), -999.0)

# Calculate happiness
phat = 0.5
ev = 0
rpe = 0

cr_d = np.zeros((N, T))
ev_d = np.zeros((N, T))
rpe_d = np.zeros((N, T))

for i in range(N):
    T_temp = 0
    for t in range(T):
        value = gain[i, t]
        if choice[i, t] == 1:
            value = loss[i, t]
            
        ev = phat * value
        rpe = outcome[i, t] - ev
        
        if t == 0:
            cr_d[i, t] = cr[i, t]
            ev_d[i, t] = ev
            rpe_d[i, t] = rpe 
        else:
            cr_d[i, t] = true_gamma[i] * cr_d[i, t - 1] + cr[i, t]
            ev_d[i, t] = true_gamma[i] * ev_d[i, t - 1] + ev
            rpe_d[i, t] = true_gamma[i] * rpe_d[i, t - 1] + rpe

        phat = phat + true_alpha[i] * (winLose[i, t] - phat)

        if t % 4 == 3:
            i_vec.append(i + 1) #stan 기준
            t_vec.append(t + 1)
            T_temp += 1
            mu = (true_w0[i] 
                  + true_w1[i] * cr_d[i, t]
                  + true_w2[i] * ev_d[i, t] 
                  + true_w3[i] * rpe_d[i, t] 
                 )
            happy[i, t] = np.random.normal(mu, 1.0)


Tsubj = [T for _ in range(N)]

stan_data = {
    "N": N,
    "T": T,
    "Tsubj": Tsubj,
    "gain": gain.tolist(),
    "loss": loss.tolist(),
    "prob1": prob1.tolist(),
    "prob2": prob2.tolist(),
    "cert": cr.tolist(),
    "gamble": gamble.tolist(),
    "choice": choice.tolist(),
    "outcome": outcome.tolist(),
    "winLose": winLose.tolist(),
    "happy": happy.tolist(),
    "happy_num": T_temp,
    "i_vec": i_vec,
    "t_vec": t_vec
}

with open("simulated_blain_1staddit.json", "w") as f:
    json.dump(stan_data, f)

model = CmdStanModel(stan_file='happy_addit.stan')

fit = model.sample(data="simulated_blain_1staddit.json", chains=4, parallel_chains=4, 
                   iter_warmup=1000, iter_sampling=1000, 
                   seed=2025, adapt_delta=0.95, max_treedepth=15,
                   show_progress=True)

results = save_stan_outputs_and_evaluation(fit, prefix="HBA_addit_bla_1st")
# Load newly uploaded file to extract all required parameters
param_df = pd.read_csv("outputs/simulation/HBA_addit_bla_1st_parameter_summary.csv")

# Recover w0 to w3
w0_mean, w0_sd = recover_param_with_sd("w0")
w1_mean, w1_sd = recover_param_with_sd("w1")
w2_mean, w2_sd = recover_param_with_sd("w2")
w3_mean, w3_sd = recover_param_with_sd("w3")
gamma_mean, gamma_sd = recover_param_with_sd("gam")
alpha_mean, alpha_sd = recover_param_with_sd("alpha")

recovered_df = pd.DataFrame({
    "participant": np.arange(1, len(w0_mean) + 1),
    "w0": w0_mean, "w0_sd": w0_sd,
    "w1": w1_mean, "w1_sd": w1_sd,
    "w2": w2_mean, "w2_sd": w2_sd,
    "w3": w3_mean, "w3_sd": w3_sd,
    "gamma": gamma_mean, "gamma_sd": gamma_sd,
    "alpha": alpha_mean, "alpha_sd": alpha_sd,
})
recovered_df.to_csv("outputs/simulation/HBA_addit_bla_1st_simulated_parameters.csv")


# 2nd phat ppe
# Load newly uploaded file to extract all required parameters
param_df = pd.read_csv("outputs/HBA_phatppe_bla_2nd_parameter_summary.csv")

# Recover w0 to w3
w0_recov = recover_param("w0")
w1_recov = recover_param("w1")
w2_recov = recover_param("w2")

# Recover gamma with Phi_approx
gamma_recov = recover_param("gam")
alpha_recov = recover_param("alpha")

# Combine into a DataFrame
recovered_df = pd.DataFrame({
    "participant": np.arange(1, len(w0_recov) + 1),
    "w0": w0_recov,
    "w1": w1_recov,
    "w2": w2_recov,
    "gamma": gamma_recov,
    "alpha": alpha_recov
})
recovered_df.to_csv("outputs/simulation/HBA_phatppe_bla_2nd_original_parameters.csv")

# 파라미터 ground truth
true_w0 = recovered_df["w0"]
true_w1 = recovered_df["w1"]
true_w2 = recovered_df["w2"]

true_gamma = recovered_df["gamma"]
true_alpha = recovered_df["alpha"]

i_vec = []
t_vec = []
T_temp = 0

# happy 계산
happy = np.full((N, T), -999.0)

# Calculate happiness
phat = 0.5

phat_d = np.zeros((N, T))
ppe_d = np.zeros((N, T))

for i in range(N):
    T_temp = 0
    for t in range(T):        
        if t == 0:
            phat_d[i, t] = phat
            ppe = winLose[i, t] - phat
            ppe_d[i, t] = ppe
        else:
            phat_d[i, t] = true_gamma[i] * phat_d[i, t - 1] + phat
            ppe = winLose[i, t] - phat
            ppe_d[i, t] = true_gamma[i] * ppe_d[i, t - 1] + ppe

        phat = phat + true_alpha[i] * ppe

        if t % 4 == 3:
            i_vec.append(i + 1) #stan 기준
            t_vec.append(t + 1)
            T_temp += 1
            mu = (true_w0[i] 
                  + true_w1[i] * phat_d[i, t]
                  + true_w2[i] * ppe_d[i, t] 
                 )
            happy[i, t] = np.random.normal(mu, 1.0)


# In[ ]:


Tsubj = [T for _ in range(N)]

stan_data = {
    "N": N,
    "T": T,
    "Tsubj": Tsubj,
    "gamble": gamble.tolist(),
    "winLose": winLose.tolist(),
    "happy": happy.tolist(),
    "happy_num": T_temp,
    "i_vec": i_vec,
    "t_vec": t_vec
}

with open("simulated_blain_2ndphatppe.json", "w") as f:
    json.dump(stan_data, f)

model = CmdStanModel(stan_file='happy_phatppe_hba.stan')

fit = model.sample(data="simulated_blain_2ndphatppe.json", chains=4, parallel_chains=4, 
                   iter_warmup=1000, iter_sampling=1000, 
                   seed=2025, adapt_delta=0.95, max_treedepth=15,
                   show_progress=True)
results = save_stan_outputs_and_evaluation(fit, prefix="HBA_phatppe_bla_2nd")

# Load newly uploaded file to extract all required parameters
param_df = pd.read_csv("outputs/simulation/HBA_phatppe_bla_2nd_parameter_summary.csv")

# Recover w0 to w3
w0_mean, w0_sd = recover_param_with_sd("w0")
w1_mean, w1_sd = recover_param_with_sd("w1")
w2_mean, w2_sd = recover_param_with_sd("w2")
gamma_mean, gamma_sd = recover_param_with_sd("gam")
alpha_mean, alpha_sd = recover_param_with_sd("alpha")

recovered_df = pd.DataFrame({
    "participant": np.arange(1, len(w0_mean) + 1),
    "w0": w0_mean, "w0_sd": w0_sd,
    "w1": w1_mean, "w1_sd": w1_sd,
    "w2": w2_mean, "w2_sd": w2_sd,
    "gamma": gamma_mean, "gamma_sd": gamma_sd,
    "alpha": alpha_mean, "alpha_sd": alpha_sd,
})
recovered_df.to_csv("outputs/simulation/HBA_phatppe_bla_2nd_simulated_parameters.csv")


# 2nd mixed addit
# Load newly uploaded file to extract all required parameters
param_df = pd.read_csv("outputs/HBA_mixed_addit_bla_2nd_parameter_summary.csv")

# Recover w0 to w3
w0_recov = recover_param("w0")
w1_recov = recover_param("w1")
w2_recov = recover_param("w2")
w3_recov = recover_param("w3")
w4_recov = recover_param("w4")


# Recover gamma with Phi_approx
gamma_recov = recover_param("gam")

alpha_recov = recover_param("alpha")

# Combine into a DataFrame
recovered_df = pd.DataFrame({
    "participant": np.arange(1, len(w0_recov) + 1),
    "w0": w0_recov,
    "w1": w1_recov,
    "w2": w2_recov,
    "w3": w3_recov,
    "w4": w4_recov,
    "gamma": gamma_recov,
    "alpha": alpha_recov
})

recovered_df.to_csv("outputs/simulation/HBA_mixed_addit_bla_2nd_original_parameters.csv")

# 파라미터 ground truth
true_w0 = recovered_df["w0"]
true_w1 = recovered_df["w1"]
true_w2 = recovered_df["w2"]
true_w3 = recovered_df["w3"]
true_w4 = recovered_df["w4"]

true_gamma = recovered_df["gamma"]
true_alpha = recovered_df["alpha"]

i_vec = []
t_vec = []
T_temp = 0

# happy 계산
happy = np.full((N, T), -999.0)

# Calculate happiness
phat = 0.5
ev = 0
rpe = 0

cr_d = np.zeros((N, T))
ev_d = np.zeros((N, T))
rpe_d = np.zeros((N, T))
ppe_d = np.zeros((N, T))

for i in range(N):
    T_temp = 0
    for t in range(T):
        value = gain[i, t]
        if choice[i, t] == 1:
            value = loss[i, t]
            
        ev = phat * value
        rpe = outcome[i, t] - ev
        ppe = winLose[i, t] - phat
        
        if t == 0:
            cr_d[i, t] = cr[i, t]
            ev_d[i, t] = ev
            rpe_d[i, t] = rpe 
            ppe_d[i, t] = ppe
        else:
            cr_d[i, t] = true_gamma[i] * cr_d[i, t - 1] + cr[i, t]
            ev_d[i, t] = true_gamma[i] * ev_d[i, t - 1] + ev
            rpe_d[i, t] = true_gamma[i] * rpe_d[i, t - 1] + rpe
            ppe_d[i, t] = true_gamma[i] * ppe_d[i, t - 1] + ppe

        phat = phat + true_alpha[i] * ppe

        if t % 4 == 3:
            i_vec.append(i + 1) #stan 기준
            t_vec.append(t + 1)
            T_temp += 1
            mu = (true_w0[i] 
                  + true_w1[i] * cr_d[i, t]
                  + true_w2[i] * ev_d[i, t] 
                  + true_w3[i] * rpe_d[i, t] 
                  + true_w4[i] * ppe_d[i, t] 
                 )
            happy[i, t] = np.random.normal(mu, 1.0)

Tsubj = [T for _ in range(N)]

stan_data = {
    "N": N,
    "T": T,
    "Tsubj": Tsubj,
    "gain": gain.tolist(),
    "loss": loss.tolist(),
    "cert": cr.tolist(),
    "gamble": gamble.tolist(),
    "choice": choice.tolist(),
    "outcome": outcome.tolist(),
    "winLose": winLose.tolist(),
    "happy": happy.tolist(),
    "happy_num": T_temp,
    "i_vec": i_vec,
    "t_vec": t_vec
}

with open("simulated_blain_2ndmixed.json", "w") as f:
    json.dump(stan_data, f)

model = CmdStanModel(stan_file='happy_Mixed_addit_hba.stan')

fit = model.sample(data="simulated_blain_2ndmixed.json", chains=4, parallel_chains=4, 
                   iter_warmup=1000, iter_sampling=1000, 
                   seed=2025, adapt_delta=0.95, max_treedepth=15,
                   show_progress=True)

results = save_stan_outputs_and_evaluation(fit, prefix="HBA_mixed_addit_bla_2nd")
print(results["eval_csv"])  # 저장된 평가 지표 파일 경로

# Load newly uploaded file to extract all required parameters
param_df = pd.read_csv("outputs/simulation/HBA_mixed_addit_bla_2nd_parameter_summary.csv")

# Recover w0 to w3
w0_mean, w0_sd = recover_param_with_sd("w0")
w1_mean, w1_sd = recover_param_with_sd("w1")
w2_mean, w2_sd = recover_param_with_sd("w2")
w3_mean, w3_sd = recover_param_with_sd("w3")
w4_mean, w4_sd = recover_param_with_sd("w4")

gamma_mean, gamma_sd = recover_param_with_sd("gam")

alpha_mean, alpha_sd = recover_param_with_sd("alpha")

recovered_df = pd.DataFrame({
    "participant": np.arange(1, len(w0_mean) + 1),
    "w0": w0_mean, "w0_sd": w0_sd,
    "w1": w1_mean, "w1_sd": w1_sd,
    "w2": w2_mean, "w2_sd": w2_sd,
    "w3": w3_mean, "w3_sd": w3_sd,
    "w4": w4_mean, "w4_sd": w4_sd,
    "gamma": gamma_mean, "gamma_sd": gamma_sd,
    "alpha": alpha_mean, "alpha_sd": alpha_sd,
})

recovered_df.to_csv("outputs/simulation/HBA_mixed_addit_bla_2nd_simulated_parameters.csv")

