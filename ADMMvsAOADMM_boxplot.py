import numpy as np
import pytensorlab as ptl
from admm_tensors import admm_tensor
from ao_admm_tensors import ao_admm_tensor
from pytensorlab import PolyadicTensor
import matplotlib.pyplot as plt
from pytensorlab import CPDAlsOptions

# -----------------------
# Problem setup
# -----------------------
m, n, p = 120, 110, 100
r = 10
num_runs = 100

# Storage for final errors
final_error_admm = []
final_error_aoadmm = []
final_error_als=[]
#Storage for time to reach 10-6 error
time_admm_record = []
time_aoadmm_record = []

# Storage for histories
obj_admm_runs = []
time_admm_runs = []

obj_aoadmm_runs = []
time_aoadmm_runs = []


for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}")

    # Generate random tensor
    A0 = np.random.rand(m, r)
    B0 = np.random.rand(n, r)
    C0 = np.random.rand(p, r)
    U = PolyadicTensor([A0, B0, C0])
    X = np.array(U)

    # Initialization
    Ainit = np.random.randn(m, r)
    Binit = np.random.randn(n, r)
    Cinit = np.random.randn(p, r)
    Uinit = PolyadicTensor([Ainit, Binit, Cinit])

    # -----------------------
    # ADMM
    # -----------------------
    parameters_admm = {
        'rho': 1,
        'maxiter': np.inf,
        'maxtime': 10,
        'Ainit': Ainit,
        'Binit': Binit,
        'Cinit': Cinit
    }

    _, _, _, _, obj_admm, _, time_admm,time_rec_admm = admm_tensor(X, r, 'KL divergence', parameters_admm)
    
    # Record only the final error and time
    final_error_admm.append(obj_admm[-1])
    time_admm_record.append(time_rec_admm)

    obj_admm_runs.append(obj_admm)
    time_admm_runs.append(time_admm)

    # -----------------------
    # AO-ADMM
    # -----------------------
    parameters_aoadmm = {
        'maxiter': np.inf,
        'inner_iter': 10,
        'maxtime': 10,
        'Ainit': Ainit,
        'Binit': Binit,
        'Cinit': Cinit
    }

    _, _, _, _, obj_aoadmm, _, time_aoadmm,time_rec_aoadmm = ao_admm_tensor(X, r, 'KL divergence', parameters_aoadmm)
    
    final_error_aoadmm.append(obj_aoadmm[-1])
    time_aoadmm_record.append(time_rec_aoadmm)

    obj_aoadmm_runs.append(obj_aoadmm)
    time_aoadmm_runs.append(time_aoadmm)

    ## Define CPD-ALS parameters
    als_options = CPDAlsOptions(max_iterations=300, tol_relfval=1e-16, tol_relstep=1e-16, tol_absfval=0.0, display=0) 
    Uopt_als,logger_als = ptl.cpd_als(X,Uinit,options=als_options)
    # Final relative reconstruction error
    final_relerr_als = np.sqrt(2 * logger_als.fval[-1]) / ptl.frob(X)
    # Record one value per run
    final_error_als.append(final_relerr_als)

# -----------------------
# Averaging (truncate to common length)
# -----------------------
min_len_admm = min(len(o) for o in obj_admm_runs)
min_len_aoadmm = min(len(o) for o in obj_aoadmm_runs)

obj_admm_median = np.median(
    np.array([o[:min_len_admm] for o in obj_admm_runs]), axis=0
)
time_admm_median = np.median(
    np.array([t[:min_len_admm] for t in time_admm_runs]), axis=0
)

obj_aoadmm_median = np.median(
    np.array([o[:min_len_aoadmm] for o in obj_aoadmm_runs]), axis=0
)
time_aoadmm_median = np.median(
    np.array([t[:min_len_aoadmm] for t in time_aoadmm_runs]), axis=0
)

# -----------------------
# Save final errors to text files
# -----------------------
np.savetxt("final_error_admm_boxplot.txt", final_error_admm)
np.savetxt("final_error_aoadmm_boxplot.txt", final_error_aoadmm)
np.savetxt("final_error_als_boxplot.txt",final_error_als)
np.savetxt("time_admm_1e_6.txt", time_admm_record)
np.savetxt("time_aoadmm_1e_6.txt", time_aoadmm_record)

# -----------------------
# Save median results
# -----------------------
np.savetxt("objective_admm_median.txt", obj_admm_median)
np.savetxt("objective_aoadmm_median.txt", obj_aoadmm_median)
np.savetxt("time_admm_median.txt", time_admm_median)
np.savetxt("time_aoadmm_median.txt", time_aoadmm_median)


## Plot figure
plt.figure(figsize=(7, 5))  # slightly wider
plt.boxplot([final_error_admm, final_error_aoadmm, final_error_als],
            tick_labels=['ADMM', 'AO-ADMM', 'ALS'])
plt.ylabel("Final reconstruction error")
plt.title("Distribution of Final Errors over 100 Runs")
plt.yscale('log')
plt.grid(True, axis='y', which='both')
plt.subplots_adjust(left=0.18, right=0.95, top=0.90, bottom=0.15)
plt.savefig("final_error_boxplot.png", dpi=200)

## Median plots
plt.figure(figsize=(7, 5))
plt.semilogy(time_admm_median, obj_admm_median, label='ADMM (avg)', linewidth=2)
plt.semilogy(time_aoadmm_median, obj_aoadmm_median, label='AO-ADMM (avg)', linewidth=2)
plt.xlabel("Time (seconds)")
plt.ylabel("Median value")
plt.title("Median Objective vs Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("median_comparison_avg.png", dpi=200)

## Percentage of times alg reaches 1e-6 and average time taken to reach
#boolean mask of non-zero elements
time_admm_record = np.asarray(time_admm_record)
time_aoadmm_record = np.asarray(time_aoadmm_record)
nonzero_mask_admm = time_admm_record !=0
nonzero_mask_aoadmm = time_aoadmm_record !=0

# number of non-zero elements/success
num_nonzero_admm = np.count_nonzero(nonzero_mask_admm)
num_nonzero_aoadmm = np.count_nonzero(nonzero_mask_aoadmm)

# average of non-zero elements
avg_nonzero_admm = np.mean(time_admm_record[nonzero_mask_admm]) if num_nonzero_admm > 0 else 0.0
avg_nonzero_aoadmm = np.mean(time_aoadmm_record[nonzero_mask_aoadmm]) if num_nonzero_aoadmm > 0 else 0.0

print("Percentage of success ADMM:", num_nonzero_admm)
print("Average time taken to reach 1e-6 ADMM:", avg_nonzero_admm)
print("Percentage of success AO-ADMM:", num_nonzero_aoadmm)
print("Average time taken to reach 1e-6 AO-ADMM:", avg_nonzero_aoadmm)

