from helper_funcs import *
import statsmodels.api as sm

def time_series_base(base_data, pre_exp_data, save_name = ""):
    # plot pre-experiment data 
    plt.plot(pre_exp_data['x_pre'], pre_exp_data['y_pre'], color = 'black', linewidth=0.5, label='Pre-Experiment Data')
    plt.plot(pre_exp_data['x_pre'], pre_exp_data['pre_fit'], color='purple', linestyle ='--', linewidth = 0.5, label='Fitted Model')
    # plot the test data before treatment assignment
    plt.plot(base_data['x'], base_data['y'], 'o', color = 'black', label='In-Experiment Data')
    plt.plot(base_data['x'], base_data['y'], color = 'black', ls = '-', linewidth = 0.5)
    # add labels
    plt.xlabel("Time")
    plt.ylabel("Log(Pageview)")
    plt.legend()
    if save_name:
        plt.savefig(save_name + ".png")
    plt.show()


def time_series_instance(exp_data, base_data, pre_exp_data, te, save_name = "", model = {}):
    # plot pre-experiment data and in-experiment data
    plt.plot(pre_exp_data['x_pre'], pre_exp_data['y_pre'], color = 'black', linewidth=0.5, label='Pre-Experiment Data')
    plt.plot(pre_exp_data['x_pre'], pre_exp_data['pre_fit'], color='purple', linestyle ='--', linewidth = 0.5, label='Fitted Model')
    plt.plot(exp_data['control_x'], exp_data['control_y'], 'o', color = control_col, label = 'Control Outcome', alpha = strong_alpha)
    plt.plot(exp_data['treat_x'], exp_data['treat_y'], 'o', color = treat_col, label = 'Treatment Outcome', alpha = strong_alpha)
    if not model: # if not dealing with particular model, plot the true test data with treatment effect applied
        plt.plot(base_data['x'], base_data['y'], color = control_col, ls = '-', linewidth = 0.5)
        plt.plot(base_data['x'], base_data['y'] + te, color = treat_col, ls = '-', linewidth = 0.5)

    else: # otherwise, plot the counterfactual estimates
        n = len(exp_data['treat_y'])
        alpha_1 = model['alpha_1']
        alpha_0 = model['alpha_0']
        mod = model['mod']
        # create a copy of exp data where we replace date x with numeric x
        temp_exp_data = exp_data.copy()
        temp_exp_data['control_x'] = temp_exp_data['control_x_num']
        temp_exp_data['treat_x'] = temp_exp_data['treat_x_num']
        cf_dict = gen_counter_data(base_data['x_num_range'], temp_exp_data, mod, alpha_1, alpha_0)
        # delete temp_exp_data after usage
        del temp_exp_data
        control_mod = cf_dict['control_mod']
        treat_mod = cf_dict['treat_mod']
        control_cfs = cf_dict['control_cfs']
        treat_cfs = cf_dict['treat_cfs']
        x = base_data['x']
        # if plotting counterfactual estimates, add them in with hollow circles. generate lines for counterfactual models
        plt.plot(exp_data['control_x'], control_cfs, 'o',fillstyle='none',color = treat_col,label='Est. Treatment Outcome',alpha=strong_alpha)
        plt.plot(exp_data['treat_x'], treat_cfs, 'o',fillstyle='none',color=control_col,label='Est. Control Outcome',alpha=strong_alpha)
        plt.plot(x, control_mod, linestyle='--', color=control_col, alpha = strong_alpha)
        plt.plot(x, treat_mod, linestyle='--', color=treat_col, alpha = strong_alpha)

        # compute ATE and SE
        ATE = np.mean(np.append(exp_data['treat_y'] - treat_cfs, control_cfs - exp_data['control_y']))
        SE = (np.var(exp_data['treat_y'] - treat_cfs, ddof = 1)/n + np.var(exp_data['control_y'] - control_cfs, ddof = 1)/n)**0.5

    # add labels
    plt.xlabel("Time")
    plt.ylabel("Log(Pageview)")
    plt.legend()
    if save_name:
        plt.savefig(save_name + ".png")
    plt.show()

    if model:
        print("ATE: ", ATE, ", SE: ", SE)

def quartic_secondary_est(exp_data, pre_exp_model):
    # ml with secondary quartic adjustment
    control_r = exp_data['control_y'] - pre_exp_model(exp_data['control_x'])
    treat_r = exp_data['treat_y'] - pre_exp_model(exp_data['treat_x'])

    # set up input matrices X and y
    x_s = np.concatenate((exp_data['treat_x'], exp_data['control_x']))
    n = len(exp_data['treat_y'])
    treat_indicator = np.concatenate((np.ones(n), np.zeros(n)))
    X_mat = np.column_stack((np.ones(2*n), treat_indicator, x_s, x_s**2, x_s**3, x_s**4))
    y_vec = np.concatenate((treat_r, control_r))
    model = sm.OLS(y_vec, X_mat)
    results = model.fit()
    params = results.params

    def quartic_mod(x):
        return  pre_exp_model(x) + params[0] + x * params[2] + x**2 * params[3] + x**3 * params[4] + x**4 * params[5]

    q_est = np.mean(exp_data['treat_y'] - quartic_mod(exp_data['treat_x'])) - np.mean(exp_data['control_y'] - quartic_mod(exp_data['control_x']))
    q_se = (np.var(exp_data['treat_y'] - quartic_mod(exp_data['treat_x']), ddof=1)/n +
            np.var(exp_data['control_y'] - quartic_mod(exp_data['control_x']),ddof=1)/n)**0.5
    del x_s, n, treat_indicator, X_mat, y_vec, model, results, params, control_r, treat_r

    return [q_est, q_se]

def gen_distribution_q(base_data, pre_exp_model, sample_sizes, sim_num, te):
    # simulating multiple experiments for all 4 models
    # begin simulating experiments
    res = {}
    x = base_data['x']
    y = base_data['y']
    for key in sample_sizes:
        temp = []
        n = sample_sizes[key]
        for _ in range(sim_num):
            exp_data = gen_exp_data(x,y,n,te)
            temp.append(np.concatenate([const_est(exp_data), 
                                        regress_est(exp_data), 
                                        flex_ml_est(exp_data, pre_exp_model), 
                                        secondary_est(exp_data, pre_exp_model),
                                        quartic_secondary_est(exp_data,pre_exp_model)]))
        res[key] = pd.DataFrame(temp, columns = ['two_est','two_se',
                                                 'reg_est','reg_se',
                                                 'ml_est','ml_se',
                                                 's_est','s_se',
                                                 'q_est','q_se']) 
        del temp
    # return the result which is a dictionary of dataframes
    return res

def plot_ATE_distribution_q(res, te, sample_sizes, save_name = ""):
    # modified for quartic secondary adjustment
    for key in res:
        plt.figure(dpi = dpi)
        sim_res = res[key]
        # use this to first determine overall bin range
        bins_max = max(max(sim_res['s_est']),max(sim_res['q_est']))
        bins_min = min(min(sim_res['s_est']),min(sim_res['q_est']))
        bins_freq = 100
        bins_range = np.arange(bins_min, bins_max, (bins_max - bins_min)/bins_freq)
        # plot the estimates
        plt.hist(sim_res['s_est'], bins = bins_range, color = colors['mix'], label=r"$\hat{\tau}_{s}$",alpha=0.5)
        plt.hist(sim_res['q_est'], bins = bins_range, color = colors['quad'], label=r"$\hat{\tau}_{q}$",alpha=0.5)

        # add labels
        top_loc = plt.gca().get_ylim()[1]
        plt.text(te, top_loc, r"$\tau = 2$",verticalalignment='top')
        plt.axvline(te, color = 'black', linestyle='-',alpha=0.5)
        plt.ylabel("Frequency")
        plt.xlabel(r"$\hat{\tau}$")
        plt.legend(loc=1)
        if save_name:
            plt.savefig(save_name + "_" + str(key) + "_ate.png")
        plt.show()
        print("Sample size:",sample_sizes[key])


def plot_SE_distribution_q(res, save_name = ""):
    for key in res:
        plt.figure(dpi=dpi)
        sim_res = res[key]
        # use this to first determine overall bin range
        bins_max = max(max(sim_res['s_se']),max(sim_res['q_se']))
        bins_min = min(min(sim_res['s_se']),min(sim_res['q_se']))
        bins_freq = 100
        bins_range = np.arange(bins_min, bins_max, (bins_max - bins_min)/bins_freq)
        # plot the SE estimates
        plt.hist(sim_res['s_se'], bins = bins_range, color = colors['mix'], label=r"$SE(\hat{\tau}_{s})$",alpha=0.5)
        plt.hist(sim_res['q_se'], bins = bins_range, color = colors['quad'], label=r"$SE(\hat{\tau}_{q})$",alpha=0.5)
        plt.axvline(np.std(sim_res['s_est']), color = colors['mix'], linestyle='--',alpha=0.5)
        plt.axvline(np.std(sim_res['q_est']), color = colors['quad'], linestyle='--',alpha=0.5)
        # add labels
        plt.ylabel("Frequency")
        plt.xlabel(r"$SE(\hat{\tau})$")
        plt.legend(loc=1)
        if save_name:
            plt.savefig(save_name + "_" + str(key) + "_se.png")
        plt.show()