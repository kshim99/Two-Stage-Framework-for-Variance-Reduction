# Helper functions for creating visualization in Thesis
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline

# set global params for plotting
# alpha is the transparency param, we use strong alpha (less transparent) and weak alpha (more transparent) to highlight certain data in the plots
weak_alpha = 0.2
strong_alpha = 0.7
# by default, treatment color will be red, and control color will be blue. Pre-experiment data is grey
treat_col = 'red'
control_col = 'blue'
pre_col = 'grey'
# when plotting results for different models, use a dictionary of colors, and use different color for different models
colors = {'two': 'gray', 'reg': 'goldenrod', 'ml': 'cadetblue', 'mix': 'slateblue', 'quad': 'firebrick'}
dpi = 200

def gen_base_data():
    
    # generate basis data (x,y) that is to be used in simulation
    x = np.arange(70,100,1)
    n = len(x)
    error = np.random.normal(0,1,n)
    y = 2*np.sin(x*2) + 0.01*((x - 70)**2) + 0.4*error
    return x, y

def gen_exp_data(x,y,n,te):
    
    # generate an instance of experiment data (n control and n treatment) given base data (x,y), sample size (n), and treatment effect (te)
    base_len = len(y)
    
    # randomly select units to apply the treatmente effect
    control_idx = random.choices(range(base_len),k=n)
    treat_idx = random.choices(range(base_len), k=n)
    
    # generate in-experiment data
    control_x = x[control_idx]
    control_y = y[control_idx]
    treat_x = x[treat_idx]
    treat_y = y[treat_idx] + te
    
    # return control x, y, and treatment x, y
    return {'control_x': control_x, 'control_y': control_y, 'control_idx': control_idx, 
            'treat_x': treat_x, 'treat_y': treat_y, 'treat_idx': treat_idx}

def gen_counter_data(x, exp_data, mod, alpha_1, alpha_0):
    # x is the basis x so we can generate counterfactual model across the entire domain 
    # mod is the counterfactual model WITHOUT the alphas - the part that is shared by both control and treatment units
    # e.g. for regression where Y_i(W_i) = alpha_1 * W_i + alpha_0 * (1-W_i) + x * beta, mod = x * beta
    # mod is passed through as a function. e.g. def mod(x): return x * beta
    # returns a dictionary of estimated treatment outcomes, estimated control outcomes, and the lines representing control and treatment models
    # and this is to be used as an input for plot_instance
    control_x = exp_data['control_x']
    treat_x = exp_data['treat_x']
    control_mod = mod(x) + alpha_0
    treat_mod = mod(x) + alpha_1
    control_cfs = mod(control_x) + alpha_1 # these are estimated TREATMENT outcomes for CONTROL observations
    treat_cfs = mod(treat_x) + alpha_0 # these are estimated CONTROL outcomes for TREATMENT observations
    
    del control_x, treat_x
    return {'control_mod': control_mod, 'treat_mod': treat_mod, 'control_cfs': control_cfs, 'treat_cfs': treat_cfs, 'x': x}


def plot_instance(in_exp_dict, x_label, y_label, cf_dict = {}, pre_exp_dict = {}, flip_alpha = False, save_name = ""):
    # set figure resolution
    plt.figure(dpi = dpi)

    # if flip_alpha = True, we bring out the pre-exp data with strong alpha and push back in_exp data with weak alpha
    # otherwise, in_exp data is highlighted with strong alpha
    if flip_alpha:
        s_alpha = weak_alpha
        w_alpha = strong_alpha
    else:
        s_alpha = strong_alpha
        w_alpha = weak_alpha

    # in_exp_dict is a dictionary of inputs for in_experiment data
    # similarly, cf_dict and pre_exp_dict are dictionaries of inputs for counterfactual and pre_experiment data 
    # the latter two are optional, but cf dict is necessary for ATE/SE calculation based on its model
    control_x = in_exp_dict['control_x']
    control_y = in_exp_dict['control_y']
    treat_x = in_exp_dict['treat_x']
    treat_y = in_exp_dict['treat_y']
    n_1, n_0 = len(treat_y), len(treat_x)
    plt.plot(control_x, control_y, 'o', color = control_col, label='Control Outcome',alpha=s_alpha)
    plt.plot(treat_x, treat_y, 'o', color = treat_col, label = 'Treatment Outcome',alpha=s_alpha)
    plt.ylim([min(np.append(control_y, treat_y)) - 1, max(np.append(control_y, treat_y)) + 1])
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    
    if cf_dict:
        control_mod = cf_dict['control_mod']
        treat_mod = cf_dict['treat_mod']
        control_cfs = cf_dict['control_cfs']
        treat_cfs = cf_dict['treat_cfs']
        x = cf_dict['x']
        # if plotting counterfactual estimates, add them in with hollow circles. generate lines for counterfactual models
        plt.plot(control_x, control_cfs, 'o',fillstyle='none',color = treat_col,label='Est. Treatment Outcome',alpha=s_alpha)
        plt.plot(treat_x, treat_cfs, 'o',fillstyle='none',color=control_col,label='Est. Control Outcome',alpha=s_alpha)
        plt.plot(x, control_mod, linestyle='--', color=control_col, alpha = s_alpha)
        plt.plot(x, treat_mod, linestyle='--', color=treat_col, alpha = s_alpha)

        # update y_lim if necessary
        cur_min, cur_max = plt.gca().get_ylim()
        cur_min = min(cur_min, min(np.append(control_cfs, treat_cfs))) - 1
        cur_max = max(cur_max, max(np.append(control_cfs, treat_cfs))) + 1
        plt.ylim([cur_min, cur_max])

        # compute ATE and SE
        ATE = np.mean(np.append(treat_y - treat_cfs, control_cfs - control_y))
        SE = (np.var(treat_y - treat_cfs, ddof = 1)/n_1 + np.var(control_y - control_cfs, ddof = 1)/n_0)**0.5

    if pre_exp_dict:
        fine_x = pre_exp_dict['fine_x']
        pre_fit = pre_exp_dict['pre_fit']
        x_pre = pre_exp_dict['x_pre']
        y_pre = pre_exp_dict['y_pre']
        plt.plot(fine_x,pre_fit, color='grey',alpha=w_alpha)
        plt.plot(x_pre, y_pre, 'o',color='grey',label="Pre-Experiment Data",alpha=w_alpha)
        # update y_lim if necessary
        cur_min, cur_max = plt.gca().get_ylim()
        cur_min = min(cur_min, min(y_pre)) - 1
        cur_max = max(cur_max, max(y_pre)) + 1
        plt.ylim([cur_min, cur_max])
    # include legend
    plt.legend()


    # if saving figure, provide name
    # ensure plots subfolder exists before this is called
    if save_name:
        plt.savefig(save_name)
    plt.show()

    # return the ATE and SE if counterfactual estimates are provided
    if cf_dict:
        print("ATE: ", ATE, ", SE: ", SE)


def gen_pre_exp_data():
    # generate pre-experiment data
    fine_x = np.arange(69.5,100.5,0.1) # this is the fine domain over which we can plot the pre-exp function we fit. We widen up the domain to avoid extreme edge behaviours for more sensible visualization
    x_pre = np.array(random.sample(list(fine_x), 100)) # this generates a sample of pre-exp x
    x_pre.sort() # need sorting for BSpline
    y_pre = 2*np.sin(x_pre*2) + 0.2*((x_pre - 70)) + np.random.normal(0,1,len(x_pre)) * 0.5 # this generates a sample of pre-exp y
    tck = splrep(x_pre, y_pre, s=15) # this fits a spline with smoothing param of s to our pre-exp sample data
    pre_fit = BSpline(*tck)(fine_x) # we can plot the pre-exp model over the fine domain 
    return {'fine_x': fine_x, 'x_pre': x_pre, 'y_pre': y_pre, 'pre_fit': pre_fit, 'model': BSpline(*tck)}

def gen_pre_exp_data_poor():
    # generate pre-experiment data
    fine_x = np.arange(69.5,100.5,0.1) # this is the fine domain over which we can plot the pre-exp function we fit. We widen up the domain to avoid extreme edge behaviours for more sensible visualization
    x_pre = np.array(random.sample(list(fine_x), 100)) # this generates a sample of pre-exp x
    x_pre.sort() # need sorting for BSpline
    y_pre = 2*np.sin(x_pre*2) + np.random.normal(0,1,len(x_pre)) * 0.5 # this generates a sample of pre-exp y
    tck = splrep(x_pre, y_pre, s=15) # this fits a spline with smoothing param of s to our pre-exp sample data
    pre_fit = BSpline(*tck)(fine_x) # we can plot the pre-exp model over the fine domain 
    return {'fine_x': fine_x, 'x_pre': x_pre, 'y_pre': y_pre, 'pre_fit': pre_fit, 'model': BSpline(*tck)}

# define functions  for getting ATE and SE for each model
def const_est(exp_data):
    # constant model
    two_est = np.mean(exp_data['treat_y']) - np.mean(exp_data['control_y'])
    two_se = (np.var(exp_data['treat_y'],ddof=1)/len(exp_data['treat_y']) + np.var(exp_data['control_y'],ddof=1)/len(exp_data['control_y']))**0.5
    return [two_est, two_se]

def regress_est(exp_data):
    # regression model
    treatxvar = np.var(exp_data['treat_x'],ddof=1)
    controlxvar = np.var(exp_data['control_x'],ddof=1)
    treatxcov = np.cov(exp_data['treat_x'],exp_data['treat_y'],ddof=1)[0,1]
    controlxcov = np.cov(exp_data['control_x'],exp_data['control_y'],ddof=1)[0,1]
    beta = (treatxcov + controlxcov)/(treatxvar + controlxvar)
    alpha_1 = np.mean(exp_data['treat_y']) - np.mean(exp_data['treat_x']) * beta
    alpha_0 = np.mean(exp_data['control_y']) - np.mean(exp_data['control_x']) * beta
    reg_est = alpha_1 - alpha_0
    reg_se = (np.var(exp_data['treat_y'] - exp_data['treat_x'] * beta, ddof=1)/len(exp_data['treat_y']) + 
                np.var(exp_data['control_y'] - exp_data['control_x'] * beta, ddof=1)/len(exp_data['control_y']))**0.5
    del treatxvar, controlxvar, treatxcov, controlxcov, beta, alpha_1, alpha_0
    return [reg_est, reg_se]

def flex_ml_est(exp_data, pre_exp_model):
    # ml model
    # make a copy of it with outcomes replaced with residual outcomes
    copy_data = exp_data.copy()
    copy_data['control_y'] = exp_data['control_y'] - pre_exp_model(exp_data['control_x'])
    copy_data['treat_y'] = exp_data['treat_y'] - pre_exp_model(exp_data['treat_x'])
    ml_est, ml_se = const_est(copy_data)
    del copy_data
    return [ml_est, ml_se]

def secondary_est(exp_data, pre_exp_model):
    # ml with secondary linear adjustment
    copy_data = exp_data
    copy_data['control_y'] = exp_data['control_y'] - pre_exp_model(exp_data['control_x'])
    copy_data['treat_y'] = exp_data['treat_y'] - pre_exp_model(exp_data['treat_x'])
    s_est, s_se = regress_est(copy_data)
    del copy_data
    return [s_est, s_se]

def gen_distribution(base_data, pre_exp_model, sample_sizes, sim_num, te):
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
            temp.append(np.concatenate([const_est(exp_data), regress_est(exp_data), flex_ml_est(exp_data, pre_exp_model), secondary_est(exp_data, pre_exp_model)]))
        res[key] = pd.DataFrame(temp, columns = ['two_est','two_se','reg_est','reg_se','ml_est','ml_se','s_est','s_se']) 
        del temp
    # return the result which is a dictionary of dataframes
    return res

def plot_ATE_distribution(res, te, sample_sizes, save_name = ""):
    for key in res:
        plt.figure(dpi = dpi)
        sim_res = res[key]
        # use this to first determine overall bin range
        bins_max = max(max(sim_res['two_est']),max(sim_res['reg_est']),max(sim_res['ml_est']),max(sim_res['s_est']))
        bins_min = min(min(sim_res['two_est']),min(sim_res['reg_est']),min(sim_res['ml_est']),min(sim_res['s_est']))
        bins_freq = 100
        bins_range = np.arange(bins_min, bins_max, (bins_max - bins_min)/bins_freq)
        # plot the estimates
        plt.hist(sim_res['two_est'], bins = bins_range, color = colors['two'], label = r"$\hat{\tau}_d$", alpha=0.5)
        plt.hist(sim_res['reg_est'], bins = bins_range, color = colors['reg'], label=r"$\hat{\tau}_r$",alpha=0.5)
        plt.hist(sim_res['ml_est'], bins = bins_range, color = colors['ml'], label=r"$\hat{\tau}_{f}$",alpha=0.5)
        plt.hist(sim_res['s_est'], bins = bins_range, color = colors['mix'], label=r"$\hat{\tau}_{s}$",alpha=0.5)
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
        # print summaries
        print("Sample size:",sample_sizes[key], 
              "\nATE_d:", '{:.2f}'.format(np.mean(sim_res['two_est'])), 
              "\nATE_r:", '{:.2f}'.format(np.mean(sim_res['reg_est'])),
              "\nATE_ml:",'{:.2f}'.format(np.mean(sim_res['ml_est'])),
              "\nATE_s:", '{:.2f}'.format(np.mean(sim_res['s_est'])))
        print("Var_d:", np.var(sim_res['two_est'],ddof=1), 
        "\nVar_red_r:", '{:.2f}%'.format(100 * (1-np.var(sim_res['reg_est'],ddof=1)/np.var(sim_res['two_est'],ddof=1))),
        "\nVar_red_ml: ", '{:.2f}%'.format(100 * (1-np.var(sim_res['ml_est'],ddof=1)/np.var(sim_res['two_est'],ddof=1))),
        "\nVar__red_s:", '{:.2f}%'.format(100 * (1-np.var(sim_res['s_est'],ddof=1)/np.var(sim_res['two_est'],ddof=1))))

def plot_SE_distribution(res, save_name = ""):
    for key in res:
        plt.figure(dpi=dpi)
        sim_res = res[key]
        # use this to first determine overall bin range
        bins_max = max(max(sim_res['two_se']),max(sim_res['reg_se']),max(sim_res['ml_se']),max(sim_res['s_se']))
        bins_min = min(min(sim_res['two_se']),min(sim_res['reg_se']),min(sim_res['ml_se']),min(sim_res['s_se']))
        bins_freq = 100
        bins_range = np.arange(bins_min, bins_max, (bins_max - bins_min)/bins_freq)
        # plot the SE estimates
        plt.hist(sim_res['two_se'], bins = bins_range, color = colors['two'], label = r"$SE(\hat{\tau}_d)$", alpha=0.5)
        plt.hist(sim_res['reg_se'], bins = bins_range, color = colors['reg'], label=r"$SE(\hat{\tau}_r)$",alpha=0.5)
        plt.hist(sim_res['ml_se'], bins = bins_range, color = colors['ml'], label=r"$SE(\hat{\tau}_{f})$",alpha=0.5)
        plt.hist(sim_res['s_se'], bins = bins_range, color = colors['mix'], label=r"$SE(\hat{\tau}_{s})$",alpha=0.5)
        plt.axvline(np.std(sim_res['two_est']), color = colors['two'], linestyle='--',alpha=0.5)
        plt.axvline(np.std(sim_res['reg_est']), color = colors['reg'], linestyle='--',alpha=0.5)
        plt.axvline(np.std(sim_res['ml_est']), color = colors['ml'], linestyle='--',alpha=0.5)
        plt.axvline(np.std(sim_res['s_est']), color = colors['mix'], linestyle='--',alpha=0.5)
        # add labels
        plt.ylabel("Frequency")
        plt.xlabel(r"$SE(\hat{\tau})$")
        plt.legend(loc=1)
        if save_name:
            plt.savefig(save_name + "_" + str(key) + "_se.png")
        plt.show()