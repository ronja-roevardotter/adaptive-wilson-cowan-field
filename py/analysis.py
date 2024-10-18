import numpy as np

import sys
import os

cdir = os.getcwd() #get current directory
os.chdir(cdir) #make sure you are in the current directory
sys.path.append(cdir) #append directory to enable import of functions from current directory files

import kernels as ks
from params import setParams

#import root and eigenvalue/-vector function
from scipy.optimize import root
from scipy.linalg import eigvals


# # THIS FILE CONTAINS ALL FUNCTIONS NECESSARY FOR THE # #
# # LINEAR STABILITY ANALYSIS, BOTH THE 2-EQUATION, AND # #
# # THE 3-EQUATION SYSTEM (local limit theory) # #

# # # - - - # # # - - - transfer functions - - - # # # - - - # # #
    
def F_e(x, params):
    return 1/(1+np.exp(-params.beta_e*(x-params.mu_e)))


def F_i(x, params):
    return 1/(1+np.exp(-params.beta_i*(x-params.mu_i)))

def F_a(x, params):
    return 1/(1+np.exp(-params.beta_a*(x-params.mu_a)))


# # # - - - # # # - - - derivatives of transfer functions - - - # # # - - - # # #

def derivF_e(x, params):
    return ((params.beta_e * np.exp(-params.beta_e*(x-params.mu_e)))/(np.exp(-params.beta_e*(x-params.mu_e))+1)**2)
    
def derivF_i(x, params):
    return ((params.beta_i * np.exp(-params.beta_i*(x-params.mu_i)))/(np.exp(-params.beta_i*(x-params.mu_i))+1)**2)

def derivF_a(x, params):
    return ((params.beta_a * np.exp(-params.beta_a*(x-params.mu_a)))/(np.exp(-params.beta_a*(x-params.mu_a))+1)**2)


# # # - - - # # # - - - inverses of transfer functions - - - # # # - - - # # #

def inverseF_e(y, params):
    return params.mu_e - (1/params.beta_e) * np.log((1/y)-1)

def inverseF_i(y, params):
    return params.mu_i - (1/params.beta_i) * np.log((1/y)-1)

def inverseF_a(y, params):
    return params.mu_a - (1/params.beta_a) * np.log((1/y)-1)


# # # - - - # # # - - - nullcline functions - - - # # # - - - # # #

def activity_ui(ue, params):
    """Returns the excitatory nullcline w.r.t. ue (\frac{due}{dt}=0)
       for the activity-based model"""
    inside = params.w_ee * ue + params.I_e - inverseF_e(ue, params) - params.b*F_a(ue, params)
    return (1/params.w_ei) * inside

def activity_ue(ui, params):
    """Returns the inhibitory nullcline w.r.t. ui (\frac{ui}{dt}=0)
       for the activity-based model"""
    inside = inverseF_i(ui, params) + params.w_ii * ui - params.I_i
    return (1/params.w_ie) * inside


#define linearization matrices

# # # - - - activity-based matrix - - - # # #

def activity_A11(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui + params.I_e - params.b * F_a(ue, params)
    return (1/params.tau_e) * (-1 + derivF_e(Be, params) * (params.w_ee - params.b * derivF_a(ue, params))) 

def activity_A12(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui + params.I_e - params.b * F_a(ue, params)
    return (1/params.tau_e) * (-params.w_ei) * derivF_e(Be, params)

def activity_A21(ue, ui, params):
    Bi = params.w_ie * ue - params.w_ii * ui + params.I_i
    return (1/params.tau_i) * params.w_ie * derivF_i(Bi, params)

def activity_A22(ue, ui, params):
    Bi = params.w_ie * ue - params.w_ii * ui + params.I_i
    return (1/params.tau_i) * (-1 + (-params.w_ii)*derivF_i(Bi, params))

def activity_A(x, params):
    ue = x[0]
    ui = x[1]
    return [[activity_A11(ue, ui, params), activity_A12(ue, ui, params)], 
            [activity_A21(ue, ui, params), activity_A22(ue, ui, params)]]


#define the activity-based model

def activity(x, params):
    ue = x[0]
    ui = x[1]
    
    exc_rhs = ((1/params.tau_e) * (-ue + F_e(params.w_ee*ue - params.w_ei*ui - params.b * F_a(ue, params) + params.I_e, params)))
    
    inh_rhs = ((1/params.tau_i) * (-ui + F_i(params.w_ie*ue - params.w_ii*ui + params.I_i, params)))
    
    return [exc_rhs, inh_rhs]


#function to determine the fixed points, depending on the model-type
def computeFPs(pDict):
    """ Derive all fixed points and collect them in the list fixed_points """
    
    params = setParams(pDict)
    fixed_points=[]
    
    start = 0
    end = 1

    for i in np.linspace(start, end, 61):
        sol = root(activity, [i, i], args=(params,), jac=activity_A, method='lm')
        if sol.success:
            closeness = all(np.isclose(activity(sol.x, params), [0.0, 0.0]))
            if closeness:
                if len(fixed_points)==0: #always append the firstly derived fixed point
                    fixed_points.append(sol.x)
                else:
                    already_derived = False
                    for k in range(len(fixed_points)):
                        if all(np.isclose(sol.x, fixed_points[k], atol=1e-9)):
                            already_derived = True
                        else: 
                            pass
                    if already_derived:
                        pass #skip the already derived fixed points
                    else:
                        fixed_points.append(sol.x)
                        
    fixed_points = np.sort(fixed_points, axis=0)
    
    return fixed_points


#function to determine the stability of fixed points, depending on the fixed points
def checkFixPtsStability(fixed_points, params):
    stability = []
    for i in range(len(fixed_points)):
        ue0 = fixed_points[i][0]
        ui0 = fixed_points[i][1]
        y=[ue0, ui0]
        if params.b == 0:
            A = activity_A(y, params)
        else:
            A = adap_A(y, params)
        w = eigvals(A)
        if all(elem.real<0 for elem in w):
            stability.append(1)
        else: 
            stability.append(0)
    return stability


# # # # # # - - - - -                                       - - - - - # # # # # #
# # # # # # - - - - - Functions for 3-equation system below - - - - - # # # # # #
# # # # # # - - - - -                                       - - - - - # # # # # #


def f_kernel(sigma, k, k_string='gaussian'):
    
    kernel_func = getattr(ks, 'f_'+k_string)
    
    return kernel_func(sigma, k)

def deriv_f_kernel(sigma, k, k_string='gaussian'):
    
    kernel_func = getattr(ks, 'deriv_f_' + k_string)
    
    return kernel_func(sigma, k)


def a_jkValues(fp, params):
    
    exc = fp[0]
    inh = fp[1]
    
    b_e = params.w_ee*exc - params.w_ei*inh - params.b*F_a(exc, params) + params.I_e
    b_i = params.w_ie*exc - params.w_ii*inh + params.I_i
    
    a_ee = params.w_ee * derivF_e(b_e, params)
    a_ei = params.w_ei * derivF_e(b_e, params)
    a_ie = params.w_ie * derivF_i(b_i, params)
    a_ii = params.w_ii * derivF_i(b_i, params)
   
        
    return a_ee, a_ei, a_ie, a_ii


# # # - - - LINEARIZATION MATRIX WITH ADAPTATIOn - from 2x2 to 3x3 - - - # # #

def adap_A11(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui - params.b * F_a(ue, params) + params.I_e
    return (1/params.tau_e) * (-1 + params.w_ee*derivF_e(Be, params))

def adap_A12(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui - params.b * F_a(ue, params) + params.I_e
    return (1/params.tau_e) * (-params.w_ei) * derivF_e(Be, params)

def adap_A13(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui - params.b * F_a(ue, params) + params.I_e
    return (1/params.tau_e) * (- params.b * derivF_e(Be, params))

def adap_A21(ue, ui, params):
    Bi = params.w_ie * ue - params.w_ii * ui + params.I_i
    return (1/params.tau_i) * params.w_ie * derivF_i(Bi, params)

def adap_A22(ue, ui, params):
    Bi = params.w_ie * ue - params.w_ii * ui + params.I_i
    return (1/params.tau_i) * (-1 + (-params.w_ii)*derivF_i(Bi, params))

def adap_A23(ue, ui, params):
    return 0

def adap_A31(ue, ui, params):
    return (1/params.tau_a) * derivF_a(ue, params)

def adap_A32(ue, ui, params):
    return 0

def adap_A33(ue, ui, params):
    return -(1/params.tau_a)

def adap_A(x, params):
    ue = x[0]
    ui = x[1]
    return [[adap_A11(ue, ui, params), adap_A12(ue, ui, params), adap_A13(ue, ui, params)], 
            [adap_A21(ue, ui, params), adap_A22(ue, ui, params), adap_A23(ue, ui, params)], 
            [adap_A31(ue, ui, params), adap_A32(ue, ui, params), adap_A33(ue, ui, params)]]