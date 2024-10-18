import sys
import os

cdir = os.getcwd() #get current directory
os.chdir(cdir) #make sure you are in the current directory
sys.path.append(cdir) #append directory to enable import of functions from current directory files

import numpy as np

#import root and eigenvalue/-vector function
from scipy.optimize import root
from scipy.linalg import eigvals

from params import setParams

from analysis import derivF_e, derivF_a
from analysis import a_jkValues, f_kernel, deriv_f_kernel
from funcs import getSwitchIndex, getCommonElement
import kernels as ks

# # THIS FILE CONTAINS ALL FUNCTIONS NECESSARY FOR THE # #
# # TURING STABILITY ANALYSIS, BOTH THE 2-EQUATION, AND # #
# # THE 3-EQUATION SYSTEM # #


# # # - - - # # # - - - extended stability analysis for the 2-dimensional system - - - # # # - - - # # #
# # # - - - # # # - - - (i.e. two-populations WITHOUT additionally considered mechanism) - - - # # # - - - # # #

# # # - - - LINEARIZATION MATRIX - - - # # #
# # # - - - without mechanism: turing-linearization matrix - - - # # #

def turing_A11(k, a_ee, params):
    return (1/params.tau_e)*(-1 + a_ee*f_kernel(params.sigma_e, k, params.kernel))

def turing_A12(k, a_ei, params):
    return (1/params.tau_e)*(-a_ei)*f_kernel(params.sigma_i, k, params.kernel)

def turing_A21(k, a_ie, params):
    return (1/params.tau_i)*a_ie*f_kernel(params.sigma_e, k, params.kernel)

def turing_A22(k, a_ii, params):
    return (1/params.tau_i)*(-1 + (-a_ii)*f_kernel(params.sigma_i, k, params.kernel))


# # # - - - TRACE - - - # # #
    
def tr(k, a_ee, a_ii, params):
    return turing_A11(k, a_ee, params) + turing_A22(k, a_ii, params)

def dtr(k, a_ee, a_ii, params):
    return (1/params.tau_e)*a_ee*deriv_f_kernel(params.sigma_e, k, params.kernel)-(1/params.tau_i)*a_ii*deriv_f_kernel(params.sigma_i, k, params.kernel)
    
    
# # # - - - DETERMINANT - - - # # # 

def det(k, a_ee, a_ei, a_ie, a_ii, params):
    return (turing_A11(k, a_ee, params)*turing_A22(k, a_ii, params))-(turing_A12(k, a_ei, params)*turing_A21(k, a_ie, params))

def ddet(k, a_ee, a_ei, a_ie, a_ii, params):
    return (- a_ii * deriv_f_kernel(params.sigma_i, k, params.kernel)
            + a_ee * deriv_f_kernel(params.sigma_e, k, params.kernel)
            - a_ee * a_ii * deriv_f_kernel(params.sigma_e, k, params.kernel) * f_kernel(params.sigma_i, k, params.kernel)
            - a_ee * a_ii * f_kernel(params.sigma_e, k, params.kernel) * deriv_f_kernel(params.sigma_i, k, params.kernel)
            + a_ei * a_ie * deriv_f_kernel(params.sigma_e, k, params.kernel) * f_kernel(params.sigma_i, k, params.kernel)
            + a_ei * a_ie * f_kernel(params.sigma_e, k, params.kernel) * deriv_f_kernel(params.sigma_i, k, params.kernel))
    
    
# # # - - - the functions to test for turing instability - - - # # #

def pos_det(a_ee, a_ei, a_ie, a_ii, params):
    if det(0, a_ee, a_ei, a_ie, a_ii, params)>0:
        return True
    else: 
        return False
    
def neg_tr(k, a_ee, a_ii, params):
    if all(tr(k, a_ee, a_ii, params)< 0):
        return True
    else:
        return False
    
def det_traj(k, a_ee, a_ei, a_ie, a_ii, params):
    if any(det(k, a_ee, a_ei, a_ie, a_ii, params)< 0): 
        return True
    else:
        return False
    
def lmbd(k_real, a_ee, a_ei, a_ie, a_ii, params):
    k = k_real.astype(complex)
    lmbd_plus = (1/2)*(tr(k, a_ee, a_ii, params) + np.sqrt(tr(k, a_ee, a_ii, params)**2 - 4*det(k, a_ee, a_ei, a_ie, a_ii, params)))
    lmbd_minus = (1/2)*(tr(k, a_ee, a_ii, params) - np.sqrt(tr(k, a_ee, a_ii, params)**2 - 4*det(k, a_ee, a_ei, a_ie, a_ii, params)))
    return [lmbd_plus, lmbd_minus]



    
# # # - - - the functions to check Turing instability and check possibility of spatiotemporal patterns - - - # # #

def violationType(k, a_ee, a_ei, a_ie, a_ii, params):
    
    """
        This function checks with what condition of the linear stability analysis on a continuum is violated.
        (i.e. run this check only for fixed points that are linearly stable in the local [i.e. one-node] system)
        Options are: det(A(k)) > 0 is violated for a k0!=0 (i.e. det(A(k0))=0). Then we have a static Turing bifurcation point (i.e. spatial pattern). This is equivalent to im(\lambda)=0.
                     tr(A(k))  < 0 is violated for a k0!=0 (i.e. tr(A(k0))=0). Then we speak of a dynamic Turing bifurcation point (i.e. spatiotemporal pattern)
                     
        For the output-wavenumber k0, we have that "The Turing bifurcation point is defined by the smalles non-zero wave number k0" 
            - by Meijer & Commbes: Travelling waves in a neural field model with refractoriness (2013)
                     
        Input:
        :k: wavenumber (array)
        :a_kj: Values necessary to determine det & tr
        :params: parameter setting of model
        
        Output: 
        :k0: wavenumber k for which the violation appears
        :k_max: wavenumber k for which the eigenvalue has the largest positive real part
        :violation_type: type of violation. Options are: 0 (no violation), 1 (static), 2 (dynamic), 3 (both).
    """
    
    violation_type = 0
    k0 = 0
    k_max = 0
    if not pos_det(a_ee, a_ei, a_ie, a_ii, params):
        return 0, 0, 0
    if det_traj(k, a_ee, a_ei, a_ie, a_ii, params):
        k0 = determinantTuringBifurcation(a_ee, a_ei, a_ie, a_ii, params)[0]
        k_max = k[np.argmin(det(k, a_ee, a_ei, a_ie, a_ii, params))]
        violation_type = 1
    elif not neg_tr(k, a_ee, a_ii, params):
        k0 = traceTuringBifurcation(a_ee, a_ii, params)[0] 
        k_max = k[np.argmax(tr(k, a_ee, a_ei, a_ie, a_ii, params))]
        violation_type = 2
    elif det_traj(k, a_ee, a_ei, a_ie, a_ii, params) and not neg_tr(k, a_ee, a_ii, params):
        k0, k_max = 0, 0
        violation_type = 3
    
    return violation_type, k0


def determinantTuringBifurcation(a_ee, a_ei, a_ie, a_ii, params):
    """ Determine the smallest non-zero wave number k0 that defines the Turing bifurcation point for which we have the condition 
        det(A(k))>0 violated."""
    
    k00=[]
    for k in np.linspace(0, 2, 11):
        sol = root(det, k, args=(a_ee, a_ei, a_ie, a_ii, params,), method='lm')
        if sol.success:
            closeness = all(np.isclose(det(sol.x, a_ee, a_ei, a_ie, a_ii, params), 0.0))
            if closeness:
                if len(k00)==0: 
                    k00.append(sol.x)
                else:
                    already_derived = False
                    for i in range(len(k00)):
                        if all(np.isclose(sol.x, k00[i], atol=1e-9)):
                            already_derived = True
                        else: 
                            pass
                    if already_derived:
                        pass #skip the already derived fixed points
                    else:
                        k00.append(sol.x)
    
    return min(abs(np.array(k00)))

def traceTuringBifurcation(a_ee, a_ii, params):
    """ Determine the smallest non-zero wave number k0 that defines the Turing bifurcation point for which we have the condition 
        det(A(k))>0 violated."""
    
    k00=[]
    for k in np.linspace(0, 2, 11):
        sol = root(tr, k, args=(a_ee, a_ii, params,), method='lm')
       # fix_point = [round(sol.x[0], 8), round(sol.x[1], 8)]
        if sol.success:
            closeness = all(np.isclose(tr(sol.x, a_ee, a_ii, params), 0.0))
            if closeness:
                if len(k00)==0: #always append the firstly derived fixed point
                    k00.append(sol.x)
                else:
                    already_derived = False
                    for i in range(len(k00)):
                        if all(np.isclose(sol.x, k00[i], atol=1e-9)):
                            already_derived = True
                        else: 
                            pass
                    if already_derived:
                        pass #skip the already derived fixed points
                    else:
                        k00.append(sol.x)
    
    return min(abs(np.array(k00)))



# # # - - - # # # - - - extended stability analysis for the 3-dimensional system - - - # # # - - - # # #
# # # - - - # # # - - - (i.e. two-populations WITH additionally considered mechanism) - - - # # # - - - # # #

# # # - - - # # # - - - adaptation matrix A(k)\in\mathbb{C}^{3\times3} - - - # # # - - - # # #

def a11(k, fp, params):
    ue = fp[0]
    ui = fp[1]
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)
    
    return (1/params.tau_e) * (-1 + a_ee*f_kernel(params.sigma_e, k, params.kernel))

def a12(k, fp, params):
    ue = fp[0]
    ui = fp[1]
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)
    
    return (1/params.tau_e) * (-a_ei) * f_kernel(params.sigma_i, k, params.kernel)

def a13(fp, params):
    ue = fp[0]
    ui = fp[1]
    be = params.w_ee * ue - params.w_ei * ui - params.b * derivF_a(ue, params) + params.I_e
    
    return (params.b/params.tau_e) * derivF_e(be, params)

def a21(k, fp, params):
    ue = fp[0]
    ui = fp[1]
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)
    
    return (1/params.tau_i) * a_ie * f_kernel(params.sigma_e, k, params.kernel)

def a22(k, fp, params):
    ue = fp[0]
    ui = fp[1]
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)

    return (1/params.tau_i)*(-1 + (-a_ii) * f_kernel(params.sigma_i, k, params.kernel))

def a23():
    return 0

def a31(fp, params):
    ue = fp[0]
    ui = fp[1]
    return (1/params.tau_a) * derivF_a(ue, params)

def a32():
    return 0

def a33(params):
    return -(1/params.tau_a)


# # # - - - # # # - - - polynomial entries - - - # # # - - - # # #

def c0(k, fp, params):
    return (-a11(k, fp, params) * a22(k, fp, params) * a33(params) + 
            a22(k, fp, params) * a13(fp, params) * a31(fp, params) + 
            a33(params) * a12(k, fp, params) * a21(k, fp, params))


def c1(k, fp, params):
    return (a11(k, fp, params) * a22(k, fp, params) + 
            a11(k, fp, params) * a33(params) + 
            a22(k, fp, params) * a33(params) - 
            a13(fp, params) * a31(fp, params) - 
            a12(k, fp, params) * a21(k, fp, params))

def c2(k, fp, params):
    return -a11(k, fp, params)-a22(k, fp, params)-a33(params)


# # # - - - # # # - - - conditions - - - # # # - - - # # #
# see  Rodica Curtu and Bard Ermentrout,
# “Pattern Formation in a Network of Excitatory and Inhibitory Cells with Adaptation"
# for more information about conditions

def checkZeroEigval(k, fp, params):
    """
    Check whther the 'bias' (i.e. c0(k)) changes sign for any k>0.
    If yes, it means that that there is a k0 for which c0(k0)=0, i.e. the polynomial has a zero-root at k0,
    i.e. the fixed point fp loses stability (is at least static Turing unstable).
    """

    if all(c0(k, fp, params) < 0) or all(c0(k, fp, params) > 0):
        return 0, 0
    else:
        k0 = k[getSwitchIndex(c0(k, fp, params))[0]]
        k_max = k[np.argmax(c0(k, fp, params))]
        return 1, k0
    

def checkImagEigval(k, fp, params):
    """
    Check whether the c1(k)c2(k)-c0(k) changes sign for any k>0.
    If yes, it means that that there is a k0 for which the subtraction is zero.
    That means, the polynomial has a purely imaginary root at k0 (Re=0),
    i.e. the fixed point fp loses stability (is dynamic Turing unstable).
    """

    temp = c1(k, fp, params) * c2(k, fp, params) - c0(k, fp, params)
    
    if all(temp < 0) or all(temp > 0):
        return 0, 0
    else:
        indeces = getSwitchIndex(temp)
        c1_sign = c1(k[indeces], fp, params)
        if any(c1_sign >= 0):
            return 1, 0
        else:
            return 0, 0
    

def checkTakensBogdanov(k, fp, params):
    c0_array = c0(k, fp, params)
    c1_array = c1(k, fp, params)
    if (all(c0_array < 0) or all(c0_array > 0)) and (all(c1_array < 0) or all(c1_array > 0)):
        return 0, 0
    else:
        c0_indeces = getSwitchIndex(c0_array)
        c1_indeces = getSwitchIndex(c1_array)
        same_k0 = getCommonElement(c0_indeces, c1_indeces)
        if same_k0:
            return 1, 0
        else:
            return 0, 0
        


# # # - - - # # # - - - check which condition is violated - - - # # # - - - # # #

def checkStability(k, fp, params):
    zeroVal = checkZeroEigval(k, fp, params)
    imagVal = checkImagEigval(k, fp, params)
    if not imagVal:
        doubleVal = checkTakensBogdanov(k, fp, params)
    else:
        doubleVal = 0, 0
    
    return zeroVal, imagVal, doubleVal
    
    
    

    













