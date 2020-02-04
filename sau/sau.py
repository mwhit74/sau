# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:39:58 2020

@author: mwhitten
"""

"""
   Functions ac1 - ac45 are taken from AISC

   Functions rc1 - rc20 are taken from Roarks' Forumlas for Stress-Strain
   
   Signularity function for multiple point loads on a simply-supported beam
   
   Singularity function for multiple uniformly distributed loads on a 
   simply-supported beam
"""

import math
import numpy as np
import tabulate as tabulate

np.set_printoptions(suppress=True)

def ac1_vx(w, l, x):
    """Shear at x - Beam simply supported - Uniformly dist. loads
    
    Calculates the shear in the beam at any location, x, along the beam due
    to a uniformly distributed load.
    
    v = w*(l/2-x)
    
    Args:
        w (float): uniformly distributed load
        
        l (float): length of beam between supports
        
        x (float): distance along beam from left support
    
    Returns:
        v (tuple(float, str)): shear at x
    
    Notes:
        1. Consistent units are the responsibility of the user. 
        2. For the reaction at the support use x = 0.0 or x = L.
        
    """
    
    v = w*(l/2.0-x)
    
    text = (f'v = w*(l/2.0-x) \n' +
            f'v = {w:.3f}*({l:.2f}/2.0-{x:.2f}) \n' + 
            f'v = {v:.2f}')
    
    return v, text

def ac1_mx(w, l, x):
    """Moment at x - Beam simply supported - Uniformly dist. loads
    
    Calculates the moment in the beam at any location, x, along the
    beam due to a uniformly distributed load.
    
    m = w*x/2*(l-x)
    
    Args:
        w (float): uniformly distributed load
        
        l (float): length of beam between supports
        
        E (float): modulus of elasticity
        
        I (float): section modulus
        
        x (float): distance along beam from left support
        
    Returns:
        m (tuple(float, str)): maximum positive moment at midspan
        
    Notes:
        1. Consistent units are the responsibility of the user.
        3. For maximum positive moment use x = L/2.
    """
    
    m = w*x/2.*(l-x)
    
    text = (f'm = w*x/2*(l-x) \n' +
            f'm = {w:.2f}*{x:.2f}/2*({l:.2f}-{x:.2f}) \n' +
            f'm = {m:.2f}')
            
    return m, text

def ac1_defl(w, l, E, I, x):
    """Deflection at x - Beam simply supported - Uniformly dist. loads
    
    Calculates the deflection in the beam at any location, x, along the
    beam due to a uniformly distributed load. 
    
    d = w*x/(24*E*I)*(math.pow(l,3) - 2*l*math.pow(x,2) + math.pow(x,3))
    
    Args:
        w (float): uniformly distributed load
        
        l (float): length of beam between supports
        
        E (float): modulus of elasticity
        
        I (float): section modulus
        
        x (float): distance along beam from left support
        
    Returns:
        d (tuple(float, str)): deflection at x
        
    Notes:
        1. Consistent units are the responsibility of the user.
    """
    
    d = w*x/(24*E*I)*(math.pow(l,3) - 2*l*math.pow(x,2) + math.pow(x,3))
    
    text = (f'd = w*x/(24*E*I)*(math.pow(l,3) - 2*l*math.pow(x,2) + ' +
            f'math.pow(x,3)) \n' + 
            f'd = {w:.2f}*{x:.1f}/(24*{e:.1}*{i:.1f})*(math.pow({l:.1f},3) - '+
            f'2*{l:.1f}*math.pow({x:.1f},2) + math.pow({x:.1f},3)) \n' + 
            f'd = {d:.3f}')
        
    return d, text





def ac15_vx(w, l, x):
    """Shear at x - Beam fixed at both ends - Uniformly dist. loads
    
    Calculates the shear in the beam at any location, x, along the
    beam due to a uniformly distributed load.
    
    v = w*(l/2.0-x)
    
    
    Args:
        w (float): uniformly distributed load
        
        l (float): length of beam between supports
        
        x (float): distance along beam from left support
    
    Returns:
        v (tuple(float, str)): shear at x
    
    Notes:
        1. Consistent units are the responsibility of the user. 
        2. For the reaction at the support use x = 0.0 or x = L.
    
    """
    
    v = w*(l/2.0-x)
    
    text = (f'v = w*(l/2.0-x) \n' +
            f'v = {w:.3f}*({l:.2f}/2.0-{x:.2f})' + 
            f'v = {v:.2f}')
    
    return v, text

def ac15_mx(w, l, x):
    """Moment at x - Beam fixed at both ends - Uniformly dist. loads
    
    Calculates the moment in the beam at any location, x, along the
    beam due to a uniformly distributed load.
    
    m = w/12.0*(6*l*x - math.pow(l,2) - 6*math.pow(x,2))
    
    Args:
        w (float): uniformly distributed load
        
        l (float): length of beam between supports
        
        E (float): modulus of elasticity
        
        I (float): section modulus
        
        x (float): distance along beam from left support
        
    Returns:
        m (tuple(float, str)): maximum positive moment at midspan
        
    Notes:
        1. Consistent units are the responsibility of the user.
        2. For maximum negative moment use x = 0.0 or x = L.
        3. For maximum positive moment use x = L/2.
    """
    
    m = w/12.0*(6*l*x - math.pow(l,2) - 6*math.pow(x,2))
    
    text = (f'm = w/12.0*(6*l*x - math.pow(l,2) - 6*math.pow(x,2)) \n' +
            f'm = {w:.3f}/12.0*(6.0*{l:.2f}*{x:.2f} - math.pow({l:.2f},2) - '
            f'6.0*math.pow({x:.2f},2)) \n' +
            f'm = {m:.2f}')
            
    return m, text


def ac15_defl(w, l, E, I, x):
    """Deflection at x - Beam fixed at both ends - Uniformly dist. loads
    
    Calculates the deflection in the beam at any location, x, along the
    beam due to a uniformly distributed load. 
    
    d = w*math.pow(x,2)/(24.0*E*I)*math.pow(l-x,2)
    
    Args:
        w (float): uniformly distributed load
        
        l (float): length of beam between supports
        
        E (float): modulus of elasticity
        
        I (float): section modulus
        
        x (float): distance along beam from left support
        
    Returns:
        d (tuple(float, str)): deflection at x
        
    Notes:
        1. Consistent units are the responsibility of the user.
    """
    
    d =  d = w*math.pow(x,2)/(24.0*E*I)*math.pow(l-x,2)
    
    text = (f'd = w*math.pow(x,2)/(24.0*E*I)*math.pow(l-x,2) \n' + 
            f'd = {w:.3f}*math.pow({x:.2f},2)/(24.0*{E:.1f}*{I:.1f})'
            f'*math.pow({l:.2f}-{x:.2f},2) \n' + 
            f'd = {d:.3f}')
        
    return d, text

def rc8a_stress_at_edge(a, b, q, e, t):
    """
    
    """
    pass
    
    
    
def rc8a_stress_at_center(a, b, q, e, t):
    """
    
    """
    pass
    
def ra8a_defl_at_center(a, b, q, e, t):
    pass

def mplob(l, loads, locs, e = 0.0, i = 0.0, defl_factor = 1.0, j = 21):
    """Multiple point loads on a beam
    
    Calculates the shear, moment, and deflection along a simply-supported 
    beam for any number of point loads applied to the beam. 
    
    Args:
        l (float): length of beam
        
        loads (list): magnitude of loads
        
        locs (list): location of each load from left end of beam
        
        e (float, optional): elastic modulus; defaults to 0.0
        
        i (float, optional): second moment of area; defaults to 0.0
        
        defl_factor (float, optional): scale factor for deflection results;
                                       defaults to 1.0
        
        j (int, optional): number of analysis points; defaults to 21
    
    Returns:
        (dls, v, m, y, text) (tuple(float, float, float, float, str)): 
            dls - location of analysis points w.r.t. left support
            v - shear at each analysis point
            m - moment at each analysis point
            y - vertical deflection at each analysis point
            text - user input and formatted table of results
                         
    Notes:
        1. Units are the responsibility of the user. 
           Internal units are consistent.
        2. Recommended units are kips and feet with a defl_factor = 12. This 
           will provide shear in kips, moment in kip-ft, and deflection in 
           inches. This means that E [ksf] and I [ft^4].
        3. If e = 0.0 or i = 0.0 no deflection will be calculated
        4. Elastic analysis, no shear deflection.
        5. Algorithm uses singularity functions to calculate values.
    
    """
    
    def v_conc(ra, p, x, a, l):
        if p == 0.0:
            return 0.0
        else:
            return ra * sf(x, 0, 0) - p * sf(x, a, 0)
        
    def m_conc(ra, p, x, a, l):
        if p == 0.0:
            return 0.0
        else:
            return ra * sf(x, 0, 1) - p * sf(x, a, 1)
        
    def y_conc(ra, p, x, a, l):
        if p == 0.0:
            return 0.0
        else:
            return (ra/6 * sf(x, 0, 3) - p/6 * sf(x,a,3) + 
                   (p/6 * math.pow(l-a,3) - ra/6 * math.pow(l,3))/l*x)
                   
    def sf(x, a, n):
        if x < a:
            return 0
        else:
            return math.pow(x-a,n)
            
    locs = np.asarray(locs, dtype=np.float32)
    loads = np.asarray(loads, dtype=np.float32)
    a_locs = locs
    b_locs = l - locs
    
    ras = loads*b_locs/l
    rbs = loads*a_locs/l
    
    dls = np.linspace(0,l,j)
    
    vs = np.empty([j], dtype=np.float32)
    ms = np.empty([j], dtype=np.float32)
    ys = np.empty([j], dtype=np.float32)
    k = 0
    
    for dl in dls:
        sum_v = 0.
        sum_m = 0.
        sum_y = 0.
        for ra, a, load in zip(ras, a_locs, loads):
            sum_v = sum_v + v_conc(ra, load, dl, a, l)
            sum_m = sum_m + m_conc(ra, load, dl, a, l)
            sum_y = sum_y + y_conc(ra, load, dl, a, l)
            
        vs[k] = sum_v
        ms[k] = sum_m
        ys[k] = sum_y
        k += 1
        
    if e == 0.0 or i == 0.0:
        ys = ys*0.0
    else:
        ys = ys/e/i*defl_factor
        
    text1 = (f'Multiple Point Loads on Simply Supported Beam Analysis \n' +
             f'(Using singularity functions) \n' +
             f'------------------------------------------------------ \n\n' +
             f'l = {l:.2f}, E = {e:.1f}, I = {i:.3f}, ' +
             f'Defl Factor = {defl_factor:.1f}, Num Nodes = {j:d} \n\n')
        
    text2 = tabulate.tabulate({f'Load':loads, 'Location':locs},
                             headers='keys',
                             floatfmt=(".2f", ".2f"))
    
    text3 = tabulate.tabulate({f"L/{j:d}":dls,
                              f"V":vs, 
                              f"M":ms, 
                              f"Y*{defl_factor:.2f}":ys},
                              headers='keys',
                              floatfmt=(".2f", ".2f", ".2f", ".4f"))
    
    text = text1 + text2 + "\n\n" + text3
        
    return dls, vs, ms, ys, text
            
def mdlob(l, loads, locs, e = 0.0, i = 0.0, defl_factor = 1.0, j = 21):
    """Multiple point loads on a beam
    
    Calculates the shear, moment, and deflection along a simply-supported 
    beam for any number of point loads applied to the beam. 
    
    Args:
        l (float): length of beam
        
        loads (list): magnitude of loads
        
        locs (list of tubles): start and end of each load from left end of beam
        
        e (float, optional): elastic modulus; defaults to 0.0
        
        i (float, optional): second moment of area; defaults to 0.0
        
        defl_factor (float, optional): scale factor for deflection results;
                                       defaults to 1.0
        
        j (int, optional): number of analysis points; defaults to 21
    
    Returns:
        (dls, v, m, y, text) (tuple(float, float, float, float, str)): 
            dls - location of analysis points w.r.t. left support
            v - shear at each analysis point
            m - moment at each analysis point
            y - vertical deflection at each analysis point
            text - user input and formatted table of results
                         
    Notes:
        1. Units are the responsibility of the user. Internal units 
           are consistent.
        2. Recommended units are kips and feet with a defl_factor = 12. This 
           will provide shear in kips, moment in kip-ft, and deflection in 
           inches. This means that E [ksf] and I [ft^4].
        3. If e = 0.0 or i = 0.0 no deflection will be calculated
        4. Elastic analysis, no shear deflection.
        5. Algorithm uses singularity functions to calculate values.
    
    """
    
    def v_dist(ra, w, x, a, b):
        if w == 0.0:
            return 0.0
        else:
            return ra * sf(x, 0, 0) - w * sf(x, a, 1) + w * sf(x, b, 1)
        
    def m_dist(ra, w, x, a, b):
        if w == 0.0:
            return 0.0
        else:
            return ra * sf(x, 0, 1) - w/2 * sf(x, a, 2) + w/2 * sf(x, b, 2)
        
    def y_dist(ra, w, x, a, b, l):
        if w == 0.0:
            return 0.0
        else:
            return (ra/6 * sf(x, 0, 3) - w/24 * sf(x, a, 4) + 
                    w/24 * sf(x, b, 4) + (w/24 * math.pow(l-a,4) - 
                    w/24 * math.pow(l-b,4) - ra/6*math.pow(l,3))/l*x)
                   
    def sf(x, a, n):
        if x < a:
            return 0
        else:
            return math.pow(x-a,n)
            
    locs = np.asarray(locs, dtype=np.float32)
    loads = np.asarray(loads, dtype=np.float32)
    s_locs = locs[...,0]
    e_locs = locs[...,1]
    a_locs = locs[...,0]
    b_locs = locs[...,1] - a_locs
    c_locs = l - a_locs - b_locs
    
    ras = loads*b_locs/(2*l)*(2*c_locs + b_locs)
    rbs = loads*b_locs/(2*l)*(2*c_locs + b_locs)
    
    dls = np.linspace(0,l,j)
    
    vs = np.empty([j], dtype=np.float32)
    ms = np.empty([j], dtype=np.float32)
    ys = np.empty([j], dtype=np.float32)
    k = 0
    
    for dl in dls:
        sum_v = 0.
        sum_m = 0.
        sum_y = 0.
        for ra, a, b, load in zip(ras, s_locs, e_locs, loads):
            sum_v = sum_v + v_dist(ra, load, dl, a, b)
            sum_m = sum_m + m_dist(ra, load, dl, a, b)
            sum_y = sum_y + y_dist(ra, load, dl, a, b, l)
            
        vs[k] = sum_v
        ms[k] = sum_m
        ys[k] = sum_y
        k += 1
        
    if e == 0.0 or i == 0.0:
        ys = ys*0.0
    else:
        ys = ys/e/i*defl_factor
        
    text1 = (f'Multiple Distributed Loads on Simply Supported Beam ' +
             f'Analysis \n' +
             f'(Using singularity functions) \n' +
             f'------------------------------------------------------ \n\n' +
             f'l = {l:.2f}, E = {e:.1f}, I = {i:.3f}, ' +
             f'Defl Factor = {defl_factor:.1f}, Num Nodes = {j:d} \n\n')
        
    text2 = tabulate.tabulate({f'Load':loads, 'Start':a_locs, 'End':b_locs},
                             headers='keys',
                             floatfmt=(".2f", ".2f", ".2f"))
    
    text3 = tabulate.tabulate({f"L/{j:d}":dls,
                              f"V":vs, 
                              f"M":ms, 
                              f"Y*{defl_factor:.2f}":ys},
                              headers='keys',
                              floatfmt=(".2f", ".2f", ".2f", ".4f"))
    
    text = text1 + text2 + "\n\n" + text3
        
    return dls, vs, ms, ys, text
            
    
            
        