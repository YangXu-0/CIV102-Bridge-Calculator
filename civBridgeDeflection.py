# civBridgeDeflection.py
# all dimensions (dim) are given in a list of lists where each sect has three values [base, height, bottom]
# assumes that all subsections of a shape are rectangular

#REFERENCE SYSTEM: up, right, ccw = positive, + moment = tension on bottom

# geometrical calculations
def area(sect):
    ''' Return the area of a rectangular shaped section'''
    A = sect[0] * sect[1]
    return A


def A_tot(dim):
    ''' Return the total area of a given shape'''
    A_tot = 0
    for sect in range(len(dim)):
        A_tot += area(dim[sect])
    return A_tot


def yi(sect):
    ''' Return the distance of the centroid of a section from the bottom of the shape'''
    centre = sect[1] / 2
    yi = sect[2] + centre
    return yi


def ybar(dim):
    ''' Return the location of the centroid of the shape'''
    sum1 = 0
    for sect in range(len(dim)):
        sum1 += yi(dim[sect]) * area(dim[sect])
    ybar = sum1 / A_tot(dim)
    return ybar


def di(sect, ybar):
    ''' Return the distance from the centroid of the section to the centroid of the shape'''
    di = yi(sect) - ybar
    return di


def I0(sect):
    ''' Return the I of a rectangular section'''
    I0 = (sect[0] * (sect[1]**3)) / 12
    return I0


def moment_of_I(dim):
    ''' Return the I of a given shape'''
    I_tot = 0
    for sect in range(len(dim)):
        d = di(dim[sect], ybar(dim))
        Ai = area(dim[sect])
        I = I0(dim[sect])
        I_tot += (Ai * (d**2)) + I
    return I_tot

# equilibrium force calculations
# loads is a dictionary of the loads(value) at each location (key)
def rxn_f(loads, loc_A, loc_B):
    '''Return the reaction forces in the two supports
    Assume all forces to be point loads at a 90degree angle
    Assume all forces to be applied at the centre point of their area of application
    Assume only one load at each location'''
    # support A: left; support B: right

    distAB = loc_B - loc_A
    sum_Fy = 0
    Ma = 0
    for load in loads.items():
        loc_from_A = load[0] - loc_A
        Ma += loc_from_A * load[1]
    By = (-1* Ma) / distAB
    for load in loads.items():
        sum_Fy += load[1]
    Ay = (-1 * sum_Fy) - By
    return {loc_A:Ay, loc_B:By}


# shear calculations
def shear_force(forces):
    ''' Takes in a dictionary {location: force} and calculates the shear force '''
    shear_forces = {}
    total = 0

    for fkey, fvalue in forces.items():
        total += fvalue
        shear_forces[fkey] = total   # just get's the cumlulative sum of all the forces essentially 

    return shear_forces # a dictionary of the intervals and the shear force


def shear_forces_dense(forces):
    ''' Takes in a dictionary of shear forces {location: force} and calculates the shear force every mm '''
    copy = {-1: 0}   # temp case
    temp = list(forces.keys())   # making a list to loop through the keys
    for i in range(0, 1280+1):
        if i in forces.keys():   # check if each mm increment has a force applied
            copy[i] = copy[i-1] + forces[i]    # adds new force to previous shear forces
        else:
            copy[i] = copy[i-1]   # otherwise just copy the old one (shear force doesn't change)

    del copy[-1]   # deleting temp case

    return copy


# bending moment calculations
def bending_moments_calc(shear_forces):
    ''' Takes in a dictionary of shear forces {location: force} and calculates the bending moments '''
    bmd = {-1: 0}
    shear_locations = list(shear_forces.keys())
    for i in range(len(shear_forces)-1):
        # calculates distance from start of shear to changes in shear, calculates area, adds to bmd  
        bmd[shear_locations[i+1]] = list(bmd.values())[i] + ((shear_locations[i+1]-shear_locations[i]) * shear_forces[shear_locations[i]])

    return bmd


def area_under_curve(forces, start, end):
    ''' calculates the area under a curve given a dictionary of the value at every mm. returns dictionary '''
    copy = {}
    for i in range(start, end+1):   # Calculates area under shear force diagram at each point
        copy[i] = np.trapz(list(forces.values())[start:i], dx=1)

    return copy

# Deflection calculations
def curvature(bending_moment, dimensions, E):
    ''' Calculates the curvature at each mm given the bending moment at every mm as a dictionary '''
    temp = {}
    I_value = moment_of_I(dimensions)
    for key in list(bending_moment.keys()):
        temp[key] = bending_moment[key]/(E * I_value)

    return temp   # returns dictionary of curvatures at every mm


def tangential_deflection(start, end, curvature):
    ''' Calculates the deflection from a point (end) to
    the tangent at another point (start) given the curvature as
    a dictionary {location: value} '''
    
    temp = curvature.copy()

    # Multiply the curvature value at each point by the distance away from end
    temp_loc = list(temp.keys())
    temp_values = list(temp.values())
    A_d = {}
    for i in range(start, end+1):
        A_d[i] = (end-i)*temp_values[i]

    # Integrate the new values from start to end
    return np.trapz(list(A_d.values()), dx=1)


def calc_forces(loc):
    ''' Given the location of the point loads, this function
    calculates the reaction forces --> the shear forces --> the
    bending moments --> and finally the curvature values '''

    # Forces
    forces = {loc: p}
    ay = list(rxn_f(forces,15,1075).values())[0]
    by = list(rxn_f(forces,15,1075).values())[1]
    test_loads = {15: ay, loc: p, 1075: by}
    test_loads = dict(sorted(test_loads.items()))

    # Calculations for shear, then bm, then curvature
    shear_forces = shear_force(test_loads)
    bending_moments_dense = area_under_curve(shear_forces_dense(test_loads), 0, 1280)
    curvature_values = curvature(bending_moments_dense, dim, E_m)

    return curvature_values


def midspan_deflection(loc):
    ''' Calculates the midspan deflection of the bridge given the curvature values {location: value} '''

    # Constant values
    BEGINNING_SUPPORT = 15
    END_SUPPORT = 1075
    MIDSPAN = 545

    # Tangential deflection calculation
    curvature_values = calc_forces(loc)
    tangential_deflection_end = tangential_deflection(BEGINNING_SUPPORT, END_SUPPORT, curvature_values)
    tangential_deflection_mid = tangential_deflection(BEGINNING_SUPPORT, MIDSPAN, curvature_values)

    # Midspan deflection calculation
    L_end = END_SUPPORT-BEGINNING_SUPPORT
    L_mid = MIDSPAN-BEGINNING_SUPPORT

    return ((L_mid/L_end)*tangential_deflection_end) - tangential_deflection_mid


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    pi = math.pi
    # material properties
    E_m = 4000 #MPa
    sig_tens = 30 #MPa
    sig_comp = 6 #MPa
    tau_m = 4 #MPa
    u_m = 0.2

    tau_g = 2 #MPa

    # **** input these values manually
    dim = [[80, 1.27, 0],[1.27, 72.46, 1.27],[1.27, 72.46, 1.27], [10, 1.27, 72.46], [10, 1.27, 72.46], [100, 1.27, 73.73]]
    length = 1280 #length of bridge
    support_A = 15
    support_B = 1075
    diaphragms = [12.5, 17.5, 577.5, 582.5, 1072.5, 1077.5, 1275, 1280] 
    # locations of the diaphragms from left to right (must be at supports + add to weakest points calculated)
    tot_height = 75
    yglue = [73.73] # height of glue location
    height_webs = 73.73
    b_centroid = 2.54
    b_glue = [20]
    case1_b = 80 # distance between webs (width of flange restrained on two side)
    case2_b = 10 # distance from web to edge (width of flange restrained on one side)
    t = 1.27 # thicknesses
    t_top = 1.27
    t_bot = 1.27
    t_webs = 1.27
    p = -200 # downward force of each point load
    # ****************

    # Location of supports
    right_p_loc = 1265
    left_p_loc = 565

    # Calculating deflection with each p
    midspan_deflection_right_p = midspan_deflection(right_p_loc)
    midspan_deflection_left_p = midspan_deflection(left_p_loc)

    # Overall deflection
    print("The deflection at the midspan is:", midspan_deflection_left_p + midspan_deflection_right_p, "mm")
