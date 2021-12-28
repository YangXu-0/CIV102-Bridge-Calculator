#civBridgeCase1.py
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

    # Find's the cumulative sum of forces at each location
    for fkey, fvalue in forces.items():
        total += fvalue
        shear_forces[fkey] = total 

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


def shear_force_diagram(forces):
    ''' Takes in a dictionary of shear forces (dense) {location: force} and plots the shear force '''

    copy = shear_forces_dense(forces)   # get's a dense version of shear forces to plot

    x_values = list(copy.keys())   # uses the locations as x-coords
    y_values = list(copy.values())   # uses the shear values as y-coords
    flat_x = [0, length]   # getting x-coordinates for failure shears

    fig, ax = plt.subplots()   # creates the plot
    ax.plot(x_values, y_values)   # plots the shear
    plt.fill_between(x_values, y_values, 0, alpha=0.3)

    # graph formatting
    ax.legend()
    plt.xlabel("Distance (mm)")
    plt.ylabel("Shear Force (N)")
    plt.title("Shear Force Diagram")
    plt.show()


def Q_max(dim):
    '''return the Q at y bar for calculating tau max'''
    # calculates Q from the bottom
    y = ybar(dim)
    Qmax = 0
    for sect in dim:
        top = sect[2] + sect[1]
        if top <= y:
            Qmax += abs(di(sect, y)) * area(sect)
    
        elif sect[2] < y and top > y:
            d_new = y - abs(((y - sect[2]) / 2) + sect[2])
            A_new = (y - sect[2]) * sect[0]
            Qmax += A_new * d_new
    return Qmax


def Q_glue(dim, yglue):
    '''return the Q at the glue position for calculating tau glue'''
    # yglue is the glue location of interest (measured from the bottom)
    Qglue = 0
    for sect in dim:
        if sect[2] >= yglue:
          d = di(sect, ybar(dim))
          A = area(sect)
          Qglue += A * d
    return Qglue


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
    ''' Calculates the area under a curve given a dictionary of the value at every mm. returns dictionary '''
    copy = {}
    for i in range(start, end+1):
        copy[i] = np.trapz(list(forces.values())[start:i], dx=1)

    return copy


def bending_moment_diagram(forces):
    ''' Takes in a dictionary of shear forces (dense) {location: force} and draws the bending moment diagram '''
    
    copy = area_under_curve(forces, 0, 1280) # integrating shear forces to get bending moments

    # getting x and y coordinates of bending moments (every mm)
    x_coords = list(forces.keys())
    y_coords = list(copy.values())

    # Plotting the curve
    fig, ax = plt.subplots()   # Plotting the curve
    plt.gca().invert_yaxis()
    ax.plot(x_coords, y_coords)
    plt.fill_between(x_coords, y_coords, 0, alpha=0.3)

    # Formatting the graph
    plt.xlabel("Distance (mm)")
    plt.ylabel("Bending Moment (Nmm)")
    plt.title("Bending Moment Diagram")
    plt.show()


def naviers(M, y, I):
    ''' Calculates the flexural stress '''
    sig = (M * y) / I
    return sig

# thin plate buckling
#case 1: flange, restrained on two sides
def flange2(E, u, b, t):
    ''' Return the stress capacity of a flange restrained on 2 sides'''
    sig_crit = ((4 * (pi**2) * E) / (12 * (1 - (u**2)))) * ((t / b)**2)
    return sig_crit

#case 2: flange, restrained on one side
def flange1(E, u, b, t):
    ''' Return the stress capacity of a flange restrained on 1 side'''
    sig_crit = ((0.425 * (pi**2) * E) / (12 * (1 - (u**2)))) * ((t / b)**2)
    return sig_crit

#case 3: web buckling
def web_buck(E, u, b, t):
    ''' Return the stress capacity for webs'''
    sig_crit = ((6 * (pi**2) * E) / (12 * (1 - (u**2)))) * ((t / b)**2)
    return sig_crit

#case 4: shear buckling
def shear_buckle(E, u, h, a, t):
    ''' Return the shear capacity before buckling'''
    sig_crit = ((5 * (pi**2) * E) / (12 * (1 - (u**2)))) * (((t / h)**2) + ((t / a)**2))
    return sig_crit


def find_a(diaphragms):
    ''' Return largest distance between diaphragms'''
    a = 0
    for dia in range(1, len(diaphragms)):
        a = max(diaphragms[dia] - diaphragms[dia - 1], a)
    return a

# Calculate max shear force and bmd 
def max_pos_bm_sf(force):
    ''' Calculate the max bending moment or shear force'''
    
    # Iterates through all points to find the max
    max = None
    index = None
    values = list(force.values())
    locations = list(force.keys())

    for i in range(len(values)):
        if max == None:
            max = values[i]
            index = i
            continue 
        
        if values[i] >= max:
            max = values[i]
            index = i

    return {locations[index]: max}   # returns a dictionary of max location (key) and magnitude (value)

# max_neg and max_pos functions are the same as previous max function
# difference is they just check for most negative and most positive
def max_neg_bm_sf(force):
    ''' Calculate the max bending moment or shear force'''
    min = None
    index = None
    values = list(force.values())
    locations = list(force.keys())

    for i in range(len(values)):
        if min == None:
            min = values[i]
            index = i
            continue 
        
        if values[i] <= min:
            min = values[i]
            index = i

    return {locations[index]: min}   # returns a dictionary of max location (key) and magnitude (value)


def max_bm_sf(force):
    ''' Calculate the max bending moment or shear force'''
    max = None
    index = None
    values = list(force.values())
    locations = list(force.keys())

    for i in range(len(values)):
        if max == None:
            max = values[i]
            index = i
            continue 
        
        if abs(values[i]) >= abs(max):
            max = values[i]
            index = i

    return {locations[index]: max}   # returns a dictionary of max location (key) and magnitude (value)


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
    t = 1.27
    t_top = 1.27
    t_bot = 1.27
    t_webs = 1.27
    # ****************

    # geometry calculations
    y_bar = ybar(dim)
    I = moment_of_I(dim)

    print(f"Total Area: {A_tot(dim)} mm^2")
    print(f"Centroidal Axis: {ybar(dim)} mm")
    print(f"Second Moment of Area (I): {moment_of_I(dim)} mm^4")

    # setup the FOS storage lists
    lowest_FOS = [None, None, None, None, None, None]
    lowest_FOS.extend(len(yglue)*[None])
    # need to extend the glue slots since then number of glued locations varies between design iterations
    lowest_FOS_locations = [None, None, None, None, None, None]
    lowest_FOS_locations.extend(len(yglue)*[None])
    local_buckle_type_pos = None
    local_buckle_type_neg = None
    lowest_buckle_FOS_pos = [None, None, None]
    lowest_buckle_FOS_neg = [None, None]

    worst_sf = []
    worst_bm = []

    # General idea:
        # increase i by 1 each loop to offset the location of the train
        # calculate values at each loop
        # return the lowest values + location that it happens at

    for i in range(67, 68):   # length+1-852 for full bridge
        temp = [0+i, 176+i, 340+i, 516+i, 680+i, 856+i]   
        if 15 in temp or 1075 in temp:   # prevents dictionary issues when force is over support
            continue

        # forces 
        p = -400/6
        forces = {0+i: p, 176+i: p, 340+i: p, 516+i: p, 680+i: p, 856+i: p}
        ay = list(rxn_f(forces,15,1075).values())[0]
        by = list(rxn_f(forces,15,1075).values())[1]
        test_loads = {15: ay, 0+i: p, 176+i: p, 340+i: p, 516+i: p, 680+i: p, 856+i: p, 1075: by}
        test_loads = dict(sorted(test_loads.items()))
        
        # Shear forces and bending moments
        shear_forces = shear_force(test_loads)
        bending_moments = bending_moments_calc(shear_forces)

        max_sf = abs(list(max_bm_sf(shear_forces).values())[0])

        max_pos_bm = max_pos_bm_sf(bending_moments) # dictionaries
        max_neg_bm = max_neg_bm_sf(bending_moments)

        # Calculating max stresses with Navier
        pos_bm_t = naviers(list(max_pos_bm.values())[0], y_bar, I) # tension on bottom
        pos_bm_c = -1*naviers(list(max_pos_bm.values())[0], tot_height-y_bar, I) # compression on top

        neg_bm_t = naviers(list(max_neg_bm.values())[0], tot_height-y_bar, I) # tension on top
        neg_bm_c = -1*naviers(list(max_neg_bm.values())[0], y_bar, I) # compression on bottom

        max_tens_stress = max(pos_bm_t, neg_bm_t)
        max_comp_stress = min(pos_bm_c, neg_bm_c)

        FOS_tens = sig_tens/max_tens_stress
        FOS_comp = sig_comp/(-1*max_comp_stress)

        # Shear failure calculations
        Q = Q_max(dim)   # Calculating Q values for middle and glue
        glue_Q = []
        for j in range(len(yglue)):
            glue_Q.append(Q_glue(dim, yglue[j]))

        shear_stress = ((max_sf)*Q)/(I*t_webs*2)   # Max shear stress
        FOS_shear_stress = tau_m/shear_stress

        shear_stress_glue = []   # Calculating shear stress in glue
        FOS_shear_glue = []

        for j in range(len(glue_Q)):
            shear_stress_glue.append(((max_sf)*glue_Q[j])/(I*b_glue[j]))
            FOS_shear_glue.append(tau_g/shear_stress_glue[j])

        # Local buckling calculations
        case_1 = flange2(E_m, u_m, case1_b, t_top)   
        case_1_2 = flange2(E_m, u_m, case1_b, t_bot)
        case_2 = flange1(E_m, u_m, case2_b, t_top)
        case_3_top = web_buck(E_m, u_m, height_webs-y_bar, t_webs)   # Top half of web
        case_3_bottom = web_buck(E_m, u_m, y_bar, t_webs)   # Bottom half of web
        case_4 = shear_buckle(E_m, u_m, height_webs, find_a(diaphragms), t)   # Calculation for shear stress

        y_top = tot_height-y_bar   # Defining distances from centroid to top and bottom
        y_bot = y_bar

        bm_case_1 = (case_1*I)/y_top   # Calculating the bending moment for each stress
        bm_case_1_2 = (case_1_2*I)/(y_bot)
        bm_case_2 = (case_2*I)/y_top
        bm_case_3_top = (case_3_top*I)/y_top
        bm_case_3_bottom = (case_3_bottom*I)/y_bot
        bm_case_4 = (case_4*I*(t*2))/Q   # Calculating the shear force for shear stress

        buckle_stresses_pos = [case_1, case_2, case_3_top]   # Sorting through to find failure stress and type
        buckle_stresses_neg = [case_3_bottom, case_1]
        buckle_bms_pos = [bm_case_1, bm_case_2, bm_case_3_top]
        buckle_bms_neg = [bm_case_1_2, bm_case_3_bottom]
        types_of_buckling_pos = ["case_1_1", "case_2", "case_3_top"]
        types_of_buckling_neg = ["case_3_bottom", "case_1_2"]

        buckle_failure_bm_pos = min(buckle_bms_pos)
        buckle_failure_stress_pos = buckle_stresses_pos[buckle_bms_pos.index(buckle_failure_bm_pos)]
        buckle_failure_type_pos = types_of_buckling_pos[buckle_bms_pos.index(buckle_failure_bm_pos)]

        buckle_failure_bm_neg = min(buckle_bms_neg)
        buckle_failure_stress_neg = buckle_stresses_neg[buckle_bms_neg.index(buckle_failure_bm_neg)]
        buckle_failure_type_neg = types_of_buckling_neg[buckle_bms_neg.index(buckle_failure_bm_neg)]
        
        # FOS calculations
        comparison_bm_pos = abs(list(max_pos_bm.values())[0])   # maximum bending moments
        comparison_bm_neg = abs(list(max_neg_bm.values())[0])

        FOS_local_buckling_pos = buckle_failure_bm_pos/comparison_bm_pos
        if comparison_bm_neg != 0:   # To prevent division by 0 errors (not always a negative bm in bridge)
            FOS_local_buckling_neg = buckle_failure_bm_neg/comparison_bm_neg
        else:
            FOS_local_buckling_neg = 9999 # since will never happen (no compression at bottom)

        max_shear_stress = ((max_sf)*Q)/(I*t*2)
        FOS_shear_buckling = case_4/max_shear_stress

        # Compiling and comparing values
        FOS_compiled = [FOS_tens, FOS_comp, FOS_shear_stress, FOS_local_buckling_pos, FOS_shear_buckling, FOS_local_buckling_neg]
        FOS_types_compiled = ["Tension", "Compression", "Shear stress", "Local buckling with positive bending moment", 
        "Shear buckling", "Local buckling with negative bending moment"]

        FOS_compiled.extend(FOS_shear_glue)   # Adding in the FOS for glue locations
        FOS_types_compiled.extend(len(FOS_shear_glue) * ["Shear in glue"])

        # Checks every FOS, if it is lower than the current lowest FOS of the bridge overall,
        # replaces the value and records the loads that caused lowest FOS
        for j in range(len(FOS_compiled)):
            if lowest_FOS[j] == None:
                lowest_FOS[j] = FOS_compiled[j]
                lowest_FOS_locations[j] = i
                local_buckle_type_pos = buckle_failure_type_pos
                local_buckle_type_neg = buckle_failure_type_neg
                worst_sf = test_loads
                worst_bm = test_loads
            elif lowest_FOS[j] > FOS_compiled[j]:
                lowest_FOS[j] = FOS_compiled[j]
                lowest_FOS_locations[j] = i
                if j == 3:
                    local_buckle_type_pos = buckle_failure_type_pos
                elif j == 5:
                    local_buckle_type_neg = buckle_failure_type_neg
                
                if j in [0, 1, 3, 5]:
                    worst_bm = test_loads
                elif j in [2, 4] or j >= 6:
                    worst_sf = test_loads

        # Compares the lowest buckling FOS under each bending moment
        for j in range(len(lowest_buckle_FOS_pos)):
            if lowest_buckle_FOS_pos[j] == None:
                lowest_buckle_FOS_pos[j] = buckle_bms_pos[j]/comparison_bm_pos
            elif lowest_buckle_FOS_pos[j] > buckle_bms_pos[j]/comparison_bm_pos:
                lowest_buckle_FOS_pos[j] = buckle_bms_pos[j]/comparison_bm_pos

        if comparison_bm_neg != 0:
            for j in range(len(lowest_buckle_FOS_neg)):
                if lowest_buckle_FOS_neg[j] == None:
                    lowest_buckle_FOS_neg[j] = buckle_bms_neg[j]/comparison_bm_neg
                elif lowest_buckle_FOS_neg[j] > buckle_bms_neg[j]/comparison_bm_neg:
                    lowest_buckle_FOS_neg[j] = buckle_bms_neg[j]/comparison_bm_neg
        else:
            lowest_buckle_FOS_neg = [9999] * len(lowest_buckle_FOS_neg)


    # Printing data

    print("\nFOS for each failure method:")
    for i in range(len(lowest_FOS)):   # Iterates through FOS list to print them all with location
        print(f"{FOS_types_compiled[i]}: {lowest_FOS[i]} at i = {lowest_FOS_locations[i]}")

    print("\nThe lowest local buckling under a positive bending moment is:", local_buckle_type_pos)
    print("The lowest local buckling under a negative bending moment is:", local_buckle_type_neg, "\n")


    # Calculating failure shear forces
    sf_fail_mat = (tau_m*I*t*2)/Q
    sf_fail_buckle = (case_4*I*t*2)/Q

    sf_fail_glue = []
    for i in range(len(b_glue)):
        sf_fail_glue.append((tau_g*I*b_glue[i])/glue_Q[i])

    print("Failure shear force for matboard:", sf_fail_mat)
    print("Failure shear force for glue:", min(sf_fail_glue))
    print("Failure shear force for all glued locations:", sf_fail_glue)
    print("Failure shear force for shear buckling:", sf_fail_buckle, "\n")

    # Calculating failure bending moments
    bm_fail_tens = min((sig_tens*I)/y_bar, (sig_tens*I)/(y_top))
    bm_fail_comp = max((sig_comp*I)/y_bar, (sig_comp*I)/(y_top))
    
    print("Failure bending moment for tension:", bm_fail_tens)
    print("Failure bending moment for compression:", bm_fail_comp)
    print("Failure bending moment for local buckling under positive bending moment:", buckle_failure_bm_pos)
    print("Failure bending moment for local buckling under negative bending moment:", buckle_failure_bm_neg)

    #Diagrams for worst scenarios
    if True:   # just a toggle for when I don't want it to print
        shear_force_diagram(worst_sf)
        bending_moment_diagram(shear_forces_dense(worst_bm))

    #print(lowest_buckle_FOS)   # Prints the FOS for all buckling cases