#civBridgeCase2.py
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
    flat_x = [0, length]

    fig, ax = plt.subplots()   # creates the plot
    ax.plot(x_values, y_values)
    plt.fill_between(x_values, y_values, 0, alpha=0.3)

    ax.plot(flat_x, [sf_fail_mat]*len(flat_x), "r-", label="Matboard Shear Failure")   # failure shear for matboard
    ax.plot(flat_x, [-1*sf_fail_mat]*len(flat_x), "r-")
    ax.plot(flat_x, [sf_fail_buckle]*len(flat_x), "m-", label="Shear Buckling Failure")   # failure shear buckling
    ax.plot(flat_x, [-1*sf_fail_buckle]*len(flat_x), "m-")
    ax.plot(flat_x, [min(sf_fail_glue)]*len(flat_x), "g-", label="Glue Shear Failure")   # failure shear for glue
    ax.plot(flat_x, [-1*min(sf_fail_glue)]*len(flat_x), "g-")

    ax.legend()
    plt.xlabel("Distance (mm)")
    plt.ylabel("Shear Force (N)")
    plt.title(f"Shear Force Diagram at P = {round(-1*i)}, Pfail = {round(-1*i, 6)}N")
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

    cross_point = -1
    for j in range(len(copy)):
        if copy[j] < 0:
            cross_point = j 
            break

    x_coords = list(forces.keys())
    y_coords = list(copy.values())

    fig, ax = plt.subplots()   # Plotting the curve
    plt.gca().invert_yaxis()
    ax.plot(x_coords, y_coords)
    plt.fill_between(x_coords, y_coords, 0, alpha=0.3)

    flat_x = [0, cross_point, cross_point+1, length]
    flat_x_pos = [0, cross_point]
    flat_x_neg = [cross_point, length]
    if False:
        ax.plot(flat_x, [bm_fail_tens_pos, bm_fail_tens_pos, -1*bm_fail_tens_neg, -1*bm_fail_tens_neg], 
        "r-", label="Matboard Tension Failure")
        ax.plot(flat_x, [bm_fail_comp_pos, bm_fail_comp_pos, -1*bm_fail_comp_neg, -1*bm_fail_comp_neg], 
        label="Matboard Compression Failure")

    if True:
        ax.plot(flat_x_pos, [bm_case_1, bm_case_1], "r-", label="Top Mid Flange Buckling")
        ax.plot(flat_x_neg, [-1*bm_case_1_2, -1*bm_case_1_2], label="Bottom Mid Flange Buckling")
        ax.plot(flat_x_pos, [bm_case_2, bm_case_2], "y-", label="Side Flange Buckling")
        ax.plot(flat_x_pos, [bm_case_3_top, bm_case_3_top], "m-", label="Top Web Buckling")
        ax.plot(flat_x_neg, [-1*bm_case_3_bottom, -1*bm_case_3_bottom], label="Bottom Web Buckling")

    ax.legend()
    plt.xlabel("Distance (mm)")
    plt.ylabel("Bending Moment (Nmm)")
    plt.title(f"Bending Moment Diagram at P = {round(-1*i)}, Pfail = {round(-1*i, 6)}N")
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

#######################
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
    Q = Q_max(dim)
    glue_Q = []
    for i in range(len(yglue)):
        glue_Q.append(Q_glue(dim, yglue[i]))

    print(f"Total Area: {A_tot(dim)} mm^2")
    print(f"Centroidal Axis: {ybar(dim)} mm")
    print(f"Second Moment of Area (I): {moment_of_I(dim)} mm^4")


    # Local buckling calculations
    case_1 = flange2(E_m, u_m, case1_b, t_top) 
    case_1_2 = flange2(E_m, u_m, case1_b, t_bot)
    case_2 = flange1(E_m, u_m, case2_b, t_top)
    case_3_top = web_buck(E_m, u_m, height_webs-y_bar, t_webs)
    case_3_bottom = web_buck(E_m, u_m, y_bar, t_webs)
    case_4 = shear_buckle(E_m, u_m, height_webs, find_a(diaphragms), t)   # Calculation for shear stress

    y_top = tot_height-y_bar   # Defining distances from centroid to top and bottom
    y_bot = y_bar

    bm_case_1 = (case_1*I)/y_top   # Calculating the bending moment for each stress
    bm_case_1_2 = (case_1_2*I)/(y_bot)
    bm_case_2 = (case_2*I)/y_top
    bm_case_3_top = (case_3_top*I)/y_top
    bm_case_3_bottom = (case_3_bottom*I)/y_bot
    bm_case_4 = (case_4*I*(t*2))/Q   # Calculating the shear force for shear stress

    # Extra calculations for plotting purposes
    bm_case_1_n = (case_1*I)/y_bot   # Calculating the bending moment for each stress
    bm_case_1_2_n = (case_1*I)/(y_top)
    bm_case_2_n = (case_2*I)/y_bot
    bm_case_3_top_n = (case_3_top*I)/y_bot
    bm_case_3_bottom_n = (case_3_bottom*I)/y_top

    buckle_stresses_pos = [case_1, case_2, case_3_top]   # Sorting through to find failure stress and type
    buckle_stresses_neg = [case_3_bottom, case_1]
    buckle_bms_pos = [bm_case_1, bm_case_2, bm_case_3_top]
    buckle_bms_neg = [bm_case_3_bottom, bm_case_1_2]
    types_of_buckling_pos = ["case_1_1", "case_2", "case_3_top"]
    types_of_buckling_neg = ["case_3_bottom", "case_1_2"]

    buckle_failure_bm_pos = min(buckle_bms_pos)
    buckle_failure_stress_pos = buckle_stresses_pos[buckle_bms_pos.index(buckle_failure_bm_pos)]
    buckle_failure_type_pos = types_of_buckling_pos[buckle_bms_pos.index(buckle_failure_bm_pos)]

    buckle_failure_bm_neg = min(buckle_bms_neg)
    buckle_failure_stress_neg = buckle_stresses_neg[buckle_bms_neg.index(buckle_failure_bm_neg)]
    buckle_failure_type_neg = types_of_buckling_neg[buckle_bms_neg.index(buckle_failure_bm_neg)]


    # Calculating failure shear forces
    sf_fail_mat = (tau_m*I*t*2)/Q
    sf_fail_buckle = (case_4*I*t*2)/Q
    sf_fail_glue = []
    for i in range(len(glue_Q)):
        sf_fail_glue.append(tau_g*I*t_webs*b_glue[i]/glue_Q[i])

    print("\nFailure shear force for matboard:", sf_fail_mat)
    print("Failure shear force for glue:", min(sf_fail_glue))
    print("Failure shear force for all glued locations:", sf_fail_glue)
    print("Failure shear force for shear buckling:", sf_fail_buckle, "\n")

    # Calculating failure bending moments
    bm_fail_tens = min((sig_tens*I)/y_bar, (sig_tens*I)/(tot_height-y_bar))
    bm_fail_comp = min((sig_comp*I)/y_bar, (sig_comp*I)/(tot_height-y_bar))

    bm_fail_tens_pos = (sig_tens*I)/y_bar
    bm_fail_comp_pos = (sig_comp*I)/(tot_height-y_bar)

    bm_fail_tens_neg = (sig_tens*I)/(tot_height-y_bar)
    bm_fail_comp_neg = (sig_comp*I)/y_bar
    
    print("Failure bending moment for tension under a positive bending moment:", bm_fail_tens_pos)
    print("Failure bending moment for compression under a positive bending moment:", bm_fail_comp_pos)
    print("Failure bending moment for tension under a negative bending moment:", bm_fail_tens_neg)
    print("Failure bending moment for compression under a negative bending moment:", bm_fail_comp_neg)
    print("Failure bending moment for local buckling with positive bending moment:", buckle_failure_bm_pos)
    print("Failure bending moment for local buckling with negative bending moment:", buckle_failure_bm_neg, "\n")

    # Compiling all failure forces and names into lists so that they can be compared
    failures = [bm_fail_tens_pos, bm_fail_comp_neg, bm_fail_tens_neg, bm_fail_comp_neg, 
    sf_fail_mat, buckle_failure_bm_pos, sf_fail_buckle, buckle_failure_bm_neg]
    failure_names =  ["Tension failure under positive bending moment", 
    "Compression failure under positive bending moment", 
    "Tension failure under negative bending moment", "Compression failure under negative bending moment",
    "Shear matboard failure", f"{buckle_failure_type_pos} failure", "Shear buckling failure", f"{buckle_failure_type_neg} failure"]

    failures.extend(sf_fail_glue)   # Adding on any glue shears since the exact number varies
    for i in range(len(sf_fail_glue)):   
        failure_names.append(f"Shear glue failure {i}")

    i = -1
    flag = False

    while True:
        # forces 
        p = i
        forces = {565: p, 1265: p}
        ay = list(rxn_f(forces,15,1075).values())[0]
        by = list(rxn_f(forces,15,1075).values())[1]
        test_loads = {15: ay, 565: p, 1075: by, 1265: p}
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

        # Shear failure calculations
        max_shear_stress = ((max_sf)*Q)/(I*t*2) # max shear stress is at middle

        shear_stress_glue = []
        for j in range(len(b_glue)):
            shear_stress_glue.append(((max_sf)*glue_Q[j])/(I*b_glue[j]))
    
        max_flexural_stress = max(max_tens_stress, abs(max_comp_stress))   # FOS calculations

        comparison_bm_pos = abs(list(max_pos_bm.values())[0]) # maximum positive and negative bending moments
        comparison_bm_neg = abs(list(max_neg_bm.values())[0])

        i -= 0.01   # iterates the forces

        # Compiling calculated max values for the specific P value
        i_failures = [comparison_bm_pos, comparison_bm_pos, comparison_bm_neg, comparison_bm_neg, 
        max_shear_stress, comparison_bm_pos, max_shear_stress, comparison_bm_neg]
        i_failures.extend(shear_stress_glue)

        # Comparing the failures to the overall values of the bridge
        # If it exceeds the failure values, returns the failure method and value of P
        for j in range(len(i_failures)):
            if abs(i_failures[j]) >= abs(failures[j]):
                print(f"{failure_names[j]} at {i}N")
                flag = True
                break

        if flag == True:
            break

    # Diagrams
    if True:   # Just a toggle for the diagrams
        shear_force_diagram(test_loads)
        bending_moment_diagram(shear_forces_dense(test_loads))
