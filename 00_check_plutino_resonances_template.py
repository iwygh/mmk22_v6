#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:50:07 2022
@author: iggymatheson
check if the 207 plutinos identified by sv20method+lm19boundaries are really 3:2 librators
"""
#%% check if object is resonant over 0-tmax time period
def check_resonance(orbels_dict,settings):
    # settings = {'tmax_yrs':tmax_yrs,'tstep_yrs':tstep_yrs,'min_stick_length_yrs':\
    # min_stick_length_yrs,'min_stick_length':min_stick_length,'bigmax':bigmax,\
    # 'smallmax':smallmax,'ordermax':ordermax,'smatol':smatol,'shift_amount':\
    # shift_amount,'mmr_tolerance':mmr_tolerance,'libration_threshold':libration_threshold,\
    # 'points_to_average':points_to_average,'averaging_proportion':averaging_proportion,\
    # 'long_or_small_threshold_pt_count':long_or_small_threshold_pt_count,\
    # 'small_division_count':small_division_count,'long_division_count':\
    # long_division_count,'length_threshold':length_threshold,'scattering_tolerance_delta_P':\
    # scattering_tolerance_delta_P,'scattering_tolerance_delta_a':scattering_tolerance_delta_a,\
    # 'tjmax':tjmax,'qmax':qmax,'a_oort':a_oort,'e_detached':e_detached,'a_outer':a_outer,\
    # 'a_inner':a_inner,'infile':infile,'outfile':outfile,'settingsfile':settingsfile,\
    # 'starting_sim_file':starting_sim_file,'integrator':integrator,'JD':JD}

    # orbels_dict = {'t':t_yrs_array,'a':a_array,'a_jupiter':a_jupiter_array,\
    # 'a_neptune':a_neptune_array,'tj':tj_array,'q':q_array,'e':e_array,\
    # 'inc':inc_array,'w':w_array,'W':W_array,'M':M_array,'l':l_array,\
    # 'l_neptune':l_neptune_array,'pomega':pomega_array,'pomega_neptune':\
    # pomega_neptune_array,'f':f_array,'f_neptune':f_neptune_array,\
    # 'P':P_yrs_array,'P_neptune':P_yrs_neptune_array,'theta':theta_array,\
    # 'theta_neptune':theta_neptune_array}
    import numpy as np
    bigmax = settings['bigmax'][0]
    smallmax = settings['smallmax'][0]
    ordermax = settings['ordermax'][0]
    min_stick_length = settings['min_stick_length'][0]
    shift_amount = settings['shift_amount'][0]
    scattering_tolerance_delta_a = settings['scattering_tolerance_delta_a'][0]
    scattering_tolerance_delta_P = settings['scattering_tolerance_delta_P'][0]
    mmr_tolerance = settings['mmr_tolerance'][0]
    points_to_average = settings['points_to_average'][0]
    averaging_proportion = settings['averaging_proportion'][0]
    libration_threshold = settings['libration_threshold'][0]
    long_or_small_threshold_pt_count = settings['long_or_small_threshold_pt_count'][0]
    long_division_count = settings['long_division_count'][0]
    small_division_count = settings['small_division_count'][0]
    length_threshold = settings['length_threshold'][0]
    a_list_neptune = orbels_dict['a_neptune']
    l_pluto_array = orbels_dict['l']
    l_neptune_array = orbels_dict['l_neptune']
    pomega_pluto_array = orbels_dict['pomega']
    a_list_1 = orbels_dict['a']
    amax = 150
    a_neptune = a_list_neptune[0]
    amin = a_neptune
    df_mmrs = make_mmrs(amin,amax,a_neptune,bigmax,smallmax,ordermax)
    Npts = len(l_pluto_array)
    Nmmr = df_mmrs.shape[0]
    startpts_list = np.arange(0,Npts-min_stick_length,shift_amount)
    stoppts_list = startpts_list + min_stick_length
    if stoppts_list[-1] < Npts:
        stoppts_list[-1] = Npts # add a couple extra points on the end
    Nwindow = len(startpts_list)
    resonant_window_list = []
    resonant_big_list = []
    resonant_small_list = []
    resonant_diff_list = []
    for iwin in range(Nwindow):
        any_resonance = False
        start_pt = startpts_list[iwin]
        stop_pt = stoppts_list[iwin]
        # do stuff in this sliding window
        a_list_2 = a_list_1[start_pt:stop_pt]
        a_list_neptune_2 = a_list_neptune[start_pt:stop_pt]
        l_list_2 = l_pluto_array[start_pt:stop_pt]
        l_list_neptune_2 = l_neptune_array[start_pt:stop_pt]
        pomega_list_2 = pomega_pluto_array[start_pt:stop_pt]
        # first check that particle isn't actively scattering
        a_list_2 = np.array(a_list_2)
        a_list_neptune_2 = np.array(a_list_neptune_2)
        l_list_2 = np.array(l_list_2)
        l_list_neptune_2 = np.array(l_list_neptune_2)
        pomega_list_2 = np.array(pomega_list_2)
        if np.min(a_list_2) > 0:
            period_ratio_list_2 = (a_list_2/a_list_neptune_2)**(3/2)
            pmax = np.max(period_ratio_list_2)
            pmin = np.min(period_ratio_list_2)
            delta_p = (pmax-pmin)/pmin
            delta_a = np.max(a_list_2) - np.min(a_list_2)
        else:
            delta_p = 99999 # arbitrarily large number
        if (delta_p > scattering_tolerance_delta_P) or \
           (delta_a > scattering_tolerance_delta_a):
            2-2 # move on to the next window
        else: # do stuff
            2-2
            # check which resonances to look for
            pmean = np.mean(period_ratio_list_2)
            for immr in range(Nmmr):
                pmmr = df_mmrs['period_ratio'][immr]
                # print(iwin+1,Nwindow,immr+1,Nmmr,pmmr)
                if abs(pmmr-pmean) <= mmr_tolerance:
                    # calculate resonant angle for this mmr in this window
                    big = df_mmrs['big'][immr]
                    small = df_mmrs['small'][immr]
                    theta_list_1 = big * l_list_2 - small * l_list_neptune_2 - (big-small) * pomega_list_2
                    theta_list_1 = np.mod(theta_list_1,360)
                    # check for libration
                    theta_sorted = np.sort(theta_list_1)
                    smallest_ten = theta_sorted[0:points_to_average]
                    largest_ten = theta_sorted[-points_to_average:]
                    smallest_mean = np.mean(smallest_ten)
                    largest_mean = np.mean(largest_ten)
                    libration_amplitude = (largest_mean-smallest_mean)/2
                    # print(libration_amplitude)
                    if libration_amplitude > libration_threshold:
                        librates_around_180 = False
                    else:
                        librates_around_180 = True
                        libration_amplitude_180 = libration_amplitude
                    theta_list_2 = theta_list_1 - 360 * (theta_list_1>180) # shift from 0,360 to -180,180
                    theta_sorted = np.sort(theta_list_2)
                    smallest_ten = theta_sorted[0:points_to_average]
                    largest_ten = theta_sorted[-points_to_average:]
                    smallest_mean = np.mean(smallest_ten)
                    largest_mean = np.mean(largest_ten)
                    libration_amplitude = (largest_mean-smallest_mean)/2
                    # print(libration_amplitude)
                    if libration_amplitude > libration_threshold:
                        librates_around_0 = False
                    else:
                        librates_around_0 = True
                        libration_amplitude_0 = libration_amplitude
                    if librates_around_180 or librates_around_0:
                    # if librates_around_180:
                        if librates_around_0:
                            libration_amplitude = libration_amplitude_0
                        if librates_around_180:
                            libration_amplitude = libration_amplitude_180
                        # record which time window and which resonance showed libration
                        if len(resonant_window_list) == 0:
                            resonant_window_list.append(iwin)
                        elif resonant_window_list[-1] != iwin:
                            resonant_window_list.append(iwin)
                        if len(resonant_big_list) == 0:
                            resonant_big_list.append([big])
                        elif len(resonant_big_list) != len(resonant_window_list):
                            resonant_big_list.append([big])
                        else:
                            resonant_big_list[-1].append(big)
                        if len(resonant_small_list) == 0:
                            resonant_small_list.append([small])
                        elif len(resonant_small_list) != len(resonant_window_list):
                            resonant_small_list.append([small])
                        else:
                            resonant_small_list[-1].append(small)
                        diff = big - small
                        if len(resonant_diff_list) == 0:
                            resonant_diff_list.append([diff])
                        elif len(resonant_diff_list) != len(resonant_window_list):
                            resonant_diff_list.append([diff])
                        else:
                            resonant_diff_list[-1].append(diff)
                        any_resonance = True
            if (any_resonance == False):
                2-2
    '''
    identify the unique resonances in the resonance list, then find the
    starts/stops/lengths/centers/etc of each resonance over time
    '''
    unique_big_list = []
    unique_small_list = []
    unique_diff_list = []
    unique_combo_list = []
    Nreswin = len(resonant_window_list)
    Nbigwin = len(resonant_big_list)
    Nsmallwin = len(resonant_small_list)
    Ndiffwin = len(resonant_diff_list)
    listwin = [Nreswin,Nbigwin,Nsmallwin,Ndiffwin]
    if np.std(listwin) != 0.0:
        2-2
    for iwin2 in range(Nreswin):
        big_list_here = resonant_big_list[iwin2]
        small_list_here = resonant_small_list[iwin2]
        diff_list_here = resonant_diff_list[iwin2]
        if (len(big_list_here) != len(small_list_here)) or (len(big_list_here) != len(diff_list_here)):
            2-2
        Nres_here = len(big_list_here)
        for ires in range(Nres_here):
            big_here = big_list_here[ires]
            small_here = small_list_here[ires]
            diff_here = diff_list_here[ires]
            combo = str(big_here) + ' to ' + str(small_here)
            if combo not in unique_combo_list:
                unique_combo_list.append(combo)
                unique_big_list.append(big_here)
                unique_small_list.append(small_here)
                unique_diff_list.append(diff_here)
    Nunique = len(unique_combo_list)
    unique_sticks_windows_lists = []
    unique_continuous_sticks_points_start_lists = []
    unique_continuous_sticks_points_end_lists = []
    unique_continuous_sticks_proportionatelength_lists = []
    unique_sticks_totalproportionatelength_list = []
    unique_continuous_sticks_amplitude_lists = []
    unique_continuous_sticks_center_lists = []
    for ires2 in range(Nunique):
        big = unique_big_list[ires2]
        small = unique_small_list[ires2]
        sticks_windows_list = []
        continuous_sticks_windows_start_list = []
        continuous_sticks_windows_end_list = []
        continuous_sticks_amplitude_list = []
        continuous_sticks_proportionatelength_list = []
        continuous_sticks_center_list = []
        for iwin3 in range(Nreswin):
            big_list_here = resonant_big_list[iwin3]
            small_list_here = resonant_small_list[iwin3]
            if (big in big_list_here) and (small in small_list_here):
                sticks_windows_list.append(iwin3)
        unique_sticks_windows_lists.append(sticks_windows_list)
        Nstickwin = len(sticks_windows_list)
        # print('\n')
        for iwin4 in range(Nstickwin):
            stickwin = sticks_windows_list[iwin4]
            if len(continuous_sticks_windows_start_list) == 0:
                continuous_sticks_windows_start_list.append(stickwin)
            if (iwin4 != 0) and (stickwin>(sticks_windows_list[iwin4-1]+1)):
                continuous_sticks_windows_start_list.append(stickwin)
            if (iwin4 == Nstickwin-1):
                continuous_sticks_windows_end_list.append(stickwin)
            elif (stickwin<(sticks_windows_list[iwin4+1]-1)):
                continuous_sticks_windows_end_list.append(stickwin)
        startpts_list_2 = []
        stoppts_list_2 = []
        Nstickwin2 =  len(continuous_sticks_windows_start_list)
        for iwin6 in range(Nstickwin2):
            startpt = startpts_list[continuous_sticks_windows_start_list[iwin6]]
            stoppt = stoppts_list[continuous_sticks_windows_end_list[iwin6]]
            startpts_list_2.append(startpt)
            stoppts_list_2.append(stoppt)
        stoppts_list_3 = []
        stopping_yet = False
        stop_index = -1
        while_ct_1 = 0
        while stopping_yet == False:
            while_ct_1 = while_ct_1 + 1
            stop_index = stop_index + 1
            stoppt = stoppts_list_2[stop_index]
            if stop_index == Nstickwin2-1:
                stopping_yet = True
            else:
                next_startpt = startpts_list_2[stop_index+1]
                if next_startpt <= stoppt:
                    stopping_yet = False
                else:
                    stopping_yet = True
        stoppts_list_3.append(stoppts_list_2[stop_index])
        while_ct_2 = 0
        while stop_index != Nstickwin2-1:
            while_ct_2 = while_ct_2 + 1
            stopping_yet = False
            inner_while_ct_2 = 0
            while stopping_yet == False:
                inner_while_ct_2 = inner_while_ct_2 + 1
                stop_index = stop_index + 1
                stoppt = stoppts_list_2[stop_index]
                if stop_index == Nstickwin2-1:
                    stopping_yet = True
                else:
                    next_startpt = startpts_list_2[stop_index+1]
                    if next_startpt <= stoppt:
                        stopping_yet = False
                    else:
                        stopping_yet = True
            stoppts_list_3.append(stoppts_list_2[stop_index])
        startpts_list_3 = [startpts_list_2[0]]
        start_index = 0
        while_ct_3 = 0
        while len(startpts_list_3) < len(stoppts_list_3):
            while_ct_3 = while_ct_3 + 1
            start_index = start_index + 1
            if start_index < len(startpts_list_2):
                if startpts_list_2[start_index] > stoppts_list_2[start_index-1]:
                    startpts_list_3.append(startpts_list_2[start_index])
            else:
                break
        unique_continuous_sticks_points_start_lists.append(startpts_list_3)
        unique_continuous_sticks_points_end_lists.append(stoppts_list_3)
        Nconwin = len(startpts_list_3)
        for iwin5 in range(Nconwin):
            startpt = startpts_list_3[iwin5]
            stoppt = stoppts_list_3[iwin5]
            Npts_here = stoppt-startpt
            proportionatelength = Npts_here/Npts
            continuous_sticks_proportionatelength_list.append(proportionatelength)
        unique_continuous_sticks_proportionatelength_lists.append(\
                continuous_sticks_proportionatelength_list)
        unique_sticks_totalproportionatelength_list.append(\
                np.sum(continuous_sticks_proportionatelength_list))
        # # # recompute amplitudes and centers of continuous sticks
        # # # this should eliminate spurious (circulating) angles
        Nstart = len(startpts_list_3)
        Nstop = len(stoppts_list_3)
        median_amplitude_list = []
        median_center_list = []
        if Nstart != Nstop:
            2-2
        for istart in range(Nstart):
            startpt = startpts_list_3[istart]
            stoppt = stoppts_list_3[istart]
            l_pluto_here = l_pluto_array[startpt:stoppt]
            l_neptune_here = l_neptune_array[startpt:stoppt]
            pomega_pluto_here = pomega_pluto_array[startpt:stoppt]
            theta_list_here = big * l_pluto_here - small * l_neptune_here - (big-small) * pomega_pluto_here
            theta_list_here = np.mod(theta_list_here,360)
            Ntheta = len(theta_list_here)
            if Ntheta >= long_or_small_threshold_pt_count:
                division_count = long_division_count
            else:
                division_count = small_division_count
            libration_amplitude_list = []
            libration_center_list = []
            Npts_per = int(np.floor(Ntheta/division_count))
            for idiv in range(division_count):
                start_pt = Npts_per*idiv
                stop_pt = start_pt + Npts_per
                if abs(stop_pt-Ntheta) < Npts_per:
                    stop_pt = Ntheta
                theta_here = theta_list_here[start_pt:stop_pt]
                theta_sorted = np.sort(theta_here)
                Npts_here = len(theta_sorted)
                points_to_average = int(np.ceil(Npts_here*averaging_proportion))
                smallest_few = theta_sorted[0:points_to_average]
                largest_few = theta_sorted[-points_to_average:]
                smallest_mean = np.mean(smallest_few)
                largest_mean = np.mean(largest_few)
                libration_amplitude = (largest_mean-smallest_mean)/2
                libration_amplitude_list.append(libration_amplitude)
                libration_center = np.mean([largest_mean,smallest_mean])
                libration_center_list.append(libration_center)
            median_amplitude = np.median(libration_amplitude_list)
            median_center = np.median(libration_center_list)
            median_amplitude_list.append(median_amplitude)
            median_center_list.append(median_center)
        unique_continuous_sticks_amplitude_lists.append(median_amplitude_list)
        unique_continuous_sticks_center_lists.append(median_center_list)
    unique_sticks_windows_lists_alt = []
    for ires4 in range(Nunique):
        sticks_windows_list = unique_sticks_windows_lists[ires4]
        sticks_windows_lists = []
        Nswl = len(sticks_windows_list)
        iswl = 0
        while iswl < Nswl:
            win = sticks_windows_list[iswl]
            if len(sticks_windows_lists) == 0:
                sticks_windows_lists.append([win])
            elif win == sticks_windows_lists[-1][-1]+1:
                sticks_windows_lists[-1].append(win)
            else:
                sticks_windows_lists.append([win])
            iswl = iswl + 1
        unique_sticks_windows_lists_alt.append(sticks_windows_lists)
    unique_combo_list_2 = []
    unique_big_list_2 = []
    unique_small_list_2 = []
    unique_diff_list_2 = []
    unique_sticks_windows_lists_2 = []
    unique_continuous_sticks_points_start_lists_2 = []
    unique_continuous_sticks_points_end_lists_2 = []
    unique_continuous_sticks_proportionatelength_lists_2 = []
    unique_sticks_totalproportionatelength_list_2 = []
    unique_continuous_sticks_amplitude_lists_2 = []
    unique_continuous_sticks_center_lists_2 = []
    for ires3 in range(Nunique):
        combo = unique_combo_list[ires3]
        big = unique_big_list[ires3]
        small = unique_small_list[ires3]
        diff = unique_diff_list[ires3]
        windows_lists = unique_sticks_windows_lists_alt[ires3]
        continuous_sticks_points_start_list = unique_continuous_sticks_points_start_lists[ires3]
        continuous_sticks_points_end_list = unique_continuous_sticks_points_end_lists[ires3]
        continuous_sticks_proportionatelength_list = unique_continuous_sticks_proportionatelength_lists[ires3]
        continuous_sticks_amplitude_list = unique_continuous_sticks_amplitude_lists[ires3]
        continuous_sticks_center_list = unique_continuous_sticks_center_lists[ires3]
        Nsticks = len(continuous_sticks_points_start_list)
        windows_lists_2 = []
        start_list_2 = []
        stop_list_2 = []
        length_list_2 = []
        amplitude_list_2 = []
        center_list_2 = []
        for istick in range(Nsticks):
            windows = windows_lists[istick]
            startpt = continuous_sticks_points_start_list[istick]
            stoppt = continuous_sticks_points_end_list[istick]
            length = continuous_sticks_proportionatelength_list[istick]
            amplitude = continuous_sticks_amplitude_list[istick]
            center = continuous_sticks_center_list[istick]
            if amplitude <= libration_threshold:
                windows_lists_2.append(windows)
                start_list_2.append(startpt)
                stop_list_2.append(stoppt)
                length_list_2.append(length)
                amplitude_list_2.append(amplitude)
                center_list_2.append(center)
        Nconwin2 = len(start_list_2)
        if Nconwin2 > 0:
            unique_combo_list_2.append(combo)
            unique_big_list_2.append(big)
            unique_small_list_2.append(small)
            unique_diff_list_2.append(diff)
            unique_sticks_windows_lists_2.append(windows_lists_2)
            unique_continuous_sticks_points_start_lists_2.append(start_list_2)
            unique_continuous_sticks_points_end_lists_2.append(stop_list_2)
            unique_continuous_sticks_proportionatelength_lists_2.append(length_list_2)
            unique_sticks_totalproportionatelength_list_2.append(np.sum(length_list_2))
            unique_continuous_sticks_amplitude_lists_2.append(amplitude_list_2)
            unique_continuous_sticks_center_lists_2.append(center_list_2)
            theta_list_plot = big * l_pluto_array - small * l_neptune_array - (big-small) * pomega_pluto_array
            theta_list_plot = np.mod(theta_list_plot,360)
    # # identify resonance with longest sticking time, if any
    Nunique2 = len(unique_combo_list_2)
    if Nunique2 == 0: # not resonant
        resonant = False
        big = 'NA'
        small = 'NA'
    else: # in one or more mean motion resonances for some amount of time
        lenlist = unique_sticks_totalproportionatelength_list_2
        lenindex = lenlist.index(np.max(lenlist))
        lenmax = lenlist[lenindex]
        big = unique_big_list_2[lenindex]
        small = unique_small_list_2[lenindex]
        big = str(big)
        small = str(small)
        if lenmax >= length_threshold:
            resonant = True
        else:
            resonant = False
            big = 'NA'
            small = 'NA'
    return resonant,big,small
#%%
def datestr(datevec):
    # datevec should be [yyyy,mm,dd,hh,mm,ss]
    datestr = ''
    for i in range(len(datevec)-1):
        thisstr = str(datevec[i])
        if len(thisstr) == 1:
            thisstr = '0' + thisstr
        if i == 0:
            datestr = datestr + thisstr
        else:
            datestr = datestr + '_' + thisstr
    thisstr = str(datevec[-1])
    if datevec[-1] < 10:
        thisstr = '0' + thisstr
    datestr = datestr + '_' + thisstr
    return datestr
#%% integrate a single object from initial conditions, return orbels_dict
def integrate_object_from_barycentric_initialconditions(starting_sim_file,tmax_yrs,tstep_yrs,a0,e0,i0,w0,W0,M0):
    # # barycentric initial conditions only!
    import rebound
    sim = rebound.Simulation(starting_sim_file)
    com = sim.calculate_com()
    sim.add(primary = com,m = 0,hash = 'tno',\
            a = a0,e = e0,inc = i0,omega = w0,Omega = W0,M = M0)
    sim.move_to_com()
    orbels_dict = integrate_single_tno(sim,tmax_yrs,tstep_yrs)
    return orbels_dict
#%% after the sim has been built with a single tno, integrate it and return orbels_dict
def integrate_single_tno(sim,tmax_yrs,tstep_yrs):
    import numpy as np
    sim.N_active = sim.N - 1 # tno is massless
    sim.move_to_com()
    tmax = tmax_yrs * 2*np.pi
    tstep = tstep_yrs * 2*np.pi
    t_list = []
    a_list = []
    a_jupiter_list = []
    a_neptune_list = []
    tj_list = []
    q_list = []
    e_list = []
    inc_list = []
    w_list = []
    W_list = []
    M_list = []
    l_list = []
    l_neptune_list = []
    pomega_list = []
    pomega_neptune_list = []
    f_list = []
    f_neptune_list = []
    P_list = []
    P_neptune_list = []
    theta_list = []
    theta_neptune_list = []
    sim.t = 0
    t = 0
    com = sim.calculate_com()
    t_list.append(t)
    jupiter = sim.particles['5']
    neptune = sim.particles['8']
    tno = sim.particles[5] # 0sun,1jup,2sat,3ur,4nep
    orbit_jupiter = jupiter.calculate_orbit(primary=com)
    orbit_neptune = neptune.calculate_orbit(primary=com)
    orbit_tno = tno.calculate_orbit(primary=com)
    a = orbit_tno.a
    a_jupiter = orbit_jupiter.a
    a_neptune = orbit_neptune.a
    e = orbit_tno.e
    q = a * (1-e)
    inc = orbit_tno.inc
    tj = tisserand(a,a_jupiter,inc,e)
    w = orbit_tno.omega
    W = orbit_tno.Omega
    M = orbit_tno.M
    l = orbit_tno.l
    l_neptune = orbit_neptune.l
    pomega = orbit_tno.pomega
    pomega_neptune = orbit_neptune.pomega
    f = orbit_tno.f
    f_neptune = orbit_neptune.f
    P = orbit_tno.P
    P_neptune = orbit_neptune.P
    theta = orbit_tno.theta
    theta_neptune = orbit_neptune.theta
    a_list.append(a)
    a_jupiter_list.append(a_jupiter)
    a_neptune_list.append(a_neptune)
    q_list.append(q)
    e_list.append(e)
    inc_list.append(inc)
    tj_list.append(tj)
    w_list.append(w)
    W_list.append(W)
    M_list.append(M)
    l_list.append(l)
    l_neptune_list.append(l_neptune)
    pomega_list.append(pomega)
    pomega_neptune_list.append(pomega_neptune)
    f_list.append(f)
    f_neptune_list.append(f_neptune)
    P_list.append(P)
    P_neptune_list.append(P_neptune)
    theta_list.append(theta)
    theta_neptune_list.append(theta_neptune)
    while t < tmax:
        t = t + tstep
        sim.integrate(t,exact_finish_time=1)
        # # record stuff
        com = sim.calculate_com()
        t_list.append(t)
        jupiter = sim.particles['5']
        neptune = sim.particles['8']
        tno = sim.particles[5] # 0sun,1jup,2sat,3ur,4nep
        orbit_jupiter = jupiter.calculate_orbit(primary=com)
        orbit_neptune = neptune.calculate_orbit(primary=com)
        orbit_tno = tno.calculate_orbit(primary=com)
        a = orbit_tno.a
        a_jupiter = orbit_jupiter.a
        a_neptune = orbit_neptune.a
        e = orbit_tno.e
        q = a * (1-e)
        inc = orbit_tno.inc
        tj = tisserand(a,a_jupiter,inc,e)
        w = orbit_tno.omega
        W = orbit_tno.Omega
        M = orbit_tno.M
        l = orbit_tno.l
        l_neptune = orbit_neptune.l
        pomega = orbit_tno.pomega
        pomega_neptune = orbit_neptune.pomega
        f = orbit_tno.f
        f_neptune = orbit_neptune.f
        P = orbit_tno.P
        P_neptune = orbit_neptune.P
        theta = orbit_tno.theta
        theta_neptune = orbit_neptune.theta
        a_list.append(a)
        a_jupiter_list.append(a_jupiter)
        a_neptune_list.append(a_neptune)
        q_list.append(q)
        e_list.append(e)
        inc_list.append(inc)
        tj_list.append(tj)
        w_list.append(w)
        W_list.append(W)
        M_list.append(M)
        l_list.append(l)
        l_neptune_list.append(l_neptune)
        pomega_list.append(pomega)
        pomega_neptune_list.append(pomega_neptune)
        f_list.append(f)
        f_neptune_list.append(f_neptune)
        P_list.append(P)
        P_neptune_list.append(P_neptune)
        theta_list.append(theta)
        theta_neptune_list.append(theta_neptune)
    # # done integrating, now save stuff to a dictionary
    t_array = np.array(t_list)
    t_yrs_array = t_array/2/np.pi
    a_array = np.array(a_list)
    a_jupiter_array = np.array(a_jupiter_list)
    a_neptune_array = np.array(a_neptune_list)
    tj_array = np.array(tj_list)
    q_array = np.array(q_list)
    e_array = np.array(e_list)
    inc_array = np.array(inc_list)
    w_array = np.array(w_list)
    W_array = np.array(W_list)
    M_array = np.array(M_list)
    l_array = np.array(l_list)
    l_neptune_array = np.array(l_neptune_list)
    pomega_array = np.array(pomega_list)
    pomega_neptune_array = np.array(pomega_neptune_list)
    f_array = np.array(f_list)
    f_neptune_array = np.array(f_neptune_list)
    P_array = np.array(P_list)
    P_neptune_array = np.array(P_neptune_list)
    theta_array = np.array(theta_list)
    theta_neptune_array = np.array(theta_neptune_list)
    w_array = np.mod(w_array,2*np.pi)
    W_array = np.mod(W_array,2*np.pi)
    M_array = np.mod(M_array,2*np.pi)
    l_array = np.mod(l_array,2*np.pi)
    l_neptune_array = np.mod(l_neptune_array,2*np.pi)
    pomega_array = np.mod(pomega_array,2*np.pi)
    pomega_neptune_array = np.mod(pomega_neptune_array,2*np.pi)
    f_array = np.mod(f_array,2*np.pi)
    f_neptune_array = np.mod(f_neptune_array,2*np.pi)
    theta_array = np.mod(theta_array,2*np.pi)
    theta_neptune_array = np.mod(theta_neptune_array,2*np.pi)
    inc_array = np.degrees(inc_array)
    w_array = np.degrees(w_array)
    W_array = np.degrees(W_array)
    M_array = np.degrees(M_array)
    l_array = np.degrees(l_array)
    l_neptune_array = np.degrees(l_neptune_array)
    pomega_array = np.degrees(pomega_array)
    pomega_neptune_array = np.degrees(pomega_neptune_array)
    f_array = np.degrees(f_array)
    f_neptune_array = np.degrees(f_neptune_array)
    P_yrs_array = P_array/2/np.pi
    P_yrs_neptune_array = P_neptune_array/2/np.pi
    theta_array = np.degrees(theta_array)
    theta_neptune_array = np.degrees(theta_neptune_array)
    orbels_dict = {'t':t_yrs_array,'a':a_array,'a_jupiter':a_jupiter_array,\
        'a_neptune':a_neptune_array,'tj':tj_array,'q':q_array,'e':e_array,\
        'inc':inc_array,'w':w_array,'W':W_array,'M':M_array,'l':l_array,\
        'l_neptune':l_neptune_array,'pomega':pomega_array,'pomega_neptune':\
        pomega_neptune_array,'f':f_array,'f_neptune':f_neptune_array,\
        'P':P_yrs_array,'P_neptune':P_yrs_neptune_array,'theta':theta_array,\
        'theta_neptune':theta_neptune_array}
    sim = None
    return orbels_dict
#%%
def JDfun(year,month,day,hour,minute,second):
    # from Vallado pg 183, valid for yrs 1900 to 2100
    import numpy as np
    JD = 367*year - np.floor(7/4*(year+np.floor(1/12*(month+9)))) + \
        np.floor(275*month/9) + day + 1721013.5 + 1/24*(hour+1/60*(second/60+minute))
    return JD
#%% make list of mmrs to check
def make_mmrs(amin,amax,a_neptune,bigmax,smallmax,ordermax):
    import math
    import pandas as pd
    big_list = []
    small_list = []
    period_ratio_list = []
    sma_list = []
    for big in range(bigmax):
        for small in range(smallmax):
            big = big + 1
            small = small + 1 # adjust for Python's 0-indexing
            gcd = math.gcd(big,small)
            atno = (big/small)**(2/3) * a_neptune
            if gcd == 1 and (amin <= atno < amax) and (big>=small) and (big-small<=ordermax):
                big_list.append(big)
                small_list.append(small)
                period_ratio_list.append(big/small)
                sma_list.append(atno)
    data = {'big':big_list,\
            'small':small_list,\
            'period_ratio':period_ratio_list,\
            'semimajor_axis':sma_list}
    df_mmrs = pd.DataFrame.from_dict(data)
    return df_mmrs
#%% simple plotting to see if the category makes sense
def plot_big_small(mpcid,unpacked,category,big,small,orbels_dict,title):
    import numpy as np
    import matplotlib.pyplot as plt
    try:
        big = int(float(big))
        small = int(float(small))
        l_tno = orbels_dict['l']
        l_neptune = orbels_dict['l_neptune']
        pomega_tno = orbels_dict['pomega']
        l_tno = np.array(l_tno)
        l_neptune = np.array(l_neptune)
        pomega_tno = np.array(pomega_tno)
        yvar = big*l_tno - small*l_neptune - (big-small)*pomega_tno
        yvar = np.mod(yvar,360)
        ylabel = 'theta_resonant'
    except:
        big = 3
        small = 2
        l_tno = orbels_dict['l']
        l_neptune = orbels_dict['l_neptune']
        pomega_tno = orbels_dict['pomega']
        l_tno = np.array(l_tno)
        l_neptune = np.array(l_neptune)
        pomega_tno = np.array(pomega_tno)
        yvar = big*l_tno - small*l_neptune - (big-small)*pomega_tno
        yvar = np.mod(yvar,360)
        ylabel = 'theta_resonant'
    xvar = orbels_dict['t']
    xvar = np.array(xvar)
    xvar = xvar/1e6 # Myr
    xvar = xvar[0:-1:100]
    yvar = yvar[0:-1:100]
    xlabel = 'Myr'
    longtitle = 'long_' + title + ' ' + mpcid + ' ' + unpacked + ' ' + category + ' ' + \
        str(big) + ' ' + str(small)
    plt.plot(xvar,yvar,'b.',markersize=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(longtitle)
    plt.savefig(longtitle+'.pdf',format='pdf',transparent=True)
    plt.show()
    return
#%% calculate tisserand parameter
def tisserand(a,a_perturber,inc,ecc):
    import numpy as np
    tp = a_perturber/a + 2 * np.cos(inc) * np.sqrt(a/a_perturber*(1-ecc*ecc))
    return tp
#%% unpack Minor Planet Center MPC ID
def unpack(designation):
    packed_designation = designation.lstrip()
    packed_designation = packed_designation.rstrip()
    Nt = len(packed_designation)
    onedigit_dict = {'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6',\
        '7':'7','8':'8','9':'9','A':'10','B':'11','C':'12','D':'13','E':'14',\
        'F':'15','G':'16','H':'17','I':'18','J':'19','K':'20','L':'21','M':'22',\
        'N':'23','O':'24','P':'25','Q':'26','R':'27','S':'28','T':'29','U':'30',\
        'V':'31','W':'32','X':'33','Y':'34','Z':'35','a':'36','b':'37','c':'38',\
        'd':'39','e':'40','f':'41','g':'42','h':'43','i':'44','j':'45','k':'46',\
        'l':'47','m':'48','n':'49','o':'50','p':'51','q':'52','r':'53','s':'54',\
        't':'55','u':'56','v':'57','w':'58','x':'59','y':'60','z':'61'}
    century_dict = {'I':'18','J':'19','K':'20'}
    if Nt == 4:
        unpacked_designation = packed_designation
    if Nt == 5:
        onedigit = packed_designation[0]
        fourdigits = packed_designation[1:5]
        if onedigit in onedigit_dict:
            onedigit = onedigit_dict[onedigit]
            unpacked_designation = onedigit + fourdigits
        else:
            unpacked_designation = '-99999'
    elif Nt == 7:
        century = packed_designation[0]
        year = packed_designation[1:3]
        letter1 = packed_designation[3]
        code1 = packed_designation[4]
        code2 = packed_designation[5]
        letter2 = packed_designation[6]
        space = ' '
        if century in century_dict:
            century = century_dict[century]
            if code1 == '0' and code2 == '0':
                code1 = ''
                code2 = ''
            else:
                if code1 in onedigit_dict:
                    code1 = onedigit_dict[code1]
                    if code1 == '0':
                        code1 = ''
            unpacked_designation = century + year + space + letter1 + letter2 + \
                code1 + code2
        else:
            unpacked_designation = '-99999'
    unpacked_designation = unpacked_designation.lstrip()
    unpacked_designation = unpacked_designation.rstrip()
    return unpacked_designation
#%%
import numpy as np
import pandas as pd
import rebound
THIS_INSTANCE = 1
date = '20221012'
year = 2022
month = 1
day = 1
hour = 0
minute = 0
second = 0
JD = JDfun(year,month,day,hour,minute,second)
JD = str(JD)
datestr = datestr([year,month,day,hour,minute,second])
csvfile = 'horizons_barycentric_nom_' + date + '_' + datestr + '.csv'
starting_sim_file = 'sim_no_tno_' + datestr + '.bin'
integrator = 'ias15'
category = 'Resonant'
df = pd.read_csv(csvfile)
Nobj = df.shape[0]
tmax_yrs = 1e7
tstep_yrs = 100
settings =  {}
settings['bigmax'] = [3]
settings['smallmax'] = [2]
settings['ordermax'] = [1]
settings['min_stick_length'] = [1000] # how many points in a window
settings['smatol'] = [0.05] # not used
settings['shift_amount'] = [100] # how many time steps to shift overlapping resonance search windows
settings['mmr_tolerance'] = [0.1] # how far away from the nominal period ratio to look for a resonance
settings['libration_threshold'] = [175] # maximum accepted resonance amplitude
settings['points_to_average'] = [10] # used in checking libration within a single window
settings['averaging_proportion'] = [2e-2] # used in recomputing centers and amplitudes of continuous sticks
settings['long_or_small_threshold_pt_count'] = [2000] # how many points makes a stick long or short when recomputing amplitude
settings['small_division_count'] = [6] # how many divisions to split a short stick into
settings['long_division_count'] = [20] # how many divisions to split a long stick into
settings['length_threshold'] = [0.5] # what total fraction of the time an object must librate to count as Resonant
settings['scattering_tolerance_delta_P'] = [0.2] # if period ratio tno/neptune changes by this much, it's Scattering
settings['scattering_tolerance_delta_a'] = [1.5] # if tno semimajor axis changes by this much, it's Scattering
a_list = df['a_au'].to_list()
e_list = df['e'].to_list()
i_list = df['i_deg'].to_list()
w_list = df['w_deg'].to_list()
W_list = df['W_deg'].to_list()
M_list = df['M_deg'].to_list()
a_array = np.array(a_list)
e_array = np.array(e_list)
i_array = np.radians(np.array(i_list))
w_array = np.radians(np.mod(np.array(w_list),360))
W_array = np.radians(np.mod(np.array(W_list),360))
M_array = np.radians(np.mod(np.array(M_list),360))
for iobj in range(THIS_INSTANCE-1,THIS_INSTANCE):
    a0 = a_array[iobj]
    e0 = e_array[iobj]
    i0 = i_array[iobj]
    w0 = w_array[iobj]
    W0 = W_array[iobj]
    M0 = M_array[iobj]
    sim = rebound.Simulation(starting_sim_file)
    orbels_dict = integrate_object_from_barycentric_initialconditions(starting_sim_file,tmax_yrs,tstep_yrs,a0,e0,i0,w0,W0,M0)
    resonant,big,small = check_resonance(orbels_dict,settings)
    mpcid = df['packed_designation'][iobj]
    unpacked = unpack(mpcid)
    print(mpcid,unpacked,resonant,big,small)
    title = str(resonant) + '_plutino'
    plot_big_small(mpcid,unpacked,category,big,small,orbels_dict,title)
    del sim
