#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:51:38 2022

@author: iggymatheson
"""
#%% set up object in an existing sim with a given JD
def add_real_object(sim,JD,name):
    # import rebound
    from astroquery.jplhorizons import Horizons
    import numpy as np
    GMdict = get_GMdict()
    m_sun_2 = GMdict['sun'] + GMdict['mercury'] + GMdict['venus'] + \
                GMdict['earth'] + GMdict['mars']
    m_jupiter_2 = GMdict['jupiter']/m_sun_2
    m_saturn_2 = GMdict['saturn']/m_sun_2
    m_uranus_2 = GMdict['uranus']/m_sun_2
    m_neptune_2 = GMdict['neptune']/m_sun_2
    m_sun_2 = m_sun_2/m_sun_2
    m_sun = m_sun_2
    m_jupiter = m_jupiter_2
    m_saturn = m_saturn_2
    m_uranus = m_uranus_2
    m_neptune = m_neptune_2
    mass_catalog = {'0':m_sun,'5':m_jupiter,'6':m_saturn,'7':m_uranus,'8':m_neptune}
    center = '500@10' # Sun (body center) aka heliocentric
    # obj = Horizons(id=name,location=center,epochs=JD,id_type=id_type)
    obj = Horizons(id=name,location=center,epochs=JD)
    el = obj.elements()
    a = float(el['a']) # au
    e = float(el['e'])
    inc = float(el['incl']) # degrees
    w = float(el['w']) # degrees
    W = float(el['Omega']) # degrees
    M = float(el['M']) # degrees
    inc = np.radians(inc)
    w = np.radians(w)
    W = np.radians(W)
    M = np.radians(M)
    w = np.mod(w,2*np.pi)
    W = np.mod(W,2*np.pi)
    M = np.mod(M,2*np.pi)
    if name in mass_catalog:
        mass = mass_catalog[name]
    else:
        mass = 0
    sim.add(primary = sim.particles[0],m = mass,hash = name,\
            a = a,e = e,inc = inc,omega = w,Omega = W,M = M)
    return sim
#%% convert mpcorb files from awkward DAT to easy to use CSV, trim unwanted info
def dat2csv(date):
    import pandas as pd
    infile = '00_MPCORB_' + date + '.dat'
    outfile = '00_MPCORB_' + date + '.csv'
    designation_list = []
    readable_list = []
    a_list = []
    e_list = []
    i_list  = []
    w_list = []
    W_list = []
    M_list = []
    n_list = []
    Nopp_list = []
    file = open(infile,'r')
    lines = file.readlines()
    Nlines = len(lines)
    for iline in range(Nlines):
        if iline > 42:
            line = lines[iline]
            if len(line) > 3: # protect against blank lines
                designation = line[0:7]
                designation = designation.lstrip()
                designation = designation.rstrip()
                designation_list.append(designation)
                M = line[26:35]
                M = M.lstrip()
                M = M.rstrip()
                M_list.append(M)
                w = line[37:46]
                w = w.lstrip()
                w = w.rstrip()
                w_list.append(w)
                W = line[48:57]
                W = W.lstrip()
                W = W.rstrip()
                W_list.append(W)
                i = line[59:68]
                i = i.lstrip()
                i = i.rstrip()
                i_list.append(i)
                e = line[70:79]
                e = e.lstrip()
                e = e.rstrip()
                e_list.append(e)
                n = line[80:91]
                n = n.lstrip()
                n = n.rstrip()
                n_list.append(n)
                a = line[92:103]
                a = a.lstrip()
                a = a.rstrip()
                a_list.append(a)
                Nopp = line[123:126]
                Nopp = Nopp.lstrip()
                Nopp = Nopp.rstrip()
                Nopp = int(Nopp)
                Nopp_list.append(Nopp)
                readable = line[166:194]
                readable = readable.lstrip()
                readable = readable.rstrip()
                readable_list.append(readable)
    this_dict = {'packed_designation':designation_list,'readable_name':readable_list,\
        'a_au':a_list,'e':e_list,'i_deg':i_list,'w_deg':w_list,'W_deg':W_list,\
        'M_deg':M_list,'n_deg_day':n_list,'Nopp':Nopp_list}
    df = pd.DataFrame.from_dict(this_dict)
    df.to_csv(outfile,index=False)
    return
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
#%%
def get_GMdict():
    GM_sun = 1
    GM_mercury = 1/6023600
    GM_venus = 1/408523.71
    GM_earthmoon = 1/328900.56
    GM_mars = 1/3098708
    GM_jupiter = 1/1047.3486
    GM_saturn = 1/3497.898
    GM_uranus = 1/22902.98
    GM_neptune = 1/19412.24
    GMdict = {'sun':GM_sun,'mercury':GM_mercury,'venus':GM_venus,\
              'earth':GM_earthmoon,'mars':GM_mars,'jupiter':GM_jupiter,\
              'saturn':GM_saturn,'uranus':GM_uranus,'neptune':GM_neptune}
    return GMdict
#%%
def get_gmv08settings():
    tjmax = 3.05 # jupiter tisserand parameter used in gmv08 scheme
    qmax = 7.35 # minimum perihelion used in gmv08 scheme
    a_oort = 2000 # oort cloud distance used in gmv08 scheme
    e_detached = 0.24 # min eccentricity for Detached objects in gmv08 scheme
    a_outer = 47.8 # boundary for outer classical belt in gmv08 scheme
    a_inner = 39.4 # boundary for inner classical belt in gmv08 scheme
    gmv08settings = {'tjmax':tjmax,'qmax':qmax,'a_oort':a_oort,\
        'e_detached':e_detached,'a_outer':a_outer,'a_inner':a_inner}
    return gmv08settings
#%%
def JDfun(year,month,day,hour,minute,second):
    # from Vallado pg 183, valid for yrs 1900 to 2100
    import numpy as np
    JD = 367*year - np.floor(7/4*(year+np.floor(1/12*(month+9)))) + \
        np.floor(275*month/9) + day + 1721013.5 + 1/24*(hour+1/60*(second/60+minute))
    return JD
#%%
def sbdb_des_to_packed_mpc_des(des1): # bring in SBDB designations, get out MPC designations
    from bidict import bidict
    onedigit_bidict = bidict({'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6',\
        '7':'7','8':'8','9':'9','A':'10','B':'11','C':'12','D':'13','E':'14',\
        'F':'15','G':'16','H':'17','I':'18','J':'19','K':'20','L':'21','M':'22',\
        'N':'23','O':'24','P':'25','Q':'26','R':'27','S':'28','T':'29','U':'30',\
        'V':'31','W':'32','X':'33','Y':'34','Z':'35','a':'36','b':'37','c':'38',\
        'd':'39','e':'40','f':'41','g':'42','h':'43','i':'44','j':'45','k':'46',\
        'l':'47','m':'48','n':'49','o':'50','p':'51','q':'52','r':'53','s':'54',\
        't':'55','u':'56','v':'57','w':'58','x':'59','y':'60','z':'61'})
    century_bidict = bidict({'I':'18','J':'19','K':'20'})
    des1 = str(des1)
    if len(des1) == 5:
        des2 = des1 # leave it unchanged, example is 15760 for Albion
    if len(des1) == 6: # change the first two numerical digits to a letter
        first_two_digits = des1[0:2]
        last_four_digits = des1[2:6]
        one_digit = onedigit_bidict.inverse[first_two_digits]
        des2 = one_digit + last_four_digits
    if len(des1) > 6:
        century = des1[0:2]
        year = des1[2:4]
        space = des1[4]
        letter1 = des1[5]
        letter2 = des1[6]
        if len(des1) == 7:
            number = 0
        elif len(des1) == 8:
            number = des1[7]
        elif len(des1) == 9:
            number = des1[7:9]
        elif len(des1) == 10:
            number = des1[7:10]
        else:
            raise Exception('unrecognized sbdb designation format',des1)
        number = int(number)
        if space != ' ':
            raise Exception('unrecognized sbdb designation format',des1)
        century = century_bidict.inverse[century]
        if number == 0:
            code = '00'
        elif number < 10:
            code = '0' + str(number)
        elif number < 100:
            code = str(number)
        elif number < 620: # 619 is the max number with the MPC packing scheme
            numberstr = str(number)
            first_two_digits = numberstr[0:2]
            last_digit = numberstr[2]
            one_digit = onedigit_bidict.inverse[first_two_digits]
            code = one_digit + last_digit
        else:
            raise Exception('unrecognized sbdb designation format',des1)
        des2 = century + year + letter1 + code + letter2
    return des2
#%% reduce MPC and JPL SBDB files to a more manageable number of non-resonant TNO candidates
def sbdb_reduce(date,jpldate,JD,datestr):
    import pandas as pd
    import numpy as np
    from astroquery.jplhorizons import Horizons
    min_oppositions = 3
    max_fractional_sigma_a = 0.05
    mpcfile = '00_MPCORB_' + date + '.csv'
    jplfile = '00_sbdb_query_results_' + jpldate + '.csv'
    nomfile = 'horizons_barycentric_nom_' + date + '_' + datestr + '.csv'
    dfmpc = pd.read_csv(mpcfile,low_memory=False)
    dfmpc = dfmpc.loc[dfmpc['a_au']>28] # rough cut
    dfmpc = dfmpc.loc[dfmpc['a_au']<200] # rough cut
    dfmpc = dfmpc.loc[dfmpc['Nopp']>=min_oppositions]
    dfjpl = pd.read_csv(jplfile,low_memory=False)
    Nstart = dfjpl.shape[0]
    heliocentric_a_list = dfjpl['a'].tolist()
    temp_sigma_a_list = dfjpl['sigma_a'].tolist()
    for iobj in range(Nstart):
        if np.isnan(temp_sigma_a_list[iobj]):
            temp_sigma_a_list[iobj] = 0 # Need this to make sure we keep Pluto.
    dfjpl['sigma_a'] = temp_sigma_a_list
    heliocentric_sigma_a_list = dfjpl['sigma_a'].tolist()
    heliocentric_a_array = np.array(heliocentric_a_list)
    heliocentric_sigma_a_array = np.array(heliocentric_sigma_a_list)
    heliocentric_fractional_sigma_a_array = heliocentric_sigma_a_array / \
        heliocentric_a_array
    dfjpl['fractional_sigma_a'] = heliocentric_fractional_sigma_a_array
    dfjpl = dfjpl[dfjpl['fractional_sigma_a']<=max_fractional_sigma_a]
    '''
    now we find objects that are in BOTH the trimmed MPC list and the trimmed JPL list
    this ensures that opposition count is high enough and fractional semimajor axis
    uncertainty is low enough
    '''
    mpcdes = dfmpc['packed_designation'].tolist()
    jpldes = dfjpl['pdes'].tolist()
    full_name = dfjpl['full_name'].tolist()
    commondes = []
    Njpl = len(jpldes)
    Nmpc = len(mpcdes)
    for i in range(Njpl):
        des = jpldes[i]
        fullname = full_name[i]
        des = des.lstrip()
        des = des.rstrip()
        fullname = fullname.lstrip()
        fullname = fullname.rstrip()
        if fullname.startswith('C/'):
            2-2 # Comet; don't add it to the list of objects in common; we don't want it.
        else:
            des2 = sbdb_des_to_packed_mpc_des(des)
            if des2 in mpcdes:
                commondes.append(des2)
    for i in range(Nmpc):
        des = mpcdes[i]
        if (des in jpldes) and (des not in commondes):
            commondes.append(des)
    Ncom = len(commondes)
    # retrieve semimajor axis of Neptune
    center = '500@0' # solar system barycenter
    obj = Horizons(id='8',location=center,epochs=JD)
    el = obj.elements()
    a_neptune = float(el['a']) # au
    # now we retrieve barycentric elements for each object
    a_list = []
    e_list = []
    i_list = []
    w_list = []
    W_list = []
    M_list = []
    for i in range(Ncom):
        des = commondes[i]
        unpacked = unpack(des)
        if des == 'D4340':
            unpacked = '9' # want pluto-charon barycenter, not pluto body center
        obj = Horizons(id=unpacked,location=center,epochs=JD)
        el = obj.elements()
        a = float(el['a']) # au
        e = float(el['e'])
        incl = float(el['incl']) # deg
        w = float(el['w']) # deg
        W = float(el['Omega']) # deg
        M = float(el['M']) # deg
        incl = np.mod(incl,360)
        w = np.mod(w,360)
        W = np.mod(W,360)
        M = np.mod(M,360)
        a_list.append(a)
        e_list.append(e)
        i_list.append(incl)
        w_list.append(w)
        W_list.append(W)
        M_list.append(M)
    orbels_dict = {'packed_designation':commondes,'a_au':a_list,'e':e_list,\
        'i_deg':i_list,'w_deg':w_list,'W_deg':W_list,'M_deg':M_list}
    dfcom = pd.DataFrame.from_dict(orbels_dict)
    # now we do a fine cut to barycentric au of 34 to 150
    # also a perihelion cut and Tisserand cut to match gmv08 resonance eligibility requirements
    gmv08settings = get_gmv08settings()
    tjmax = gmv08settings['tjmax']
    qmax = gmv08settings['qmax']
    a_array = np.array(a_list)
    e_array = np.array(e_list)
    q_array = a_array * (1-e_array)
    name = '5' # jupiter
    obj = Horizons(id=name,location=center,epochs=JD)
    el = obj.elements()
    a = float(el['a'])
    a_perturber = a
    elliptical_comet_list = []
    elliptical_comet_count = 0
    for i in range(Ncom):
        a = a_array[i]
        e = e_array[i]
        inc = np.radians(i_list[i])
        tj = tisserand(a,a_perturber,inc,e)
        q = q_array[i]
        if q<qmax and tj<tjmax:
            elliptical_comet_list.append(2)
            elliptical_comet_count = elliptical_comet_count + 1
        else:
            elliptical_comet_list.append(0)
    dfcom['elliptical_comet'] = elliptical_comet_list
    dfcom = dfcom.loc[dfcom['elliptical_comet']<1] # drop elliptical comets
    dfcom = dfcom.drop(columns=['elliptical_comet']) # don't care about saving this info
    dfcom = dfcom.loc[dfcom['e']<1] # drop hyperbolic comets
    dfcom = dfcom.loc[dfcom['a_au']>=1.2*a_neptune] # enforce semimajor axis limits
    dfcom = dfcom.loc[dfcom['a_au']<=1.4*a_neptune] # enforce semimajor axis limits
    commondes = dfcom['packed_designation'].tolist()
    Ncom = len(commondes)
    dfcom.to_csv(nomfile,index=False)
    return Nstart,Ncom
#%% set up solar system (giant planets only) on a given JD, save to file
def set_up_solar_system(JD,integrator,savefile):
    import rebound
    sim = rebound.Simulation()
    sim.add(m = 1,hash = '0')
    sim = add_real_object(sim,JD,'5')
    sim = add_real_object(sim,JD,'6')
    sim = add_real_object(sim,JD,'7')
    sim = add_real_object(sim,JD,'8')
    sim.integrator = integrator
    sim.N_active = 5
    sim.move_to_com()
    sim.save(savefile) # eg 'snapshot.bin'
    return sim
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
from astroquery.jplhorizons import Horizons
date = '20221012'
year = 2022
month = 1
day = 1
hour = 0
minute = 0
second = 0
dat2csv(date)
JD = JDfun(year,month,day,hour,minute,second)
datestr = datestr([year,month,day,hour,minute,second])
outfile = 'sun_and_planets_Horizons_barycentric_' + datestr + '.csv'
center = '500@0' # solar system barycenter
GMdict = get_GMdict()
GM_list = [GMdict['sun'],GMdict['mercury'],GMdict['venus'],GMdict['earth'],\
           GMdict['mars'],GMdict['jupiter'],GMdict['saturn'],GMdict['uranus'],\
           GMdict['neptune']]
names_list = ['10','1','2','3','4','5','6','7','8'] # center of the sun, barycenter of each planet+moons system
Nobj = len(names_list)
a_list = []
e_list = []
i_list = []
w_list = []
W_list = []
M_list = []
for iobj in range(Nobj):
    name_in = names_list[iobj]
    obj = Horizons(id=name_in,location=center,epochs=JD)
    el = obj.elements()
    a = float(el['a']) # au
    e = float(el['e'])
    inc = float(el['incl']) # degrees
    w = float(el['w']) # degrees
    W = float(el['Omega']) # degrees
    M = float(el['M']) # degrees
    w = np.mod(w,360)
    W = np.mod(W,360)
    M = np.mod(M,360)
    a_list.append(a)
    e_list.append(e)
    i_list.append(inc)
    w_list.append(w)
    W_list.append(W)
    M_list.append(M)
dictionary = {'GM_solar_masses':GM_list,
              'semimajor_axis_au':a_list,
              'eccentricity':e_list,
              'inclination_degrees':i_list,
              'argument_of_pericenter_degrees':w_list,
              'longitude_of_node_degrees':W_list,
              'mean_anomaly_degrees':M_list,
              }
df_out = pd.DataFrame(dictionary)
df_out.to_csv(outfile,index=False)
savefile = 'sim_no_tno_' + datestr + '.bin'
integrator = 'ias15'
sim = set_up_solar_system(JD,integrator,savefile)
jpldate = date
Nstart,Njobs = sbdb_reduce(date,jpldate,JD,datestr)
print('starting number of objects',Nstart)
print('reduced number of objects',Njobs)
template_file = '00_check_plutino_resonances_template.py'
with open(template_file,'r') as file:
    template_data = file.read()
for i in range(Njobs):
    ct = i + 1
    ctstr = str(ct)
    outfile = 'check_plutino_resonances_' + ctstr + '.py'
    filedata = template_data
    filedata = filedata.replace('THIS_INSTANCE = 1','THIS_INSTANCE = '+ctstr)
    with open(outfile,'w') as file:
        file.write(filedata)
template_file = '00_slurm_check_plutino_resonances_template.slurm'
with open(template_file,'r') as file:
    template_data = file.read()
outfile = 'slurm_check_plutino_resonances.slurm'
filedata = template_data
filedata = filedata.replace('Nobj',str(Njobs))
with open(outfile,'w') as file:
    file.write(filedata)
