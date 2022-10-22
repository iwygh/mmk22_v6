#%% add to category lists sv20
def add_category_lists_sv20classifer(int_dict,prediction,detached_list,resonant_list,scattering_list,classical_list):
    if int_dict[0] == 'Detached':
        detached_list.append(prediction[0][0])
    elif int_dict[0] == 'Resonant':
        resonant_list.append(prediction[0][0])
    elif int_dict[0] == 'Scattering':
        scattering_list.append(prediction[0][0])
    elif int_dict[0] == 'Classical':
        classical_list.append(prediction[0][0])
    else:
        raise Exception('int_dict is all weird',int_dict)
    if int_dict[1] == 'Detached':
        detached_list.append(prediction[0][1])
    elif int_dict[1] == 'Resonant':
        resonant_list.append(prediction[0][1])
    elif int_dict[1] == 'Scattering':
        scattering_list.append(prediction[0][1])
    elif int_dict[1] == 'Classical':
        classical_list.append(prediction[0][1])
    else:
        raise Exception('int_dict is all weird',int_dict)
    if int_dict[2] == 'Detached':
        detached_list.append(prediction[0][2])
    elif int_dict[2] == 'Resonant':
        resonant_list.append(prediction[0][2])
    elif int_dict[2] == 'Scattering':
        scattering_list.append(prediction[0][2])
    elif int_dict[2] == 'Classical':
        classical_list.append(prediction[0][2])
    else:
        raise Exception('int_dict is all weird',int_dict)
    if int_dict[3] == 'Detached':
        detached_list.append(prediction[0][3])
    elif int_dict[3] == 'Resonant':
        resonant_list.append(prediction[0][3])
    elif int_dict[3] == 'Scattering':
        scattering_list.append(prediction[0][3])
    elif int_dict[3] == 'Classical':
        classical_list.append(prediction[0][3])
    else:
        raise Exception('int_dict is all weird',int_dict)
    return detached_list,resonant_list,scattering_list,classical_list
#%%
def classify_clones(THIS_INSTANCE,Njobs):
    import numpy as np
    import pandas as pd
    import rebound
    classifier, int_dict, types_dict = get_sv20classifier()
    date = 'YYYYMMDD'
    datestr = 'DATESTR'
    JD = 1234567.5
    sim_template_file = 'sim_no_tno_' + datestr + '.bin'
    horizons_file = 'horizons_barycentric_cov_' + date + '_' + datestr + '.csv'
    df = pd.read_csv(horizons_file)
    Nobj = df.shape[0]
    des_list = df['packed_designation'].tolist()
    obj_per_instance = int(np.ceil(Nobj/Njobs))
    start_obj = (THIS_INSTANCE-1) * obj_per_instance
    stop_obj = THIS_INSTANCE * obj_per_instance
    if stop_obj > Nobj:
        stop_obj = Nobj
    for iobj in range(start_obj,stop_obj):
        des = des_list[iobj]
        clones_file = 'clones_' + des + '.csv'
        df = pd.read_csv(clones_file)
        alist = df['a_au'].tolist()
        elist = df['e'].tolist()
        ilist = df['i_deg'].tolist()
        wlist = df['w_deg'].tolist()
        Wlist = df['W_deg'].tolist()
        tplist = df['tp_jd'].tolist()
        Nclones = df.shape[0]
        clone_class_list = []
        classical_list = []
        resonant_list = []
        scattering_list = []
        detached_list = []
        for iclone in range(Nclones):
            sim_template = rebound.Simulation(sim_template_file)
            data = runsim_sv20classifier(sim_template,iclone,alist,elist,ilist,wlist,Wlist,tplist,JD)
            sim_template = None
            category,prediction = parsedata_sv20classifier(data,classifier,int_dict)
            clone_class_list.append(category)
            detached_list,resonant_list,scattering_list,classical_list = \
                add_category_lists_sv20classifer(int_dict,prediction,detached_list,\
                                resonant_list,scattering_list,classical_list)
        df['category'] = clone_class_list
        df['classical_probability'] = classical_list
        df['resonant_probability'] = resonant_list
        df['scattering_probability'] = scattering_list
        df['detached_probability'] = detached_list
        df.to_csv(clones_file,index=False)
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
def get_sv20classifier():
    # classifier is 100% unchanged from sv20 sample code
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    training_file = 'KBO_features.csv'
    all_KBOs = pd.read_csv(training_file,skipinitialspace=True)
    secure_KBOs = all_KBOs[all_KBOs['Securely Classified']==True]
    all_types = list(set(secure_KBOs['Class']))
    types_dict = { all_types[i] : i for i in range( len(all_types) ) }
    int_dict = { i : all_types[i] for i in range( len(all_types) ) }
    classes = secure_KBOs['Class'].map(types_dict)
    features_train, features_test, classes_train, classes_test = train_test_split(secure_KBOs, classes, test_size=0.3, random_state=30)
    features_train.drop(['MPC ID', 'Securely Classified', 'Class'], axis=1, inplace=True)
    features_train = features_train.to_numpy()
    features_test.drop(['MPC ID', 'Securely Classified', 'Class'], axis=1, inplace=True)
    features_test = features_test.to_numpy()
    classifier = GradientBoostingClassifier( learning_rate=0.1, loss='deviance', max_depth=3, max_features='log2', n_estimators=130, random_state=30 )
    classifier.fit(features_train, classes_train)
    return classifier, int_dict, types_dict
#%%
def parsedata_sv20classifier(data,classifier,int_dict):
    '''
    parse(data) computes the necessary features to classify
    data MUST be a 101 row x 6 column array
    columns are t, a, e, i, Omega, omega
    rows are different time outputs: MUST be 1000yr outputs, ie [0, 1E3, 2E3....99E3,100E3]
    Returns features for classification
    from sv20
    Smullen, Rachel A., and Kathryn Volk.
    'Machine learning classification of Kuiper belt populations.'
    Monthly Notices of the Royal Astronomical Society 497.2 (2020): 1391-1403.
    classification simulation and data parsing is copied from sv20 code with
    minimal changes to not have to query Horizons through Rebound when setting up a sim,
    because Rebound's Horizons query is SOOOO SLOOOOW
    '''
    import numpy as np
    # Take stats of simulations
    initials = data[0,1:] # a, e, i, Omega, omega
    finals = data[-1,1:]
    mins = np.amin(data[:,1:],axis = 0)
    maxes = np.amax(data[:,1:],axis = 0)
    dels = maxes-mins
    means = np.mean(data[:,1:],axis = 0)
    stdev = np.std(data[:,1:],axis = 0)
    # Take time derivatives
    diffs = data[1:,:]-data[:-1,:]
    dxdt = diffs[:,1:]/diffs[:,0, np.newaxis] # add on new axis to time to give same dimensionality as the numerator
    mindxdt = np.amin(dxdt,axis = 0)
    meandxdt = np.mean(dxdt,axis = 0)
    maxdxdt = np.amax(dxdt,axis = 0)
    deldxdt = maxdxdt-mindxdt
    # rearrange data into the order I want
    arrs = [initials,finals,mins,means,maxes,stdev,dels,mindxdt,meandxdt,maxdxdt,deldxdt]
    inds = [0,1,2,3,4] # a, e, i, Omega, omega
    features = []
    ## features contains all x values, then all y, etc: xi, xf, xmin, xmean, xmax, xsigma, Deltax, xdotmin, xdotmean, xdotmax
    for i in inds:
        for a in arrs:
            features += [a[i]]
    features_out = np.array(features).reshape(1,-1) # make sure features is a 2d array
    prediction = classifier.predict_proba(features_out) # Predict the probabilities of class membership for object
    if np.max(prediction) == prediction[0][0]:
        category = int_dict[0]
    elif np.max(prediction) == prediction[0][1]:
        category = int_dict[1]
    elif np.max(prediction) == prediction[0][2]:
        category = int_dict[2]
    elif np.max(prediction) == prediction[0][3]:
        category = int_dict[3]
    print('This object has the following probabilities of class membership:')
    p=prediction[0]
    for i,k in enumerate(list(int_dict.keys())):
        print(int_dict[k],':',p[i]*100,'%')
    return category,prediction
#%%
def runsim_sv20classifier(sim_template,iobj,alist,elist,ilist,wlist,Wlist,tplist,JD):
    # run sim to classify object using gradient boosting classifier according to sv20
    import numpy as np
    sim = sim_template
    sim.integrator = 'ias15'
    primary = sim.calculate_com()
    ahere = alist[iobj]
    ehere = elist[iobj]
    ihere = ilist[iobj]
    where = wlist[iobj]
    Where = Wlist[iobj]
    tphere = tplist[iobj] # time of pericenter passage, JD
    delta_t = JD - tphere # time since pericenter passage, JD
    delta_t = delta_t * (2*np.pi/365.25) # convert to yr/(2pi)
    n = np.sqrt(1/ahere**3)
    Mhere = n * delta_t
    Mhere = np.mod(Mhere,2*np.pi)
    sim.add(a=ahere,e=ehere,inc=np.radians(ihere),omega=np.radians(where),\
            Omega=np.radians(Where),M=Mhere,primary=primary,m=0)
    sim.move_to_com()
    time_outs = np.linspace(0,100E3,101)*2*np.pi
    data = []
    for i,t in enumerate(time_outs):
        if t>0:
            sim.move_to_com()
            sim.integrate(t, exact_finish_time=True) # integrate to next output
        orbits = sim.calculate_orbits(primary=sim.calculate_com())
        o = orbits[-1] # take KBO
        step = np.array([t/2/np.pi, o.a, o.e, np.degrees(o.inc), np.degrees(o.Omega)%360, np.degrees(o.omega)%360]) # save t, a, e, i, Omega, omega - time in data needs to be in years, so divide by 2pi
        # add step to data
        if len(data)==0: data = step
        else: data = np.vstack((data,step))
    return data
#%%
THIS_INSTANCE = 1
Njobs = NJOBS
classify_clones(THIS_INSTANCE,Njobs)
