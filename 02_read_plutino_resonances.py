#%%
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python
    Produce a circular histogram of angles on ax.
    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').
    x : array
        Angles to plot, expected in units of radians.
    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.
    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.
    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.
    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.
    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.
    bins : array
        The edges of the bins.
    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi
    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)
    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)
    # Compute width of each bin
    widths = np.diff(bins)
    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n
    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)
    # Set the direction of the zero angle
    ax.set_theta_offset(offset)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
    return n, bins, patches
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
def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    # also from Stack Exchange or a Google search, I forgot to copy the link
    import scipy.interpolate as interpolate
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)
#%%
def JDfun(year,month,day,hour,minute,second):
    # from Vallado pg 183, valid for yrs 1900 to 2100
    import numpy as np
    JD = 367*year - np.floor(7/4*(year+np.floor(1/12*(month+9)))) + \
        np.floor(275*month/9) + day + 1721013.5 + 1/24*(hour+1/60*(second/60+minute))
    return JD
#%%
def rand_t_marginal(kappa,p,N=1):
    """
    https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html
        rand_t_marginal(kappa,p,N=1)
        ============================
        Samples the marginal distribution of t using rejection sampling of Wood [3].
        INPUT:
            * kappa (float) - concentration
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
            * N (int) - number of samples
        OUTPUT:
            * samples (array of floats of shape (N,1)) - samples of the marginal distribution of t
    """
    import numpy as np
    # Check kappa >= 0 is numeric
    # if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
    if (kappa < 0):
        raise Exception("kappa must be a non-negative number.")
    if (p<=0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")
    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")
    # Start of algorithm
    b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (p - 1.0)**2 ))
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0**2)
    samples = np.zeros((N,1))
    # Loop over number of samples
    for i in range(N):
        # Continue unil you have an acceptable sample
        while True:
            # Sample Beta distribution
            Z = np.random.beta( (p - 1.0)/2.0, (p - 1.0)/2.0 )
            # Sample Uniform distribution
            U = np.random.uniform(low=0.0,high=1.0)
            # W is essentially t
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
            # Check whether to accept or reject
            if kappa * W + (p - 1.0)*np.log(1.0 - x0*W) - c >= np.log(U):
                # Accept sample
                samples[i] = W
                break
    return samples
#%%
def rand_uniform_hypersphere(N,p):
    """
    https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html
        rand_uniform_hypersphere(N,p)
        =============================
        Generate random samples from the uniform distribution on the (p-1)-dimensional
        hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$. We use the method by
        Muller [1], see also Ref. [2] for other methods.
        INPUT:
            * N (int) - Number of samples
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
    References:
    [1] Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres."
    Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.
    [2] https://mathworld.wolfram.com/SpherePointPicking.html
    """
    import numpy as np
    if (p<=0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")
    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")
    v = np.random.normal(0,1,(N,p))
    v = np.divide(v,np.linalg.norm(v,axis=1,keepdims=True))
    return v
#%%
def rand_von_mises_fisher(mu,kappa,N=1):
    """
    https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html
        rand_von_mises_fisher(mu,kappa,N=1)
        ===================================
        Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa.
        INPUT:
            * mu (array of floats of shape (p,1)) - mean direction. This should be a unit vector.
            * kappa (float) - concentration.
            * N (int) - Number of samples.
        OUTPUT:
            * samples (array of floats of shape (N,p)) - samples of the von Mises-Fisher distribution
            with mean direction mu and concentration kappa.
    """
    import numpy as np
    import numpy.matlib
    from scipy.linalg import null_space
    # Check that mu is a unit vector
    eps = 10**(-8) # Precision
    norm_mu = np.linalg.norm(mu)
    if abs(norm_mu - 1.0) > eps:
        raise Exception("mu must be a unit vector.")
    # Check kappa >= 0 is numeric
    # if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
    if (kappa < 0):
        raise Exception("kappa must be a non-negative number.")
    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")
    # Dimension p
    p = len(mu)
    # Make sure that mu has a shape of px1
    mu = np.reshape(mu,(p,1))
    # Array to store samples
    samples = np.zeros((N,p))
    #  Component in the direction of mu (Nx1)
    t = rand_t_marginal(kappa,p,N)
    # Component orthogonal to mu (Nx(p-1))
    xi = rand_uniform_hypersphere(N,p-1)
    # von-Mises-Fisher samples Nxp
    # Component in the direction of mu (Nx1).
    # Note that here we are choosing an
    # intermediate mu = [1, 0, 0, 0, ..., 0] later
    # we rotate to the desired mu below
    samples[:,[0]] = t
    # Component orthogonal to mu (Nx(p-1))
    samples[:,1:] = np.matlib.repmat(np.sqrt(1 - t**2), 1, p-1) * xi
    # Rotation of samples to desired mu
    O = null_space(mu.T)
    R = np.concatenate((mu,O),axis=1)
    samples = np.dot(R,samples.T).T
    return samples
#%%
def rodrigues_rotation(vold,k,theta):
    # vold is a vector in R3
    # k is a unit vector describing an axis of rotation about which vold rotates
    # theta is the angle of rotation according to the right hand rule
    import numpy as np
    # equation A2 in the paper
    term1 = vold * np.cos(theta)
    term2 = np.cross(k,vold) * np.sin(theta)
    term3 = k * np.dot(k,vold) * (1-np.cos(theta))
    vnew = term1 + term2 + term3
    return vnew
#%%
# def truncated_rayleigh_pdf(x, sigma):
def truncated_rayleigh_pdf(x,kappa):
    import numpy as np
    # equation 26 in the paper
    sigma = 1/np.sqrt(kappa)
    # equation 28 in the paper
    C_R = 1/(1-np.exp(-np.pi**2/2/sigma**2))
    # equation 27 in the paper
    return C_R/sigma**2 * x * np.exp(-x**2/2/sigma**2)
#%% K04K19H
# K14Se4J
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from scipy import special
from scipy import stats
import scipy.optimize as optimization
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astroquery.jplhorizons import Horizons
date = '20221012'
year = 2022
month = 1
day = 1
hour = 0
minute = 0
second = 0
JD = JDfun(year,month,day,hour,minute,second)
datestr = datestr([year,month,day,hour,minute,second])
horizons_file = 'horizons_barycentric_nom_' + date + '_' + datestr + '.csv'
resfile = 'resonant_barycentric_nom' + date + '_' + datestr + '.csv'
path = os.getcwd()
df = pd.read_csv(horizons_file)
# packed_designation,a_au,e,i_deg,w_deg,W_deg,M_deg
Nobj = df.shape[0]
resdes = []
res_a = []
res_e = []
res_i = []
res_w = []
res_W = []
res_M = []
for iobj in range(Nobj):
    des = df['packed_designation'][iobj]
    if des == 'K04K19H': # this object looks resonant but is tagged False, add it in
        resdes.append(des)
        res_a.append(df['a_au'][iobj])
        res_e.append(df['e'][iobj])
        res_i.append(df['i_deg'][iobj])
        res_w.append(df['w_deg'][iobj])
        res_W.append(df['W_deg'][iobj])
        res_M.append(df['M_deg'][iobj])
    elif des != 'K14Se4J': # this object doesn't look resonant but is tagged True, exclude it
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and ('long_True_plutino ' + des) in i:
                resdes.append(des)
                res_a.append(df['a_au'][iobj])
                res_e.append(df['e'][iobj])
                res_i.append(df['i_deg'][iobj])
                res_w.append(df['w_deg'][iobj])
                res_W.append(df['W_deg'][iobj])
                res_M.append(df['M_deg'][iobj])
Nres = len(resdes)
print('number of source objects=',Nobj,'number of plutinos=',Nres)
dictionary = {'packed_designation':resdes,'a_au':res_a,'e':res_e,'i_deg':res_i,\
              'w_deg':res_w,'W_deg':res_W,'M_deg':res_M}
df = pd.DataFrame.from_dict(dictionary)
df.to_csv(resfile,index=False)
df = pd.read_csv(resfile)
des_list = df['packed_designation'].tolist()
a_list = df['a_au'].to_list()
e_list = df['e'].to_list()
i_list = df['i_deg'].to_list()
w_list = df['w_deg'].to_list()
W_list = df['W_deg'].to_list()
M_list = df['M_deg'].to_list()
# # make new dataframe with reordered columns
df2 = pd.DataFrame()
df2['Packed MPC designation'] = df['packed_designation']
df2['Semimajor axis au barycentric'] = a_list
df2['Eccentricity barycentric'] = e_list
df2['Inclination ecliptic J2000 barycentric degrees'] = i_list
df2['Longitude of ascending node ecliptic J2000 barycentric degrees'] = W_list
df2['Argument of perihelion ecliptic J2000 barycentric degrees'] = w_list
df2['Mean anomaly ecliptic J2000 barycentric degrees'] = M_list
df2['Epoch JD'] = [JD for i in range(len(a_list))]
df2['Barycentric element source'] = ['JPL Horizons via Astroquery' for i in range(len(a_list))]
df2.to_csv('plutinos_for_mnras.csv',index=False)
JD = str(JD)
name = '8' # neptune barycenter
center = '500@0' # solar system barycenter
obj = Horizons(id=name,location=center,epochs=JD)
el = obj.elements()
i_neptune = np.radians(float(el['incl']))
W_neptune = np.radians(float(el['Omega']))
# equation 1 in the paper
hx_neptune = np.sin(i_neptune)*np.sin(W_neptune)
hy_neptune = -np.sin(i_neptune)*np.cos(W_neptune)
hz_neptune = np.cos(i_neptune)
a_array = np.array(a_list)
e_array = np.array(e_list)
i_array = np.array(i_list)
w_array = np.array(w_list)
W_array = np.array(W_list)
M_array = np.array(M_list)
w_array = np.mod(w_array,360)
W_array = np.mod(W_array,360)
M_array = np.mod(M_array,360)
i_array = np.radians(i_array)
w_array = np.radians(w_array)
W_array = np.radians(W_array)
M_array = np.radians(M_array)
# equation 1 in paper
hx_array = np.sin(i_array)*np.sin(W_array)
hy_array = -np.sin(i_array)*np.cos(W_array)
hz_array = np.cos(i_array)
Nobj = len(hz_array)
# want to tell apart plutinos on the north hemisphere and plutinos on the south
# hemisphere in case it makes fig 1 and fig 2 in the paper a little easier to read
hx_array_north = []
hx_array_south = []
hy_array_north = []
hy_array_south = []
hz_array_north = []
hz_array_south = []
for iobj in range(Nobj):
    if hz_array[iobj] >= 0:
        hx_array_north.append(hx_array[iobj])
        hy_array_north.append(hy_array[iobj])
        hz_array_north.append(hz_array[iobj])
    else:
        hx_array_south.append(hx_array[iobj])
        hy_array_south.append(hy_array[iobj])
        hz_array_south.append(hz_array[iobj])
hx_array_north = np.array(hx_array_north)
hy_array_north = np.array(hy_array_north)
hz_array_north = np.array(hz_array_north)
hx_array_south = np.array(hx_array_south)
hy_array_south = np.array(hy_array_south)
hz_array_south = np.array(hz_array_south)
del a_list,e_list,i_list,w_list,W_list,M_list
# equation 9 in paper
hxbar = np.mean(hx_array)
hybar = np.mean(hy_array)
hzbar = np.mean(hz_array)
# equation 10 in paper
R = np.sqrt(hxbar**2+hybar**2+hzbar**2)
# equation 11 in paper
muhat_hx = hxbar/R
muhat_hy = hybar/R
muhat_hz = hzbar/R
print('[muhat_hx,muhat_hy,muhat_hz]=',[muhat_hx,muhat_hy,muhat_hz])
# equation A1 in paper
i_mu = np.sqrt(muhat_hx**2+muhat_hy**2)
W_mu = np.mod(np.arctan2(muhat_hx/np.sin(i_mu),-muhat_hy/np.sin(i_mu)),2*np.pi)
print('i_mu=',np.degrees(i_mu),'deg')
print('W_mu=',np.degrees(W_mu),'deg')
# equation 12 in paper
kappa = R*(3-R**2)/(1-R**2)
diff = 1
tol = 1e-8
# equation 13 in paper
while abs(diff) > tol:
    top = special.iv(3/2, kappa)
    bottom = special.iv(1/2, kappa)
    # equation 14 in paper
    A3 = top/bottom
    diff =(A3-R)/(1-A3**2-2/kappa*A3)
    kappa = kappa - diff
print('kappa=',kappa)
print('sigma=',kappa**-0.5,'=',np.degrees(kappa**-0.5),'deg')
sigma = 1/np.sqrt(kappa)
del diff,tol,top,bottom,A3
i_invariable = np.radians(1.578694)
W_invariable = np.radians(107.582222)
# equation 1 in paper
hx_invariable = np.sin(i_invariable)*np.sin(W_invariable)
hy_invariable = -np.sin(i_invariable)*np.cos(W_invariable)
hz_invariable = np.cos(i_invariable)
hx_ecliptic = 0
hy_ecliptic = 0
hz_ecliptic = 1
polevec_invariable = np.array([hx_invariable,hy_invariable,hz_invariable])
polevec_ecliptic = np.array([hx_ecliptic,hy_ecliptic,hz_ecliptic])
polevec_vmf = np.array([muhat_hx,muhat_hy,muhat_hz])
# equation A3 in paper
a = np.cross(polevec_vmf,polevec_ecliptic)
a = a / np.linalg.norm(a)
# equation 18 in paper
theta_rot = np.arccos(np.dot(polevec_ecliptic,polevec_vmf))
# check that the vmf pole is rotated onto the ecliptic pole
rodrigues_check = rodrigues_rotation(polevec_vmf, a, theta_rot)
# equation A5 in the paper
khat = rodrigues_rotation(np.array([0,0,1]),-a,theta_rot)
ihat = rodrigues_rotation(np.array([1,0,0]),-a,theta_rot)
jhat = np.cross(khat,ihat)
# rotate the plutino poles around so their meanvec is the ecliptic, then find relative Omega
hx_array_post = []
hy_array_post = []
hz_array_post = []
i_array_post = []
W_array_post = []
for iobj in range(Nobj):
    vold = np.array([hx_array[iobj],hy_array[iobj],hz_array[iobj]])
    vnew = rodrigues_rotation(vold,a,theta_rot)
    hx_array_post.append(vnew[0])
    hy_array_post.append(vnew[1])
    hz_array_post.append(vnew[2])
    # equation A4 in the paper
    i_post = np.arcsin(np.sqrt(vnew[0]**2+vnew[1]**2))
    W_post = np.mod(np.arctan2(vnew[0]/np.sin(i_post),-vnew[1]/np.sin(i_post)),2*np.pi)
    i_array_post.append(i_post)
    W_array_post.append(W_post)
hx_array_post = np.array(hx_array_post)
hy_array_post = np.array(hy_array_post)
hz_array_post = np.array(hz_array_post)
i_array_post = np.array(i_array_post)
W_array_post = np.array(W_array_post)
# equation 16 in the paper
# method 1 (ie total,term,d,delta,theta) should be the same as method 2 (total2,term2,d2,delta2,theta2)
total  = 0
total2 = 0
for iobj in range(Nobj):
    term  = hz_array_post[iobj]
    term2 = muhat_hx*hx_array[iobj] + muhat_hy*hy_array[iobj] + muhat_hz*hz_array[iobj]
    total  = total  + term
    total2 = total2 + term2
d  = 1 - total /Nobj
d2 = 1 - total2/Nobj
# equation 15 in the paper
delta  = np.sqrt(d /(Nobj*R**2))
delta2 = np.sqrt(d2/(Nobj*R**2))
# equation 17 in the paper
alpha = 0.05 # 95% confidence region
theta  = np.arcsin(delta *np.sqrt(-np.log(alpha)))
print('theta=',np.degrees(theta),'deg')
theta2 = np.arcsin(delta2*np.sqrt(-np.log(alpha)))
del alpha
# define points on the 95% confidence circle centered on the origin
Npts_circle = 100
circle_clock_angles = np.linspace(start=0,stop=2*np.pi,num=Npts_circle,endpoint=True)
radius_circle = np.sin(theta)
hx_circle_pre = np.cos(circle_clock_angles)*radius_circle
hy_circle_pre = np.sin(circle_clock_angles)*radius_circle
hz_circle_pre = np.cos(theta)*np.ones(Npts_circle)
# rotate the confidence circle around to center on the VMF mean pole
hx_circle_post = []
hy_circle_post = []
hz_circle_post = []
for ipt in range(Npts_circle):
    vold = np.array([hx_circle_pre[ipt],hy_circle_pre[ipt],hz_circle_pre[ipt]])
    vnew = rodrigues_rotation(vold, -a, theta_rot)
    hx_circle_post.append(vnew[0])
    hy_circle_post.append(vnew[1])
    hz_circle_post.append(vnew[2])
hx_circle_post = np.array(hx_circle_post)
hy_circle_post = np.array(hy_circle_post)
hz_circle_post = np.array(hz_circle_post)
#%% 3d rendering of orbit poles on the unit sphere
u, v = np.mgrid[0:np.radians(360):40j, 0:np.radians(90):40j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
scale = 1
x = x * scale
y = y * scale
z = z * scale
fig = plt.figure(figsize=(2.5,2.5))
plt.rcParams['font.size'] = 8
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(x,y,z,rstride=1,cstride=1,color='whitesmoke',alpha=0.1,edgecolor='gray',linewidth=0.25) # unit sphere
ax.scatter(hx_array_north,hy_array_north,hz_array_north,color='tomato',s=2) # plutino poles in northern hemisphere
ax.scatter(hx_array_south,hy_array_south,hz_array_south,color='cadetblue',s=2) # plutino poles in southern hemisphere
ax.set_box_aspect((1,1,0.5))
ax.set_xticks(ticks=[-0.5,0.5], minor=False)
ax.set_yticks(ticks=[-0.5,0,0.5], minor=False)
ax.set_zticks(ticks=[0,0.5,1], minor=False)
ax.view_init(60, 45)
plt.tight_layout()
ax.set_xlabel('$h_x$ = sin(i)sin(Ω)')
ax.set_ylabel('$h_y$ = -sin(i)cos(Ω)')
ax.set_zlabel('$h_z$ = cos(i)')
titlestr = 'fig1_' + str(Nres)
plt.tight_layout()
plt.savefig(titlestr + '.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()
#%% plot orbit poles in (hx,hy) plane
fig = plt.figure()
ax = fig.add_subplot(121)
th = np.linspace(start=0,stop=2*np.pi,num=100,endpoint=True)
costh = np.cos(th)
sinth=  np.sin(th)
ax.plot(np.cos(np.radians(90))*costh,np.cos(np.radians(90))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(80))*costh,np.cos(np.radians(80))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(70))*costh,np.cos(np.radians(70))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(60))*costh,np.cos(np.radians(60))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(50))*costh,np.cos(np.radians(50))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(40))*costh,np.cos(np.radians(40))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(30))*costh,np.cos(np.radians(30))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(20))*costh,np.cos(np.radians(20))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(10))*costh,np.cos(np.radians(10))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(0))*costh,np.cos(np.radians(0))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(0*30))],[0,np.sin(np.radians(0*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(1*30))],[0,np.sin(np.radians(1*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(2*30))],[0,np.sin(np.radians(2*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(3*30))],[0,np.sin(np.radians(3*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(4*30))],[0,np.sin(np.radians(4*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(5*30))],[0,np.sin(np.radians(5*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(6*30))],[0,np.sin(np.radians(6*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(7*30))],[0,np.sin(np.radians(7*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(8*30))],[0,np.sin(np.radians(8*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(9*30))],[0,np.sin(np.radians(9*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(10*30))],[0,np.sin(np.radians(10*30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(11*30))],[0,np.sin(np.radians(11*30))],color='gray',linestyle='-',linewidth=0.25)
ax.axhline(color='black',linestyle='-',linewidth=1)
ax.axvline(color='black',linestyle='-',linewidth=1)
ax.scatter(hx_array_north,hy_array_north,color='tomato',s=3) # plutino poles in the northern hemisphere
ax.scatter(hx_array_south,hy_array_south,color='cadetblue',s=3) # plutino poles in the southern hemisphere
plt.tight_layout()
plt.axis('equal')
ax.set_xlabel('$h_x$')
ax.set_ylabel('$h_y$')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_box_aspect(1)
# detail near the origin
ax = fig.add_subplot(122)
th = np.linspace(start=0,stop=2*np.pi,num=100,endpoint=True)
costh = np.cos(th)
sinth=  np.sin(th)
ax.plot(np.cos(np.radians(80))*costh,np.cos(np.radians(80))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(70))*costh,np.cos(np.radians(70))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(60))*costh,np.cos(np.radians(60))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(50))*costh,np.cos(np.radians(50))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(40))*costh,np.cos(np.radians(40))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(30))*costh,np.cos(np.radians(30))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(20))*costh,np.cos(np.radians(20))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(10))*costh,np.cos(np.radians(10))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-9))*costh,np.cos(np.radians(90-9))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-8))*costh,np.cos(np.radians(90-8))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-7))*costh,np.cos(np.radians(90-7))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-6))*costh,np.cos(np.radians(90-6))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-5))*costh,np.cos(np.radians(90-5))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-4))*costh,np.cos(np.radians(90-4))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-3))*costh,np.cos(np.radians(90-3))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-2))*costh,np.cos(np.radians(90-2))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot(np.cos(np.radians(90-1))*costh,np.cos(np.radians(90-1))*sinth,color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(10))],[0,np.sin(np.radians(10))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(20))],[0,np.sin(np.radians(20))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(30))],[0,np.sin(np.radians(30))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(40))],[0,np.sin(np.radians(40))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(50))],[0,np.sin(np.radians(50))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(60))],[0,np.sin(np.radians(60))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(70))],[0,np.sin(np.radians(70))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(80))],[0,np.sin(np.radians(80))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(90))],[0,np.sin(np.radians(90))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(100))],[0,np.sin(np.radians(100))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(110))],[0,np.sin(np.radians(110))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(120))],[0,np.sin(np.radians(120))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(130))],[0,np.sin(np.radians(130))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(140))],[0,np.sin(np.radians(140))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(150))],[0,np.sin(np.radians(150))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(160))],[0,np.sin(np.radians(160))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(170))],[0,np.sin(np.radians(170))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(180))],[0,np.sin(np.radians(180))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(190))],[0,np.sin(np.radians(190))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(200))],[0,np.sin(np.radians(200))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(210))],[0,np.sin(np.radians(210))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(220))],[0,np.sin(np.radians(220))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(230))],[0,np.sin(np.radians(230))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(240))],[0,np.sin(np.radians(240))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(250))],[0,np.sin(np.radians(250))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(260))],[0,np.sin(np.radians(260))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(270))],[0,np.sin(np.radians(270))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(280))],[0,np.sin(np.radians(280))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(290))],[0,np.sin(np.radians(290))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(300))],[0,np.sin(np.radians(300))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(310))],[0,np.sin(np.radians(310))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(320))],[0,np.sin(np.radians(320))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(330))],[0,np.sin(np.radians(330))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(340))],[0,np.sin(np.radians(340))],color='gray',linestyle='-',linewidth=0.25)
ax.plot([0,np.cos(np.radians(350))],[0,np.sin(np.radians(350))],color='gray',linestyle='-',linewidth=0.25)
ax.axhline(color='black',linestyle='-',linewidth=1)
ax.axvline(color='black',linestyle='-',linewidth=1)
ax.scatter(hx_array_north,hy_array_north,color='tomato',s=5) # plutino poles on the northern hemisphere
ax.scatter(hx_array_south,hy_array_south,color='cadetblue',s=5) # plutino poles on the southern hemisphere
ax.plot(hx_circle_post,hy_circle_post,color='blue',linestyle='-',linewidth=1) # confidence circle
ax.scatter(muhat_hx,muhat_hy,color='blue',s=50,marker='o') # vmf midplane of plutinos
ax.scatter(hx_invariable,hy_invariable,color='red',s=50,marker='v') # invariable pole of the solar system
ax.scatter(hx_neptune,hy_neptune,color='blue',s=50,marker='>') # orbit pole of neptune
plt.tight_layout()
plt.axis('equal')
ax.set_xlabel('$h_x$')
ax.set_ylabel('$h_y$')
ax.set_xlim([-0.01,0.09])
ax.set_ylim([-0.01,0.09])
ax.set_box_aspect(1)
titlestr = 'fig2_' + str(Nres)
plt.tight_layout()
plt.savefig(titlestr + '.eps',dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()
#%%
'''
make separate plot of plutino inclinations relative to vMF midplane
histogram of inclinations, vmf inclination distribution, on same plot
'''
vmf_relative_inclinations_degrees = []
for iobj in range(Nobj):
    polevec_obj = np.array([hx_array[iobj],hy_array[iobj],hz_array[iobj]])
    dotted = np.dot(polevec_vmf,polevec_obj)
    normalized = dotted/np.linalg.norm(polevec_vmf)/np.linalg.norm(polevec_obj)
    inclination_radians = np.arccos(normalized)
    inclination_degrees = np.degrees(inclination_radians)
    vmf_relative_inclinations_degrees.append(inclination_degrees)
vmf_relative_inclinations_degrees = np.array(vmf_relative_inclinations_degrees)
vmf_relative_inclinations_radians = np.radians(vmf_relative_inclinations_degrees)
Nhere = 1000000
rand_vmf = rand_von_mises_fisher(mu=polevec_vmf,kappa=kappa,N=Nhere)
rand_vmf_relative_inclinations_degrees = []
for iobj in range(Nhere):
    polevec_obj = rand_vmf[iobj]
    dotted = np.dot(polevec_vmf,polevec_obj)
    normalized = dotted/np.linalg.norm(polevec_vmf)/np.linalg.norm(polevec_obj)
    inclination_radians = np.arccos(normalized)
    inclination_degrees = np.degrees(inclination_radians)
    rand_vmf_relative_inclinations_degrees.append(inclination_degrees)
rand_vmf_relative_inclinations_degrees = np.array(rand_vmf_relative_inclinations_degrees)
rand_vmf_relative_inclinations_radians = np.radians(rand_vmf_relative_inclinations_degrees)
# Anderson-Darling test mentioned in first paragraph of section 3.4
AD_result = stats.anderson_ksamp([vmf_relative_inclinations_radians,\
                                  rand_vmf_relative_inclinations_radians])
degrees_vec = np.linspace(0,np.max(rand_vmf_relative_inclinations_degrees),num=1000000,endpoint=True)
radians_vec = np.linspace(0,np.max(rand_vmf_relative_inclinations_radians),num=1000000,endpoint=True)
# equation 22 in paper
curve_vec = kappa/(np.exp(kappa)-np.exp(-kappa))*\
    np.exp(kappa*np.cos(radians_vec))*np.sin(radians_vec)
fig = plt.figure(figsize=(2,2))
ax1  = fig.add_subplot(111)
ax1.set_xlabel('Relative inclination (degrees)')
ax1.set_xticks(np.radians([0,10,20,30,40]))
ax1.set_xticklabels(['0','10','20','30','40'])
plt.rcParams['font.size'] = 6
data = vmf_relative_inclinations_radians
myHist  = ax1.hist(data, bins=20,density=True,histtype='bar',ec='black',alpha=0.25,zorder=3)
x = np.linspace(0,np.max(vmf_relative_inclinations_radians),num=1000)
# plot exact relative inclination pdf of vmf distribution using kappa from sample
g = ax1.plot(radians_vec[0:-1:100],curve_vec[0:-1:100],color='gold',zorder=3,lw=1)
# plot truncated rayleigh pdf distribution using least-squares kappa
initial_guess = kappa
xdata = myHist[1]
xdata = xdata[0:len(xdata)-1]
ydata = myHist[0]
dx = xdata[1]-xdata[0]
print('bin_width = ',np.degrees(dx),'deg')
xdata = xdata
ydata = ydata
popt,pcov = optimization.curve_fit(f=truncated_rayleigh_pdf, xdata=xdata, ydata=ydata, \
       p0=initial_guess)
best_fit_kappa = popt[0]
print('best_fit_kappa=',best_fit_kappa)
print('best_fit_sigma=',np.degrees(best_fit_kappa**-0.5),'deg')
curve_vec_2 = truncated_rayleigh_pdf(x,best_fit_kappa)
h = ax1.plot(x[0:-1:10],curve_vec_2[0:-1:10],color='yellowgreen',zorder=15,lw=1)
maxx = np.max([np.max(vmf_relative_inclinations_radians),np.max(i_array_post)])
maxy = np.max([np.max(curve_vec),np.max(curve_vec_2),np.max(myHist[0])])
maxx = 1.1*maxx
maxy = 1.1*maxy
ax1.set_xlim([0,maxx])
ax1.set_ylim([0,maxy])
titlestr = 'fig3_' + str(Nres)
plt.savefig(titlestr + '.eps',dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()
#%%
'''
check for uniformity of W using Rayleigh z test
http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%2016%20-%20Directional%20Statistics.pdf
https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics
https://arxiv.org/pdf/1310.5457.pdf
'''
sinWbar = (np.sum(np.sin(W_array)))/Nobj
cosWbar = (np.sum(np.cos(W_array)))/Nobj
r_rayleighztest = np.sqrt(sinWbar**2 + cosWbar**2)
R_rayleighztest = Nobj * r_rayleighztest
pval = np.exp(np.sqrt(1+4*Nobj+4*(Nobj**2-R_rayleighztest**2))-(1+2*Nobj))
print('rayleigh z-test pval=',pval)
# Construct figure and axis to plot on
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
# Visualise by area of bins
n,bins,patches=circular_hist(ax,W_array,bins=10)
titlestr = 'fig4_' + str(Nres)
plt.savefig(titlestr + '.eps',dpi=300,bbox_inches='tight',pad_inches=0)
#%%
'''
get a simple bootstrap confidence interval for kappa
'''
Nboot = 1000
bootstrapped_kappas = []
for iboot in range(Nboot):
    hx_array_boot = []
    hy_array_boot = []
    hz_array_boot = []
    for iobj in range(Nobj):
        rand = random.randint(0,Nobj-1)
        hx_array_boot.append(hx_array[rand])
        hy_array_boot.append(hy_array[rand])
        hz_array_boot.append(hz_array[rand])
    hx_array_boot = np.array(hx_array_boot)
    hy_array_boot = np.array(hy_array_boot)
    hz_array_boot = np.array(hz_array_boot)
    # equation 9 in thee paper
    hxbar_boot = np.mean(hx_array_boot)
    hybar_boot = np.mean(hy_array_boot)
    hzbar_boot = np.mean(hz_array_boot)
    # equation 10 in the paper
    R_boot = np.sqrt(hxbar_boot**2+hybar_boot**2+hzbar_boot**2)
    # equation 11 in the paper
    muhat_hx_boot = hxbar_boot/R_boot
    muhat_hy_boot = hybar_boot/R_boot
    muhat_hz_boot = hzbar_boot/R_boot
    # equation 12 in the paper
    kappa_boot = R_boot*(3-R_boot**2)/(1-R_boot**2)
    diff_boot = 1
    tol_boot = 1e-8
    while abs(diff_boot) > tol_boot:
        top_boot = special.iv(3/2, kappa_boot)
        bottom_boot = special.iv(1/2, kappa_boot)
        # equation 14 in the papeer
        A3_boot = top_boot/bottom_boot
        # equation 13 in th paper
        diff_boot =(A3_boot-R_boot)/(1-A3_boot**2-2/kappa_boot*A3_boot)
        kappa_boot = kappa_boot - diff_boot
    bootstrapped_kappas.append(kappa_boot)
bootstrapped_kappas = np.array(bootstrapped_kappas)
bootstrapped_sigmas = np.degrees(1/np.sqrt(bootstrapped_kappas))
print('16th percentile bootstrapped kappa=',np.percentile(bootstrapped_kappas,16))
print('84th percentile bootstrapped kappa=',np.percentile(bootstrapped_kappas,84))
print('16th percentile bootstrapped sigma=',np.percentile(bootstrapped_sigmas,16))
print('84th percentile bootstrapped sigma=',np.percentile(bootstrapped_sigmas,84))
#%%
'''
Rayleigh parameter estimation from Wikipedia, which references
https://nvlpubs.nist.gov/nistpubs/jres/66D/jresv66Dn2p167_A1b.pdf
JOURNAL OF RESEARCH of the National Bureau of Standards-D. Radio Propagation
Vol. 66D, No.2, March- April 1962
Some Problems Connected With Rayleigh Distributions
M. M. Siddiqui
Contribution from Boulder Laboratories, National Bureau of Standards, Boulder, Colo.
(October 19, 1961)
'''
# paragraph between equations 3.10 and 3.11 in the reference
zvec = np.radians(vmf_relative_inclinations_degrees)**2 # with variance gamma/2, so std = sqrt(gamma)/sqrt(2)
# equation 4.7 in the reference
ccc = np.sum(zvec)/Nobj # maximum likelihood estimate of gamma
# equation 4.10 in the reference
gamma_bound_1 = 2*Nobj*ccc/stats.chi2.isf(0.84,df=2*Nobj)
gamma_bound_2 = 2*Nobj*ccc/stats.chi2.isf(0.16,df=2*Nobj)
std = np.sqrt(ccc)/np.sqrt(2)
std_bound_1 = np.sqrt(gamma_bound_1)/np.sqrt(2)
std_bound_2 = np.sqrt(gamma_bound_2)/np.sqrt(2)
print('siddiqui statistics',np.degrees([std_bound_2,std,std_bound_1]))
#%% compare vmf inclination and truncated rayleigh inclination for appendix B
kappa_vec = np.array([30,15,4,1])
colors = ['green','brown','magenta','blue']
degree_vec = np.linspace(start=0,stop=180,num=1000,endpoint=False)
fig = plt.figure(figsize=(3,2))
ax1  = fig.add_subplot(111)
ax1.set_xlabel('Inclination (degrees)')
ax1.set_ylabel('pdf')
plt.rcParams['font.size'] = 6
axins = inset_axes(ax1,width="50%",height="50%",borderpad=1)
axins.set_xlabel('Inclination (degrees)')
axins.set_ylabel('Rayleigh - VMF')
mindiff = []
maxdiff = []
maxy = []
for ik in range(4):
    kappa_here = kappa_vec[ik]
    color = colors[ik]
    sigma_here = 1/np.sqrt(kappa_here)
    # equation 22 in paper
    curve_vec = kappa_here/(np.exp(kappa_here)-np.exp(-kappa_here))*\
        np.exp(kappa_here*np.cos(np.radians(degree_vec)))*np.sin(np.radians(degree_vec))
    curve_vec = np.radians(curve_vec)
    # equation 28 in paper
    C_R = 1/( 1 - np.exp(-np.pi**2/2/sigma_here**2) )
    # equation 27 in paper
    curve_vec_2 = C_R/sigma_here**2 * np.radians(degree_vec) * \
        np.exp(-np.radians(degree_vec)**2/2/sigma_here**2)
    curve_vec_2 = np.radians(curve_vec_2)
    h = ax1.plot(degree_vec[0:-1:10],curve_vec[0:-1:10],color=color,zorder=1,lw=0.5,linestyle='solid')
    h2 = ax1.plot(degree_vec[0:-1:10],curve_vec_2[0:-1:10],color=color,zorder=2,lw=0.5,linestyle='dotted')
    h3 = axins.plot(degree_vec[0:-1:10],curve_vec_2[0:-1:10]-curve_vec[0:-1:10],color=color,\
                    lw=0.5,linestyle='solid')
    mindiff.append(np.min(curve_vec_2-curve_vec))
    maxdiff.append(np.max(curve_vec_2-curve_vec))
    maxy.append(1.1*np.max([np.max(curve_vec),np.max(curve_vec_2)]))
ax1.set_xlim([0,180])
ax1.set_ylim([0,np.max(maxy)])
axins.set_xlim([0,180])
titlestr = 'figB1'
plt.savefig(titlestr + '.eps',dpi=300,bbox_inches='tight',pad_inches=0)
plt.show()
#%%
thisdir = os.getcwd()
files_in_dir = os.listdir(thisdir)
for this_file in files_in_dir:
    if this_file.endswith('.out'):
        os.remove(os.path.join(thisdir,this_file))
#%%
khat_old = np.array([0,0,1])
khat_new = rodrigues_rotation(khat_old, -a, theta_rot)
ihat_old = np.array([1,0,0])
ihat_new = rodrigues_rotation(ihat_old, -a, theta_rot)
print('khat__rel=',khat_new)
print('ihat__rel=',ihat_new)
