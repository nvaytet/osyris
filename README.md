![demo.png](https://bitbucket.org/repo/jq5boX/images/2936418214-demo.png)

# Osiris #

A python visualization utility for RAMSES data. It's purpose is to plot quick diagnostics while a simulation is running, and also produce publication grade figures.

### Installation ###

You will need matplotlib installed on your system. Clone the Osiris repository and append its location to your PYTHONPATH.

### From within ipython ###

```
#!python
import osiris
mydata = osiris.RamsesData(71,scale="au")
osiris.plot_slice(mydata.log_rho,direction="z",vec=mydata.velocity,dx=100)
```

### Demo ###

You can download the sample data [here](http://www.nbi.dk/~nvaytet/osiris/ramses_sample_data.tar.gz).

```
#!python

import matplotlib.pyplot as plt
import osiris

# Change default time unit to kyr
osiris.conf.default_values["time_unit"]="kyr"

# Load data
mydata = osiris.RamsesData(nout=71,center="max:density",scale="au")

# Create figure
fig = plt.figure()
ratio = 0.5
sizex = 20.0
fig.set_size_inches(sizex,ratio*sizex)
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

# Density vs B field with AMR level contours
osiris.plot_histogram(mydata.log_rho,mydata.log_B,axes=ax1,cmap="log,YlGnBu")
osiris.plot_histogram(mydata.log_rho,mydata.log_B,var_z=mydata.level,contour=True,axes=ax1,contour_args={"fmt":"%i","label":True,"colors":"k","cmap":None,"levels":range(5,20)},cbar=False,zmin=6,zmax=16)

# Create new field with log of velocity
mydata.new_field(name="log_vel",operation="np.log10(np.sqrt(velocity_x**2+velocity_y**2+velocity_z**2))",unit="cm/s",label="log(Velocity)")

# Density vs log_vel
osiris.plot_histogram(mydata.log_rho,mydata.log_vel,mydata.log_T,axes=ax2,cmap="gnuplot",scatter=True,outline=True,scatter_args={"iskip":100})

#x,z density slice with B field streamlines
osiris.plot_slice(mydata.density,direction="y",stream=mydata.B,dx=100,axes=ax3,scalar_args={"cmap":"log"})
# x,y density slice with velocity vectors in color
osiris.plot_slice(scalar=mydata.log_rho,direction="z",vec=mydata.velocity,dx=100,axes=ax4,vec_args={"cmap":"seismic","vskip":4})
# x,y temperature slice with velocity vectors
osiris.plot_slice(mydata.log_T,direction="z",vec=mydata.velocity,dx=100,axes=ax5,scalar_args={"cmap":"hot"},contour=mydata.level,contour_args={"fmt":"%i","label":True,"colors":"w","cmap":None,"levels":range(9,17)})

fig.savefig("demo.pdf",bbox_inches="tight")
```

### Have a problem or need a new feature? ###

Use the [Issue tracker](https://bitbucket.org/nvaytet/osiris/issues) on the Bitbucket website.

### Contributors ###

* Neil Vaytet (StarPlan/NBI)
* Tommaso Grassi (StarPlan/NBI)
* Matthias Gonzalez (CEA Saclay)
* Troels Haugbolle (StarPlan/NBI)

### License

Osiris is distributed under the GPLv3 license.

### Funding

Neil Vaytet gratefully acknowledges support from the European Commission through the Horizon 2020 Marie Sklodowska-Curie Actions Individual Fellowship 2014 programme (Grant Agreement no. 659706).