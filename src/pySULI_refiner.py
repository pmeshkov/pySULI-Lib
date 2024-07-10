import os
import sys
import shutil
import scipy
import time
import glob
import ipympl
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr

import functions_lib_version

from IPython.display import clear_output

def set_GSAS_II_user(user):
    # we need to append GSASII directory to the system path
    match user:
        case "mtopsakal":
            sys.path += ['/home/mt/software/miniforge3/envs/GSASII/GSAS-II/GSASII']
        case "pmeshkov":
            sys.path += ['/opt/anaconda3/envs/GSASII/GSAS-II/GSASII']
        case "kmorell":
            sys.path += ['/Users/kevinmorell/Downloads/anaconda3/envs/GSASII/GSAS-II/GSASII']
        case "user":
            sys.path += ['your path here']
            
    # we then import GSASIIscriptable
    import GSASIIscriptable as G2sc
    import pybaselines # this comes with gsas2_package
    
    return G2sc, pybaselines

def set_GSAS_II_path(GSASII_path):
    sys.path += [GSASII_path]
    
    # we then import GSASIIscriptable
    import GSASIIscriptable as G2sc
    import pybaselines # this comes with gsas2_package
    
    return G2sc, pybaselines
    
class Refiner:
    def __init__(self, nc_path, phases, gsas2_scratch, q_range, GSASII_path = None, da_input_bkg=None):
        
        if GSASII_path is not None:
            G2sc, pybaselines = set_GSAS_II_path(GSASII_path)
        else:
            print("GSAS-II path not found.\n Install GSAS-II, and insert the GSAS-II_path into the refiner class.")
            return 0
            
        # set instance variables
        self.nc_path = nc_path
        self.phases = phases
        self.gsas2_scratch = gsas2_scratch
        self.q_range = q_range
        self.da_input_bkg = da_input_bkg
        self.da_i2d = None
        self.da_i2d_m = None
        self.bkg_auto = None        
        
        if not os.path.isfile('%s/gsas.instprm'%self.gsas2_scratch):
            print('ERROR:!! gsas.instprm is not found. Please copy one into %s folder.\n\n'%self.gsas2_scratch)
            return

        with xr.open_dataset(self.nc_path) as ds:

            for k in ['Y_obs','Y_calc','Y_bkg_auto','Y_bkg_gsas','Y_bkg','Y_bkg_arpls','Y_bkg_auto']:
                if k in ds.keys():
                    del ds[k]

            self.da_i2d = ds.i2d.sel(radial=slice(self.q_range[0],self.q_range[1])).astype('float32')
            self.da_i2d_m = self.da_i2d.mean(dim='azimuthal')

            # set background function 
            if self.da_input_bkg is None:
                arpls_ = pybaselines.Baseline(x_data=self.da_i2d_m.radial.values).arpls((self.da_i2d_m).values, lam=1e5)[0]
                shift_ = min((self.da_i2d_m).values - arpls_)
                bkg_arpls = (arpls_+shift_)
                self.bkg_auto = bkg_arpls

            else:
                print(self.da_input_bkg)
                self.da_input_bkg = self.da_input_bkg.sel(radial=slice(self.q_range[0],self.q_range[1]))
                print(self.da_input_bkg)
                blank_scale = (self.da_i2d_m[0] / self.da_input_bkg[0]).values
                while (min((self.da_i2d_m.values-blank_scale*self.da_input_bkg.values)) < 0):
                    blank_scale = blank_scale*0.95

                da_input_bkg_scaled = blank_scale*self.da_input_bkg

                arpls_ = pybaselines.Baseline(x_data=self.da_i2d_m.radial.values).arpls((self.da_i2d_m-da_input_bkg_scaled).values, lam=1e5)[0]
                shift_ = min((self.da_i2d_m-da_input_bkg_scaled).values - arpls_)
                bkg_arpls = (arpls_+shift_)
                self.bkg_auto = bkg_arpls + da_input_bkg_scaled.values
            Y_to_gsas = self.da_i2d_m.values-self.bkg_auto
            
            self.y_scale = 1000/max(Y_to_gsas)
            self.y_baseline = 10
            
            # set gsas intensity profile and save as xy file
            Y_to_gsas = self.y_baseline+Y_to_gsas*self.y_scale
            X_to_gsas = np.rad2deg(functions_lib_version.q_to_twotheta(self.da_i2d_m.radial.values, wavelength=(ds.attrs['wavelength']*1.0e10)))
            np.savetxt('%s/data.xy'%gsas2_scratch, np.column_stack( (X_to_gsas,Y_to_gsas) ), fmt='%.4f %.4f')

            # save ds as an instance variable 
            self.ds = ds
            
            # save gpx as an instance variable
            self.gpx = G2sc.G2Project(newgpx='%s/gsas.gpx'%gsas2_scratch)
            self.gpx.data['Controls']['data']['max cyc'] = 100
            try:
                self.gpx.add_powder_histogram('%s/data.xy'%gsas2_scratch,'%s/gsas.instprm'%gsas2_scratch)
            except:
                self.gpx.add_powder_histogram('%s/data.xy'%gsas2_scratch,'%s/gsas.instprm'%gsas2_scratch)

            hist   = self.gpx.histograms()[0]
            for p in phases:
                self.gpx.add_phase(p['cif_abs_path'],phasename=p['phase_name'], histograms=[hist],fmthint='CIF')
            self.gpx.save()

    def refine(self, refinement_recipe):
        # apply refinement recipt to the instance variable gpx
        # possibly make this simpler later
        self.gpx = refinement_recipe(self)

        # get x, yobs, ycalc, and background from gpx histograms
        histogram = self.gpx.histograms()[0]

        gsas_X  = histogram.getdata('x').astype('float32')
        gsas_Yo = histogram.getdata('yobs').astype('float32')
        gsas_Yc = histogram.getdata('ycalc').astype('float32')
        gsas_B  = histogram.getdata('Background').astype('float32')

        # now convert Y back
        Y_Obs_from_gsas  = ((gsas_Yo -self.y_baseline  )/self.y_scale )
        Y_Bkg_from_gsas  = ((gsas_B  -self.y_baseline  )/self.y_scale )
        Y_calc_from_gsas = ((gsas_Yc -self.y_baseline  )/self.y_scale ) # this also includes background from GSAS

        # add data to the instance variable ds
        self.ds = self.ds.assign_coords(
            {"X_in_q": self.da_i2d_m.radial.values.astype('float32')},
            )
        self.ds = self.ds.assign_coords(
            {"X_in_tth": gsas_X.astype('float32')},
            )
        self.ds = self.ds.assign_coords(
            {"X_in_d": functions_lib_version.q_to_d(self.da_i2d_m.radial.values.astype('float32'))},
            )

        self.ds['Y_obs'] = xr.DataArray(
                                    data=self.da_i2d_m.values,
                                    dims=['X_in_q'],
                                )
        self.ds['Y_calc'] = xr.DataArray(
                                    data=Y_calc_from_gsas-Y_Bkg_from_gsas,
                                    dims=['X_in_q'],
                                )# by Ycalc, we mean only calculated peaks, no background

        self.ds['Y_bkg_gsas'] = xr.DataArray(
                                    data=Y_Bkg_from_gsas,
                                    dims=['X_in_q'],
                                )
        self.ds['Y_bkg_auto'] = xr.DataArray(
                                    data=self.bkg_auto,
                                    dims=['X_in_q'],
                                )

        with open('%s/gsas.lst'%(self.gsas2_scratch)) as lst_file:
            gsas_lst = lst_file.read()
            self.ds.attrs['gsas_lst'] = gsas_lst
            
        return [self.gpx,self.ds]
        
    def update_ds_file(self):
        """
        Updates the ds file in nc_path
        """
        self.ds.to_netcdf(self.nc_path+'.new.nc',engine="scipy")
        time.sleep(0.1)
        shutil.move(self.nc_path+'.new.nc',self.nc_path)
    
    def update_gpx(self):
        """
        Updates the gpx file in nc_path
        """
        shutil.copyfile(self.gpx['Controls']['data']['LastSavedAs'],'%s.gpx'%self.nc_path[:-3])
        
    def get_gpx(self):
        return self.gpx
    
    def get_ds(self):
        return self.ds
    
    def plot_refinement_results(self, plt_range=None):
        
        # if x-axis range isn't given, just use original q_range
        if plt_range is None:
            plt_range = self.q_range
        # otherwise, ensure the plt_range is valid, and if it isn't, cut to original q_range
        elif plt_range[0] < self.q_range[0]:
            plt_range[0] = self.q_range[0]
        elif plt_range[1] > self.plt_range[1]:
            plt_range[1] = self.q_range[1]
        
        # create a mosaic and plot based on its pattern
        fig = plt.figure(figsize=(8,8),dpi=96)
        mosaic = """
        RRR
        YYY
        YYY
        PPP
        DDD
        """
        ax_dict = fig.subplot_mosaic(mosaic)

        # plot the cake pattern in subplot 1 from the top
        ax = ax_dict["R"]
        np.log(self.da_i2d).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=0)
        ax.set_ylabel(self.ds.i2d.ylabel)
        ax.set_xlabel(None)
        ax.set_xticklabels([])
        ax.set_xlim([plt_range[0],plt_range[1]])
        Rwp, GoF = self.gpx.data['Covariance']['data']['Rvals']['Rwp'], self.gpx.data['Covariance']['data']['Rvals']['GOF']
        ax.set_title('%s\n(R$_{wp}$=%.3f GoF=%.3f)'%(self.nc_path.split('/')[-1],Rwp,GoF))
        ax.set_facecolor('#FFFED4')

        # plot observed and calculated I vs 2Theta in subplot 2 from the top
        ax = ax_dict["Y"]
        X, Yobs, Ycalc, Ybkg= self.ds['X_in_q'].values, self.ds['Y_obs'].values, self.ds['Y_calc'].values, (self.ds['Y_bkg_auto']+self.ds['Y_bkg_gsas']).values
        ax.plot(X, np.log( Yobs ),label='Y$_{obs}$',lw=2,color='k')
        ax.plot(X, np.log( Ycalc+Ybkg ),label='Y$_{calc}$+Y$_{bkg}$',lw=1,color='y')
        ax.fill_between(X, np.log( Ybkg ),label='Y$_{bkg}$',alpha=0.2)
        ax.set_ylim(bottom=np.log(min(min(Ybkg),min(Ybkg)))-0.1)
        ax.set_xticklabels([])
        ax.set_ylabel('Log$_{10}$(counts) (a.u.)')
        ax.set_xlim([plt_range[0],plt_range[1]])

        # find lattice constants for each phase and save them to use as label strings in subplot 2
        phases = functions_lib_version.get_valid_phases(self.gpx)
        phase_ct = len(phases)
        label_strs = [None] * phase_ct
        label_colors = [None] * phase_ct
        for ep, phase in enumerate(phases):
            label_strs[ep] = phase
            consts = iter(['a', 'b', 'c'])
            for value in np.unique(list(functions_lib_version.get_cell_consts(self.gpx, phase).values())):
                label_strs[ep] = label_strs[ep] + "\n(" + next(consts) + " = " + str(round(value,6)) + ")"
            label_colors[ep] = "C%d" % ep
            
        # create Line2D objects to use as labels for subplot 2 legend
        from matplotlib.lines import Line2D
        custom_handles = []
        for i in np.arange(0,phase_ct,1):
            custom_handles.append(Line2D([0], [0], marker='o',color=label_colors[i], label='Scatter',
                          markerfacecolor=label_colors[i], markersize=5))

        # get set legend handles from the plot and set legend with all handles and labels
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles + custom_handles
        print(label_strs)
        all_labels = labels + label_strs
        ax.legend(all_handles, all_labels)
        
        # use gpx_plotter() to plot intensities as points, stems, in subplot 3, subplot 2, respectively
        ax = ax_dict["Y"]
        ax_bottom = ax_dict["P"]
        functions_lib_version.gpx_plotter(
            self.gpx,
            line_axes=[ax,ax_bottom],
            stem_axes=[ax_bottom],
            radial_range=plt_range,
            phases=functions_lib_version.get_valid_phases(self.gpx),
            marker="o",
            stem=True,
            unit="d",
            plot_unit="q_A^-1",
            y_shift=0.1
        )
        
        # plot difference in observed vs calculated diffraction pattern in subplot 4
        ax = ax_dict["D"]
        ax.plot(X, Yobs-Ycalc-Ybkg,label='diff.',lw=1,color='r')
        ax.set_xlabel('Scattering vector $q$ ($\AA^{-1}$)')
        ax.set_ylabel('counts (a.u.)')
        ax.set_xlim([plt_range[0],plt_range[1]])
        ax.legend(loc='best')
    
    def print_wR(self, header=''):
        clear_output()
        hist = self.gpx.histograms()[0]
        print('\n'+header)
        print("\t{:20s}: {:.2f}".format(hist.name,hist.get_wR()))
        print("")
        
    def set_LeBail(self, LeBail=False, verbose=True):
        self.gpx.set_refinement({"set":{'LeBail': LeBail}})
        self.gpx.save()
        if(verbose):
            print('\nLeBail is set to %s\n '%(LeBail))

    def refine_background(self,num_coeffs,set_to_false=True):
        try:
            rwp_old = self.gpx['Covariance']['data']['Rvals']['Rwp']
        except:
            rwp_old = 'na'

        ParDict = {'set': {'Background': {'refine': True,
                                        'type': 'chebyschev-1',
                                        'no. coeffs': num_coeffs
                                        }}}
        self.gpx.set_refinement(ParDict)

        self.gpx.refine(); rwp_new = self.gpx['Covariance']['data']['Rvals']['Rwp']

        if set_to_false:
            self.gpx.set_refinement({'set': {'Background': {'refine': False}}})
        self.gpx.save()
        
        try:
            print('\n\n\nBackground is refined: Rwp=%.3f (was %.3f)\n\n\n '%(rwp_new,rwp_old))
        except:
            print('\n\n\nBackground is refined: Rwp=%.3f \n\n\n '%(rwp_new))


    def refine_cell_params(self,phase_ind=None,set_to_false=True):
        rwp_old = self.gpx['Covariance']['data']['Rvals']['Rwp']
        phases = self.gpx.phases()
        for e,p in enumerate(phases):
            if phase_ind is None:
                self.gpx['Phases'][p.name]['General']['Cell'][0]= True
            else:
                if e == phase_ind:
                    self.gpx['Phases'][p.name]['General']['Cell'][0]= True
        self.gpx.refine()
        if set_to_false:
            phases = self.gpx.phases()
            for p in phases:
                self.gpx['Phases'][p.name]['General']['Cell'][0]= False
        rwp_new = self.gpx['Covariance']['data']['Rvals']['Rwp']
        self.gpx.save()
        
        if phase_ind is None:
            print('\n\n\nCell parameters of all phases are refined simultaneously: Rwp=%.3f (was %.3f)\n\n\n '%(rwp_new,rwp_old))
        else:
            print('\n\n\nCell parameters of phase #d is refined: Rwp=%.3f (was %.3f)\n\n\n '%(phase_ind,rwp_new,rwp_old))


    def refine_strain_broadening(self,set_to_false=True):
        rwp_old = self.gpx['Covariance']['data']['Rvals']['Rwp']
        ParDict = {'set': {'Mustrain': {'type': 'isotropic',
                                        'refine': True
                                        }}}
        self.gpx.set_refinement(ParDict)
        self.gpx.refine()

        if set_to_false:
            ParDict = {'set': {'Mustrain': {'type': 'isotropic',
                                    'refine': False
                                    }}}
        self.gpx.set_refinement(ParDict)
        self.gpx.refine()
        rwp_new = self.gpx['Covariance']['data']['Rvals']['Rwp']
        self.gpx.save()

        print('\n\n\nStrain broadeing is refined: Rwp=%.3f (was %.3f)\n\n\n '%(rwp_new,rwp_old))


    def refine_size_broadening(self,set_to_false=True):
        rwp_old = self.gpx['Covariance']['data']['Rvals']['Rwp']
        ParDict = {'set': {'Size': {'type': 'isotropic',
                                        'refine': True
                                        }}}
        self.gpx.set_refinement(ParDict)
        self.gpx.refine()
        if set_to_false:
            ParDict = {'set': {'Size': {'type': 'isotropic',
                                    'refine': False
                                    }}}
        self.gpx.set_refinement(ParDict)
        self.gpx.refine()
        rwp_new = self.gpx['Covariance']['data']['Rvals']['Rwp']
        self.gpx.save()

        print('\nSize broadening is refined: Rwp=%.3f (was %.3f)\n '%(rwp_new,rwp_old))


    def refine_inst_parameters(self,inst_pars_to_refine=['X', 'Y', 'Z', 'Zero', 'SH/L', 'U', 'V', 'W'],set_to_false=True):
        rwp_old = self.gpx['Covariance']['data']['Rvals']['Rwp']
        self.gpx.set_refinement({"set": {'Instrument Parameters': inst_pars_to_refine}})
        self.gpx.refine()
        if set_to_false:
            ParDict = {"clear": {'Instrument Parameters': ['X', 'Y', 'Z', 'Zero', 'SH/L', 'U', 'V', 'W']}}
            self.gpx.set_refinement(ParDict)
            self.gpx.refine()
        rwp_new = self.gpx['Covariance']['data']['Rvals']['Rwp']
        self.gpx.save()

        print('\nInstrument parameters %s are refined: Rwp=%.3f (was %.3f)\n '%(inst_pars_to_refine,rwp_new,rwp_old))

    def instprm_updater(self, gsas2_scratch='gsas2_scratch'):
        instprm_dict = self.gpx['PWDR data.xy']['Instrument Parameters'][0]
        with open('%s/gsas.instprm'%gsas2_scratch, 'w') as f:
            f.write('#GSAS-II instrument parameter file; do not add/delete items!\n')
            f.write('Type:PXC\n')
            f.write('Bank:1.0\n')
            f.write('Lam:%s\n'%(instprm_dict['Lam'][1]))
            f.write('Polariz.:%s\n'%(instprm_dict['Polariz.'][1]))
            f.write('Azimuth:%s\n'%(instprm_dict['Azimuth'][1]))
            f.write('Zero:%s\n'%(instprm_dict['Zero'][1]))
            f.write('U:%s\n'%(instprm_dict['U'][1]))
            f.write('V:%s\n'%(instprm_dict['V'][1]))
            f.write('W:%s\n'%(instprm_dict['W'][1]))
            f.write('X:%s\n'%(instprm_dict['X'][1]))
            f.write('Y:%s\n'%(instprm_dict['Y'][1]))
            f.write('Z:%s\n'%(instprm_dict['Z'][1]))
            f.write('SH/L:%s\n'%(instprm_dict['SH/L'][1]))