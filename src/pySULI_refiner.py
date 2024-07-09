import os
import sys
import shutil
import scipy
import time
import glob
import ipympl

import numpy as np
import xarray as xr

import pySULI_general


from IPython.display import clear_output

user = "pmeshkov"

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

# importing matplotlib for plots.
import matplotlib.pyplot as plt

def PrintwR(gpx,header=''):
    clear_output()
    hist   = gpx.histograms()[0]
    print('\n'+header)
    print("\t{:20s}: {:.2f}".format(hist.name,hist.get_wR()))
    print("")


# def set_limits(gpx,limits=None):
#     gpx.set_refinement({"set":{'Limits': limits}})
#     gpx.save()
#     print('\nlimits are set\n ')

def set_LeBail(gpx,LeBail=False):
    gpx.set_refinement({"set":{'LeBail': LeBail}})
    gpx.save()
    
    print('\nLeBail is set to %s\n '%(LeBail))


def refine_background(gpx,num_coeffs,set_to_false=True):
    try:
        rwp_old = gpx['Covariance']['data']['Rvals']['Rwp']
    except:
        rwp_old = 'na'

    ParDict = {'set': {'Background': {'refine': True,
                                      'type': 'chebyschev-1',
                                      'no. coeffs': num_coeffs
                                     }}}
    gpx.set_refinement(ParDict)

    gpx.refine(); rwp_new = gpx['Covariance']['data']['Rvals']['Rwp']

    if set_to_false:
        gpx.set_refinement({'set': {'Background': {'refine': False}}})
    gpx.save()
    
    try:
        print('\n\n\nBackground is refined: Rwp=%.3f (was %.3f)\n\n\n '%(rwp_new,rwp_old))
    except:
        print('\n\n\nBackground is refined: Rwp=%.3f \n\n\n '%(rwp_new))


def refine_cell_params(gpx,phase_ind=None,set_to_false=True):
    rwp_old = gpx['Covariance']['data']['Rvals']['Rwp']
    phases = gpx.phases()
    for e,p in enumerate(phases):
        if phase_ind is None:
            gpx['Phases'][p.name]['General']['Cell'][0]= True
        else:
            if e == phase_ind:
                gpx['Phases'][p.name]['General']['Cell'][0]= True
    gpx.refine()
    if set_to_false:
        phases = gpx.phases()
        for p in phases:
            gpx['Phases'][p.name]['General']['Cell'][0]= False
    rwp_new = gpx['Covariance']['data']['Rvals']['Rwp']
    gpx.save()
    
    if phase_ind is None:
        print('\n\n\nCell parameters of all phases are refined simultaneously: Rwp=%.3f (was %.3f)\n\n\n '%(rwp_new,rwp_old))
    else:
        print('\n\n\nCell parameters of phase #d is refined: Rwp=%.3f (was %.3f)\n\n\n '%(phase_ind,rwp_new,rwp_old))


def refine_strain_broadening(gpx,set_to_false=True):
    rwp_old = gpx['Covariance']['data']['Rvals']['Rwp']
    ParDict = {'set': {'Mustrain': {'type': 'isotropic',
                                      'refine': True
                                     }}}
    gpx.set_refinement(ParDict)
    gpx.refine()

    if set_to_false:
        ParDict = {'set': {'Mustrain': {'type': 'isotropic',
                                  'refine': False
                                 }}}
    gpx.set_refinement(ParDict)
    gpx.refine()
    rwp_new = gpx['Covariance']['data']['Rvals']['Rwp']
    gpx.save()

    print('\n\n\nStrain broadeing is refined: Rwp=%.3f (was %.3f)\n\n\n '%(rwp_new,rwp_old))


def refine_size_broadening(gpx,set_to_false=True):
    rwp_old = gpx['Covariance']['data']['Rvals']['Rwp']
    ParDict = {'set': {'Size': {'type': 'isotropic',
                                      'refine': True
                                     }}}
    gpx.set_refinement(ParDict)
    gpx.refine()
    if set_to_false:
        ParDict = {'set': {'Size': {'type': 'isotropic',
                                  'refine': False
                                 }}}
    gpx.set_refinement(ParDict)
    gpx.refine()
    rwp_new = gpx['Covariance']['data']['Rvals']['Rwp']
    gpx.save()

    print('\nSize broadening is refined: Rwp=%.3f (was %.3f)\n '%(rwp_new,rwp_old))


def refine_inst_parameters(gpx,inst_pars_to_refine=['X', 'Y', 'Z', 'Zero', 'SH/L', 'U', 'V', 'W'],set_to_false=True):
    rwp_old = gpx['Covariance']['data']['Rvals']['Rwp']
    gpx.set_refinement({"set": {'Instrument Parameters': inst_pars_to_refine}})
    gpx.refine()
    if set_to_false:
        ParDict = {"clear": {'Instrument Parameters': ['X', 'Y', 'Z', 'Zero', 'SH/L', 'U', 'V', 'W']}}
        gpx.set_refinement(ParDict)
        gpx.refine()
    rwp_new = gpx['Covariance']['data']['Rvals']['Rwp']
    gpx.save()

    print('\nInstrument parameters %s are refined: Rwp=%.3f (was %.3f)\n '%(inst_pars_to_refine,rwp_new,rwp_old))




def instprm_updater(gpx, gsas2_scratch='gsas2_scratch'):
    instprm_dict = gpx['PWDR data.xy']['Instrument Parameters'][0]
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


def refiner(nc_path,
            phases,  # should be a dict like [{'cif_abs_path':'/content/drive/MyDrive/XRD-on-colab/_cifs/LaB6.cif','phase_name':'LaB6'}]
            da_input_bkg=None,
            q_range=[0.1, 10.1],
            gsas2_scratch  = 'gsas2_scratch',
            refinement_recipe = None,
            update_ds = True,
            update_gpx=True,
            plot=True
            ):


    if not os.path.isfile('%s/gsas.instprm'%gsas2_scratch):
        print('ERROR:!! gsas.instprm is not found. Please copy one into %s folder.\n\n'%gsas2_scratch)
        return


    with xr.open_dataset(nc_path) as ds:


        for k in ['Y_obs','Y_calc','Y_bkg_auto','Y_bkg_gsas','Y_bkg','Y_bkg_arpls','Y_bkg_auto']:
            if k in ds.keys():
                del ds[k]

        da_i2d = ds.i2d.sel(radial=slice(q_range[0],q_range[1])).astype('float32')
        da_i2d_m = da_i2d.mean(dim='azimuthal')


        if da_input_bkg is None:
            arpls_ = pybaselines.Baseline(x_data=da_i2d_m.radial.values).arpls((da_i2d_m).values, lam=1e5)[0]
            shift_ = min((da_i2d_m).values - arpls_)
            bkg_arpls = (arpls_+shift_)
            bkg_auto = bkg_arpls

        else:
            da_input_bkg = da_input_bkg.sel(radial=slice(q_range[0],q_range[1]))
            blank_scale = (da_i2d_m[0] / da_input_bkg[0]).values
            while (min((da_i2d_m.values-blank_scale*da_input_bkg.values)) < 0):
                blank_scale = blank_scale*0.95

            da_input_bkg_scaled = blank_scale*da_input_bkg

            arpls_ = pybaselines.Baseline(x_data=da_i2d_m.radial.values).arpls((da_i2d_m-da_input_bkg_scaled).values, lam=1e5)[0]
            shift_ = min((da_i2d_m-da_input_bkg_scaled).values - arpls_)
            bkg_arpls = (arpls_+shift_)
            bkg_auto = bkg_arpls + da_input_bkg_scaled.values


        Y_to_gsas = da_i2d_m.values-bkg_auto
        y_scale = 1000/max(Y_to_gsas)
        y_baseline = 10
        Y_to_gsas = y_baseline+Y_to_gsas*y_scale

        X_to_gsas = np.rad2deg(pySULI_general.q_to_twotheta(da_i2d_m.radial.values, wavelength=(ds.attrs['wavelength']*1.0e10)))
        np.savetxt('%s/data.xy'%gsas2_scratch, np.column_stack( (X_to_gsas,Y_to_gsas) ), fmt='%.4f %.4f')


        # gpx part
        gpx = G2sc.G2Project(newgpx='%s/gsas.gpx'%gsas2_scratch)
        gpx.data['Controls']['data']['max cyc'] = 100
        gpx.add_powder_histogram('%s/data.xy'%gsas2_scratch,'%s/gsas.instprm'%gsas2_scratch)

        hist   = gpx.histograms()[0]
        for p in phases:
            gpx.add_phase(p['cif_abs_path'],phasename=p['phase_name'], histograms=[hist],fmthint='CIF')
        gpx.save()

        gpx = refinement_recipe(gpx)


        histogram = gpx.histograms()[0]

        gsas_X  = histogram.getdata('x').astype('float32')
        gsas_Yo = histogram.getdata('yobs').astype('float32')
        gsas_Yc = histogram.getdata('ycalc').astype('float32')
        gsas_B  = histogram.getdata('Background').astype('float32')



        # now convert Y back
        Y_Obs_from_gsas  = ((gsas_Yo -y_baseline  )/y_scale )
        Y_Bkg_from_gsas  = ((gsas_B  -y_baseline  )/y_scale )
        Y_calc_from_gsas = ((gsas_Yc -y_baseline  )/y_scale ) # this also includes background from GSAS


        ds = ds.assign_coords(
            {"X_in_q": da_i2d_m.radial.values.astype('float32')},
            )
        ds = ds.assign_coords(
            {"X_in_tth": gsas_X.astype('float32')},
            )
        ds = ds.assign_coords(
            {"X_in_d": pySULI_general.q_to_d(da_i2d_m.radial.values.astype('float32'))},
            )


        ds['Y_obs'] = xr.DataArray(
                                    data=da_i2d_m.values,
                                    dims=['X_in_q'],
                                )
        ds['Y_calc'] = xr.DataArray(
                                    data=Y_calc_from_gsas-Y_Bkg_from_gsas,
                                    dims=['X_in_q'],
                                )# by Ycalc, we mean only calculated peaks, no background

        ds['Y_bkg_gsas'] = xr.DataArray(
                                    data=Y_Bkg_from_gsas,
                                    dims=['X_in_q'],
                                )
        ds['Y_bkg_auto'] = xr.DataArray(
                                    data=bkg_auto,
                                    dims=['X_in_q'],
                                )



        with open('%s/gsas.lst'%(gsas2_scratch)) as lst_file:
            gsas_lst = lst_file.read()
            ds.attrs['gsas_lst'] = gsas_lst


    # for e,p in enumerate(gpx.phases()):
    #     p.export_CIF(outputname='%s/%s_from_refinement.cif'%(gsas2_scratch,p.name))
    #     with open('%s/%s_from_refinement.cif'%(gsas2_scratch,p.name)) as cif_file:
    #         ds.attrs['%s_refined_structure'%(p.name)] = cif_file.read()
    #     os.remove('%s/%s_from_refinement.cif'%(gsas2_scratch,p.name))


    if update_ds:
        ds.to_netcdf(nc_path+'.new.nc',engine="scipy")
        time.sleep(0.1)
        shutil.move(nc_path+'.new.nc',nc_path)


    if update_gpx:
        shutil.copyfile(gpx['Controls']['data']['LastSavedAs'],'%s.gpx'%nc_path[:-3])


    if plot:

        # now create a mosaic and plot based on its pattern
        fig = plt.figure(figsize=(8,8),dpi=96)
        mosaic = """
        RRR
        YYY
        YYY
        PPP
        DDD
        """
        ax_dict = fig.subplot_mosaic(mosaic)

        ax = ax_dict["R"]

        np.log(da_i2d).plot.imshow(ax=ax,robust=True,add_colorbar=False,cmap='Greys',vmin=0)
        ax.set_ylabel(ds.i2d.ylabel)
        ax.set_xlabel(None)
        ax.set_xticklabels([])
        ax.set_xlim([q_range[0],q_range[1]])
        Rwp, GoF = gpx.data['Covariance']['data']['Rvals']['Rwp'], gpx.data['Covariance']['data']['Rvals']['GOF']
        ax.set_title('%s\n(R$_{wp}$=%.3f GoF=%.3f)'%(nc_path.split('/')[-1],Rwp,GoF))
        ax.set_facecolor('#FFFED4')

        ax = ax_dict["Y"]
        X, Yobs, Ycalc, Ybkg= ds['X_in_q'].values, ds['Y_obs'].values, ds['Y_calc'].values, (ds['Y_bkg_auto']+ds['Y_bkg_gsas']).values
        ax.plot(X, np.log( Yobs ),label='Y$_{obs}$',lw=2,color='k')
        ax.plot(X, np.log( Ycalc+Ybkg ),label='Y$_{calc}$+Y$_{bkg}$',lw=1,color='y')
        ax.fill_between(X, np.log( Ybkg ),label='Y$_{bkg}$',alpha=0.2)
        ax.set_ylim(bottom=np.log(min(min(Ybkg),min(Ybkg)))-0.1)
        ax.set_xticklabels([])
        ax.set_ylabel('Log$_{10}$(counts) (a.u.)')
        ax.set_xlim([q_range[0],q_range[1]])

        # find lattice constants for each phase and use them for label strings
        phases = pySULI_general.get_valid_phases(gpx)
        phase_ct = len(phases)
        label_strs = [None] * phase_ct
        label_colors = [None] * phase_ct
        for ep, phase in enumerate(phases):
            label_strs[ep] = phase
            consts = iter(['a', 'b', 'c'])
            for value in np.unique(list(pySULI_general.get_cell_consts(gpx, phase).values())):
                label_strs[ep] = label_strs[ep] + "\n(" + next(consts) + " = " + str(round(value,6)) + ")"
            label_colors[ep] = "C%d" % ep
            
        # create labels for legend as Line2D objects
        from matplotlib.lines import Line2D
        custom_handles = []
        for i in np.arange(0,phase_ct,1):
            custom_handles.append(Line2D([0], [0], marker='o',color=label_colors[i], label='Scatter',
                          markerfacecolor=label_colors[i], markersize=5))

        # get legend handles from the plot and set legend
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles + custom_handles
        print(label_strs)
        all_labels = labels + label_strs
        ax.legend(all_handles, all_labels)
        
        ax = ax_dict["Y"]
        ax_bottom = ax_dict["P"]
        pySULI_general.gpx_plotter(
            gpx,
            line_axes=[ax,ax_bottom],
            stem_axes=[ax_bottom],
            radial_range=q_range,
            phases=pySULI_general.get_valid_phases(gpx),
            marker="o",
            stem=True,
            unit="d",
            plot_unit="q_A^-1",
            y_shift=0.1
        )

        ax = ax_dict["Y"]
        ax_bottom = ax_dict["P"]

        ax = ax_dict["D"]
        ax.plot(X, Yobs-Ycalc-Ybkg,label='diff.',lw=1,color='r')
        ax.set_xlabel('Scattering vector $q$ ($\AA^{-1}$)')
        ax.set_ylabel('counts (a.u.)')
        ax.set_xlim([q_range[0],q_range[1]])
        ax.legend(loc='best')


    return [gpx,ds]


class Refiner:
    def __init__(self):
        refinement_recipe = []