import pySULI_refiner as rf
import os
import sys
import xarray as xr

# Assuming notebook and data folder are here.
here = '/Users/petermeshkov/Repos/pySULI-Lib/example_data'
example_data_path = 'xrd_refinement_with_gsas2/'

os.chdir('%s/%s'%(here,example_data_path))
os.listdir()

os.makedirs('gsas2_scratch',exist_ok=True)
f = open("gsas2_scratch/gsas.instprm", "w")
f.write("""#GSAS-II instrument parameter file; do not add/delete items!
Type:PXC
Bank:1.0
Lam:0.1824
Polariz.:7.277695011573669
Azimuth:0.0
Zero:-0.00025200085768731445
U:129.15268004188428
V:1.816645496453032
W:0.43034683098418736
X:-0.0661572816525536
Y:-0.9270864296622138
Z:0.02775503495558348
SH/L:0.002""")

def refinement_recipe(refiner,update_instprm=True):

    refiner.refine_background(num_coeffs=2,set_to_false=True)
    refiner.set_LeBail(LeBail=True)
    refiner.refine_cell_params(set_to_false=True)
    # refine_size_broadening(gpx,set_to_false=True)
    # refine_strain_broadening(gpx,set_to_false=True)

    for par in ['U', 'V', 'W']:
        refiner.refine_inst_parameters(inst_pars_to_refine=[par])
    for par in ['X', 'Y', 'Z']:
        refiner.refine_inst_parameters(inst_pars_to_refine=[par])
    for par in ['Zero']:
        refiner.refine_inst_parameters(inst_pars_to_refine=[par])
    if update_instprm:
        refiner.instprm_updater()
        
    refiner.get_gpx().save

    refiner.refine_background(num_coeffs=10,set_to_false=True)
    refiner.refine_cell_params(set_to_false=True)

    return refiner.get_gpx()

with xr.open_dataset('Background.nc') as ds:
    da_input_bkg = ds.i2d.mean(dim='azimuthal')

LaB6_refiner = rf.Refiner(
    nc_path = 'LaB6.nc',
    phases = [{'cif_abs_path':'LaB6.cif','phase_name':'LaB6','scale':1},],
    gsas2_scratch = 'gsas2_scratch',
    q_range = [0.5,4.1],
    da_input_bkg = da_input_bkg
)

LaB6_refiner.refine(refinement_recipe)
LaB6_refiner.update_ds_file()
LaB6_refiner.update_gpx()
LaB6_refiner.plot_refinement_results()