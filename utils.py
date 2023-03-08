import xesmf as xe
import os
import numpy as np
import xarray as xr
import pandas as pd

from geopy.geocoders import Nominatim
from dask.diagnostics import ProgressBar
from xclim import ensembles
from xclim.ensembles import create_ensemble
import xclim.indices as xci

def get_coords(city:str):
    '''
    get lat, lon from city
    see: https://geopy.readthedocs.io/en/stable/#nominatim
    '''
    geolocator = Nominatim(user_agent='http')
    location = geolocator.geocode(city)
    latitude, longitude = location.latitude, location.longitude
    print("Location, (lat, lon): ",location, (latitude, longitude))
    return (latitude, longitude)

def load_mf_dataset(path, models:list):
    '''
    given path to multi-model ensemble dataset and list of models,
    load xr files into a dict of mf_datasets for each model
    '''
    model_files = {}
    for model in models.split(","):
        model_filenames=[]
        for filename in os.listdir(path):
            if model in filename:
                model_filenames.append(filename)
        model_files[model] = model_filenames

    os.chdir(path)
    data = {}
    for model, files in model_files.items():
        print(model, len(files))
        data[model] = xr.open_mfdataset(files, engine='netcdf4', chunks={'time': 120})
    return data

def multimodel_ensemble(data):
    ''' given a dict of models i.e. model_name[data]=dataset,
    create an xclim ensemble, selecting start and end dates, normalizing time, 
    and optionally convert to pandas dataframe
    '''
    ensemble = create_ensemble([model for model in data.values()]).load()
    ensemble.close()
    return ensemble

def extreme_temperature_indices(ens, aggregate=False):
    if aggregate==False:
        tas=ens.tas
        tasmin=ens.tasmin
        tasmax=ens.tasmax
    elif aggregate==True:
        tas=ens.tas_mean
        tasmin=ens.tasmin_mean
        tasmax=ens.tasmax_mean
    '''
    calculate extreme temperature indices from xclim ensemble
    '''
    indices = xci.daily_temperature_range(tasmin, tasmax).to_dataset(name='dtr')
    indices['dtrv'] = xci.daily_temperature_range_variability(tasmin, tasmax)
    indices['etr'] = xci.extreme_temperature_range(tasmin, tasmax)
    indices['hwf'] = xci.heat_wave_frequency(tasmin, tasmax)
    indices['hwi'] = xci.heat_wave_index(tasmax)
    indices['hwtl'] = xci.heat_wave_total_length(tasmin, tasmax)
    indices['hsf'] = xci.hot_spell_frequency(tasmax)
    #indices['wsdi'] = xci.warm_spell_duration_index(tasmin, tasmax)
    indices['hwml'] = xci.heat_wave_max_length(tasmin, tasmax)
    return indices