import numpy as np
import xarray as xr

# Load the NetCDF dataset
file_path = "BeiJing_2023.nc"
dataset = xr.open_dataset(file_path)

# List the variables in the dataset
variables = list(dataset.data_vars.keys())

# Compute the wind speed as the vector magnitude of u10 and v10
wind_speed = np.sqrt(dataset['u10']**2 + dataset['v10']**2)

# Calculate mean annual values
mean_annual_temperature = dataset['t2m'].mean().item() - 273.15  # Convert from Kelvin to Celsius
mean_annual_humidity = dataset['d2m'].mean().item() - 273.15     # Convert from Kelvin to Celsius (proxy for humidity)
mean_annual_pressure = dataset['sp'].mean().item() / 100         # Convert from Pa to hPa
mean_annual_wind_speed = wind_speed.mean().item()                # In m/s

"""mean_annual_temperature, mean_annual_humidity, mean_annual_pressure, mean_annual_wind_speed"""
