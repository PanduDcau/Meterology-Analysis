import numpy as np
import xarray as xr

# Load the NetCDF dataset
file_path = "BeiJing_2023.nc"

try:
    dataset = xr.open_dataset(file_path)

    # List the variables in the dataset
    variables = list(dataset.data_vars.keys())
    print("Variables in the dataset:")
    for var in variables:
        print(f"- {var}")
    print("-" * 30)

    # Compute the wind speed as the vector magnitude of u10 and v10
    if 'u10' in dataset and 'v10' in dataset:
        wind_speed = np.sqrt(dataset['u10']**2 + dataset['v10']**2)
        mean_annual_wind_speed = wind_speed.mean().item()
        print(f"Mean annual wind speed: {mean_annual_wind_speed:.2f} m/s")
    else:
        print("u10 or v10 variables not found. Cannot calculate wind speed.")
    print("-" * 30)

    # Mean annual temperature
    if 't2m' in dataset:
        mean_annual_temperature = dataset['t2m'].mean().item() - 273.15  # K to °C
        print(f"Mean annual temperature: {mean_annual_temperature:.2f} °C")
    else:
        print("t2m variable not found. Cannot calculate mean annual temperature.")
    print("-" * 30)

    # Mean annual dewpoint temperature (proxy for humidity)
    if 'd2m' in dataset:
        mean_annual_humidity = dataset['d2m'].mean().item() - 273.15  # K to °C
        print(f"Mean annual dew point temperature (proxy for humidity): {mean_annual_humidity:.2f} °C")
    else:
        print("d2m variable not found. Cannot calculate mean annual dew point temperature.")
    print("-" * 30)

    # Mean annual surface pressure
    if 'sp' in dataset:
        mean_annual_pressure = dataset['sp'].mean().item() / 100  # Pa to hPa
        print(f"Mean annual surface pressure: {mean_annual_pressure:.2f} hPa")
    else:
        print("sp variable not found. Cannot calculate mean annual surface pressure.")
    print("-" * 30)

    # Total annual precipitation
    if 'tp' in dataset:
        total_precipitation = dataset['tp'].sum().item() * 1000  # m to mm
        print(f"Total annual precipitation: {total_precipitation:.2f} mm")
    else:
        print("tp variable not found. Cannot calculate total annual precipitation.")
    print("-" * 30)

    # Print dataset info
    print("\nDataset information:")
    print(dataset)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except ValueError as e:
    print(f"Error opening or processing the dataset: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
