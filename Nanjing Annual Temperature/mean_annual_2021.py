import xarray as xr
import pandas as pd

# Load the dataset
file_path = r"E:\pythoncode\NLP\Classwork_tem\2021_nanjing.nc"
ds = xr.open_dataset(file_path)

# Assign the proper time coordinate
ds = ds.rename({'valid_time': 'time'})

# Extract the temperature variable and convert from Kelvin to Celsius
temp_c = ds['t2m'] - 273.15

# Group by month and calculate the mean
monthly_mean = temp_c.groupby('time.month').mean(dim=['time', 'latitude', 'longitude'])

# Annual mean across all time, latitude, longitude
annual_mean = temp_c.mean(dim=['time', 'latitude', 'longitude'])

# Output results
print("Monthly Mean Temperatures (°C):")
for month, value in enumerate(monthly_mean.values, start=1):
    print(f"Month {month:02d}: {value:.2f}°C")

print(f"\nAnnual Mean Temperature (°C): {annual_mean.values:.2f}")
