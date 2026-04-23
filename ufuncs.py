def lon_to_180(ds):
    ds = ds.copy()
    ds['longitude'] = ((ds['longitude'] + 180) % 360) - 180
    ds = ds.sortby('longitude')
    return ds