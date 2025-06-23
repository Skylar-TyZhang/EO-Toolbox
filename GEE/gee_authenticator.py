from IPython.display import Image
import matplotlib.pyplot as plt
import folium
import ee
import numpy as np
class Remote_Sensing:
    def __init__(self) -> None
        pass
        
    def gee_authenticator(self):
        # make sure you have created a Google Earth Engine account before this
        # Trigger the authentication flow.
        ee.Authenticate()

        # Initialize the library.
        ee.Initialize()
    
    def download_image(self, collection:str, bounds, date_start, date_end ):
        # define the data collection
        collection = ee.ImageCollection(collection)
        
        # load a single representative image from the collection 
        # Use `filterBounds()`, `filterDate` to filterout the data
        img = ee.Image(
            collection
            # get only those images that contain our boundary for London
            .filterBounds(bounds)
            .filterDate(date_start, date_end)  # get images from selected time period
            .filter(ee.Filter.lt('CLOUD_COVER', 10))
            .sort('CLOUD_COVER')  # sort in ascending order by cloud cover
            .first()  # get the first image only (least cloudy)
            .clip(bounds)  # clip to the London area extent
        )
        return img
        
        