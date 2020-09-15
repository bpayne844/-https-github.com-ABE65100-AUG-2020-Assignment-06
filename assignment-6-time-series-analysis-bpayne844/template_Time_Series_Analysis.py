"""
Add your own header text here...
"""

# import required modules
import pandas as pd
import matplotlib.pyplot as plt

# user defined functions

def get_daily_mean_flow( dataDF ):
    # calculate daily mean value, returns a new dataframe
    
def get_highest_flow( dataDF, Nvals ):
    # identify the Nval highest value, returns a new dataframe with Nvals rows
    
def get_monthly_mean_flow( dataDF ):
    # calculate monthly mean value, returns a new dataframe
    
def read_data( inFileName ):
    # open the data file and read contents, retuns a dataframe

def plot_streamflow( dataDF ):
    # create plots of the time series data

# the following condition checks whether we are
# running as a script, in which case run the test code,
# or being imported, in which case don't.

if __name__ == '__main__':

    # define filename
    #inFileName = 'WabashRiver_Discharge_20150317-20160324.txt'
    
    # read in data file
    #dataDF = read_data( inFileName )
    
    # create plots of time series data
    #plot_streamflow( dataDF )