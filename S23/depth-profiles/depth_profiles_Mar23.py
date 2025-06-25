#!/usr/bin/env python
# coding: utf-8

'''
Script to parse SnowEx March 2023 snow depth profile data
The purpose of this script is to pull the transcribed snow depth profile data from the SnowEx 2023 March campaign add the spatial information given the SE coordinate reference point.
The manual snow depths were added to the bottom portion of the digital pit sheets. This is a one stop shop for associated data (i.e. SE plot perimeter coordinate)
and all the manual depth observations with comments.
'''

# standard imports
from pathlib import Path
import glob
import os
import shutil
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utm

# custom imports
from metadata_headers_depthprofiles import metadata_headers_depthprofile

# -----------------------------------------------------------------------------
# 1. Function to write summary file header rows
def writeHeaderRows(fname_summaryFile, metadata_headers):
        with open(fname_summaryFile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in metadata_headers:
                writer.writerow(row)
# -----------------------------------------------------------------------------
#. Function to extract snow depths and other pit metadata
def extract_snow_depths(fname):

    d = pd.read_excel(fname, sheet_name='FRONT')
    Date = d['Unnamed: 6'][1]
    Time = d['Unnamed: 6'][4]
    Site = d['Unnamed: 0'][4].split(':')[0]
    PlotID = d['Unnamed: 0'][6]
    Plot_No = int(PlotID[-3:])
    Easting = float(d['Unnamed: 8'][6])
    Northing = float(d['Unnamed: 12'][6])
    if 'CPC' in Site:
        Site = 'CPCRW'
    rIx = (d.iloc[:,0] == 'Depth Measurements').idxmax() #locate 'Depth Measurements' cell in spreadsheet (row Index)
    d = d.loc[rIx:,:].reset_index(drop=True)
    Observer = d['Unnamed: 3'][1]
    # get manual depth values
    df = d.iloc[5:, [0,2,4,5,6,8,9]] # yuck, hard coding.... 5 is to the 'box' row and 70 should be ~5+NaNs that get dropped later
    df.columns = ['PointID', 'HS top (cm)', 'HS bottom (cm)', 'Canopy Cover',
                  'HS top (cm).1', 'HS bottom (cm).1', 'Comment']
    df = df[df['PointID'].notna()].reset_index(drop=True) # keep everything that is not nan in the PointID row
    df['PointID'] = df['PointID'].astype(str) # convert point ID to string (e.g. '00', '01') so that it can search for 'box' and 'north' below
    if df['PointID'].str.contains('Box').any():
        rIx_box = df.index[df['PointID'] == 'Box'][0] # box index (note, secondary box for site sans MagnaProbe and Mesa2, n=3)
        df = df.iloc[:rIx_box].reset_index(drop=True)

    elif df['PointID'].str.contains('North').any():
        rIx_north = df.index[df['PointID'] == 'North'][0] # north index (n=19), remove N (and W) legs because they are not part of the depth profile (top and bottom sampling)
        df = df.iloc[:rIx_north].reset_index(drop=True)
    # add Columns to dataframe
    df['Study Area'] = Site
    df['Plot ID'] = PlotID
    df['Date'] = Date
    df['Time'] = Time
    df['Easting'] = Easting
    df['Northing'] = Northing
    # format type
    df['Date'] = df['Date'].dt.strftime("%m/%d/%y")
    df['Time'] = df['Time'].apply(lambda x: x.strftime('%H:%M'))

    # pad with leading zero (e.g. 00, 01, 02, etc.)
    # df['PointID'] = df['PointID'].astype('str').str.zfill(2) # maybe don't need the .astype('str')
    df['PointID'] = df['PointID'].str.zfill(2)
    # add pointType (i.e P=Preimeter)
    df['PointID'] = 'P' + df['PointID']#.astype(str)

    # assign county info
    if Plot_No < 500:
        df['County'] = "Fairbanks North Star Borough"
    elif Plot_No >=500:
        df['County'] = "North Slope Borough"

    return df
#df = df[['PointID', 'Easting', 'Northing']]

# -----------------------------------------------------------------------------
# Function to check dataframe length -- probs can delete
def df_length(df, fname):
    print(len(df), fname.name)

# -----------------------------------------------------------------------------
# 3. Function to check dataframe length -- probs can deletedef update_xy(df):
def update_xy(df):
    if len(df) ==20: # 1 m interval pit perimeter box spacing

            # create x,y grid: a local coordinate system centered on the pit location, aligned with magnetic north
            df['x']=np.nan
            df['y']=np.nan

            # this works and is adequate for the size dataset we have
            # it could be made more pythonic as iterating over rows in a table is not best practice because it is slow
            for index,row in df.iterrows():
                pointType=row['PointID'][0]
                pointNum=float(row['PointID'][1:])
                if pointType == 'P':
                    if pointNum < 5: # walk north
                        row['x']=0.
                        row['y']=pointNum
                    elif pointNum < 10: # walk west
                        row['x']=-(pointNum-5)
                        row['y']=5.
                    elif pointNum < 15: # walk south
                        row['x']=-5.
                        row['y']=5-(pointNum-10)
                    elif pointNum <= 19: # walk east
                        row['x']=-5+(pointNum-15)
                        row['y']=0
                    else:
                        print('Unexpected pointNum for perimeter')

                df.loc[index]=row

            # # plot positions to check we are doing this about right
            # df.plot.scatter('x','y')

    elif len(df)>20: # applies to North Slope sites where 4 corners + 25 cm transect spacing on one of the box segments (majority NW to SW; n=1 SW to SE )

        # create x,y grid: a local coordinate system centered on the pit location, aligned with magnetic north
        df['x']=0.# these can NOT be np.nan or else the rotation func. will not work
        df['y']=0.# these can NOT be np.nan or else the rotation func. will not work

        # Use np.where with masks to set values based on conditions
        mask_P01 = df['PointID'] == 'P01' # NE corner (0, 5)
        df.loc[mask_P01, 'y'] = 5

        mask_P02 = df['PointID'] == 'P02' # NW corner (-5, 5)
        df.loc[mask_P02, 'x'] = -5
        df.loc[mask_P02, 'y'] = 5

        mask_last_point = df['PointID'] == df['PointID'].iloc[-1] # SW corner (-5, 0)
        df.loc[mask_last_point, 'x'] = -5

        len_transect = len(df.iloc[3:-1]) # length of transect rows (P03 to end-1)
        mask_transect = (df.index >= 3) & (df.index < len(df) - 1)
        df.loc[mask_transect, 'x'] = -5 # x off set -5 meters
        df.loc[mask_transect, 'y'] = 5 - np.array([5/len_transect] * len_transect).cumsum() # y off set 5m/cumsum length of transect
        # print(df)

    return df

# -----------------------------------------------------------------------------
# 5. Rotate xy data
def rotate_xy(df, declination_degrees):

    # Convert declination from degrees to radians
    declination_radians = np.radians(declination_degrees)

    # Define the rotation matrix
    rotation_matrix = np.array([[np.cos(declination_radians), np.sin(declination_radians)],
                                [-np.sin(declination_radians), np.cos(declination_radians)]])


    # Apply rotation to x, y data
    xy_data = df[['x', 'y']].values  # Extract x, y data as numpy array
    rotated_xy_data = np.dot(xy_data, rotation_matrix.T)  # Rotate the data

    # Replace original x, y data with rotated data in the DataFrame
    df[['x', 'y']] = rotated_xy_data

    return df

# -----------------------------------------------------------------------------
#  Function to convert to csv and save.

def append_to_csv(filename, df):
    df.to_csv(filename, mode='a', na_rep=-9999, header=False, index=False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":

    # paths
    base = Path('/Users/cvuyovic/Documents/Projects/SnowEx/2022/Data/Snow Depth Profiles/Mar 2023')
    path_in  = Path('/Users/cvuyovic/Documents/Projects/SnowEx/2022/Data/Snow Depth Profiles/Mar 2023/Data_revised')
    path_out = Path('/Users/cvuyovic/Documents/Projects/SnowEx/2022/Data/Snow Depth Profiles/Mar 2023/outputs')
    path_out.mkdir(parents=True, exist_ok=True)

    # create empty dataframe for summary file
    df_main = pd.DataFrame()

    # load data
    for fname in sorted(path_in.rglob('*.xlsx')):
        print(fname.name)

        df = extract_snow_depths(fname)
        # df_length(df, fname)
        df = update_xy(df) # inside for loop to go site by site, hence len(df)==20 or >20
        df = rotate_xy(df, declination_degrees=15.98)
        df_main = df_main.append(df, ignore_index=True)

    # sum UTM with off-set and rotation
    df_main['Easting'] = df_main[['Easting', 'x']].sum(axis=1) # sum pit coordinate with offset
    df_main['Northing'] = df_main[['Northing', 'y']].sum(axis=1) # sum pit coordinate with offset

    # add Lat / Lon
    df_main['Latitude'] =  utm.to_latlon(df_main['Easting'], df_main['Northing'], 6, 'Northern')[0]
    df_main['Longitude'] = utm.to_latlon(df_main['Easting'], df_main['Northing'], 6, 'Northern')[1]

    df=df_main

    # add static information to df
    df['State'] = 'Alaska'
    df['Elevation'] = None
    df['Version'] = 'v1'
    df.rename(columns={"PointID": "ID"}, inplace=True)

    # sum total thickness
    df['HS bottom (cm)'] = -df['HS bottom (cm)'] #convert bottom row to negative
    df['HS bottom (cm).1'] = -df['HS bottom (cm).1'] # convert 2nd layer bottom row to negative
    # CAREFUL ABOVE, THIS MAY FILL WITH 0s...annoying!
    df['Total Snow (cm)'] = df[['HS top (cm)', 'HS bottom (cm)', 'HS top (cm).1', 'HS bottom (cm).1']].sum(axis=1, min_count=1)
    # put data back as positive
    df['HS bottom (cm)'] = -df['HS bottom (cm)'] #convert bottom back to positive
    df['HS bottom (cm).1'] = -df['HS bottom (cm).1'] # convert 2nd layer bottom back to positive

    # reorder DataFrame columns
    reorder_cols = ['State', 'County', 'Study Area', 'Plot ID', 'ID', 'Date', 'Time',
    'Latitude', 'Longitude', 'Northing', 'Easting', 'Elevation', 'Total Snow (cm)', 'HS top (cm)',
    'HS bottom (cm)', 'HS top (cm).1', 'HS bottom (cm).1', 'Canopy Cover', 'Comment', 'Version']

    df = df[reorder_cols]

    # name output file
    filename = path_out.joinpath('_AKIOP_Mar_PlotPerimeter_DepthProfiles.csv')
    filename2 = path_out.joinpath('_AKIOP_Mar_PlotPerimeter_DepthProfiles_pandasReady.csv')
    # write metadata header rows
    writeHeaderRows(filename, metadata_headers_depthprofile)

    # append DataFrame to summary file - file to be published with metadata
    append_to_csv(filename, df)

    # file sans metadata and pandas Ready
    df.to_csv(filename2, index=False)

    print('done done')




# ~~~~~~~~~~~~~~~~ old plotting stuff

        # # plot positions to check we are doing this about right
        # fig,ax=plt.subplots()
        # plt.scatter(df['xr'],df['yr'])
        # ax.set_aspect('equal')


    #     # fig,ax=plt.subplots()
    #     # plt.scatter(df['UTM Easting [m]'],df['UTM Northing [m]'])
    #     # ax.set_aspect('equal')

    #
    #     # more clean up - tree 'wall' to well
    #     # df.replace(regex=['tree wall'], value='Tree well', inplace=True)
    #     # df.replace(regex=['Tree Wall'], value='Tree well', inplace=True)
    #     # df.replace(regex=['Tree wall'], value='Tree well', inplace=True)
    #     # df.replace(regex=['TW'], value='Tree Well', inplace=True)
    #     # df.replace(regex=['Canopy Cover'], value='Canopy', inplace=True)
    #     # df['Canopy Cover'].replace(regex=['C'], value='Canopy', inplace=True)
    #     # df.replace(regex=['deadfall'], value='Deadfall', inplace=True)
    #     # df.replace(regex=['DF'], value='Deadfall', inplace=True)
    #     #
    #     # df.replace(regex=['V'], value='Buried Tree', inplace=True)
    #     # df.replace(regex=['BT'], value='Buried Tree', inplace=True)
    #     # df.replace(regex=['ES'], value='Elevated Snow', inplace=True)
    #     # df.replace(regex=['NC'], value='No Calibration', inplace=True)
    #
    #
    #     # df.plot.scatter(x='Longitude', y='Latitude', c='Total Snow (cm)', colormap='viridis')

    #
    #     # print(fname.name)
    #     # plt.figure(figsize=(16, 5))
    #     # plt.plot(df[['HS top (cm)', 'HS bottom (cm)']], label=('HS top', 'HS bottom'), marker='o', linestyle='dotted')
    #     # plt.plot(df[['HS top (cm).1', 'HS bottom (cm).1']], label=('HS-second layer top', 'HS-second layer bottom'), marker='o', linestyle='dotted')
    #     # plt.plot(df['Total Snow (cm)'], label='Total snowpack thickness (cm)')
    #     # plt.title('Snow depth profile: {}, {}'.format(title, Date))
    #     # plt.xlabel('Flattened Perimeter')
    #     # plt.ylabel('Snow Depth (cm)')
    #     # plt.legend()
    #     # plt.show()
