# Environmental Informatics

## Assignment 06 - Time Series Analysis with Pandas

### Lab Objectives

On completion of this lab, students will be able to:

1. Use Time and Date functions with pandas DataFrames to conduct time series analysis of a dataset.
2. Import a data file into a Pandas DataFrame that uses Datetime as the index.
3. Use Pandas Datetime methods to summarize time series data.

### Reading Assignment

Data Analysis with Open Source Tools:

- Chapter 4: Time as a Variable: Time Series Analysis

### The Lab Assignment

1. Start by cloning this assignment to the folder where you have been completing assignments.

2. Next, complete the tutorial [Time Series Analysis with Pandas](http://earthpy.org/pandas-basics.html), but include all of the tutorial code in a Python program called **PandasDatesDemo.py** that is included in this repository.

   - The "set_printoptions()" function does not appear to work with the current version of pandas.
   - Do not set your graphics to inline (see "ln[7]:"), graphics are already prepared in the spyder interface.
   - The `!wget` command will download the data file from within the console in spyder.
   - What dates should you use to define the correct record length?  Open the file in Notepad+ or another text editor (the tutorial was written in 2013, but the data file continues to be updated).  
   - The length of record for the AO and NAO files may not be the same, despite what the tutorial document says.  If they are not the same length, what happens when they are combined into a single DataFrame?
   - Submit the following plots for evaluation: 
     - Daily Atlantic Oscillation (AO) plot (Out [23]:)
     - Annual median values for AO (Out [48]:)
     - Rolling mean for both AO and NAO (Out [52]:)
     
3. Once you have completed the tutorial, copy the template **template_Time_Series_Analysis.py** to the submission file name **Time_Series_Analysis.py**, and use your new skills to write a Python script that will do the following. 

   - Read the contents of the file **WabashRiver_Discharge_20150317-20160324.txt** into a Pandas dataframe.
     - This file contains discharge for the Wabash River at the Lafayette, Indiana gauge from March 17, 2015 through March 24, 2016.
     - Use the column labeled "datetime" to form a single Datetime element - use this as the Index of the DataFrame.  It is not necessary to work with the time zone for this assignment.
     - Use the column labeled "01_00060", which is discharge in cubic feet per second, for the value - it should be called \"Discharge\".
     - The function should return a Pandas Series, with datetime as the index and Discharge as the value.  Make sure that the data column is named **Discharge**.  In Pandas, a Series is effectively a DataFrame with one column.  Because there is only one column, some of the indexing with a series is different, but most functionality transfers between the two data types.
     - For this process, the following additional information may prove useful:
       - The Pandas documentation on the [read_table](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html) DataFrame method.
       - More information can also be found at the [IO Tools page](http://pandas.pydata.org/pandas-docs/stable/io.html), in particular help on how to get the read function to parse the date for you automatically can be found [here](http://pandas.pydata.org/pandas-docs/stable/io.html#datetime-handling).
       - Information on how to get read_tables to handle variable white space as a separator can be found at this [discussion page](http://stackoverflow.com/questions/12021730/can-pandas-handle-variable-length-whitespace-as-column-delimeters).
       - Also note that all of this does not have to all be done in a single line statement.  You may find it easier to import the data in a more raw format, and then work it into a usable final format.  I find this approach useful when datafile quality is suspect, as it provides more options to assess problems and catch errors.  The file I have given you is pretty clean, so not necessarily representative of the types of data you will find in real life. 
   - Write a function to convert the 15-minute streamflow records into average daily values, and return a new series.  Daily discharge values should have the header **Discharge** in the new Series.
   - Create a plot of daily average streamflow for the period of record, written to a PDF or PS file.
   - Write a function to identify the 10 highest daily flow events during the extent of the data record, and return a new Series where the discharge is labeled **Highest Flow**.
   - Create a new plot with daily average flow data, and identify the 10 days with highest flow using a symbol.  Include a legend for the discharge and highest flow data, and save the plot to a PDF or PS file.
   - Write a function to convert average daily flow into average monthly flow, and return a new Series.  Monthly flow data should be labeled **Discharge**.
   - Create a plot of monthly average streamflow for the period of record, written to a PDF or PS file.

4. Be sure that the script has a complete header comment block, appropriate in-line comments, and runs without intervention relative to where the datafile is stored in the repository.

#### What to turn in...

The following should be included in your GitHub repository:

1. A working tutorial solution file called **PandasDatesDemo.py**.

2. The original data file, **WabashRiver_Discharge_20150317-20160324.txt**, provided with the repository.

3. A working program called **Time_Series_Analysis.py**.

4. Put your input file, output files, and processing script in the assignment repository and push to GitHub to submit.

#### Grading Rubric (50 pts Total)

| Question | Description | Score |
| -------- | ----------- | ----- |
| 1. |Complete the tutorial Time series analysis with Pandas, and submit the following for evaluation | (15 pts) |
| 1.1. | Daily Atlantic Oscillation (AO) plot (Out [23]:) | 5 pts |
| 1.2. | Annual median values for AO (Out [48]:) | 5 pts |
| 1.3. | Rolling mean for both AO and NAO (Out [52]:) | 5 pts |
| 2. | Python script passes the autograder | 10 pts |
| 3. | Create a plot of daily average streamflow for the period of record, written to a PDF or PS file. | 5 pts |
| 4. | Using the daily average flow data, identify and plot the 10 days with highest flow, written to a PDF or PS file.  Use symbols to represent the data on the same time axis used for the full daily flow record. | 5 pts |
| 5. | Create a plot of monthly average streamflow for the period of record, written to a PDF or PS file. | 5 pts |
| 6. | Submit all input and output files. | 5 pts |
| 7. | Program has complete header and adequate in-line comments. | 5 pts |
