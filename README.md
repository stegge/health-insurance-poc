**# health-insurance-poc**
- The purpose of the below steps is to simulate a local file upload to DBFS from an external vendor at a health insurance company. 

**#Update file generation script**
- Step 1: Navigate to /Data Build & Movement/ and open health_ins_poc_file_gen.py. 
- Step 2: Change the OUT_DIR to your desired local path where the 55 csv files will be written to. OUT_DIR = "[LOCAL_PATH]"
- Step 3: You can also modify the START_DATE and END_DATE to generate a wider range of data across broader time scales. 
- Step 4: Each file generation step has a range parameter that can be adjusted based on your needs. 
- Step 5: Save changes to the script. 

**#Update script to copy files to DBFS**
- Step 1: Update the param() information changing the information in the below example to your requirements. 
    - LocalPath = the path the python script wriote the 55 csv files to in the previous script.
    - BaseDest = the path to your DBFS volume where the files will be written.
    - DBXProfile = your dbx profile where the volume is located. 
      param(
        [string]$LocalPath = "C:\Users\STEGGE\Desktop\health_ins_poc_full\files",
        [string]$BaseDest  = "dbfs:/Volumes/workspace/tegge-insurance-data/health_ins_poc_raw",
        [switch]$Overwrite,
        [string]$DBXProfile   = "https://dbc-1522605f-073b.cloud.databricks.com"  # default to your saved profile name
        )

**#Update Pipeline_2 file**
- Step 1: Update the params
  PythonScriptPath = path of your python file generation script.
  DbfsLoaderScriptPath = path to the copy to DBFS script
  WorkingDir = path to the 55 csv files
    param(
        [string]$PythonScriptPath = "C:\Users\STEGGE\Desktop\script\full_load\health_ins_poc_file_gen.py",
        [string]$PythonExe = "python",
        [string]$DbfsLoaderScriptPath = "C:\Users\STEGGE\Desktop\script\full_load\health_ins_poc_full_files_to_dbfs.ps1",
        [string]$WorkingDir = "C:\Users\STEGGE\Desktop\health_ins_poc_full\files"
      )
- Step 2: Validate you have not changed anything else. **If this is modified incorrectly, the generation script and/or the copy script will not run. **

**#Running the Pipeline Script manually.**
- Step 1: open the terminal.
- Step 2: change directories to the location of the pipeline_2.ps1 script. cd "PATH_TO_SCRIPT"
- Step 3: type .\pipeline_2.ps1 and hit enter. 
