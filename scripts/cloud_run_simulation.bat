@echo off
setlocal

:: ============================================================================
:: GCP SIMULATION AUTOMATION SCRIPT (DEBUGGING MODE - NO SHUTDOWN)
:: ============================================================================
:: WARNING: THIS SCRIPT WILL NOT SHUT DOWN THE VM AUTOMATICALLY.
:: Its only purpose is to get a complete error log.
:: YOU MUST MANUALLY DELETE THE VM AFTER RUNNING THIS SCRIPT.
:: ============================================================================

:: ----------------------------------------------------------------------------
:: (1) CONFIGURE YOUR SETTINGS HERE
:: ----------------------------------------------------------------------------
set "PROJECT_ID=storied-epigram-458906-v7"
set "VM_ZONE=asia-southeast1-a"
set "MACHINE_TYPE=e2-standard-16"
set "VM_IMAGE_FAMILY=debian-11"
set "VM_IMAGE_PROJECT=debian-cloud"

:: --- File paths relative to this script's location ---
set "LOCAL_REQUIREMENTS_PATH=..\requirements.txt"
set "LOCAL_SCRIPT_PATH=..\src\main_simulation\cloud_run_simulation.py"
set "LOCAL_DATA_FILE_PATH=..\data\data.xlsx"
set "STARTUP_SCRIPT_PATH=%~dp0startup-script.sh"

:: --- Auto-generated names for cloud resources ---
set "RANDOM_ID=%RANDOM%"
set "VM_NAME=qsdsan-sim-vm-%RANDOM_ID%"
set "BUCKET_NAME=%PROJECT_ID%-qsdsan-sim-%RANDOM_ID%"
for %%F in ("%LOCAL_SCRIPT_PATH%") do set "REMOTE_SCRIPT_NAME=%%~nxF"
set "REMOTE_RESULTS_FILENAME=simulation_results_final.xlsx"
set "REMOTE_LOG_FILE=simulation_log.txt"

echo =================================================
echo.
echo  !!!!!!!!!!   W A R N I N G   !!!!!!!!!!
echo.
echo  This script is in DEBUGGING MODE.
echo  The VM created (%VM_NAME%) WILL NOT SHUT DOWN automatically.
echo  You MUST delete it manually from the Google Cloud Console
echo  after the script finishes to avoid extra charges.
echo.
echo =================================================
echo.
pause

:: ----------------------------------------------------------------------------
:: (2) GENERATE THE VM STARTUP SCRIPT
:: ----------------------------------------------------------------------------
echo Generating VM startup script for debugging...

(echo #!/bin/bash) > "%STARTUP_SCRIPT_PATH%"
(echo exec ^> /var/log/startup-script-output.log 2^>^&1) >> "%STARTUP_SCRIPT_PATH%"
(echo set -e) >> "%STARTUP_SCRIPT_PATH%"
(echo.) >> "%STARTUP_SCRIPT_PATH%"
(echo echo "--- Installing System Dependencies ---") >> "%STARTUP_SCRIPT_PATH%"
(echo apt-get update -y) >> "%STARTUP_SCRIPT_PATH%"
(echo apt-get install -y apt-transport-https ca-certificates gnupg curl python3-pip python3-venv) >> "%STARTUP_SCRIPT_PATH%"
(echo.) >> "%STARTUP_SCRIPT_PATH%"
(echo echo "--- Installing Google Cloud SDK ---") >> "%STARTUP_SCRIPT_PATH%"
(echo echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" ^| tee /etc/apt/sources.list.d/google-cloud-sdk.list) >> "%STARTUP_SCRIPT_PATH%"
(echo curl https://packages.cloud.google.com/apt/doc/apt-key.gpg ^| apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -) >> "%STARTUP_SCRIPT_PATH%"
(echo apt-get update -y ^&^& apt-get install -y google-cloud-sdk) >> "%STARTUP_SCRIPT_PATH%"
(echo.) >> "%STARTUP_SCRIPT_PATH%"
(echo echo "--- Creating Python Virtual Environment ---") >> "%STARTUP_SCRIPT_PATH%"
(echo python3 -m venv /opt/sim-env) >> "%STARTUP_SCRIPT_PATH%"
(echo.) >> "%STARTUP_SCRIPT_PATH%"
(echo BUCKET_NAME=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/BUCKET_NAME" -H "Metadata-Flavor: Google" ^) ) >> "%STARTUP_SCRIPT_PATH%"
(echo SCRIPT_NAME=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/SCRIPT_NAME" -H "Metadata-Flavor: Google" ^) ) >> "%STARTUP_SCRIPT_PATH%"
(echo RESULTS_FILE=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/RESULTS_FILE" -H "Metadata-Flavor: Google" ^) ) >> "%STARTUP_SCRIPT_PATH%"
(echo LOG_FILE=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/LOG_FILE" -H "Metadata-Flavor: Google" ^) ) >> "%STARTUP_SCRIPT_PATH%"
(echo.) >> "%STARTUP_SCRIPT_PATH%"
(echo echo "--- Downloading requirements.txt and installing Python packages ---") >> "%STARTUP_SCRIPT_PATH%"
(echo gsutil -q cp "gs://\${BUCKET_NAME}/requirements.txt" .) >> "%STARTUP_SCRIPT_PATH%"
(echo /opt/sim-env/bin/pip install --upgrade pip) >> "%STARTUP_SCRIPT_PATH%"
(echo /opt/sim-env/bin/pip install -r requirements.txt) >> "%STARTUP_SCRIPT_PATH%"
(echo.) >> "%STARTUP_SCRIPT_PATH%"
(echo echo "--- Downloading simulation files and starting script ---") >> "%STARTUP_SCRIPT_PATH%"
(echo gsutil -q cp "gs://\${BUCKET_NAME}/\${SCRIPT_NAME}" .) >> "%STARTUP_SCRIPT_PATH%"
(echo gsutil -q cp "gs://\${BUCKET_NAME}/data.xlsx" .) >> "%STARTUP_SCRIPT_PATH%"
(echo /opt/sim-env/bin/python3 "\${SCRIPT_NAME}" --bucket "gs://\${BUCKET_NAME}" --output "\${RESULTS_FILE}") >> "%STARTUP_SCRIPT_PATH%"
(echo.) >> "%STARTUP_SCRIPT_PATH%"
(echo echo "--- Startup script finished successfully. ---") >> "%STARTUP_SCRIPT_PATH%"
(echo echo "--- Uploading log file. ---") >> "%STARTUP_SCRIPT_PATH%"
(echo gsutil -q cp /var/log/startup-script-output.log "gs://\${BUCKET_NAME}/\${LOG_FILE}") >> "%STARTUP_SCRIPT_PATH%"
(echo echo "--- VM will now remain running for inspection. Please delete it manually. ---") >> "%STARTUP_SCRIPT_PATH%"

IF NOT EXIST "%STARTUP_SCRIPT_PATH%" (
    echo. & echo FATAL ERROR: Failed to generate the startup script file. & goto :EOF
)
echo  ...Startup script generated successfully.
echo.

:: ----------------------------------------------------------------------------
:: (3) CREATE CLOUD RESOURCES AND START SIMULATION
:: ----------------------------------------------------------------------------
echo Creating Cloud Storage Bucket and uploading files...
CALL gsutil mb -p %PROJECT_ID% -l %VM_ZONE:~0,-2% gs://%BUCKET_NAME% >nul
if %errorlevel% neq 0 ( echo ERROR: Failed to create Cloud Storage bucket. & goto :CleanupError )

echo  ...Uploading requirements.txt, script, and data file...
CALL gsutil -q cp "%LOCAL_REQUIREMENTS_PATH%" gs://%BUCKET_NAME%/requirements.txt
CALL gsutil -q cp "%LOCAL_SCRIPT_PATH%" gs://%BUCKET_NAME%/%REMOTE_SCRIPT_NAME%
CALL gsutil -q cp "%LOCAL_DATA_FILE_PATH%" gs://%BUCKET_NAME%/data.xlsx
if %errorlevel% neq 0 ( echo ERROR: Failed to upload files to bucket. & goto :CleanupError )

echo Creating VM and starting simulation...
CALL gcloud compute instances create %VM_NAME% ^
    --project=%PROJECT_ID% ^
    --zone=%VM_ZONE% ^
    --machine-type=%MACHINE_TYPE% ^
    --image-family=%VM_IMAGE_FAMILY% ^
    --image-project=%VM_IMAGE_PROJECT% ^
    --boot-disk-size=30GB ^
    --scopes=https://www.googleapis.com/auth/cloud-platform ^
    --metadata-from-file=startup-script="%STARTUP_SCRIPT_PATH%" ^
    --metadata=BUCKET_NAME=%BUCKET_NAME%,SCRIPT_NAME=%REMOTE_SCRIPT_NAME%,RESULTS_FILE=%REMOTE_RESULTS_FILENAME%,LOG_FILE=%REMOTE_LOG_FILE%
    
if %errorlevel% neq 0 ( echo ERROR: Failed to create Compute Engine VM. & goto :CleanupError )

echo.
echo ==========================================================================
echo.
echo  SUCCESS: VM is created. It will remain running for debugging.
echo  Please follow the manual steps to retrieve the log and delete the VM.
echo.
echo ==========================================================================
goto :CleanupSuccess

:CleanupError
echo. & echo An error occurred during setup.
:CleanupSuccess
del "%STARTUP_SCRIPT_PATH%" 2>nul
:Finish
echo.
endlocal