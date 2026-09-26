@echo off
title Push BraTS 2021 Task 2 (MGMT Radiogenomics) to GitHub
echo ======================================================================
echo Pushing BraTS 2021 Task 2 Research Repository to GitHub
echo Remote: https://github.com/Tarekswd/brats2021-mgmt-radiogenomics.git
echo Author: Tarek Al Ahmadieh
echo ======================================================================
echo.

cd /d "c:\Users\Admin\Desktop\brats2021\brats2021_research\brats2021_task2"

echo Staging all Task 2 files...
git add .

echo Checking for changes to commit...
git commit -m "BraTS 2021 Task 2 MGMT Radiogenomics 500M Foundation Model framework by Tarek Al Ahmadieh" 2>nul

echo.
echo Running: git push -u origin main
echo (If prompted, click 'Sign in with your browser' to authorize)
echo.

git push -u origin main

echo.
if %ERRORLEVEL% equ 0 (
    echo ======================================================================
    echo SUCCESS: All Task 2 files pushed to:
    echo https://github.com/Tarekswd/brats2021-mgmt-radiogenomics
    echo ======================================================================
) else (
    echo ======================================================================
    echo Push encountered an error or was cancelled.
    echo If prompted for authentication, ensure you sign in with your GitHub account.
    echo ======================================================================
)

pause
