# milton_da on Windows (C:\Users\taylo\milton_da)

The package folder itself is `milton_da`, so every command runs from its PARENT, `C:\Users\taylo`,
with `python -m milton_da....`. Nothing needs to be pip-installed as a package.

## 1. One-time setup
Double-click `milton_da\setup_windows.bat` (or run it from a terminal). It creates `C:\Users\taylo\venv`,
installs torch (CPU build; edit the index URL for CUDA), installs the requirements, and runs the 9 tests.

## 2. Credentials (two small text files in C:\Users\taylo)

`C:\Users\taylo\_netrc`   (underscore, no extension; Notepad: "Save as type: All files")

    machine urs.earthdata.nasa.gov login YOUR_EARTHDATA_USERNAME password YOUR_EARTHDATA_PASSWORD

Then log in at https://urs.earthdata.nasa.gov -> Applications -> Authorized Apps and approve
"NASA GESDISC DATA ARCHIVE". Without this, ATMS and IMERG downloads return HTTP 401.

`C:\Users\taylo\.cdsapirc`   (needs a CDS account: https://cds.climate.copernicus.eu)

    url: https://cds.climate.copernicus.eu/api
    key: YOUR_CDS_PERSONAL_ACCESS_TOKEN

Open https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels once, go to the
Download tab, and accept the licence at the bottom. The token is on your profile page.

If you prefer environment variables to `_netrc`:
    setx EARTHDATA_USERNAME your_user
    setx EARTHDATA_PASSWORD your_pass
(open a new terminal afterwards).

## 3. Smoke test, then the case
Double-click `milton_da\run_smoke.bat`. It does a dry run (track only) and then one full analysis time.
The final line of the script prints the command for all 16 Milton times.

## 4. Windows notes
* DataLoader workers: keep `num_workers = 0` on Windows (the default TrainConfig has 4; set
  `cfg.train.num_workers = 0` in any training script you write, or run training under WSL2).
* Long paths: if you see "path too long", enable long paths in Windows or use a short --root such as D:\milton.
* Disk: Milton case < 1 GB; the training archive needs ~100 GB, so use --root on a drive with space.
* GPU: training the prior needs one. Build scenes here, upload data\milton\scenes and data\archive\scenes
  to a rented GPU (Colab Pro, Lambda, RunPod), train there, bring unet.pt and score.pt back.
