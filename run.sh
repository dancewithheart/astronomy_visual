python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas plotly astropy astroquery
pip install pyarrow
pip install pyvista imageio
python gaia_orion_flythrough.py
