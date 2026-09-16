
# tech stack

* [Astropy](https://docs.astropy.org/en/stable/coordinates/index.html) for coordinates, units, tables, and astronomy-aware transforms.
* Plotly for an interactive browser-based 3D version.
* PyVista for the prettier rendered version, especially if you want glowing clouds, volume effects, or a cinematic camera path.
* Glue is excellent if you want exploratory linking between views, though it is more of an analysis GUI than a polished final “showpiece.”

## 3D Gaia-based stellar-nursery fly-through, rendered with Plotly or PyVista, enhanced with dust volumes and labels

https://www.esa.int/ESA_Multimedia/Videos/2025/09/The_most_accurate_3D_map_of_stellar_nurseries_in_the_Milky_Way

* on Ubuntu
```
sudo apt-get install -y libgl1 libglx-mesa0 libegl1
```

```
python3 -m venv .venv
source .venv/bin/activate
```

```
pip install numpy pandas plotly astropy astroquery
pip install pyarrow
```

python gaia_orion_flythrough.py

```
pip install pyvista imageio
```
python gaia_orion_pyvista.py



pip install "imageio[ffmpeg]"
pip install tqdm