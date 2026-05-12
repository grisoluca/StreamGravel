# StreamGravel

Streamlit app for neutron spectrum unfolding with GRAVEL and MLEM.

## Run locally

Install the dependencies and start the app from the repository root:

```powershell
pip install -r requirements.txt
streamlit run StreamGravel.py
```

`StreamGravel.py` is intentionally kept in the repository root so Streamlit Cloud and the devcontainer can keep using the same entrypoint.

## Repository layout

```text
StreamGravel.py          Streamlit app entrypoint
gravel.py                GRAVEL unfolding algorithm
mlem.py                  MLEM unfolding algorithm
rebin.py                 Rebinning helper
response_matrix.py       Response matrix and uploaded counts loader
requirements.txt         Python dependencies for Streamlit
LogoNML-Black.png        Streamlit page icon, kept at root for compatibility
examples/                Downloadable and reference example input files
unfolding_inputs/        Working input datasets and analysis material
.devcontainer/           Codespaces/devcontainer configuration
```

## Input files

The app expects four uploaded text files:

- Response matrix: tab-separated response functions.
- Measured counts: first column counts, optional second column relative uncertainty.
- Energy bins: left edge, right edge, and central energy.
- Initial guess spectrum: first column energy in MeV, second column differential spectrum `dPhi/dE`.

If the guess spectrum is provided per unit lethargy, enable the corresponding checkbox so the app converts it before unfolding.

## Streamlit path notes

The app keeps the Streamlit entrypoint and icon in the repository root:

- `StreamGravel.py`
- `LogoNML-Black.png`

The downloadable LINAC example is loaded from:

- `examples/example-LINAC.zip`

If these files are moved again, update the path constants near the top of `StreamGravel.py` before deploying.
