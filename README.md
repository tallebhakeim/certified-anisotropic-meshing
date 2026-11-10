# Guaranteed bounds on the simulated resistance of thin-layer device structures

Scripts behind every figure, table and number of

> H. Talleb, A. Elomari, M. Ben Chouikha, *Guaranteed bounds on the simulated resistance of thin-layer device structures: contact resistance on thin films and isolation sheets*, submitted to the Journal of
> Computational Electronics (2026).

On one mesh, a potential solution (P1) gives a lower bound of a two-terminal resistance and a
current solution (P1 stream function in 2D, Raviart-Thomas RT0 face currents in 3D) gives an
upper bound. The exact resistance of the model lies between the two numbers, whatever the shape
of the cells.

## Citation
Archived on Zenodo: concept DOI https://doi.org/10.5281/zenodo.24117327 (all versions); version 1.0.0: https://doi.org/10.5281/zenodo.24117328.

## Requirements
Python 3.10+, NumPy, SciPy, Matplotlib (`pip install -r requirements.txt`).

## Reproduce
| Command | Produces | Time |
|---|---|---|
| `python3 step_example.py` | `jce_step.json`: step conductor checked by hand (Table 1, Fig. 2) | a few seconds |
| `python3 jce_contact.py` | `jce_contact.json`: contact resistance on a thin film (Tables 4 to 6, Figs. 5 and 6) | about 1 min |
| `python3 jce_numbers.py` | `jce_numbers.json`: potential drop across the sheet of the three-layer stack (Sect. 4.2) | < 1 min |
| `python3 jce_brackets.py` | `jce_brackets.json`: bounds of the stack in 3D (Table 7, Fig. 9) | about 25 min (`--quick` skips the largest mesh) |
| `python3 make_figures.py` | `figures/Fig1-9.{pdf,eps}` and `jce_aspect.json` (Table 2) | a few minutes |

The JSON files of the published run are included.

## Files
- `bounds2d.py`: the two bounds in 2D (structured grid, P1 potential, P1 stream function). About 80 lines.
- `step_example.py`: step conductor, bounds 1.5 and 2 by hand, computed intervals.
- `tlm_contact.py`, `jce_contact.py`: contact on a thin film with a 1 nm interfacial layer; comparison with the
  one-dimensional transfer-length formula; refinement sequences G and E.
- `dgm/`: minimal 3D engine (tetrahedral mesh, P1 assembly, RT0 mixed assembly).
- `conductance3d.py`: P1 and RT0 bounds of a two-terminal conductance in 3D.
- `bilayer_certified.py`: film on a body, layer-parallel flow (Table 2, Fig. 4).
- `certified_bracket.py`, `jce_brackets.py`: three-layer stack in 3D, square contact on grid lines.
- `structure_certified.py`: axisymmetric three-layer stack (Fig. 8).
- `band_diagram.py`: 1D Poisson-Boltzmann equilibrium band diagram (Fig. 7).
- `pointsource_certified.py`: point source on a film over a substrate against its image series (additional check, not in the article).
- `verify_refs_jce.py`: Crossref check of references.

The scripts were written with the help of a generative AI coding assistant; they are
deterministic and were checked by the authors.

## License
MIT, see `LICENSE`.
