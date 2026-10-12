# Certified anisotropic meshing of thin, high-aspect-ratio regions

Scripts behind every figure, table and number of

> H. Talleb, A. Elomari, M. Ben Chouikha, *Certified anisotropic meshing of thin,
> high-aspect-ratio regions in device modeling*, submitted to the Journal of
> Computational Electronics (2026).

A thin layer is resolved on anisotropic structured cells (Kuhn tetrahedra), and the
two-terminal resistance is enclosed between a conforming P1 bound and an equilibrated
RT0 (Raviart-Thomas) bound computed on the same mesh.

## Requirements
Python 3.10+, NumPy, SciPy, Matplotlib (`pip install -r requirements.txt`).

## Reproduce
| Command | Produces | Time |
|---|---|---|
| `python3 jce_numbers.py` | `jce_numbers.json`: point-source deviations (Sect. 3.2), stack drops and resistances (Sect. 3.4) | < 1 min |
| `python3 jce_brackets.py` | `jce_brackets.json`: certified enclosures of Table 3 (Sect. 3.3, 3.4) | about 25 min (`--quick` skips the 144 000-tetrahedron mesh) |
| `python3 make_figures_jce.py` | `figures/Fig1-7.{pdf,eps}` and `jce_aspect.json` (Table 2) | a few minutes |

The JSON files of the published run are included.

## Files
- `dgm/`: minimal subset of the DGM engine (tetrahedral mesh, P1 assembly, RT0 mixed assembly).
- `conductance3d.py`: P1 upper and RT0 lower bounds on a two-terminal conductance.
- `bilayer_certified.py`: film-on-body mesh and the parallel-sheet benchmark (problems 1 and 1').
- `certified_bracket.py`: three-layer stack mesh (problem 4).
- `jce_brackets.py`: enclosures with a square contact lying on grid lines for every mesh (problems 3 and 4).
- `pointsource_certified.py`: axisymmetric point source on a film / substrate and its image series (problem 2).
- `structure_certified.py`: axisymmetric three-layer stack (problem 4, Fig. 6).
- `band_diagram.py`: 1D Poisson-Boltzmann equilibrium band diagram (Fig. 5).
- `verify_refs_jce.py`: Crossref check of the references.

The scripts were written with the help of a generative AI coding assistant; they are
deterministic and were checked by the authors.

## License
MIT, see `LICENSE`.
