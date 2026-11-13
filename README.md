# Quark matter in neutron-star cores

Numerical study of cold hybrid neutron stars using the CompOSE RG(SLy4) hadronic equation of state and a three-flavour, massless MIT bag model.

The code computes thermodynamic consistency checks, Maxwell-transition candidates, Tolman–Oppenheimer–Volkoff (TOV) equilibrium sequences, and selected fundamental radial oscillation modes for rapid and slow phase conversion.

## Run

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python run_analysis.py
```

To reproduce only the equation-of-state and TOV tables, without the slower radial-mode calculations:

```bash
python run_analysis.py --skip-radial
```

Generated tables and figures are written to `results/`. The input CompOSE table used in the investigation is included in `data/rg_sly4_compose.zip` (thermodynamic and grid files from [CompOSE RG(SLy4) #134](https://compose.obspm.fr/eos/134)).

## Structure

- `quarkmatter/model.py`: CompOSE reader, hadronic and quark equations of state, Maxwell crossings, TOV solver.
- `quarkmatter/radial.py`: stellar background, relativistic radial perturbations, rapid/slow interface matching, eigenvalue search.
- `run_analysis.py`: parameter sweep, thermodynamic checks, CSV results and graphs.
- `tests/`: numeric regression and physical consistency tests.
- `original_code/`: preserved source scripts from the research archive for exact comparison with the original figures and audit tables.
- `original_results/`: CSV tables and figures supplied with the research archive.

## Tests

```bash
python -m unittest discover -s tests -v
```

The refactored main implementation was checked against the original phase transition pressures, TOV equilibrium masses and a reference radial-mode calculation. The archived scripts remain available for reproducing the exact layout and additional checks used during the research.

## Assumptions and interpretation

- The quark model has massless `u`, `d`, `s` quarks with no interactions beyond the bag constant. The scanned values are `B^(1/4) = 145, 155, 165, 175, 185, 190 MeV`.
- The Maxwell crossing is sought only within the chemical-potential interval of the hadronic table. At 145 and 155 MeV, quark matter is favoured across this interval; these are not conventional hadronic-envelope hybrid-star solutions in this model.
- The 165 MeV solution crosses at exceptionally low pressure, producing unrealistic low-density configurations. It is retained as a formal model result, not evidence for a physically stable hybrid star.
- TOV solutions describe **hydrostatic equilibrium**, not necessarily **dynamical stability**. Radial modes are evaluated only for selected central pressures. A missing eigenvalue is marked `unresolved`, not `stable`.
- Stellar surfaces are set at the minimum **positive** pressure in the CompOSE table, rather than at exactly zero pressure. Radial-mode eigenvalues also depend on the chosen outer cutoff and radial resolution.
- The stability analysis is exploratory and is not an independent observational validation or an exhaustive scan of every equilibrium branch.

## References

- CompOSE: [RG(SLy4) EOS #134](https://compose.obspm.fr/eos/134).
- Farhi & Jaffe (1984), *Strange matter*, Phys. Rev. D 30, 2379.
- Alford, Han & Prakash (2013), *Generic conditions for stable hybrid stars*, Phys. Rev. D 88, 083013.
- Pereira, Flores & Lugones (2018), *Phase transition effects on the dynamical stability of hybrid neutron stars*, ApJ 860, 12.
