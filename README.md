# LitMove: looking at how Li-ions diffuse - one dance at a time!

## Documentation
For code packages documentation check this: [Atomic Positionism](https://atomic-positionism.netlify.app/)

## Description
> TLDR: **Stability is a mess, be chaotic!**

A work in progress as a part of master's thesis identifying atoms' diffusion; behavior in interacting between themselves – how an atom prefers to stay within/ join a group instead of the others.

## Input
### 1. `CONTCARs/ POSCARs` file of VASP results
that's located within the directory `Init_System` and arranged inside as `/geometry/path/file`, whose numeration starts from 0. Its information is then stored as a DataFrame.
| Geo | Path    | Location         |
| --- | ------- | ---------------- |
| 0   | 0       | "../0/0/CONTCAR" |
|     | ..      | ..               |
|     | n       | "../0/n/CONTCAR" |
| n   | 0       | "../n/0/CONTCAR" |
|     | ..      | ..               | 
|     | n       | "../n/n/CONTCAR" |

The collection of VASP and NEB results is stored in [a repository called "CONTCARs"](https://github.com/a-firdaus/CONTCARs.git).

### 2. Excel file of total energy `toted_final.ods`
which then is read by the software and stored in the DataFrame as an additional column `toten [eV]`.


## Output
- Distribution of Li-ion position based on Wyckoff sites
- Diffusion mechanism undergone by Li-ions: intratriad, intracage, and intercage
