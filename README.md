# C(++)ECCO
C++ Error Control COding (CECCO) is a header-only library for ECC simulations and experiments, modeling complete coding systems for the rationals and arbitrary finite fields, dealing with complex inter-field relationships. CECCO provides an intuitive, object-oriented interface for fields, vectors, polynomials, matrices, and many linear error-correcting codes (Hamming, Repetition, Cordaro-Wagner, Simplex, Single-Parity-Check, Golay, Reed-Muller, Generalized Reed-Solomon, Bose-Chaudhuri-Hocquenghem, Goppa, Alternant, Convolutional) with a wide range of hard- and soft input general and specialized decoders ((List-)Viterbi, BCJR, BP, recursive, standard array, Meggitt, FHWT, GMD, (LC-)OSD, Welch-Berlekamp, Berlekamp-Massey, Guruswami-Sudan, Koetter-Vardy, etc., including error-erasure variants). Its low-level functionality can be used to implement arbitrary linear codes and their decoders, while its high-level functionality is convenient for creating complete simulation chains. Text/LaTeX/TiKZ export and print functionalities are useful for writing technical reports, theses, and papers. 

Source code is available on Github <a href="https://github.com/christiansenger/cecco">here</a>.

A growing <a href="https://cecco.senger.eu/#demos">collection of demos and examples</a> is available on the <a href="https://cecco.senger.eu">project website</a>.

Detailed Doxygen documentation is available <a href="https://christiansenger.github.io/cecco/doxygen/html">here</a>.

CECCO requires C++20. It is developed and tested with GCC 15 and Apple Clang 21. Older compilers may work, but are not part of the supported compatibility target. Clang 14 and older are not supported.

In case you want to use CECCO for your research or teaching: Please cite the following DOI (a BibTeX exporter is behind the DOI link, on the lower right).

<a href="https://doi.org/10.5281/zenodo.15685869"><img src="https://zenodo.org/badge/1003774077.svg" alt="DOI"></a>

<p align="center">
<a href="https://github.com/christiansenger/cecco"><img src="https://raw.githubusercontent.com/christiansenger/cecco/refs/heads/main/.github/assets/cecco.png" height="200" width="200" ></a>
</p>

## License

This software, authored by Christian Senger, is licensed under a modified BSD License for **noncommercial use only**. You may use, modify, and redistribute this software for teaching, academic research, and personal non-profit purposes. **Commercial use is strictly prohibited** without a separate commercial license.

For details, see the [LICENSE](./LICENSE) file.

To request a commercial license, contact the author at [senger@inue.uni-stuttgart.de](mailto:senger@inue.uni-stuttgart.de).

![License: Noncommercial-Only](https://img.shields.io/badge/license-noncommercial--only-red)

The file InfInt.hpp, originally authored by Sercan Tutar, is under MPL 2.0 (see [LICENSE-MPL](./LICENSE-MPL)).
