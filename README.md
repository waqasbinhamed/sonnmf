# Nonnegative Matrix Factorization with Sum-of-Norms Clustering Regularization (NMF-SON)

NMF-SON is a modified Nonnegative Matrix Factorization method that uses a reglarization term that minimizes the sum of norms between the columns of the basis matrix. This regularization eliminates the need to specify the ideal rank (model order) for NMF beforehand. 

To learn more about NMF-SON please refer to https://uwaterloo.ca/computational-mathematics/sites/ca.computational-mathematics/files/uploads/files/waqas_bin_hamed_research_paper.pdf. Note that the mentioned research paper desribes an old version of NMF-SON, and the method currently being further improved.

### Installation

This package is still in development, so you need to install it as an experimental package. To install:

1. Clone repository.
2. Navigate to the main directory `nmf_method`.
3. Run `pip install -r requirements.txt` to install dependencies. Note that the packages required to run the notebooks are commented.
4. Run `pip install -e .`.

### Notes

- This package is a work in progress. Apologizes for any bugs.
- The `experimental` directory contains files related to ongoing improvements to the NMF methods and the package. 
- Please feel free to email me at waqasbinhamed@gmail.com for any concerns related to this package.
