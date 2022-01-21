import numpy as np


def pca_variance(X, num_dims=None, flip=True, verbose=False, frac=0.99):

    _, singvals, _ = np.linalg.svd(X)
    singvals = np.sort(singvals)

    if flip:
        singvals = np.flip(singvals)

    fraction_variance_explained = np.cumsum(singvals) / np.sum(singvals)
    truncation = np.argmax(fraction_variance_explained > frac)

    if verbose:
        print(
            "Retaining {} of {} components required for 99\% variance".format(
                truncation, X.shape[1]
            )
        )

    return fraction_variance_explained, truncation
