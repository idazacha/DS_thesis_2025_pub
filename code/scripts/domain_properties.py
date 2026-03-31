from scipy.stats import skew, kurtosis

def sparsity_report(df):
    zero_frac = (df == 0).sum() / len(df)
    print("Fraction zeros per feature:\n", zero_frac.sort_values(ascending=False))
    print("Overall sparsity:", (df==0).sum().sum() / df.size)


def distribution_report(df):
    print("Skewness:\n", df.apply(skew))
    print("Kurtosis:\n", df.apply(kurtosis))

