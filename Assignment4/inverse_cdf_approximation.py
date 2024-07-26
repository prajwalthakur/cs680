import torch
Zs = torch.arange(-5, 5, 0.1)

length_Zs = len(Zs)

print(length_Zs)

# def inverse_cdf_normal(p, tol=1e-6, max_iter=100):
#   """
#   Approximates the inverse CDF of the standard normal distribution using the bisection method.

#   Args:
#     p: A tensor of probabilities in the range [0, 1].
#     tol: Tolerance for the approximation.
#     max_iter: Maximum number of iterations for the bisection method.

#   Returns:
#     A tensor of the same shape as p containing the approximate z-scores.
#   """

#   def norm_cdf(x):
#     two_  = torch.tensor(2, dtype=torch.int8)
#     return 0.5 * (1 + torch.erf(x / two_))

#   # Handle potential numerical issues near 0 and 1
#   p = torch.clamp(p, 1e-9, 1 - 1e-9)

#   # Initialize lower and upper bounds
#   lower = torch.zeros_like(p) - 5
#   upper = torch.zeros_like(p) + 5

#   for _ in range(max_iter):
#     mid = (lower + upper) / 2
#     cdf_mid = norm_cdf(mid)
#     lower = torch.where(cdf_mid < p, mid, lower)
#     upper = torch.where(cdf_mid >= p, mid, upper)

#   return (lower + upper) / 2

# # Example usage
# p = torch.tensor([0.5, 0.95])
# z = inverse_cdf_normal(p)
# print(z)
