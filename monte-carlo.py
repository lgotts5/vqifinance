import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker = "AAPL"
years = 1
num_sims = 1000
steps_per_year = 252

data = yf.download(ticker, period="3y")

data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
mu = data["log_return"].mean() * steps_per_year
sigma = data["log_return"].std() * np.sqrt(steps_per_year)
S0 = float(data["Close"].iloc[-1])  # make sure it’s a float

print(f"Current {ticker} price: ${S0:.2f}")
print(f"Estimated annual return (mu): {mu:.2%}")
print(f"Estimated volatility (sigma): {sigma:.2%}")

# Monte Carlo Simulation
dt = 1 / steps_per_year
S = np.zeros((steps_per_year * years, num_sims))
S[0] = S0

for t in range(1, len(S)):
    Z = np.random.standard_normal(num_sims)
    S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

plt.figure(figsize=(10, 6))
plt.plot(S[:, :50], lw=1, alpha=0.5)
plt.title(f"Monte Carlo Simulation of {ticker} Stock Prices ({num_sims} paths, {years} year)")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.grid(True)

final_prices = S[-1]
mean_final = final_prices.mean()
p5 = np.percentile(final_prices, 5)
p95 = np.percentile(final_prices, 95)
plt.axhline(mean_final, color='black', linestyle='--', label=f"Mean: ${mean_final:.2f}")
plt.axhline(p5, color='red', linestyle='--', label=f"5%: ${p5:.2f}")
plt.axhline(p95, color='green', linestyle='--', label=f"95%: ${p95:.2f}")
plt.legend()
plt.show()

print("\n--- Summary ---")
print(f"Expected final price: ${mean_final:.2f}")
print(f"5th percentile (pessimistic): ${p5:.2f}")
print(f"95th percentile (optimistic): ${p95:.2f}")