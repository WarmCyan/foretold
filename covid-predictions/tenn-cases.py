import numpy as np
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.style.use('dark_background')

with open("./tenn-case-data.json", 'r') as infile:
    data = json.load(infile)

def simulate(variance=(.01, .01), forecast_day=30):
    x = np.array(data["day"])
    y = np.array(data["cases"])

    def f(x, a, b):
        return a * np.exp(b*x)

    popt, pcov = curve_fit(f, x, y)

    plt.figure()
    plt.grid()
    plt.scatter(x, y)
    plt.plot(x, f(x, *popt))
    plt.title("Current data")

    high = np.copy(popt)
    high[1] += variance[1]
    low = np.copy(popt)
    low[1] -= variance[0]
    plt.plot(x, f(x, *high))
    plt.plot(x, f(x, *low))

    x2 = np.append(x, list(range(len(x) - 1, forecast_day+1)))
    plt.figure()
    plt.grid()
    plt.scatter(x, y)
    plt.plot(x2, f(x2, *popt))
    plt.plot(x2, f(x2, *high))
    plt.plot(x2, f(x2, *low))
    plt.title("Forecast")

    print(f(forecast_day, *high))
    print(f(forecast_day, *popt))
    print(f(forecast_day, *low))

simulate((.04, .01))
