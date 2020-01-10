# Import Modules
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load CSV File (Data)
data = pandas.read_csv('../cost-revenue-clean.csv')

# Config Graph
X = DataFrame(data, columns=['production_budget_usd'])
Y = DataFrame(data, columns=['worldwide_gross_usd'])

# Linear Regression Init
regression = LinearRegression()
regression.fit(X, Y)

# Draw the Graph
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.3)
plt.plot(X, regression.predict(X), color='red', linewidth=4)

plt.title('Film Cost Vs. Global Revenue')
plt.xlabel('Production Cost (USD)')
plt.ylabel('Revenue (USD)')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()



