import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.array([2020, 2021, 2022, 2023, 2024, 2025])
y_cel = np.array([7, 8, 9, 10, 14, 20])

# Convert Celsius to Fahrenheit
y_fahrenheit = (y_cel * 9/5) + 32

# First subplot - Fahrenheit
plt.subplot(1, 3, 1)
plt.plot(x, y_fahrenheit, 'bo-', label='Fahrenheit')
plt.xlabel('Years')
plt.ylabel('Temperature (°F)')
plt.title('Temperature in Fahrenheit')

plt.legend()

# Second subplot - Celsius
plt.subplot(1, 3, 2)
plt.plot(x, y_cel, 'r*-', label='Celsius')
plt.xlabel('Years')
plt.ylabel('Temperature (°C)')
plt.title('Temperature in Celsius')
plt.legend()


# Third subplot - Kelvin
plt.subplot(1, 3, 3)
plt.plot(x, y_cel, 'bo-', label='Kelvin')
plt.xlabel('Years')
plt.ylabel('Temperature (°K)')
plt.title('Temperature in Kelvin')
plt.legend()

plt.savefig('Week02_practice.png')

# Show the plot
plt.show()
