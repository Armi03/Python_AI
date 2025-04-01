from matplotlib import pyplot as plt
import numpy as np


#####==== Exercise01 ====#####

# Define the x range
x = np.linspace(0, 10, 100)

# Define the equations
y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3

# Plot the lines with different styles
plt.plot(x, y1, linestyle='-', color='black', label='y = 2x + 1')  # Solid black line
plt.plot(x, y2, linestyle='--', color='gray', label='y = 2x + 2')  # Dashed gray line
plt.plot(x, y3, linestyle='-.', color='dimgray', label='y = 2x + 3')  # Dash-dot dimgray line

# Labels and title
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Graph of y for different values of b")

# Add a legend
plt.legend()
plt.show()


#####==== Exercise02 ====#####

# Create the vectors x and y
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])

# Create the scatter plot
plt.scatter(x, y, marker='+', color='red')
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("Scatter Plot of Given Points")
plt.grid(True)
plt.show()



#####==== Exercise03 ====#####

# Load data, handling mixed types and skipping headers
filename = "weight-height.csv"
data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(1, 2))  # Assuming length and weight are in cols 1 & 2

# Extract length (height) and weight columns
length = data[:, 0]  # Height in inches
weight = data[:, 1]  # Weight in pounds

# Convert lengths to centimeters and weights to kilograms
length_cm = length * 2.54  # 1 inch = 2.54 cm
weight_kg = weight * 0.453592  # 1 pound = 0.453592 kg

# Calculate means
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean length (cm): {mean_length:.2f}")
print(f"Mean weight (kg): {mean_weight:.2f}")

# Plot histogram of lengths
plt.hist(length_cm, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Length (cm)")
plt.ylabel("Frequency")
plt.title("Histogram of Student Lengths")
plt.grid(True)
plt.show()



#####==== Exercise04 ====#####

A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
A_inv = np.linalg.inv(A)

# Verify the identity matrix by computing A * A_inv and A_inv * A
identity1 = np.dot(A, A_inv)
identity2 = np.dot(A_inv, A)

# Print results
print("A * A_inv (should be close to identity matrix):")
print(identity1)
print("\nA_inv * A (should be close to identity matrix):")
print(identity2)