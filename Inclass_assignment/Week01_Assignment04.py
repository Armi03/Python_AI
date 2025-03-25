def calculate_grade():
    try:
        score = float(input("Enter the student's score: "))

        grade = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'D' if score >= 60 else 'E'
        print(f"The student's grade is: {grade}")
    except ValueError:
        print("Invalid input. Please enter a numeric score.")

if __name__ == "__main__":
    calculate_grade()

