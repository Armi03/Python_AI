def can_administer_medicine():
    try:
        age = int(input("Enter the patient's age: "))

        if age >= 18 or (15 <= age < 18 and float(input("Enter the patient's weight (kg): ")) >= 55):
            print("Medicine can be given.")
        else:
            print("Medicine cannot be given.")
    except ValueError:
        print("Invalid input. Please enter numeric values for age and weight.")


if __name__ == "__main__":
    can_administer_medicine()
