age = int(input("Enter your age: "))  
ask_nationality = input("Do you want to enter your nationality? (yes/no): ").strip().lower()  

if age < 18:  
    if ask_nationality == "yes":  
        print("You are a minor and not eligible for nationality.")  
    else:  
        print("You are a minor.")  
else:  
    if ask_nationality == "yes":  
        nationality = input("Enter your nationality: ")  
        print("You are eligible to cast a vote.")  
    else:  
        print("You chose not to enter nationality.")  
