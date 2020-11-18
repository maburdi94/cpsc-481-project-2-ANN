


filename = input("Input file: ")
file = open(filename or "input.txt", "r")
print(file.read())




file.close()
