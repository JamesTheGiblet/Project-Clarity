// Simple function with Result type
function divide(a, b):
    if b == 0:
        return Error("Cannot divide by zero")
    else:
        return Success(a / b)

let result = divide(10, 2)
print("Division result:", result)

match result:
    Success -> print("Success!")
    Error -> print("Failed!")
    _ -> print("Unknown")

test "division works":
    let good = divide(6, 2)
    assert good == Success(3)
