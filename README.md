# Clarity Programming Language

*A language designed to make wrong code unwritable*

## Philosophy

Clarity is built on the principle that **code should express intent, not implementation details**. It eliminates entire categories of bugs through language design rather than runtime checking.

## Core Principles

### 1. Explicit Intent
- No hidden side effects
- No implicit type conversions
- All state changes are visible at the call site
- Error conditions must be handled

### 2. Immutability by Default
- All values are immutable unless explicitly marked
- Mutation is scoped and validated
- Shared mutable state requires explicit synchronization

### 3. Correctness Through Types
- The type system prevents invalid states
- Pattern matching ensures completeness
- Contracts are enforced at compile time

### 4. Progressive Disclosure
- Start simple, add complexity as needed
- Defaults handle common cases safely
- Advanced features don't complicate basic usage

## Language Features

### Type System

```clarity
// Basic types
Number, String, Boolean, Byte
List<T>, Map<K,V>, Set<T>
Optional<T>  // Never null, either Some(value) or None

// Custom types
type User = {
    name: String,
    age: Number,
    email: String
}

// Union types for modeling real-world states
type PaymentStatus = Pending | Processed | Failed(reason: String)

// Constrained types
type PositiveNumber = Number where value > 0
type NonEmptyString = String where length > 0
```

### Error Handling

```clarity
// All functions that can fail return Result<Success, Error>
function divide(a: Number, b: Number) -> Result<Number, MathError>:
    match b:
        0 -> Error(DivisionByZero)
        _ -> Success(a / b)

// Error propagation operator
function calculate_average(numbers: List<String>) -> Result<Number, ParseError>:
    let parsed = numbers.map(s -> parse_number(s)?)  // ? propagates errors
    Success(parsed.sum() / parsed.length)

// Multiple error types
function process_user_input(input: String) -> Result<User, ValidationError | ParseError>:
    let data = parse_json(input)?          // ParseError
    let user = validate_user(data)?        // ValidationError
    Success(user)
```

### Pattern Matching

```clarity
// Exhaustive pattern matching
function handle_response(response: HttpResponse) -> String:
    match response.status:
        200 -> "Success"
        404 -> "Not found"
        500 -> "Server error"
        code -> "Unknown status: {code}"  // Catch-all required

// Destructuring patterns
match user:
    User { name: "admin", ... } -> grant_admin_access()
    User { age, ... } where age < 18 -> require_parental_consent()
    User { email, ... } where email.ends_with(".edu") -> apply_student_discount()
    _ -> standard_processing()
```

### Memory Management

```clarity
// Ownership and borrowing (like Rust but more ergonomic)
function process_data(data: List<String>) -> ProcessedData:
    let processed = data
        .borrow()           // Explicit borrow
        .map(transform)     // Pure transformation
        .collect()          // New owned data
    
    // 'data' is still valid here
    log("Processed {} items", data.length)
    processed

// Reference counting for shared data
let shared_config = Rc::new(load_config())
spawn_worker(shared_config.clone())  // Explicit clone
```

### Concurrency

```clarity
// Async/await with explicit error handling
async function fetch_user_data(id: UserId) -> Result<UserData, NetworkError>:
    let response = await http_get("/users/{id}")?
    let data = await response.json()?
    Success(data)

// Channels for communication
let (sender, receiver) = channel<WorkItem>()

async function worker():
    for item in receiver:
        match process(item):
            Success(result) -> log("Processed: {result}")
            Error(err) -> log("Failed: {err}")

// Parallel processing
let results = data
    .par_chunks(100)        // Parallel iteration
    .map(process_chunk)     // Pure function required
    .collect()
```

### State Management

```clarity
// Immutable by default
let user = User { name: "Alice", age: 25 }
// user.age = 26  // Compile error!

// Explicit mutation scopes
function update_user_age(user: User, new_age: Number) -> User:
    user.with { age: new_age }  // Creates new instance

// Mutable variables when needed
let mut counter = 0
counter += 1  // OK

// Controlled mutation with validation
function transfer_money(mut account: BankAccount, amount: Money) -> Result<(), TransferError>:
    ensure!(amount > 0, "Amount must be positive")
    ensure!(account.balance >= amount, "Insufficient funds")
    
    with mut account:
        account.balance -= amount
        account.validate()?  // Required after mutation
    
    Success(())
```

### Testing Integration

```clarity
function fibonacci(n: Number) -> Number:
    match n:
        0 -> 0
        1 -> 1
        _ -> fibonacci(n-1) + fibonacci(n-2)

// Tests live alongside code
test "fibonacci basics":
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5

test "fibonacci properties":
    for n in 1..10:
        assert fibonacci(n) >= fibonacci(n-1)  // Monotonic growth

// Property-based testing
test "addition is commutative" with random(a: Number, b: Number):
    assert a + b == b + a
```

### Module System

```clarity
// Explicit exports
module MathUtils:
    export function gcd(a: Number, b: Number) -> Number
    export type Fraction = { numerator: Number, denominator: Number }
    
    // Private helper
    function simplify(frac: Fraction) -> Fraction:
        let g = gcd(frac.numerator, frac.denominator)
        Fraction { 
            numerator: frac.numerator / g,
            denominator: frac.denominator / g 
        }

// Explicit imports
import { gcd, Fraction } from MathUtils
import HttpClient from WebUtils
```

### Contracts and Invariants

```clarity
// Preconditions and postconditions
function binary_search<T>(arr: List<T>, target: T) -> Optional<Number>
    requires: arr.is_sorted()
    ensures: result.is_some() -> arr[result.unwrap()] == target:
    
    // Implementation here
    // Compiler checks that postcondition holds

// Type invariants
type SortedList<T> = List<T>
    invariant: self.is_sorted()

// State machine types
state_machine FileHandle:
    Closed -> open() -> Opened
    Opened -> read() -> Opened  
    Opened -> write(data) -> Opened
    Opened -> close() -> Closed
    
    // Calling read() on Closed = compile error
```

## Advanced Features

### Metaprogramming

```clarity
// Compile-time code generation
macro derive_json for type T:
    function to_json(self: T) -> Json:
        // Generated based on T's structure
    
    function from_json(json: Json) -> Result<T, ParseError>:
        // Generated parser

type User = { name: String, age: Number }
    derive: [Json, Equals, Debug]
```

### Effect System

```clarity
// Track side effects in types
function pure_computation(x: Number) -> Number:
    x * 2  // No effects

function log_and_compute(x: Number) -> Number with IO:
    print("Computing for {x}")  // IO effect
    x * 2

function network_fetch(url: String) -> Result<String, NetworkError> with IO, Network:
    // Both IO and Network effects
```

## Standard Library

The standard library provides safe, efficient implementations of common data structures and algorithms, all following Clarity's principles of explicit intent and error handling.

```clarity
// Collections with safe operations
let numbers = List.of([1, 2, 3, 4, 5])
let doubled = numbers.map(x -> x * 2)
let first = numbers.first()  // Returns Optional<Number>

// String processing with proper Unicode handling
let text = "Hello, 世界!"
let chars = text.chars()     // Iterator over Unicode codepoints
let upper = text.to_upper()  // Proper Unicode case conversion

// File I/O with automatic resource management
function read_config(path: String) -> Result<Config, FileError>:
    with file = File.open(path)?:  // Automatically closed
        let content = file.read_all()?
        parse_config(content)
```

## Design Goals Achieved

1. **Memory Safety**: Ownership system prevents use-after-free, double-free, and data races
2. **Null Safety**: Optional types eliminate null pointer exceptions
3. **Error Safety**: Result types force error handling
4. **Concurrency Safety**: Message passing and immutability prevent race conditions
5. **Type Safety**: Rich type system catches logic errors at compile time
6. **Performance**: Zero-cost abstractions and compile-time optimization

## Example: Complete Web Server

```clarity
import { HttpServer, Router, Json } from WebFramework
import Database from DatabaseDriver

type User = { id: UserId, name: String, email: String }

async function get_user(id: UserId) -> Result<Json, ApiError> with Database:
    let user = await Database.find_user(id)?
    match user:
        Some(u) -> Success(u.to_json())
        None -> Error(NotFound("User not found"))

async function create_user(data: Json) -> Result<Json, ApiError> with Database:
    let user_data = User.from_json(data)?
    let user = await Database.create_user(user_data)?
    Success(user.to_json())

function main():
    let router = Router.new()
        .get("/users/:id", get_user)
        .post("/users", create_user)
    
    let server = HttpServer.new(router)
        .port(3000)
        .with_cors()
        .with_logging()
    
    server.start()
```

Clarity makes it impossible to write common bugs while keeping the code readable and maintainable. The type system guides you toward correct solutions, and the tooling catches errors before they reach production.
