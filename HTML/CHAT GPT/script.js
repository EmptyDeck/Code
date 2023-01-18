function calculateFibonacci() {
  var inputNumber = document.getElementById("inputNumber").value;
  var result = document.getElementById("result");

  // Check if input is a number
  if (isNaN(inputNumber)) {
    result.innerHTML = "Please enter a valid number.";
    return;
  }

  // Calculate Fibonacci number
  var fib = [0, 1];
  for (var i = 2; i <= inputNumber; i++) {
    fib[i] = fib[i-1] + fib[i-2];
  }

  result.innerHTML = "The Fibonacci number at index " + inputNumber + " is " + fib[inputNumber];
}
