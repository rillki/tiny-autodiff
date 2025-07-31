<img src="imgs/icon-tautodiff.png" width="95" height="52" align="left"></img>
# Tiny AutoDiff
A tiny autograd library. Implements backpropagation autodiff. It supports all you need to build small neural networks. 

## Library
Add library to your project using DUB:
```
dub add tiny-autodiff
```

## Precision
Use the `versions` configuration to specify the precision:
* `TAUTODIFF_USE_FLOAT`
* `TAUTODIFF_USE_DOUBLE`
* `TAUTODIFF_USE_REAL`
```
// dub.sdl
versions "TAUTODIFF_USE_FLOAT"
```
```
// dub.json
versions: ["TAUTODIFF_USE_FLOAT"]
```

## Example usage
**Checking version:**
```d
import std.stdio;
import rk.tautodiff;

writeln("Using `tiny-autodiff` version: ", Version);
```

### Value

```d
import rk.tautodiff;

auto a = value(2);
auto b = value(-3);
auto c = value(10);
auto f = value(-2);
auto e = a * b;
auto d = e + c;
auto g = f * d;

// backward
g.backward();

// check grad after backward
assert(g.grad == 1);
assert(f.grad == 4);
assert(d.grad == -2);
assert(e.grad == -2);
assert(c.grad == -2);
assert(b.grad == -4);
assert(a.grad == 6);
```

### ChainSolver
Use `ChainSolver` to solve equations step by step.
```d
import rk.tautodiff;

// create solver
auto solver = ChainSolver(0); // 0 is initial value

// operations using the produced result 
solver += 5; // 0 + 5 = 5
solver *= 2; // 3 * 2 = 6

// append new value and work with it
solver ~= solver / value(2);
assert(solver.data == 3);

// backward
solver.backward();
assert(solver.grad == 1);

// zero grad
solver.zeroGrad();
assert(solver.grad == 0);

// reset
solver.reset();
assert(solver.data == 0);
assert(solver.grad == 0);

// total length (allocated elements)
assert(solver.values.length == 4);
```

### Tape
Create `tapes` of equations and update the resulting value:
```d
// init
auto tape = new Tape();
assert(tape.values == []);
assert(tape.values.length == 0);
assert(tape.locked == false);
assert(!tape.isLocked);

// d = a * b - c
auto a = 5.value;
auto b = 10.value;
auto c = 25.value;
auto d = a * b;
auto e = d - c;
assert(e.data == 25);

// push
tape.pushBack(a);
tape ~= b;
tape ~= [c, d, e];
assert(tape.values == [a, b, c, d, e]);
assert(tape.values.length == 5);
assert(tape.lastValue.data == 25);

// lock tape
tape.lock();
// tape ~= 24.value; // assert error: reset the tape to push new values

// modify value
a.data = 6;

// update tape
tape.update();
assert(tape.lastValue.data == 35);

// reset tape to push new values
tape.reset();
tape ~= 35.value; // good
```

### Multi-layer perceptron

```d
import rk.tautodiff;

import std.array : array;
import std.stdio : writefln;
import std.algorithm : map;

// define data
auto input = [  // binary
    [0, 0, 0, 0], // 0
    [0, 0, 0, 1], // 1
    [0, 0, 1, 0], // 2
    [0, 0, 1, 1], // 3
    [0, 1, 0, 0], // 4
    [0, 1, 0, 1], // 5
    [0, 1, 1, 0], // 6
    [0, 1, 1, 1], // 7
    [1, 0, 0, 0], // 8
    [1, 0, 0, 1], // 9
    [1, 0, 1, 0], // 10
    [1, 0, 1, 1], // 11
    [1, 1, 0, 0], // 12
    [1, 1, 0, 1], // 13
    [1, 1, 1, 0], // 14
    [1, 1, 1, 1], // 15
].map!(x => x.map!(y => y.value).array).array;

auto target = [ // 1: even, 0: odd
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
].map!(x => x.value).array;

// split train, test
auto input_train = input[0 .. 12];
auto input_test = input[12 .. $];

// define model
auto model = new MLP([4, 8, 1], &activateRelu, &activateSigmoid);

// define loss function
auto lossL2(Value[] preds)
{
    import std.algorithm : reduce;

    // voldemort type
    struct L2Loss { Value loss; float accuracy; }

    // mse loss
    Value[] losses; 
    foreach (i; 0..preds.length) losses ~= (preds[i] - target[i]) * (preds[i] - target[i]);
    auto dataLoss = losses.reduce!((a, b) => a + b) / preds.length;

    // accuracy
    float accuracy = 0.0;
    foreach (i; 0..preds.length) accuracy += ((preds[i].data > 0.5) == target[i].data);
    accuracy /= preds.length;

    // return voldemort type with cost and accuracy
    return L2Loss(dataLoss, accuracy); 
}

// train
enum lr = 0.05;
enum epochs = 100;
foreach (epoch; 0..epochs)
{
    // forward
    Value[] preds;
    foreach (x; input_train) preds ~= model.forward(x);

    // loss
    auto l2 = lossL2(preds);
    
    // backward
    model.zeroGrad();
    l2.loss.backward();
    
    // update
    model.update(lr);

    // debug print
    if (epoch % 10 == 0) writefln("epoch %3s loss %.4f accuracy %.2f", epoch, l2.loss.data, l2.accuracy);
}

// test
foreach (i, x; input_test) 
{
    auto pred = model.forward(x)[0];
    assert((pred.data > 0.5) == target[i].data);
}
```
Output:
```sh
epoch   0 loss 1.9461 accuracy 0.50
epoch  10 loss 0.1177 accuracy 0.75
epoch  20 loss 0.0605 accuracy 1.00
epoch  30 loss 0.0395 accuracy 1.00
...
epoch  90 loss 0.0010 accuracy 1.00
```

## References
* [Golem **(D)**](https://github.com/lempiji/golem)
* [Grain **(D)**](https://github.com/ShigekiKarita/grain)
* [Micrograd **(Py)**](https://github.com/karpathy/micrograd)
* [Teenygrad **(Py)**](https://github.com/tinygrad/teenygrad)
* [Tinygrad **(Py)**](https://github.com/tinygrad/tinygrad)

## LICENSE
All code is licensed under the BSL license. 

