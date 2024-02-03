<img src="imgs/icon-tautodiff.png" width="95" height="52" align="left"></img>
# Tiny AutoDiff
A tiny autograd library. Implements backpropagation autodiff. It supports all you need to build small neural networks.

## Library
Add library to your project using DUB:
```
dub add tiny-autodiff
```

## Example usage
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

### Multi-layer perceptron

```d
import rk.tautodiff;

import std.array : array;
import std.stdio : writeln;
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
    import std.algorithm : reduce, sum;

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

## LICENSE
All code is licensed under the BSL license. 

