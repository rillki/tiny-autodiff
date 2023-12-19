# Tiny Grad
A tiny autograd library. Implements backpropagation autodiff.

## Example usage
### Variable

```d
import rk.tgrad;

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
import rk.tgrad;

import std.stdio;
import std.array : array;
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

// define model
auto model = new MLP([4, 8, 1], &activateRelu, &activateRelu);

auto mseLoss(Value[] preds)
{
    import std.typecons : tuple;
    import std.algorithm : reduce;

    // mse loss
    Value[] losses; 
    foreach (i; 0..preds.length) losses ~= (preds[i] - target[i]) * (preds[i] - target[i]);
    auto data_loss = losses.reduce!((a, b) => a + b) / preds.length;

    // accuracy
    ElementType accuracy = 0.0;
    foreach (i; 0..preds.length) accuracy += ((preds[i].data > 0.5) == target[i].data);

    return tuple(data_loss, accuracy/preds.length);
}

// train
enum lr = 0.0005;
enum epochs = 100;
foreach (epoch; 0..epochs)
{
    // forward
    Value[] preds;
    foreach (x; input) preds ~= model.forward(x);

    // loss
    auto ret = mseLoss(preds);
    auto loss = ret[0];
    auto accuracy = ret[1];
    
    // backward
    model.zeroGrad();
    loss.backward();
    
    // update
    model.update(lr);

    if (epoch % 10 == 0) writefln("epoch %3s loss %.4f accuracy %.2f", epoch, loss.data, accuracy);
}

// predict
auto pred = model.forward(input[0])[0];
assert(pred.data > 0.5);
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


