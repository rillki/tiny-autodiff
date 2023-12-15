module rk.tgrad.nn.neuron;

import rk.tgrad.core.common;
import rk.tgrad.core.value;

class Neuron
{
    import std.parallelism : parallel;

    Value[] params;
    Value function(Value) activate;

    this(in size_t inputSize, Value function(Value) activate = (x){return x;}) {
        foreach (i; 0..inputSize+1) this.params ~= value();
        this.activate = activate;
    }

    Value forward(Value[] input) in (input.length+1 == params.length)
    {
        auto sum = value(0);
        foreach (i, x; input) sum = sum + params[i]*x;
        return activate(sum + params[$-1]);
    }

    void zeroGrad()
    {
        foreach (p; params.parallel) p.grad = 0;
    }

    void update(in ElementType lr)
    {
        foreach (p; params.parallel) p.data -= lr * p.grad;
    }
}

unittest
{
    import std.stdio;
    import std.math : round;

    // define model
    auto neuron = new Neuron(2);

    // define data
    auto input = [
        [value(0), value(0)],
        [value(1), value(0)],
        [value(0), value(1)],
        [value(1), value(1)],
    ];
    auto target = [
        0.value, 1.value, 1.value, 1.value
    ];

    // train
    enum lr = 0.01;
    enum epochs = 120;
    foreach (epoch; 0..epochs)
    {
        auto loss = value(0);
        float accuracy = 0;
        foreach (i, x; input)
        {
            // forward
            auto yhat = neuron.forward(x);

            // loss and accuracy
            loss = loss + (yhat - target[i]);
            accuracy += yhat.data.round == target[i].data;
        }
        loss = loss / input.length;
        accuracy /= input.length;
        
        // backward
        neuron.zeroGrad();
        loss.backward();

        // update
        neuron.update(lr * loss.data);
    }

    // predict
    auto pred = neuron.forward([1.value, 1.value]);
    assert(pred.data.round == 1);
}

