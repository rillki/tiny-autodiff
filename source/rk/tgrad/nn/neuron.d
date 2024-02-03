module rk.tgrad.nn.neuron;

import rk.tgrad.core.common;
import rk.tgrad.core.value;
import rk.tgrad.aux.activation;

class Neuron : INeuron
{
    Value[] params;
    Value function(Value) activate;

    /// Initialize a new neuron with a custom activation function
    this(in size_t inputSize, Value function(Value) activate = &activateLinear) {
        this.activate = activate;
        foreach (i; 0 .. inputSize+1) this.params ~= value();
    }

    /// Forward operation
    Value forward(Value[] input) in (input.length+1 == parameters.length)
    {
        auto sum = value(0);

        // multiply weights with input: w[i] * x[i]
        foreach (i; 0 .. input.length) sum = sum + params[i]*input[i];

        // add bias
        sum = sum + params[$-1];

        // activate
        return this.activate ? this.activate(sum) : sum;
    }

    Value[] parameters()
    {
        return params;
    }

    ElementType[] parameterValues()
    {
        ElementType[] paramsList;
        foreach (p; this.params) paramsList ~= p.parameterValues;
        return paramsList;
    }

    ElementType[] parameterGrads()
    {
        ElementType[] paramsList;
        foreach (p; this.params) paramsList ~= p.parameterGrads;
        return paramsList;
    }
}

unittest
{
    import std.stdio; 

    // define model
    auto neuron = new Neuron(2);

    // define data: AND operator
    auto input = [
        [value(0), value(0)],
        [value(1), value(0)],
        [value(0), value(1)],
        [value(1), value(1)],
    ];
    auto target = [
        0.value, 0.value, 0.value, 1.value
    ];

    // train
    enum lr = 0.05;
    enum epochs = 100;
    foreach (epoch; 0..epochs)
    {
        auto loss = value(0);
        float accuracy = 0;
        foreach (i, x; input)
        {
            // forward
            auto yhat = neuron.forward(x);

            // loss
            loss = loss + (yhat - target[i]) * (yhat - target[i]);

            // accuracy
            accuracy += (yhat.data > 0.5) == target[i].data;
        }

        // adjust by input size
        loss = loss / input.length;
        accuracy /= input.length;
        
        // backward
        neuron.zeroGrad();
        loss.backward();

        // update
        neuron.update(lr);

        // debug print
        // if (epoch % 10 == 0) writefln("epoch %3s loss %.4f accuracy %.4f", epoch, loss.data, accuracy);
    }

    // predict
    auto pred = neuron.forward([1.value, 1.value]);
    assert(pred.data > 0.5);

    // test parameters property
    neuron.parameters[0].grad = 2;
    assert(neuron.params[0].grad == 2);
}

