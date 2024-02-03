module rk.tautodiff.nn.layer;

import rk.tautodiff.core.common;
import rk.tautodiff.core.value;
import rk.tautodiff.nn.neuron;
import rk.tautodiff.aux.activation;

class Layer : INeuron
{
    Neuron[] neurons;
    Value function(Value) activate;

    /// Initialize a new layer with a custom activation function
    this(in size_t[2] inOutSize, Value function(Value) activate = &activateLinear)
    {
        this.activate = activate;
        foreach (i; 0 .. inOutSize[1]) this.neurons ~= new Neuron(inOutSize[0], activate);
    }
    
    /// Forward operation
    Value[] forward(Value[] input) in (input.length+1 == neurons[0].parameters.length)
    {
        Value[] output;
        foreach (n; neurons) output ~= n.forward(input);
        return output;
    }

    Value[] parameters()
    {
        Value[] paramsList;
        foreach (n; neurons) paramsList ~= n.parameters;
        return paramsList;
    }

    ElementType[] parameterValues()
    {
        ElementType[] paramsList;
        foreach (n; neurons) paramsList ~= n.parameterValues;
        return paramsList;
    }

    ElementType[] parameterGrads()
    {
        ElementType[] paramsList;
        foreach (n; neurons) paramsList ~= n.parameterGrads;
        return paramsList;
    }
}

unittest
{
    import std.array : array;
    import std.algorithm : map;
    import std.stdio : writefln, writeln;

    // define layer
    auto layer = new Layer([4, 1], &activateLinear);

    // define data
    auto input = [  // binary
        [0, 0, 0, 0].map!(x => x.value).array, // 0
        [0, 0, 0, 1].map!(x => x.value).array, // 1
        [0, 0, 1, 0].map!(x => x.value).array, // 2
        [0, 0, 1, 1].map!(x => x.value).array, // 3
        [0, 1, 0, 0].map!(x => x.value).array, // 4
        [0, 1, 0, 1].map!(x => x.value).array, // 5
        [0, 1, 1, 0].map!(x => x.value).array, // 6
        [0, 1, 1, 1].map!(x => x.value).array, // 7
        [1, 0, 0, 0].map!(x => x.value).array, // 8
        [1, 0, 0, 1].map!(x => x.value).array, // 9
        [1, 0, 1, 0].map!(x => x.value).array, // 10
        [1, 0, 1, 1].map!(x => x.value).array, // 11
        [1, 1, 0, 0].map!(x => x.value).array, // 12
        [1, 1, 0, 1].map!(x => x.value).array, // 13
        [1, 1, 1, 0].map!(x => x.value).array, // 14
        [1, 1, 1, 1].map!(x => x.value).array, // 15
    ];
    auto target = [ // 1: even, 0: odd
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    ].map!(x => x.value).array;

    // train
    enum lr = 0.0005;
    enum epochs = 100;
    foreach (epoch; 0..epochs)
    {
        auto loss = value(0);
        float accuracy = 0;
        foreach (i, x; input)
        {
            // forward
            auto yhat = layer.forward(x)[0];

            // loss
            loss = loss + (yhat - target[i]) * (yhat - target[i]);

            // accuracy
            accuracy += (yhat.data > 0.5) == target[i].data;
        }

        // adjust by input size
        loss = loss / input.length;
        accuracy /= input.length;
        
        // backward
        layer.zeroGrad();
        loss.backward();

        // update
        layer.update(lr);

        // debug print
        // if (epoch % 10 == 0) writefln("epoch %3s loss %.4f accuracy %.4f", epoch, loss.data, accuracy);
    }

    // predict
    auto pred = layer.forward(input[0])[0];
    assert(pred.data > 0.5);

    // test parameters property
    layer.parameters[0].grad = 2;
    assert(layer.neurons[0].params[0].grad == 2);
}

