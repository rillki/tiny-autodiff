module rk.tgrad.nn.layer;

import rk.tgrad.core.common;
import rk.tgrad.core.value;
import rk.tgrad.nn.neuron;
import rk.tgrad.aux.activation;

class Layer : INeuron
{
    Neuron[] neurons;
    Value function(Value) activate;

    this(in size_t[2] inOutSize, Value function(Value) activate = &activateLinear)
    {
        foreach (i; 0..inOutSize[1]) this.neurons ~= new Neuron(inOutSize[0], activate);
        this.activate = activate;
    }

    Value[] forward(Value[] input) in (input.length+1 == neurons[0].parameters.length)
    {
        Value[] output;
        foreach (n; neurons) output ~= n.forward(input);
        return output;
    }

    void backward()
    {
        foreach (p; this.parameters) p.backward();
    }

    void update(in ElementType lr)
    {
        foreach (n; neurons) n.update(lr);
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
    import rk.tgrad.aux.math;
    import std.math : log, abs, round;
    import std.array : array;
    import std.algorithm : clamp, map;
    import std.stdio;

    // define layer
    auto layer0 = new Layer([8, 4], &activateLinear);
    auto layer1 = new Layer([4, 1], &activateSigmoid);

    // define data
    auto input = [  // chocolate properties
        [0,1,0,0,1,0,1,0].map!(x => x.value).array, // 1
        [0,0,0,1,0,0,1,0].map!(x => x.value).array, // 1
        [0,0,0,0,0,0,0,0].map!(x => x.value).array, // 0
        [0,0,0,0,0,0,0,0].map!(x => x.value).array, // 0
        [1,0,0,0,0,0,0,0].map!(x => x.value).array, // 0
        [0,0,1,0,0,0,1,0].map!(x => x.value).array, // 1
        [0,1,1,1,0,0,1,0].map!(x => x.value).array, // 1
        [0,0,1,0,0,0,0,1].map!(x => x.value).array, // 0
    ];
    auto target = [ // 1: yes, 0: no
        1.value,
        1.value,
        0.value,
        0.value,
        0.value,
        1.value,
        1.value,
        0.value,
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
            auto out0 = layer0.forward(x);
            auto yhat = layer1.forward(out0);

            // loss
            loss = loss + (yhat[0] - target[0]);

            // accuracy
            accuracy += yhat[0].data.round == target[i].data;
        }
        // writeln("params l0: ", layer0.parameterValues);
        // writeln("params g0: ", layer0.parameterGrads);
        // writeln;
        // writeln("params l1: ", layer1.parameterValues);
        // writeln("params g1: ", layer1.parameterGrads);
        // writeln;
        // writeln;
        loss = loss / input.length;
        accuracy /= input.length;
        
        // backward
        layer0.zeroGrad();
        layer1.zeroGrad();
        loss.backward();

        // update
        layer1.update(lr * loss.data);
        layer0.update(lr * loss.data);

        if (epoch % 10 == 0) writefln("epoch %s loss %s accuracy %s", epoch, loss.data, accuracy);
    }

    // predict
    auto out0 = layer0.forward(input[0]);
    auto pred = layer1.forward(out0);
    writeln(pred[0].data);

    // // test parameters property
    // layer1.parameters[0].grad = 2;
    // assert(layer1.neurons[0].params[0].grad == 2);
}

