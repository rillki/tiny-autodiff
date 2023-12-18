module rk.tgrad.nn.mlp;

import rk.tgrad.core.common;
import rk.tgrad.core.value;
import rk.tgrad.nn.layer;
import rk.tgrad.aux.activation;

class MLP : INeuron
{
    Layer[] layers;

    this(
        in size_t[] shape, 
        Value function(Value) activateHidden = &activateLinear, 
        Value function(Value) activateOutput = &activateLinear
    ) {
        foreach (i; 1..shape.length) layers ~= new Layer(
            [shape[i-1], shape[i]], 
            i+1 != shape.length ? activateHidden : activateOutput
        );
    }

    Value[] forward(Value[] input) in (input.length+1 == layers[0].neurons[0].parameters.length)
    {
        Value[] output = input;
        foreach (l; this.layers) output = l.forward(output);
        return output;
    }
    
    Value[] parameters()
    {
        Value[] paramsList;
        foreach (l; layers) paramsList ~= l.parameters;
        return paramsList;
    }

    ElementType[] parameterValues()
    {
        ElementType[] paramsList;
        foreach (l; layers) paramsList ~= l.parameterValues;
        return paramsList;
    }

    ElementType[] parameterGrads()
    {
        ElementType[] paramsList;
        foreach (l; layers) paramsList ~= l.parameterGrads;
        return paramsList;
    }
}

unittest
{
    import std.stdio;
    import std.array : array;
    import std.algorithm : map;

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

    // define model
    auto model = new MLP([4, 8, 1], &activateRelu, &activateRelu);

    auto L2Loss(Value[] preds)
    {
        import std.typecons : tuple;
        import std.algorithm : reduce;

        // mae loss
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
        auto ret = L2Loss(preds);
        auto loss = ret[0];
        auto accuracy = ret[1];
        
        // backward
        model.zeroGrad();
        loss.backward();
        
        // update
        model.update(lr);

        // if (epoch % 10 == 0) writefln("epoch %3s loss %.4f accuracy %.2f", epoch, loss.data, accuracy);
    }

    // predict
    auto pred = model.forward(input[0])[0];
    assert(pred.data > 0.5);
}

