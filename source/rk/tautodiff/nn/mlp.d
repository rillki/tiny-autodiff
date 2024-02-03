module rk.tautodiff.nn.mlp;

import rk.tautodiff.core.common;
import rk.tautodiff.core.value;
import rk.tautodiff.nn.layer;
import rk.tautodiff.aux.activation;

class MLP : INeuron
{
    Layer[] layers;

    /// Initialize a new layer with a custom activation function
    this(
        in size_t[] shape, 
        Value function(Value) activateHidden = &activateRelu, 
        Value function(Value) activateOutput = &activateLinear
    ) {
        foreach (i; 1 .. shape.length) layers ~= new Layer(
            [shape[i-1], shape[i]], 
            i+1 != shape.length ? activateHidden : activateOutput
        );
    }

    /// Forward operation
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
        // auto dataLoss = sum(losses) / preds.length;

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
        // if (epoch % 10 == 0) writefln("epoch %3s loss %.4f accuracy %.2f", epoch, l2.loss.data, l2.accuracy);
    }

    // test
    foreach (i, x; input_test) 
    {
        auto pred = model.forward(x)[0];
        assert((pred.data > 0.5) == target[i].data);
    }
}

