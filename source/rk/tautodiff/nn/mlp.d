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
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
    ].map!(x => x.map!(y => y.value).array).array;
    
    auto target = [ // 1: even, 0: odd
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    ].map!(x => x.value).array;

    // split train, test
    auto input_train = input[0 .. 12];
    auto input_test = input[12 .. $];
    auto target_train = target[0 .. 12];
    auto target_test = target[12 .. $];

    // define model
    auto model = new MLP([4, 8, 1], &activateRelu, &activateSigmoid);

    // define loss function
    auto lossL2(Value[] preds)
    {
        import std.algorithm : reduce;

        struct L2Loss { Value loss; float accuracy; }

        // mse loss
        Value[] losses; 
        foreach (i; 0..preds.length) losses ~= (preds[i] - target_train[i]) * (preds[i] - target_train[i]);
        auto sum = losses.reduce!((a, b) => a + b);
        auto dataLoss = sum / preds.length;

        // accuracy
        float accuracy = 0.0;
        foreach (i; 0..preds.length) accuracy += ((preds[i].data > 0.5) == target_train[i].data);
        accuracy /= preds.length;

        return L2Loss(dataLoss, accuracy); 
    }

    // train
    enum lr = 0.05;
    enum epochs = 1000;
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
        if (epoch % 100 == 0) writefln("epoch %3s loss %.4f accuracy %.4f", epoch, l2.loss.data, l2.accuracy);
    }

    // test
    foreach (i, x; input_test) 
    {
        auto pred = model.forward(x)[0];
        assert((pred.data > 0.5) == target_test[i].data);
    }
}

