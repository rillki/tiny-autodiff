module rk.tgrad.nn.ann;

import rk.tgrad.core.common;
import rk.tgrad.core.value;
import rk.tgrad.nn.layer;
import rk.tgrad.aux.activation;

class ANN : INeuron
{
    Layer[] layers;
    size_t[] shape;

    this(
        in size_t[] shape, 
        Value function(Value) activateHidden = &activateSigmoid, 
        Value function(Value) activateOutput = &activateLinear
    ) {
        this.shape = shape.dup;
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
    import std.array : array;
    import std.algorithm : map;
    import std.stdio : writefln, writeln;

    auto input = [
        [ 1.49928018, -0.33561585],
        [ 0.15461869,  1.051595  ],
        [ 1.77486444, -0.23983261],
        [ 0.85171071,  0.3476219 ],
        [ 0.69586119,  0.68299791],
        [-0.82775393,  0.15475422],
        [ 0.14068448,  1.06993949],
        [ 0.73628177, -0.43885805],
        [ 0.55217771,  0.9354338 ],
        [-1.02158662,  0.23943295],
        [-0.80992061,  0.27182436],
        [ 1.93250481,  0.26269972],
        [ 0.80255226,  0.48441665],
        [ 1.58273182, -0.37550597],
        [ 1.68856625, -0.36372212],
        [ 0.81562222, -0.36699182],
        [-0.34774187,  0.99005479],
        [ 1.31054327, -0.67393862],
        [-0.23766105,  1.01348406],
        [ 1.48099337, -0.44745054],
        [ 2.00827033, -0.00629419],
        [-0.95849272,  0.33630233],
        [ 0.59550462, -0.61918555],
        [ 0.69742798,  0.53244912],
        [ 1.70837324, -0.38823502],
        [ 0.20623876, -0.17261582],
        [ 1.19680822, -0.51085551],
        [ 1.76425207, -0.32921988],
        [ 0.58277429, -0.27556463],
        [ 1.85059836,  0.275165  ],
        [ 1.24965827, -0.44566863],
        [ 1.74784038, -0.21781901],
        [ 0.76134198,  0.502691  ],
        [ 0.64766335, -0.59492903],
        [-0.45242504,  0.74513425],
        [-0.91822631, -0.00676402],
        [-0.50280357,  0.81512931],
        [-0.05908822,  0.29514475],
        [ 1.21089211, -0.72718958],
        [ 1.03840619, -0.04986757],
        [-0.62526991,  0.52356489],
        [ 1.94132823,  0.01771246],
        [ 1.00377967,  0.22102984],
        [ 2.04277777,  0.55836941],
        [ 0.25951979,  0.86004764],
        [ 0.93498906, -0.39677462],
        [ 0.89105486,  0.37412085],
        [ 0.16629977,  1.06170903],
        [ 2.25477498,  0.05695808],
        [ 0.11441211,  0.07519445],
        [ 0.01359948,  0.04565586],
        [ 1.0178053 ,  0.24473705],
        [ 1.71602741, -0.02972543],
        [-0.81899877,  0.17976269],
        [ 0.33492858,  0.87561818],
        [-0.8346961 ,  0.69449814],
        [ 0.71723573,  0.67703771],
        [-0.19614128,  0.51852735],
        [-0.89413751,  0.19611231],
        [ 2.01857612,  0.58233289],
        [ 0.80120458,  0.47555331],
        [-0.2166673 ,  1.03073446],
        [-0.21297449,  0.96817192],
        [-0.42372258,  1.02537431],
        [-0.08420483,  0.47139526],
        [-1.1570004 ,  0.1793537 ],
        [ 0.65543212,  0.65024375],
        [ 0.91944602, -0.00376513],
        [ 1.34704016, -0.45228099],
        [-0.88802013,  0.51239227],
        [ 0.02011324, -0.04717865],
        [ 0.56874643, -0.40130802],
        [ 0.85957534,  0.44077822],
        [-0.87355234,  0.64202188],
        [-0.11910786,  1.21776786],
        [ 0.40598291,  0.92874681],
        [ 0.14132577,  0.03872508],
        [ 0.41487147,  0.94146833],
        [ 1.98977526,  0.26991227],
        [-0.06889296,  0.96707422],
        [ 0.47849824, -0.26146136],
        [-0.03506317,  0.28798087],
        [-0.64499926,  0.70660755],
        [ 1.15041898, -0.40172719],
        [-0.79582091,  0.65872656],
        [ 0.03405215,  0.39339559],
        [ 0.3401256 , -0.09746349],
        [ 0.44648763, -0.36929805],
        [-0.65440413,  0.73595733],
        [ 0.15859575, -0.19370566],
        [ 1.00091035,  0.34816382],
        [-0.36856214,  0.87769497],
        [ 1.30812984, -0.4117994 ],
        [ 0.85161003, -0.40471352],
        [ 1.89990821,  0.24488301],
        [ 0.5370575 ,  0.76316245],
        [ 1.18257737, -0.54051314],
        [ 0.18953163, -0.07941639],
        [ 0.0974822 ,  0.33684532],
        [ 0.20120651,  1.03995993],
    ].map!(x => x.map!(y => y.value).array).array;
    auto target = [ 
        1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1,
        1, -1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,
        -1, -1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1,  1,  1,  1,
        -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1,
        1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,
        1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1, -1
    ].map!(x => x.value).array;

    // define model
    enum mshape = [2, 4, 2, 1];
    auto model = new ANN(mshape, &activateSigmoid, &activateLinear);
    assert(model.shape == mshape);

    // define cost
    auto cost(Value[] preds, in ElementType alpha = 1e-4)
    {
        import std.typecons : tuple;
        import std.algorithm : reduce, sum;

        // svm "max-margin" loss
        Value[] losses; 
        foreach (i; 0..preds.length) losses ~= (1 - target[i]*preds[i]).activateRelu;
        auto dataLoss = losses.reduce!((a, b) => a + b) / preds.length;

        // L2 regularization
        auto regLoss = alpha * model.parameterValues.map!(a => a*a).array.sum;
        auto totalLoss = dataLoss + regLoss;

        // accuracy
        ElementType accuracy = 0.0;
        foreach (i; 0..preds.length) accuracy += ((preds[i].data > 0) == (target[i].data > 0));

        return tuple(totalLoss, accuracy/preds.length);
    }

    // train
    enum lr = 0.005;
    enum epochs = 1000;
    foreach (epoch; 0..epochs)
    {
        // forward
        Value[] preds;
        foreach (x; input) preds ~= model.forward(x);

        // loss
        auto ret = cost(preds);
        auto loss = ret[0];
        auto accuracy = ret[1];
        
        // backward
        model.zeroGrad();
        loss.backward();
        
        // update
        model.update(lr);

        if (epoch % 1 == 0) writefln("epoch %3s loss %.4f accuracy %.2f", epoch, loss.data, accuracy);
        writeln("params v: ", model.parameterValues);
        writeln("params g: ", model.parameterGrads);
        writeln("preds  v: ", preds.map!(a => a.data).array);
        writeln;
        writeln;
    }

}

