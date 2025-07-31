module app;

import std.stdio;
import rk.tautodiff;

void main()
{
    import std.array : array;
    import std.stdio : writeln, writefln;
    import std.algorithm : map;

    // print version
    writeln("Using `tiny-autodiff` version: ", Version);
    writeln("---");

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
    auto target_train = target[0 .. 12];
    auto target_test = target[12 .. $];

    // define model
    auto model = new MLP([4, 8, 1], &activateRelu, &activateSigmoid);

    // define loss function
    auto lossL2(Value[] preds)
    {
        import std.algorithm : reduce;

        // voldemort type
        struct L2Loss { Value loss; float accuracy; }

        // mse loss
        Value[] losses; 
        foreach (i; 0..preds.length) 
            losses ~= (preds[i] - target_train[i]) * (preds[i] - target_train[i]);
        
        auto sum = losses.reduce!((a, b) => a + b);
        auto dataLoss = sum / preds.length;

        // accuracy
        float accuracy = 0.0;
        foreach (i; 0..preds.length) accuracy += ((preds[i].data > 0.5) == target_train[i].data);
        accuracy /= preds.length;

        // return voldemort type with cost and accuracy
        return L2Loss(dataLoss, accuracy); 
    }

    // train
    enum lr = 0.5;
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
        assert((pred.data > 0.5) == target_test[i].data);
    }
}

