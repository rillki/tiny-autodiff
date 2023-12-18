module app;

import std.stdio;
import rk.tgrad;

void main()
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
    enum wd = 0.001;    // weight decay
    enum lr = 0.05;     // learning rate
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
        model.update(lr * wd);

        if (epoch % 10 == 0) writefln("epoch %3s loss %.4f accuracy %.2f", epoch, loss.data, accuracy);
    }

    // predict
    auto pred = model.forward(input[0])[0];
    assert(pred.data > 0.5);
}

