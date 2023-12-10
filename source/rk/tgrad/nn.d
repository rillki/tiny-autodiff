module rk.tgrad.nn;

import rk.tgrad.value;
import rk.tgrad.activation;
import std.traits : isFloatingPoint;

template neuron(size_t inSize, T = float)
{
    auto neuron(void function(Value!T) activate = (v) { v = activateRelu(v); })
    {
        import std.random : uniform;

        // randomize weights
        Value!(T)[inSize] weights;
        foreach (w; weights)
        {
            w = value(uniform(0.0f, 1.0f));
        }

        return new Neuron!(T, inSize)(value(1), weights, activate);
    }
}

// auto neuron(T = float)(size_t inSize, void delegate(Value!T) activate = activateRelu)
// {
//     import std.random : uniform;

//     // randomize weights
//     Value!(T)[inSize] weights;
//     foreach (w; weights)
//     {
//         w = value(uniform(0.0f, 1.0f));
//     }

//     return new Neuron!(T)(value(1), weights, activate);
// }

interface INeuron
{
    void zeroGrad();
    S parameters(S)();
}

class Neuron(T = float, size_t inSize) if (isFloatingPoint!T) : INeuron
{
    Value!(T) bias;
    Value!(T)[inSize] weights;
    void delegate(ref Value!T) activate;

    this(Value!(T) bias, Value!(T)[inSize] weights, void delegate(ref Value!T) activate)
    {
        this.bias = bias;
        this.weights = weights;
        this.activate = activate;
    }
    
    auto opCall(Value!(T)[inSize] x)
    {
        auto sum = value(0);
        foreach (i; 0..weights.length)
        {
            sum += weights[i]*x[i];
        }

        return activate(sum);
    }

    void zeroGrad()
    {
        foreach (w; weights)
        {
            w.zeroGrad();
        }
    }

    S parameters(S)() 
    {
        return [w, b];
    }
}

// class Layer(T, size_t[2] sizeInOut) if (isFloatingPoint!T)
// {
//     Value!(T)[sizeInOut]
// }

unittest
{
    auto n = neuron!(3)(activateRelu);
    // auto n = new Neuron!(float, 1)(value(0.0), [value(0)], (x){});
}

