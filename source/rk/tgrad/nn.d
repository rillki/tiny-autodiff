module rk.tgrad.nn;

import rk.tgrad.value;
import rk.tgrad.activation;
import std.traits : isFloatingPoint;

interface INeuron
{
    void zeroGrad();
}

class Neuron(T = float, size_t inSize) if (isFloatingPoint!T) : INeuron
{
    Value!(T) bias;
    Value!(T)[inSize] weights;
    Value!(T) function(ref Value!T) activate;

    this(Value!(T) bias, Value!(T)[inSize] weights, Value!(T) function(ref Value!T) activate)
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

    auto parameters() 
    {
        return weights ~ bias;
    }
}

// class Layer(T, size_t[2] sizeInOut) if (isFloatingPoint!T)
// {
//     Value!(T)[sizeInOut]
// }

unittest
{
    auto n = new Neuron!(float, 1)(value(0.0), [value(2)], (x){ return activateRelu(x); });
    auto r = n([value(2)]);
    import std.stdio; r.writeln; writeln(n.parameters());
    r.grad.writeln;
}

