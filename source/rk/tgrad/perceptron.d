module rk.tgrad.perceptron;

import rk.tgrad.value;
import std.traits : isFloatingPoint;

abstract class INeuron(T)
{
    abstract Value!(T)[] parameters();
    void zeroGrad()
    {
        foreach (p; parameters)
        {
            p.zeroGrad();
        }
    }
}

class Perceptron(size_t inSize, T = float) if (isFloatingPoint!T) : INeuron!T
{
    Value!(T) bias;
    Value!(T)[inSize] weights;
    Value!(T) function(ref Value!T) activate;

    this(Value!(T) bias, Value!(T)[inSize] weights, Value!(T) function(ref Value!T) activate) 
    in (weights.length == inSize)
    {
        this.bias = bias;
        this.weights = weights;
        this.activate = activate;
    }

    this(Value!(T) function(ref Value!T) activate)
    {
        this.bias = value(0);
        foreach (ref w; this.weights) w = value(0);
        this.activate = activate;
    }
    
    auto opCall(Value!(T)[] x)
    in (x.length == inSize)
    {
        auto sum = value(0);
        foreach (i; 0..weights.length)
        {
            sum += weights[i]*x[i];
        }

        return activate(sum);
    }

    override Value!(T)[] parameters() 
    {
        return weights ~ bias;
    }
}

unittest
{
    import std.stdio; 
    import rk.tgrad.activation;

    auto n = new Perceptron!(1)((x){ return activateRelu(x); });
    // auto n = new Perceptron!(1)(value(0.0), [value(0)], (x){ return activateRelu(x); });
    auto r = n([value(2)]);
    
    // r.writeln;
    // r.grad.writeln;
    // r.backward();
    // r.grad.writeln;
    // r.children[0].writeln;
}

