module rk.tgrad.core.value;

import rk.tgrad.core.common;
import std.traits: isNumeric;

interface INeuron
{
    final void zeroGrad()
    {
        import std.parallelism : parallel;
        foreach (ref p; parameters.parallel) p.grad = 0;
    }

    final void update(in ElementType lr)
    {
        import std.parallelism : parallel;
        foreach (ref p; parameters.parallel) p.data -= lr * p.grad;
    }

    Value[] parameters();
    ElementType[] parameterValues();
    ElementType[] parameterGrads();
}

template value(T = ElementType) if (isNumeric!T)
{
    /// Initialize with a uniform random distribution [0; 1)
    auto value()
    {
        return new Value();
    }

    /// Initialize with a custom variable
    auto value(in T data)
    {
        return new Value(data);
    }

    /// Initialize with custom parameters
    auto value(in T data, Value[] parents, void function(Value) backward = null)
    {
        return new Value(data, parents, backward);
    }
}

class Value : INeuron
{
    ElementType data = 0;
    ElementType grad = 0;
    Value[] parents = null;
    void function(Value) _backward = null;

    /// Initialize with a uniform random distribution [0; 1)
    this() 
    {
        import std.random : uniform;
        this.data = uniform!("[)", ElementType, ElementType)(0, 1);
    }

    /// Initialize with a custom variable
    this(in ElementType data) 
    {
        this.data = data;
    }

    /// Initialize with custom parameters
    this(in ElementType data, Value[] parents, void function(Value) backward = null) 
    {
        this(data);
        this.parents = parents;
        this._backward = backward;
    }

    /// backward operation 
    void backward() 
    {        
        this.grad = 1;
        foreach (node; buildNodeList(this)) if (node._backward) node._backward(node);
    }

    /// Returns model parameters as Value object
    Value[] parameters()
    {
        return [this];
    }

    /// Returns model parameter values
    ElementType[] parameterValues()
    {
        return [data];
    }

    /// Returns model gradients values
    ElementType[] parameterGrads()
    {
        return [grad];
    }

    auto opBinary(string op)(Value rhs) 
    {
        static if (op == "+" || op == "-")
        {
            return value(mixin("this.data" ~ op ~ "rhs.data"), [this, rhs], &opBackwardAddSub);
        }
        else static if (op == "*" || op == "/")
        {
            return value(mixin("this.data" ~ op ~ "rhs.data"), [this, rhs], &opBackwardMulDiv);
        }
        else static assert(0, "Operator <"~op~"> not supported!");
    }

    auto opBinary(string op)(in ElementType rhs)
    {
        static if (op == "+" || op == "-")
        {
            return value(mixin("this.data" ~ op ~ "rhs"), [this, value(rhs)], &opBackwardAddSub);
        }
        else static if (op == "*" || op == "/")
        {
            return value(mixin("this.data" ~ op ~ "rhs"), [this, value(rhs)], &opBackwardMulDiv);
        }
        else static assert(0, "Operator <"~op~"> not supported!");
    }

    auto opBinaryRight(string op)(in ElementType lhs)
    {
        static if (op == "+" || op == "-")
        {
            return value(mixin("lhs" ~ op ~ "this.data"), [value(lhs), this], &opBackwardAddSub);
        }
        else static if (op == "*" || op == "/")
        {
            return value(mixin("lhs" ~ op ~ "this.data"), [value(lhs), this], &opBackwardMulDiv);
        }
        else static assert(0, "Operator <"~op~"> not supported!");
    }

    /// Inplace operations
    void opInplace(string op)(Value lhs, Value rhs)
    {
        static if (op == "+" || op == "-")
        {
            this.reinit(mixin("rhs.data" ~ op ~ "lhs.data"), [lhs, rhs], &opBackwardAddSub);
        }
        else static if (op == "*" || op == "/")
        {
            this.reinit(mixin("rhs.data" ~ op ~ "lhs.data"), [lhs, rhs], &opBackwardMulDiv);
        }
        else static assert(0, "Operator <"~op~"> not supported!");
    }
    
    /// Re-initialize object parameters
    void reinit(in ElementType data, Value[] parents = null, void function(Value) backward = (x){})
    {
        this.grad = 0;
        this.data = data;
        this.parents = parents;
        this._backward = backward;
    }

    /// Build node tree of all Values
    auto buildNodeList(Value startNode)
    {
        import std.algorithm : canFind;

        // define a list where to save all nodes
        Value[] nodeList = [];

        // define deep walk funtion to traverse each node and add to list
        void deepWalk(Value node)
        {
            // add to node list
            if (!nodeList.canFind(node)) nodeList ~= node;

            // traverse each child iteratively
            foreach (child; node.parents) deepWalk(child);
        }

        // traverse node tree
        deepWalk(startNode);

        return nodeList;
    }
}

/// Backward operation for addition and substraction
private void opBackwardAddSub(Value opResult)
{
    // get parents
    auto lhs = opResult.parents[0];
    auto rhs = opResult.parents[1];

    // perform backward operation
    lhs.grad += 1.0 * opResult.grad;
    rhs.grad += 1.0 * opResult.grad;
}

/// Backward operation for multiplication and division
private void opBackwardMulDiv(Value opResult)
{
    // get parents
    auto lhs = opResult.parents[0];
    auto rhs = opResult.parents[1];

    // perform backward operation
    lhs.grad += rhs.data * opResult.grad;
    rhs.grad += lhs.data * opResult.grad;
}

unittest
{
    // create
    auto a = value(2);
    auto b = value(-3);
    auto c = value(10);
    auto f = value(-2);
    auto e = a * b;
    auto d = e + c;
    auto g = f * d;

    // check values
    assert(g.data == -8);
    assert(f.data == -2);
    assert(d.data == 4);
    assert(e.data == -6);
    assert(c.data == 10);
    assert(b.data == -3);
    assert(a.data == 2);

    // check grad
    assert(g.grad == 0);
    assert(f.grad == 0);
    assert(d.grad == 0);
    assert(e.grad == 0);
    assert(c.grad == 0);
    assert(b.grad == 0);
    assert(a.grad == 0);

    // backward
    g.backward();

    // check grad after backward
    assert(g.grad == 1);
    assert(f.grad == 4);
    assert(d.grad == -2);
    assert(e.grad == -2);
    assert(c.grad == -2);
    assert(b.grad == -4);
    assert(a.grad == 6);

    // zero grad
    g.zeroGrad();
    f.zeroGrad();
    d.zeroGrad();
    e.zeroGrad();
    c.zeroGrad();
    b.zeroGrad();
    a.zeroGrad();
    assert(g.grad == 0);
    assert(f.grad == 0);
    assert(d.grad == 0);
    assert(e.grad == 0);
    assert(c.grad == 0);
    assert(b.grad == 0);
    assert(a.grad == 0);

    // test parameters property returns by reference
    g.parameters[0].grad = 2;
    assert(g.grad == 2);

    // check inplace operation
    auto h = value(0);
    h.opInplace!"*"(a, b);
    a.zeroGrad();
    b.zeroGrad();
    h.backward();
    assert(h.data == -6);
    assert(b.data == -3);
    assert(a.data == 2);
}

