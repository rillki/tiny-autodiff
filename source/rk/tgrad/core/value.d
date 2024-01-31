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
    auto value()
    {
        return new Value();
    }

    auto value(in T data)
    {
        return new Value(data);
    }

    auto value(in T data, Value[] parents)
    {
        return new Value(data, parents);
    }

    auto value(in T data, Value[] parents, void function(Value) backward)
    {
        return new Value(data, parents, backward);
    }
}

class Value : INeuron
{
    ElementType data;
    ElementType grad;
    Value[] parents = null;
    void function(Value) _backward = (x){};

    this() 
    {
        import std.random : uniform;
        this.data = uniform!("()", ElementType, ElementType)(-1, 1);
        this.grad = 0;
    }

    this(in ElementType data) 
    {
        this.data = data;
        this.grad = 0;
    }

    this(in ElementType data, Value[] parents) 
    {
        this(data);
        this.parents = parents;
    }

    this(in ElementType data, Value[] parents, void function(Value) backward) 
    {
        this(data, parents);
        this._backward = backward;
    }

    void backward() 
    {
        import std.parallelism : parallel;
        
        this.grad = 1;
        foreach (node; buildNodeList(this).parallel) node._backward(node);
    }

    Value[] parameters()
    {
        return [this];
    }

    ElementType[] parameterValues()
    {
        return [data];
    }

    ElementType[] parameterGrads()
    {
        return [grad];
    }

    auto opBinary(string op)(Value rhs)
    {
        auto result = value(mixin("this.data" ~ op ~ "rhs.data"), [this, rhs]);

        // set backward function
        static if (op == "+" || op == "-")
        {
            result._backward = (x) {
                // get lhs, rhs
                auto lhs = x.parents[0];
                auto rhs = x.parents[1];

                // perform backward operation
                lhs.grad += 1.0 * x.grad;
                rhs.grad += 1.0 * x.grad;
            };
        }
        else static if (op == "*" || op == "/")
        {
            result._backward = (x) {
                // get lhs, rhs
                auto lhs = x.parents[0];
                auto rhs = x.parents[1];

                // perform backward operation
                lhs.grad += rhs.data * x.grad;
                rhs.grad += lhs.data * x.grad;
            };
        }
        else static assert(0, "Operator <"~op~"> not supported!");

        return result;
    }

    auto opBinary(string op)(in ElementType rhs)
    {
        auto result = value(mixin("this.data" ~ op ~ "rhs"), [this, value(rhs)]);

        // set backward function
        static if (op == "+" || op == "-")
        {
            result._backward = (x) {
                // get lhs, rhs
                auto lhs = x.parents[0];
                auto rhs = x.parents[1];

                // perform backward operation
                lhs.grad += 1.0 * x.grad;
                rhs.grad += 1.0 * x.grad;
            };
        }
        else static if (op == "*" || op == "/")
        {
            result._backward = (x) {
                // get lhs, rhs
                auto lhs = x.parents[0];
                auto rhs = x.parents[1];

                // perform backward operation
                lhs.grad += rhs.data * x.grad;
                rhs.grad += lhs.data * x.grad;
            };
        }
        else static assert(0, "Operator <"~op~"> not supported!");

        return result;
    }

    auto opBinaryRight(string op)(in ElementType lhs)
    {
        auto result = value(mixin("this.data" ~ op ~ "lhs"), [this, value(lhs)]);

        // set backward function
        static if (op == "+" || op == "-")
        {
            result._backward = (x) {
                // get lhs, rhs
                auto lhs = x.parents[0];
                auto rhs = x.parents[1];

                // perform backward operation
                lhs.grad += 1.0 * x.grad;
                rhs.grad += 1.0 * x.grad;
            };
        }
        else static if (op == "*" || op == "/")
        {
            result._backward = (x) {
                // get lhs, rhs
                auto lhs = x.parents[0];
                auto rhs = x.parents[1];

                // perform backward operation
                lhs.grad += rhs.data * x.grad;
                rhs.grad += lhs.data * x.grad;
            };
        }
        else static assert(0, "Operator <"~op~"> not supported!");

        return result;
    }

    void opInto(string op)(Value rhs, Value lhs)
    {
        this.data = mixin("rhs.data" ~ op ~ "lhs.data");
        this.parents = [rhs, lhs];

        // set backward function
        static if (op == "+" || op == "-")
        {
            this._backward = (x) {
                // get lhs, rhs
                auto lhs = x.parents[0];
                auto rhs = x.parents[1];

                // perform backward operation
                lhs.grad += 1.0 * x.grad;
                rhs.grad += 1.0 * x.grad;
            };
        }
        else static if (op == "*" || op == "/")
        {
            this._backward = (x) {
                // get lhs, rhs
                auto lhs = x.parents[0];
                auto rhs = x.parents[1];

                // perform backward operation
                lhs.grad += rhs.data * x.grad;
                rhs.grad += lhs.data * x.grad;
            };
        }
        else static assert(0, "Operator <"~op~"> not supported!");
    }

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

unittest
{
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
    h.opInto!"*"(a, b);
    a.zeroGrad();
    b.zeroGrad();
    h.backward();
    assert(h.data == -6);
    assert(b.data == -3);
    assert(a.data == 2);
}

