module rk.tautodiff.core.value;

import rk.tautodiff.core.common;
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
        return new Value(data, 0, parents, backward);
    }

    /// Initialize with custom parameters
    auto value(in T data, in char op, Value[] parents, void function(Value) backward = null)
    {
        return new Value(data, op, parents, backward);
    }
}

class Value : INeuron
{
    char op;
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
    this(in ElementType data, in char op, Value[] parents, void function(Value) backward = null) 
    {
        this(data);
        this.op = op;
        this.parents = parents;
        this._backward = backward;
    }

    /// backward operation 
    void backward()
    {
        this.grad = 1;
        foreach (node; buildNodeList(this)) if (node._backward) node._backward(node);
    }

    /// Update variable value from cached `op` and `parents` information
    void update() {
        if (this.parents.length == 2 && this.parents[0] && this.parents[1])
        {
            auto lhs = this.parents[0];
            auto rhs = this.parents[1];
            switch (this.op)
            {
                case '+': this.opInplace!"+"(lhs, rhs); break;
                case '-': this.opInplace!"-"(lhs, rhs); break;
                case '*': this.opInplace!"*"(lhs, rhs); break;
                case '/': this.opInplace!"/"(lhs, rhs); break;
                default: break;
            }
        }
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
        void function(Value) backward = null;
        static if (op == "+") backward = &opBackwardAdd;
        else static if (op == "-") backward = &opBackwardSub;
        else static if (op == "*") backward = &opBackwardMul;
        else static if (op == "/") backward = &opBackwardDiv;
        else static assert(0, "Operator <"~op~"> not supported!");
        
        return value(mixin("this.data" ~ op ~ "rhs.data"), op[0], [this, rhs], backward);
    }

    auto opBinary(string op)(in ElementType rhs)
    {
        void function(Value) backward = null;
        static if (op == "+") backward = &opBackwardAdd;
        else static if (op == "-") backward = &opBackwardSub;
        else static if (op == "*") backward = &opBackwardMul;
        else static if (op == "/") backward = &opBackwardDiv;
        else static assert(0, "Operator <"~op~"> not supported!");
        
        return value(mixin("this.data" ~ op ~ "rhs"), op[0], [this, value(rhs)], backward);
    }

    override string toString() const @safe pure
    {
        import std.string : format;
        return "Value(data=%s, grad=%s, op=%s)".format(this.data, this.grad, this.op);
    }

    auto opBinaryRight(string op)(in ElementType lhs)
    {
        void function(Value) backward = null;
        static if (op == "+") backward = &opBackwardAdd;
        else static if (op == "-") backward = &opBackwardSub;
        else static if (op == "*") backward = &opBackwardMul;
        else static if (op == "/") backward = &opBackwardDiv;
        else static assert(0, "Operator <"~op~"> not supported!");
        
        return value(mixin("lhs" ~ op ~ "this.data"), op[0], [value(lhs), this], backward);
    }

    /// Inplace operations
    void opInplace(string op)(Value lhs, Value rhs)
    {
        void function(Value) backward = null;
        static if (op == "+") backward = &opBackwardAdd;
        else static if (op == "-") backward = &opBackwardSub;
        else static if (op == "*") backward = &opBackwardMul;
        else static if (op == "/") backward = &opBackwardDiv;
        else static assert(0, "Operator <"~op~"> not supported!");
        
        this.reinit(mixin("lhs.data" ~ op ~ "rhs.data"), op[0], [lhs, rhs], backward);
    }
    
    /// Re-initialize object parameters
    void reinit(in ElementType data, in char op, Value[] parents = null, void function(Value) backward = (x){})
    {
        this.op = op;
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

/// Backward operation for addition only
void opBackwardAdd(Value opResult)
{
    auto lhs = opResult.parents[0];
    auto rhs = opResult.parents[1];

    lhs.grad += 1.0 * opResult.grad;
    rhs.grad += 1.0 * opResult.grad;
}

/// Backward operation for subtraction only
void opBackwardSub(Value opResult)
{
    auto lhs = opResult.parents[0];
    auto rhs = opResult.parents[1];

    lhs.grad += 1.0 * opResult.grad;
    rhs.grad += -1.0 * opResult.grad;
}

/// Backward operation for multiplication only
void opBackwardMul(Value opResult)
{
    auto lhs = opResult.parents[0];
    auto rhs = opResult.parents[1];

    lhs.grad += rhs.data * opResult.grad;
    rhs.grad += lhs.data * opResult.grad;
}

/// Backward operation for division only
void opBackwardDiv(Value opResult)
{
    auto lhs = opResult.parents[0];
    auto rhs = opResult.parents[1];

    lhs.grad += (1.0 / rhs.data) * opResult.grad;
    rhs.grad += (-lhs.data / (rhs.data * rhs.data)) * opResult.grad;
}

unittest
{
    // check sum and mul operations
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

    // test sub and div operations
    {
        auto x = value(5.0);
        auto y = value(3.0);
        auto z = x - y;
        z.backward();
        
        assert(z.data == 2.0);
        assert(x.grad == 1.0);
        assert(y.grad == -1.0);
        
        // Test subtraction with scalar (right side)
        x.grad = 0;
        auto w = x - 2.0;
        w.backward();
        
        assert(w.data == 3.0);
        assert(x.grad == 1.0);
        
        // test subtraction with scalar (left side)  
        x.grad = 0;
        auto v = 10.0 - x;
        v.backward();
        
        assert(v.data == 5.0);
        assert(x.grad == -1.0);
        
        // test with a more complex expression
        auto a = value(4.0);
        auto b = value(2.0);
        auto c = value(3.0);
        
        // f = (a - b) * c = (4 - 2) * 3 = 6
        auto f = (a - b) * c;
        f.backward(); 
        
        assert(f.data == 6.0);
        assert(a.grad == 3.0);
        assert(b.grad == -3.0);
        assert(c.grad == 2.0);

        // test division
        auto g = value(6.0);
        auto h = value(2.0);
        auto j = g / h;
        
        j.backward();
        
        import std.math : abs;
        assert(j.data == 3.0);
        assert(abs(g.grad - 0.5) < 1e-10);
        assert(abs(h.grad - (-1.5)) < 1e-10);
    }
}

