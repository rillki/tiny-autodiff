module rk.tgrad.value;

import std.traits : isFloatingPoint, isNumeric;

template value(T = float) if (isFloatingPoint!T)
{
    auto value(S)(in S data) if (isNumeric!S)
    {
        return new Value!(T, S)(data);
    }

    auto value(S)(in S data, Value!(T, S)[] children) if (isNumeric!S)
    {
        return new Value!(T, S)(data, children);
    }

    auto value(S)(in S data, void delegate() _backward) if (isNumeric!S)
    {
        return new Value!(T, S)(data, _backward);
    }
}

class Value(T = float, S) if (isFloatingPoint!T && isNumeric!S)
{
    T data;
    T grad;
    typeof(this)[] children;
    void delegate() _backward;

    alias data this;

    this(in T data) 
    {
        this.data = data;
        this.grad = 0;
        this._backward = (){};
    }

    this(in T data, typeof(this)[] children) 
    {
        this(data);
        this.children = children;
    }

    this(in T data, void delegate() _backward) 
    {
        this(data);
        this._backward = _backward;
    }

    void backward() {
        this.grad = 1;
        foreach (node; buildNodeList(this)) node._backward();
    }

    void resetGrads() {
        foreach (node; buildNodeList(this)) node.grad = 0;
    }

    /// for Value type
    typeof(this) opBinary(string op)(ref typeof(this) rhs) 
    {
        auto result = new typeof(this)(mixin("this.data" ~ op ~ "rhs.data"), [this, rhs]);
        result._backward = () 
        {
            static if (op == "+" || op == "-") 
            {
                this.grad += result.grad;
                rhs.grad += result.grad;
            } 
            else static if (op == "*" || op == "/") 
            {
                this.grad += mixin("rhs.data" ~ op ~ "result.grad");
                rhs.grad += mixin("this.data" ~ op ~ "result.grad");
            } 
            else static assert(0, "Operator <"~op~"> not supported!");
        };
        return result;
    }

    // /// for numerical values
    typeof(this) opBinary(string op)(in T rhs)
    {
        auto rhs_value = new typeof(this)(rhs);
        auto result = new typeof(this)(mixin("this.data" ~ op ~ "rhs_value.data"), [this, rhs_value]);
        result._backward = () 
        {
            static if (op == "+" || op == "-") 
            {
                this.grad += result.grad;
                rhs_value.grad += result.grad;
            } 
            else static if (op == "*" || op == "/") 
            {
                this.grad += mixin("rhs_value.data" ~ op ~ "result.grad");
                rhs_value.grad += mixin("this.data" ~ op ~ "result.grad");
            } 
            else static assert(0, "Operator <"~op~"> not supported!");
        };
        return result;
    }

    private auto buildNodeList(typeof(this) startNode)
    {
        import std.algorithm : canFind;

        // define a list where to save all nodes
        typeof(this)[] nodeList = [];

        // define deep walk funtion to traverse each node and add to list
        void deepWalk(typeof(this) node)
        {
            // add to node list
            if (!nodeList.canFind(node)) nodeList ~= node;

            // traverse each child iteratively
            foreach (child; node.children) deepWalk(child);
        }

        // traverse node tree
        deepWalk(startNode);

        return nodeList;
    }
}

