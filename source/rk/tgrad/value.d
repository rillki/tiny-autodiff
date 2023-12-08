module rk.tgrad.value;

import std.traits : isFloatingPoint;

class Value(T = float) if (isFloatingPoint!T)
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

    void backward() {
        import std.algorithm : canFind;

        // build topo list
        typeof(this)[] topoList;
        buildTopoList(this, topoList);

        // perform backward propagation
        this.grad = 1;
        foreach (child; topoList)
        {   
            child._backward();
        }
    }

    void resetGrads() {
        // build topo list
        typeof(this)[] topoList;
        buildTopoList(this, topoList);

        // reset grads
        foreach (child; topoList)
        {
            child.grad = 0;
        }
    }

    /// for Value 
    typeof(this) opBinary(string op)(ref typeof(this) rhs) 
    {
        auto result = new Value!T(mixin("this.data" ~ op ~ "rhs.data"), [this, rhs]);
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

    /// for numerical values
    typeof(this) opBinary(string op)(in T rhs)
    {
        auto rhs_value = new Value!T(rhs);
        auto result = new Value!T(mixin("this.data" ~ op ~ "rhs_value.data"), [this, rhs_value]);
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

    private void buildTopoList(typeof(this) visitor, ref typeof(this)[] topoList) 
    {
        import std.algorithm : canFind;

        if (!topoList.canFind(visitor)) topoList ~= visitor;
        foreach (v; visitor.children)
        {   
            buildTopoList(v, topoList);
        }
    }
}




